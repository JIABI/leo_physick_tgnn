"""Training engine for the paper-aligned Intensity--Flow protocol.

This module intentionally does not reuse the legacy ``leo_pg.train.Trainer``.
Paper records contain typed, persistent-edge targets and paper rollouts must let
the fixed controller's executed actions change the simulator state.  The two
training objectives implemented here are therefore:

``one_step``
    Supervised recurrent training over ``PaperEpisodeDataset`` records using
    :func:`leo_pg.train.intensity_flow.intensity_flow_one_step_loss`.

``action_coupled``
    On-policy multi-step training in :class:`PaperAlignedLEOEnv`.  An action at
    epoch ``t`` is selected from descriptors staged before the current model
    call.  The prediction made at ``t`` is aligned and staged for ``t+1``; the
    resulting action is committed through ``step_action`` and therefore changes
    congestion, association history, and every later target.

Every loss coefficient and every enabled schedule is read from the explicit
``paper_training`` configuration (or its normalized ``paper_train`` form).  No
manuscript value is guessed in this file.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Callable, Dict

import numpy as np
import torch
from torch import nn

from leo_pg.control.policy import FixedRankPolicy
from leo_pg.eval.closed_loop import (
    ConstantPolicyStreamInitializer,
    PolicyStreamInitializer,
)
from leo_pg.eval.substitution import (
    carry_policy_stream,
    cold_start_policy_stream,
    persistent_candidate_mask,
)
from leo_pg.models.heads.intensity_flow import IntensityFlowOutput
from leo_pg.sim.paper_environment import PaperAlignedLEOEnv, protocol_fingerprint
from leo_pg.sim.state import (
    ControlObservation,
    PolicyDescriptors,
    SimulatorDescriptors,
)
from leo_pg.train.checkpoint import (
    load_ckpt,
    model_signature_from_config,
    save_ckpt,
)
from leo_pg.train.intensity_flow import (
    IntensityFlowLoss,
    IntensityFlowLossWeights,
    IntensityFlowTarget,
    build_next_step_target,
    intensity_flow_one_step_loss,
)


PAPER_TRAINER_SCHEMA_VERSION = 4
PAPER_TRAINER_LEGACY_SCHEMA_VERSIONS = (3,)
TRAINING_PLAN_FINGERPRINT_VERSION = 2
RESUMABLE_TRAINING_PLAN_FIELDS = ("epochs", "save_dir")

ACTION_BUDGET_REFERENCE = "snapshot_physick_complete_episode_passes"
ACTION_BUDGET_ALLOCATION = "round_half_up_deterministic_seed_subset"
ACTION_ROLLOUT_COVERAGE = "complete_episode_partitioned_into_h_step_windows"


def _mapping(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _required(mapping: Mapping[str, Any], name: str, path: str) -> Any:
    if name not in mapping:
        raise ValueError(f"{path}.{name} is required")
    return mapping[name]


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be an integer")
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return result


@dataclass(frozen=True)
class ScalarSchedule:
    """An explicit epoch schedule used for sampling and objective weights."""

    kind: str
    start: float
    end: float
    start_epoch: int
    end_epoch: int

    @classmethod
    def from_config(
        cls,
        value: Any,
        *,
        path: str,
        lower: float | None = None,
        upper: float | None = None,
    ) -> "ScalarSchedule":
        cfg = _mapping(value, path)
        kind = str(_required(cfg, "kind", path)).strip().lower()
        if kind == "constant":
            constant = _finite(_required(cfg, "value", path), f"{path}.value")
            schedule = cls(kind, constant, constant, 1, 1)
        elif kind in {"linear", "cosine"}:
            start = _finite(_required(cfg, "start", path), f"{path}.start")
            end = _finite(_required(cfg, "end", path), f"{path}.end")
            start_epoch = _positive_int(
                _required(cfg, "start_epoch", path), f"{path}.start_epoch"
            )
            end_epoch = _positive_int(
                _required(cfg, "end_epoch", path), f"{path}.end_epoch"
            )
            if end_epoch < start_epoch:
                raise ValueError(f"{path}.end_epoch must be >= start_epoch")
            schedule = cls(kind, start, end, start_epoch, end_epoch)
        else:
            raise ValueError(
                f"{path}.kind must be constant, linear, or cosine; got {kind!r}"
            )
        for label, number in (("start", schedule.start), ("end", schedule.end)):
            if lower is not None and number < lower:
                raise ValueError(f"{path}.{label} must be >= {lower}")
            if upper is not None and number > upper:
                raise ValueError(f"{path}.{label} must be <= {upper}")
        return schedule

    def value(self, epoch: int) -> float:
        if epoch <= self.start_epoch:
            return self.start
        if epoch >= self.end_epoch:
            return self.end
        span = float(self.end_epoch - self.start_epoch)
        alpha = (float(epoch) - self.start_epoch) / span
        if self.kind == "cosine":
            alpha = 0.5 - 0.5 * math.cos(math.pi * alpha)
        return self.start + alpha * (self.end - self.start)


@dataclass(frozen=True)
class OneStepObjectiveConfig:
    enabled: bool
    weight: ScalarSchedule | None = None
    scheduled_sampling: ScalarSchedule | None = None
    tbptt_steps: int = 1
    new_edge_source: str = "teacher"


@dataclass(frozen=True)
class ActionCoupledObjectiveConfig:
    enabled: bool
    weight: ScalarSchedule | None = None
    scheduled_sampling: ScalarSchedule | None = None
    rollout_horizon: int = 1
    tbptt_steps: int = 1
    rollout_sequence_batch_size: int = 1
    budget_multiplier: float = 1.0
    validation_budget_multiplier: float = 1.0
    budget_reference: str = ACTION_BUDGET_REFERENCE
    budget_allocation: str = ACTION_BUDGET_ALLOCATION
    rollout_coverage: str = ACTION_ROLLOUT_COVERAGE
    new_edge_source: str = "configured_initializer"
    train_seeds: tuple[int, ...] = ()
    validation_seeds: tuple[int, ...] = ()
    step_weights: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ValidationConfig:
    every_epochs: int
    selection_metric: str
    one_step_model_probability: float
    action_coupled_model_probability: float
    one_step_weight: float
    action_coupled_weight: float
    sampling_seed: int


@dataclass(frozen=True)
class PaperTrainingConfig:
    epochs: int
    save_dir: Path
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    clip_grad_norm: float
    batch_size_control_graphs: int
    amp: bool
    amp_dtype: str
    sampling_seed: int
    learning_rate_multiplier: ScalarSchedule
    loss_weights: IntensityFlowLossWeights
    decision_aware_weight: ScalarSchedule
    one_step: OneStepObjectiveConfig
    action_coupled: ActionCoupledObjectiveConfig
    validation: ValidationConfig

    @classmethod
    def from_config(cls, root: Mapping[str, Any]) -> "PaperTrainingConfig":
        cfg = _normalized_training_config(root)
        epochs = _positive_int(_required(cfg, "epochs", "paper_train"), "paper_train.epochs")
        save_dir = Path(str(_required(cfg, "save_dir", "paper_train"))).expanduser()

        optimizer = _mapping(
            _required(cfg, "optimizer", "paper_train"), "paper_train.optimizer"
        )
        optimizer_name = str(
            _required(optimizer, "name", "paper_train.optimizer")
        ).strip().lower()
        if optimizer_name not in {"adam", "adamw"}:
            raise ValueError("paper_train.optimizer.name must be adam or adamw")
        learning_rate = _finite(
            _required(optimizer, "lr", "paper_train.optimizer"),
            "paper_train.optimizer.lr",
        )
        if learning_rate <= 0.0:
            raise ValueError("paper_train.optimizer.lr must be positive")
        weight_decay = _finite(
            _required(optimizer, "weight_decay", "paper_train.optimizer"),
            "paper_train.optimizer.weight_decay",
            nonnegative=True,
        )
        clip_grad_norm = _finite(
            _required(cfg, "clip_grad_norm", "paper_train"),
            "paper_train.clip_grad_norm",
        )
        if clip_grad_norm <= 0.0:
            raise ValueError("paper_train.clip_grad_norm must be positive")
        batch_size_control_graphs = _positive_int(
            _required(cfg, "batch_size_control_graphs", "paper_train"),
            "paper_train.batch_size_control_graphs",
        )

        amp_cfg = _mapping(_required(cfg, "amp", "paper_train"), "paper_train.amp")
        amp = _required(amp_cfg, "enabled", "paper_train.amp")
        if not isinstance(amp, bool):
            raise TypeError("paper_train.amp.enabled must be boolean")
        amp_dtype = str(_required(amp_cfg, "dtype", "paper_train.amp")).lower()
        if amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("paper_train.amp.dtype must be float16 or bfloat16")

        sampling_seed_raw = _required(cfg, "sampling_seed", "paper_train")
        if isinstance(sampling_seed_raw, bool) or int(sampling_seed_raw) < 0:
            raise ValueError("paper_train.sampling_seed must be a non-negative integer")
        sampling_seed = int(sampling_seed_raw)
        learning_rate_multiplier = ScalarSchedule.from_config(
            _required(cfg, "learning_rate_multiplier", "paper_train"),
            path="paper_train.learning_rate_multiplier",
            lower=0.0,
        )

        loss_cfg = _mapping(
            _required(cfg, "loss_weights", "paper_train"),
            "paper_train.loss_weights",
        )
        loss_weights = IntensityFlowLossWeights(
            gamma=_finite(
                _required(loss_cfg, "gamma", "paper_train.loss_weights"),
                "paper_train.loss_weights.gamma",
                nonnegative=True,
            ),
            intensity=_finite(
                _required(loss_cfg, "intensity", "paper_train.loss_weights"),
                "paper_train.loss_weights.intensity",
                nonnegative=True,
            ),
            flow=_finite(
                _required(loss_cfg, "flow", "paper_train.loss_weights"),
                "paper_train.loss_weights.flow",
                nonnegative=True,
            ),
            feasibility=_finite(
                _required(loss_cfg, "feasibility", "paper_train.loss_weights"),
                "paper_train.loss_weights.feasibility",
                nonnegative=True,
            ),
        )
        decision_schedule = ScalarSchedule.from_config(
            _required(cfg, "decision_aware_weight", "paper_train"),
            path="paper_train.decision_aware_weight",
            lower=0.0,
        )

        objectives = _mapping(
            _required(cfg, "objectives", "paper_train"), "paper_train.objectives"
        )
        one_step = cls._parse_one_step(
            _required(objectives, "one_step", "paper_train.objectives")
        )
        action_coupled = cls._parse_action_coupled(
            _required(objectives, "action_coupled", "paper_train.objectives"),
            batch_size_control_graphs=batch_size_control_graphs,
        )
        if not one_step.enabled and not action_coupled.enabled:
            raise ValueError("at least one paper training objective must be enabled")
        if set(action_coupled.train_seeds).intersection(action_coupled.validation_seeds):
            raise ValueError("action-coupled train and validation seeds must be disjoint")

        validation_cfg = _mapping(
            _required(cfg, "validation", "paper_train"), "paper_train.validation"
        )
        selection_metric = str(
            _required(validation_cfg, "selection_metric", "paper_train.validation")
        ).strip().lower()
        if selection_metric not in {"one_step", "action_coupled", "combined"}:
            raise ValueError(
                "paper_train.validation.selection_metric must be one_step, "
                "action_coupled, or combined"
            )
        validation = ValidationConfig(
            every_epochs=_positive_int(
                _required(validation_cfg, "every_epochs", "paper_train.validation"),
                "paper_train.validation.every_epochs",
            ),
            selection_metric=selection_metric,
            one_step_model_probability=cls._probability(
                _required(
                    validation_cfg,
                    "one_step_model_probability",
                    "paper_train.validation",
                ),
                "paper_train.validation.one_step_model_probability",
            ),
            action_coupled_model_probability=cls._probability(
                _required(
                    validation_cfg,
                    "action_coupled_model_probability",
                    "paper_train.validation",
                ),
                "paper_train.validation.action_coupled_model_probability",
            ),
            one_step_weight=_finite(
                _required(validation_cfg, "one_step_weight", "paper_train.validation"),
                "paper_train.validation.one_step_weight",
                nonnegative=True,
            ),
            action_coupled_weight=_finite(
                _required(
                    validation_cfg,
                    "action_coupled_weight",
                    "paper_train.validation",
                ),
                "paper_train.validation.action_coupled_weight",
                nonnegative=True,
            ),
            sampling_seed=int(
                _required(validation_cfg, "sampling_seed", "paper_train.validation")
            ),
        )
        if validation.sampling_seed < 0:
            raise ValueError("paper_train.validation.sampling_seed must be non-negative")
        if selection_metric == "combined":
            if validation.one_step_weight + validation.action_coupled_weight <= 0.0:
                raise ValueError("combined validation weights must have a positive sum")

        return cls(
            epochs=epochs,
            save_dir=save_dir,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
            batch_size_control_graphs=batch_size_control_graphs,
            amp=amp,
            amp_dtype=amp_dtype,
            sampling_seed=sampling_seed,
            learning_rate_multiplier=learning_rate_multiplier,
            loss_weights=loss_weights,
            decision_aware_weight=decision_schedule,
            one_step=one_step,
            action_coupled=action_coupled,
            validation=validation,
        )

    @staticmethod
    def _probability(value: Any, path: str) -> float:
        result = _finite(value, path)
        if not 0.0 <= result <= 1.0:
            raise ValueError(f"{path} must be in [0,1]")
        return result

    @classmethod
    def _parse_one_step(cls, value: Any) -> OneStepObjectiveConfig:
        path = "paper_train.objectives.one_step"
        cfg = _mapping(value, path)
        enabled = _required(cfg, "enabled", path)
        if not isinstance(enabled, bool):
            raise TypeError(f"{path}.enabled must be boolean")
        if not enabled:
            return OneStepObjectiveConfig(enabled=False)
        new_edge_source = str(_required(cfg, "new_edge_source", path)).lower()
        if new_edge_source not in {"teacher", "configured_initializer"}:
            raise ValueError(
                f"{path}.new_edge_source must be teacher or configured_initializer"
            )
        return OneStepObjectiveConfig(
            enabled=True,
            weight=ScalarSchedule.from_config(
                _required(cfg, "weight", path), path=f"{path}.weight", lower=0.0
            ),
            scheduled_sampling=ScalarSchedule.from_config(
                _required(cfg, "scheduled_sampling", path),
                path=f"{path}.scheduled_sampling",
                lower=0.0,
                upper=1.0,
            ),
            tbptt_steps=_positive_int(
                _required(cfg, "tbptt_steps", path), f"{path}.tbptt_steps"
            ),
            new_edge_source=new_edge_source,
        )

    @classmethod
    def _parse_action_coupled(
        cls,
        value: Any,
        *,
        batch_size_control_graphs: int,
    ) -> ActionCoupledObjectiveConfig:
        path = "paper_train.objectives.action_coupled"
        cfg = _mapping(value, path)
        enabled = _required(cfg, "enabled", path)
        if not isinstance(enabled, bool):
            raise TypeError(f"{path}.enabled must be boolean")
        if not enabled:
            return ActionCoupledObjectiveConfig(enabled=False)
        new_edge_source = str(_required(cfg, "new_edge_source", path)).lower()
        if new_edge_source not in {"teacher", "configured_initializer"}:
            raise ValueError(
                f"{path}.new_edge_source must be teacher or configured_initializer"
            )
        train_seeds = cls._seed_list(_required(cfg, "train_seeds", path), f"{path}.train_seeds")
        validation_seeds = cls._seed_list(
            _required(cfg, "validation_seeds", path), f"{path}.validation_seeds"
        )
        step_weights = _mapping(
            _required(cfg, "step_weights", path), f"{path}.step_weights"
        )
        rollout_horizon = _positive_int(
            _required(cfg, "rollout_horizon", path), f"{path}.rollout_horizon"
        )
        if rollout_horizon < 2:
            raise ValueError(
                f"{path}.rollout_horizon must be at least 2 so a staged prediction "
                "can affect a later action and simulator state"
            )
        # Validate the scheme now; the horizon-specific tensor is built later.
        build_multistep_weights(
            rollout_horizon,
            step_weights,
            device=torch.device("cpu"),
        )
        tbptt_steps = _positive_int(
            _required(cfg, "tbptt_steps", path), f"{path}.tbptt_steps"
        )
        if tbptt_steps != rollout_horizon:
            raise ValueError(
                f"{path}.tbptt_steps must equal rollout_horizon: H defines each "
                "truncated-BPTT window in the paper action-coupled protocol"
            )
        if rollout_horizon > batch_size_control_graphs:
            raise ValueError(
                f"{path}.rollout_horizon cannot exceed "
                "paper_train.batch_size_control_graphs"
            )
        sequence_batch_size = _positive_int(
            cfg.get(
                "rollout_sequence_batch_size",
                max(1, batch_size_control_graphs // rollout_horizon),
            ),
            f"{path}.rollout_sequence_batch_size",
        )
        if sequence_batch_size * rollout_horizon > batch_size_control_graphs:
            raise ValueError(
                f"{path}.rollout_sequence_batch_size * rollout_horizon must not "
                "exceed paper_train.batch_size_control_graphs"
            )
        budget_multiplier = _finite(
            cfg.get("budget_multiplier", 1.0),
            f"{path}.budget_multiplier",
        )
        if budget_multiplier <= 0.0:
            raise ValueError(f"{path}.budget_multiplier must be positive")
        validation_budget_multiplier = _finite(
            cfg.get("validation_budget_multiplier", 1.0),
            f"{path}.validation_budget_multiplier",
        )
        if validation_budget_multiplier != 1.0:
            raise ValueError(
                f"{path}.validation_budget_multiplier must be 1.0: validation "
                "runs every configured seed exactly once"
            )
        budget_reference = str(
            cfg.get("budget_reference", ACTION_BUDGET_REFERENCE)
        ).strip().lower()
        if budget_reference != ACTION_BUDGET_REFERENCE:
            raise ValueError(
                f"{path}.budget_reference must be {ACTION_BUDGET_REFERENCE!r}"
            )
        budget_allocation = str(
            cfg.get("budget_allocation", ACTION_BUDGET_ALLOCATION)
        ).strip().lower()
        if budget_allocation != ACTION_BUDGET_ALLOCATION:
            raise ValueError(
                f"{path}.budget_allocation must be {ACTION_BUDGET_ALLOCATION!r}"
            )
        rollout_coverage = str(
            cfg.get("rollout_coverage", ACTION_ROLLOUT_COVERAGE)
        ).strip().lower()
        if rollout_coverage != ACTION_ROLLOUT_COVERAGE:
            raise ValueError(
                f"{path}.rollout_coverage must be {ACTION_ROLLOUT_COVERAGE!r}"
            )
        return ActionCoupledObjectiveConfig(
            enabled=True,
            weight=ScalarSchedule.from_config(
                _required(cfg, "weight", path), path=f"{path}.weight", lower=0.0
            ),
            scheduled_sampling=ScalarSchedule.from_config(
                _required(cfg, "scheduled_sampling", path),
                path=f"{path}.scheduled_sampling",
                lower=0.0,
                upper=1.0,
            ),
            rollout_horizon=rollout_horizon,
            tbptt_steps=tbptt_steps,
            rollout_sequence_batch_size=sequence_batch_size,
            budget_multiplier=budget_multiplier,
            validation_budget_multiplier=validation_budget_multiplier,
            budget_reference=budget_reference,
            budget_allocation=budget_allocation,
            rollout_coverage=rollout_coverage,
            new_edge_source=new_edge_source,
            train_seeds=train_seeds,
            validation_seeds=validation_seeds,
            step_weights=step_weights,
        )

    @staticmethod
    def _seed_list(value: Any, path: str) -> tuple[int, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"{path} must be a list of non-negative integers")
        seeds = []
        for index, seed in enumerate(value):
            if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
                raise ValueError(f"{path}[{index}] must be a non-negative integer")
            seeds.append(int(seed))
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{path} must not contain duplicates")
        return tuple(seeds)


def _normalized_training_config(root: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize the released flat config without inventing experiment values.

    ``paper_train`` is the strict internal form used by this trainer.  The
    repository's user-facing formal configuration predates it and stores the
    same choices under ``paper_training``.  This adapter is deliberately
    mechanical: disabled schedules become an explicit zero schedule, scalar
    weights become constant schedules, and action-coupled seeds remain empty so
    that the immutable dataset split manifest supplies them later.
    """

    if "paper_train" in root:
        return _mapping(root["paper_train"], "paper_train")
    raw = _mapping(
        _required(root, "paper_training", "config"), "paper_training"
    )
    optimizer_name = str(_required(raw, "optimizer", "paper_training"))
    loss_weights = _mapping(
        _required(raw, "loss_weights", "paper_training"),
        "paper_training.loss_weights",
    )
    objective = _mapping(
        _required(raw, "objective", "paper_training"),
        "paper_training.objective",
    )
    sampling = _mapping(
        _required(raw, "scheduled_sampling", "paper_training"),
        "paper_training.scheduled_sampling",
    )
    action = _mapping(
        _required(raw, "action_coupled_multistep", "paper_training"),
        "paper_training.action_coupled_multistep",
    )
    checkpoint_selection = str(
        _required(raw, "checkpoint_selection", "paper_training")
    ).strip().lower()
    if checkpoint_selection != "minimum_validation_loss":
        raise ValueError(
            "paper_training.checkpoint_selection must be minimum_validation_loss"
        )
    baseline = root.get("paper_baseline", {})
    baseline_training: Dict[str, Any] = {}
    if isinstance(baseline, Mapping) and isinstance(baseline.get("training"), Mapping):
        baseline_training = dict(baseline["training"])

    objective_kind = str(objective.get("kind", "one_step_teacher_forcing")).lower()
    baseline_objective = str(baseline_training.get("objective", "")).lower()
    if baseline_objective:
        action_enabled = "action_coupled" in baseline_objective
        one_step_enabled = "one_step" in baseline_objective
    else:
        action_enabled = bool(action.get("enabled", False)) or "action_coupled" in objective_kind
        one_step_enabled = "one_step" in objective_kind
    if not one_step_enabled and not action_enabled:
        raise ValueError(
            "paper_training must enable one-step or action-coupled training"
        )

    sampling_enabled = sampling.get("enabled", False)
    if not isinstance(sampling_enabled, bool):
        raise TypeError("paper_training.scheduled_sampling.enabled must be boolean")
    if sampling_enabled:
        schedule_name = str(_required(sampling, "schedule", "scheduled_sampling")).lower()
        if schedule_name not in {"linear", "cosine"}:
            raise ValueError(
                "paper_training.scheduled_sampling.schedule must be linear or cosine"
            )
        sampling_schedule: Dict[str, Any] = {
            "kind": schedule_name,
            "start": _required(sampling, "start_probability", "scheduled_sampling"),
            "end": _required(sampling, "maximum_probability", "scheduled_sampling"),
            # Training epochs are one-based; an explicit epoch zero means the
            # schedule is already active at the first optimizer epoch.
            "start_epoch": max(1, int(_required(sampling, "start_epoch", "scheduled_sampling"))),
            "end_epoch": max(1, int(_required(sampling, "end_epoch", "scheduled_sampling"))),
        }
    else:
        sampling_schedule = {"kind": "constant", "value": 0.0}

    feedback_probability = action.get("model_feedback_probability")
    if feedback_probability is not None:
        action_sampling_schedule: Dict[str, Any] = {
            "kind": "constant",
            "value": _finite(
                feedback_probability,
                "paper_training.action_coupled_multistep.model_feedback_probability",
            ),
        }
    elif sampling_enabled:
        action_sampling_schedule = dict(sampling_schedule)
    else:
        # Disabling scheduled sampling means no annealing. The genuinely
        # action-coupled objective still uses full staged model feedback rather
        # than silently becoming an oracle/teacher rollout.
        action_sampling_schedule = {"kind": "constant", "value": 1.0}

    # A one-step objective commonly declares ``rollout_horizon: 1``.  It must
    # not silently override an enabled action-coupled objective.  Method-level
    # settings are most specific; otherwise use the explicit action setting or
    # the last value of its declared horizon ladder (5/10/20 in the released
    # protocol).  The one-step value is only a final compatibility fallback.
    configured_horizon = baseline_training.get("rollout_horizon")
    if configured_horizon is None and action_enabled:
        configured_horizon = action.get("rollout_horizon")
    if configured_horizon is None and action_enabled:
        horizons = action.get("horizons")
        if (
            isinstance(horizons, Sequence)
            and not isinstance(horizons, (str, bytes))
            and horizons
        ):
            configured_horizon = horizons[-1]
    if configured_horizon is None:
        configured_horizon = objective.get("rollout_horizon")
    if action_enabled and configured_horizon is None:
        raise ValueError(
            "action-coupled training requires an explicit rollout_horizon"
        )
    rollout_horizon = int(configured_horizon or 1)
    batch_size_control_graphs = _positive_int(
        _required(raw, "batch_size_control_graphs", "paper_training"),
        "paper_training.batch_size_control_graphs",
    )
    rollout_sequence_batch_size = int(
        baseline_training.get(
            "rollout_sequence_batch_size",
            max(1, batch_size_control_graphs // max(1, rollout_horizon)),
        )
    )
    budget_multiplier = _finite(
        baseline_training.get("budget_multiplier", 1.0),
        "paper_baseline.training.budget_multiplier",
    )
    if budget_multiplier <= 0.0:
        raise ValueError(
            "paper_baseline.training.budget_multiplier must be positive"
        )
    discount = _finite(
        action.get("discount", 1.0),
        "paper_training.action_coupled_multistep.discount",
        nonnegative=True,
    )
    if discount == 1.0:
        step_weights: Dict[str, Any] = {"scheme": "uniform", "normalize": True}
    else:
        step_weights = {
            "scheme": "discount",
            "discount": discount,
            "normalize": True,
        }

    decision_value = _finite(
        _required(loss_weights, "decision_aware", "paper_training.loss_weights"),
        "paper_training.loss_weights.decision_aware",
        nonnegative=True,
    )
    if baseline_objective != "decision_aware_action_coupled":
        decision_value = 0.0
    validation_metric = (
        "action_coupled"
        if action_enabled and not one_step_enabled
        else "combined"
        if action_enabled
        else "one_step"
    )
    one_step_validation_probability = float(
        sampling_schedule.get("value", sampling_schedule.get("end", 0.0))
    )
    action_validation_probability = float(
        action_sampling_schedule.get(
            "value", action_sampling_schedule.get("end", 1.0)
        )
    )
    raw_lr_schedule = _required(raw, "learning_rate_schedule", "paper_training")
    if isinstance(raw_lr_schedule, str):
        if raw_lr_schedule.strip().lower() != "constant":
            raise ValueError(
                "string paper_training.learning_rate_schedule must be 'constant'; "
                "use a mapping for an annealed schedule"
            )
        learning_rate_multiplier: Dict[str, Any] = {
            "kind": "constant",
            "value": 1.0,
        }
    elif isinstance(raw_lr_schedule, Mapping):
        learning_rate_multiplier = dict(raw_lr_schedule)
    else:
        raise TypeError(
            "paper_training.learning_rate_schedule must be 'constant' or a schedule mapping"
        )
    normalized: Dict[str, Any] = {
        "epochs": _required(raw, "epochs", "paper_training"),
        "save_dir": _required(raw, "output_dir", "paper_training"),
        "optimizer": {
            "name": optimizer_name,
            "lr": _required(raw, "learning_rate", "paper_training"),
            "weight_decay": _required(raw, "weight_decay", "paper_training"),
        },
        "clip_grad_norm": _required(raw, "gradient_clip_norm", "paper_training"),
        "batch_size_control_graphs": batch_size_control_graphs,
        "amp": {
            "enabled": bool(raw.get("amp_enabled", False)),
            "dtype": str(raw.get("amp_dtype", "float16")),
        },
        "sampling_seed": int(raw.get("sampling_seed", root.get("seed", 0))),
        "learning_rate_multiplier": learning_rate_multiplier,
        "loss_weights": {
            name: _required(loss_weights, name, "paper_training.loss_weights")
            for name in ("gamma", "intensity", "flow", "feasibility")
        },
        "decision_aware_weight": {"kind": "constant", "value": decision_value},
        "objectives": {
            "one_step": {
                "enabled": one_step_enabled,
                "weight": {"kind": "constant", "value": 1.0},
                "scheduled_sampling": dict(sampling_schedule),
                "tbptt_steps": int(raw.get("tbptt_steps", 1)),
                "new_edge_source": "teacher",
            },
            "action_coupled": {
                "enabled": action_enabled,
                "weight": {
                    "kind": "constant",
                    "value": _finite(
                        action.get("state_loss_weight", 1.0),
                        "paper_training.action_coupled_multistep.state_loss_weight",
                        nonnegative=True,
                    ),
                },
                "scheduled_sampling": action_sampling_schedule,
                "rollout_horizon": rollout_horizon,
                "tbptt_steps": int(action.get("tbptt_steps", rollout_horizon)),
                "rollout_sequence_batch_size": rollout_sequence_batch_size,
                "budget_multiplier": budget_multiplier,
                "validation_budget_multiplier": 1.0,
                "budget_reference": ACTION_BUDGET_REFERENCE,
                "budget_allocation": ACTION_BUDGET_ALLOCATION,
                "rollout_coverage": ACTION_ROLLOUT_COVERAGE,
                "new_edge_source": "configured_initializer",
                # Empty means: use the dataset manifest's immutable train/val
                # episode seed assignments. The trainer rejects missing
                # manifest seeds instead of synthesizing replacements.
                "train_seeds": list(action.get("train_seeds", [])),
                "validation_seeds": list(action.get("validation_seeds", [])),
                "step_weights": step_weights,
            },
        },
        "validation": {
            "every_epochs": int(raw.get("validation_every_epochs", 1)),
            "selection_metric": validation_metric,
            "one_step_model_probability": one_step_validation_probability,
            "action_coupled_model_probability": action_validation_probability,
            "one_step_weight": 1.0 if one_step_enabled else 0.0,
            "action_coupled_weight": 1.0 if action_enabled else 0.0,
            "sampling_seed": int(raw.get("validation_sampling_seed", root.get("seed", 0))),
        },
    }
    return normalized


def _canonical_mapping_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _semantic_training_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the immutable optimizer/objective/batch/RNG part of a plan."""

    semantic_plan = copy.deepcopy(dict(plan))
    for name in RESUMABLE_TRAINING_PLAN_FIELDS:
        if name not in semantic_plan:
            raise ValueError(f"paper training plan is missing resumable field {name!r}")
        semantic_plan.pop(name)
    return semantic_plan


def _resumable_training_fields(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize the small set of fields that may change during resume."""

    source = dict(plan)
    missing = [name for name in RESUMABLE_TRAINING_PLAN_FIELDS if name not in source]
    if missing:
        raise ValueError(
            "paper training plan is missing resumable fields: " + ", ".join(missing)
        )
    epochs = source["epochs"]
    if isinstance(epochs, bool) or int(epochs) != epochs or int(epochs) <= 0:
        raise ValueError("paper training plan epochs must be a positive integer")
    return {
        "epochs": int(epochs),
        "save_dir": str(Path(str(source["save_dir"])).expanduser()),
    }


def _training_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    """Hash immutable training semantics, excluding only documented mutable fields."""

    return _canonical_mapping_fingerprint(_semantic_training_plan(plan))


def _legacy_full_training_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    """Fingerprint emitted before epochs/save_dir were made resumable."""

    return _canonical_mapping_fingerprint(copy.deepcopy(dict(plan)))


def _dataset_semantics(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind dataset bytes/contracts/splits while allowing file relocation."""

    semantics = copy.deepcopy(dict(metadata))
    semantics.pop("path", None)
    return semantics


def _dataset_semantic_fingerprint(metadata: Mapping[str, Any]) -> str:
    return _canonical_mapping_fingerprint(_dataset_semantics(metadata))


def _action_environment_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Select simulator/controller fields consumed by action-coupled rollouts."""

    return {
        name: copy.deepcopy(config[name])
        for name in ("K_users", "S_sats", "ephemeris", "paper_protocol")
        if name in config
    }


def _action_environment_fingerprint(config: Mapping[str, Any]) -> str:
    # protocol_fingerprint also binds configured callables and TLE file bytes.
    return protocol_fingerprint(_action_environment_config(config))


def _validated_resume_plans(
    *,
    trainer_state: Mapping[str, Any],
    current_plan: Mapping[str, Any],
    completed_epoch: int,
    schema_version: int,
) -> None:
    """Validate a current or v3 training plan without weakening old migrations."""

    saved_plan_value = trainer_state.get("training_plan")
    if not isinstance(saved_plan_value, Mapping):
        raise ValueError("resume checkpoint has no paper training plan")
    saved_plan = copy.deepcopy(dict(saved_plan_value))
    current = copy.deepcopy(dict(current_plan))
    saved_semantic = _semantic_training_plan(saved_plan)
    current_semantic = _semantic_training_plan(current)
    saved_mutable = _resumable_training_fields(saved_plan)
    current_mutable = _resumable_training_fields(current)

    stored_fingerprint = trainer_state.get("training_plan_fingerprint")
    if not isinstance(stored_fingerprint, str):
        raise ValueError(
            "checkpoint does not contain a paper training-plan fingerprint"
        )
    fingerprint_version = trainer_state.get("training_plan_fingerprint_version")
    semantic_fingerprint = _training_plan_fingerprint(saved_plan)
    legacy_fingerprint = _legacy_full_training_plan_fingerprint(saved_plan)
    if (
        fingerprint_version is None
        and schema_version in PAPER_TRAINER_LEGACY_SCHEMA_VERSIONS
    ):
        if stored_fingerprint not in {semantic_fingerprint, legacy_fingerprint}:
            raise ValueError("checkpoint paper training-plan fingerprint is invalid")
    elif fingerprint_version == TRAINING_PLAN_FINGERPRINT_VERSION:
        if stored_fingerprint != semantic_fingerprint:
            raise ValueError("checkpoint paper training-plan fingerprint is invalid")
    elif (
        fingerprint_version == 1
        and schema_version in PAPER_TRAINER_LEGACY_SCHEMA_VERSIONS
    ):
        if stored_fingerprint != legacy_fingerprint:
            raise ValueError("checkpoint paper training-plan fingerprint is invalid")
    else:
        raise ValueError(
            "unsupported paper training-plan fingerprint version: "
            f"{fingerprint_version!r}"
        )

    if schema_version == PAPER_TRAINER_SCHEMA_VERSION:
        immutable_value = trainer_state.get("immutable_training_plan")
        immutable_fingerprint = trainer_state.get(
            "immutable_training_plan_fingerprint"
        )
        resume_fields = trainer_state.get("resumable_training_fields")
        if not isinstance(immutable_value, Mapping):
            raise ValueError("resume checkpoint has no immutable training plan")
        if dict(immutable_value) != saved_semantic:
            raise ValueError("resume checkpoint immutable training plan is inconsistent")
        if immutable_fingerprint != semantic_fingerprint:
            raise ValueError(
                "resume checkpoint immutable training-plan fingerprint is invalid"
            )
        if not isinstance(resume_fields, Mapping):
            raise ValueError("resume checkpoint has no resumable training fields")
        if resume_fields.get("epochs") != saved_mutable["epochs"]:
            raise ValueError("resume checkpoint epoch budget metadata is inconsistent")
        if resume_fields.get("save_dir") != saved_mutable["save_dir"]:
            raise ValueError("resume checkpoint save-dir metadata is inconsistent")
        for name in ("best_checkpoint_path", "last_checkpoint_path"):
            value = resume_fields.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"resume checkpoint has invalid mutable path metadata {name!r}"
                )

    if saved_semantic != current_semantic:
        raise ValueError("resume checkpoint paper training plan changed")
    if completed_epoch < 0:
        raise ValueError("resume checkpoint completed epoch must be non-negative")
    if saved_mutable["epochs"] < completed_epoch:
        raise ValueError(
            "resume checkpoint completed epoch exceeds its saved epoch budget"
        )
    if current_mutable["epochs"] < completed_epoch:
        raise ValueError(
            "resume epoch budget cannot be below the completed checkpoint epoch "
            f"({current_mutable['epochs']} < {completed_epoch})"
        )


def _is_compatible_best_checkpoint(
    path: Path,
    *,
    model_signature: Mapping[str, Any],
    paper_method: str,
    model_class: str,
    best_epoch: int,
    training_semantics: Mapping[str, Any],
    dataset_semantics: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    if not isinstance(payload, Mapping):
        return False
    if payload.get("model_signature") != dict(model_signature):
        return False
    state = payload.get("trainer_state")
    if not isinstance(state, Mapping):
        return False
    if state.get("paper_method") != paper_method or state.get("model_class") != model_class:
        return False
    if state.get("epoch") != best_epoch or state.get("best_epoch") != best_epoch:
        return False
    candidate_plan = state.get("training_plan")
    candidate_dataset = state.get("dataset")
    if not isinstance(candidate_plan, Mapping) or not isinstance(
        candidate_dataset, Mapping
    ):
        return False
    return (
        _semantic_training_plan(candidate_plan) == dict(training_semantics)
        and _dataset_semantics(candidate_dataset) == dict(dataset_semantics)
    )


def _budgeted_seed_passes(
    seeds: Sequence[int],
    *,
    multiplier: float,
    generator: torch.Generator,
) -> tuple[int, ...]:
    """Schedule complete episode passes for an exact relative training budget.

    Every integer part is a complete pass over ``seeds``.  The fractional part
    selects a deterministic, without-replacement subset after round-half-up at
    the episode-pass level.  A final deterministic permutation prevents the
    fractional subset from always appearing at the end of an epoch.
    """

    if not seeds:
        raise ValueError("action-coupled training budget requires non-empty seeds")
    value = _finite(multiplier, "action-coupled budget_multiplier")
    if value <= 0.0:
        raise ValueError("action-coupled budget_multiplier must be positive")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if any(seed < 0 for seed in normalized_seeds):
        raise ValueError("action-coupled training seeds must be non-negative")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("action-coupled training seeds must be unique")

    complete_passes = int(math.floor(value))
    target_passes = max(
        1,
        int(math.floor(value * len(normalized_seeds) + 0.5)),
    )
    expanded = [
        seed
        for _pass_index in range(complete_passes)
        for seed in normalized_seeds
    ]
    fractional_count = target_passes - len(expanded)
    if fractional_count < 0 or fractional_count > len(normalized_seeds):
        raise RuntimeError("invalid fractional action-coupled budget allocation")
    if fractional_count:
        selection = torch.randperm(
            len(normalized_seeds), generator=generator
        ).tolist()
        expanded.extend(normalized_seeds[index] for index in selection[:fractional_count])
    order = torch.randperm(len(expanded), generator=generator).tolist()
    return tuple(expanded[index] for index in order)


def build_multistep_weights(
    length: int,
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build explicit per-transition weights for an action-coupled rollout."""

    length = _positive_int(length, "rollout length")
    cfg = _mapping(config, "step_weights")
    scheme = str(_required(cfg, "scheme", "step_weights")).lower()
    normalize = _required(cfg, "normalize", "step_weights")
    if not isinstance(normalize, bool):
        raise TypeError("step_weights.normalize must be boolean")
    index = torch.arange(length, dtype=torch.float32, device=device)
    if scheme == "uniform":
        weights = torch.ones(length, dtype=torch.float32, device=device)
    elif scheme == "poly":
        power = _finite(_required(cfg, "power", "step_weights"), "step_weights.power")
        weights = (index + 1.0).pow(power)
    elif scheme == "exp":
        beta = _finite(_required(cfg, "beta", "step_weights"), "step_weights.beta")
        denominator = float(max(1, length - 1))
        weights = torch.exp(beta * index / denominator)
    elif scheme == "discount":
        discount = _finite(
            _required(cfg, "discount", "step_weights"),
            "step_weights.discount",
            nonnegative=True,
        )
        weights = torch.pow(
            torch.full_like(index, discount),
            index,
        )
    elif scheme == "explicit":
        values = _required(cfg, "values", "step_weights")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError("step_weights.values must be a numeric list")
        if len(values) != length:
            raise ValueError(
                f"step_weights.values must have {length} entries, got {len(values)}"
            )
        weights = torch.tensor(
            [
                _finite(value, f"step_weights.values[{i}]", nonnegative=True)
                for i, value in enumerate(values)
            ],
            dtype=torch.float32,
            device=device,
        )
    else:
        raise ValueError(
            "step_weights.scheme must be uniform, poly, exp, discount, or explicit"
        )
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("step weights must be finite and non-negative")
    total = weights.sum()
    if float(total.item()) <= 0.0:
        raise ValueError("step weights must have a positive sum")
    return weights / total if normalize else weights


def _move_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {str(key): _move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    return value


def _policy_from_payload(
    payload: Mapping[str, Any], device: torch.device, *, path: str
) -> PolicyDescriptors:
    cfg = _mapping(payload, path)
    descriptors = PolicyDescriptors(
        gamma_edge=torch.as_tensor(_required(cfg, "gamma_edge", path), device=device),
        intensity_edge=torch.as_tensor(
            _required(cfg, "intensity_edge", path), device=device
        ),
        flow_node=torch.as_tensor(_required(cfg, "flow_node", path), device=device),
    )
    return descriptors


def observation_from_record(
    record: Mapping[str, Any], device: torch.device | str
) -> ControlObservation:
    """Materialize a serialized paper record as a validated observation."""

    target_device = torch.device(device)
    raw = record.get("observation", record)
    payload = _mapping(raw, "record.observation")
    observation_id = _required(payload, "observation_id", "record.observation")
    if not isinstance(observation_id, Sequence) or len(observation_id) != 2:
        raise ValueError("record.observation.observation_id must contain seed and epoch")
    sim_payload = _mapping(
        _required(payload, "sim_descriptors", "record.observation"),
        "record.observation.sim_descriptors",
    )
    policy_payload = _mapping(
        _required(payload, "policy_descriptors", "record.observation"),
        "record.observation.policy_descriptors",
    )
    edge_type = torch.as_tensor(
        _required(payload, "edge_type", "record.observation"),
        device=target_device,
    )
    if edge_type.ndim != 1 or bool((edge_type != 0).any()):
        raise ValueError("paper observation edge_type must be a one-dimensional zero vector")
    observation = ControlObservation(
        observation_id=(int(observation_id[0]), int(observation_id[1])),
        node_x=torch.as_tensor(
            _required(payload, "node_x", "record.observation"), device=target_device
        ),
        candidate_edge_index=torch.as_tensor(
            _required(payload, "edge_index", "record.observation"),
            device=target_device,
        ).long(),
        candidate_edge_ids=torch.as_tensor(
            _required(payload, "candidate_edge_ids", "record.observation"),
            device=target_device,
        ).long(),
        edge_features=torch.as_tensor(
            _required(payload, "edge_z", "record.observation"), device=target_device
        ),
        elevation_deg=torch.as_tensor(
            _required(payload, "elevation_deg", "record.observation"),
            device=target_device,
        ),
        sim_descriptors=SimulatorDescriptors(
            policy_fields=_policy_from_payload(
                sim_payload,
                target_device,
                path="record.observation.sim_descriptors",
            ),
            feasible_edge=torch.as_tensor(
                _required(
                    sim_payload,
                    "feasible_edge",
                    "record.observation.sim_descriptors",
                ),
                device=target_device,
            ).bool(),
        ),
        policy_descriptors=_policy_from_payload(
            policy_payload,
            target_device,
            path="record.observation.policy_descriptors",
        ),
        current_serving=torch.as_tensor(
            _required(payload, "current_serving", "record.observation"),
            device=target_device,
        ).long(),
        hold_steps=torch.as_tensor(
            _required(payload, "hold_steps", "record.observation"),
            device=target_device,
        ).long(),
        user_order=torch.as_tensor(
            _required(payload, "user_order", "record.observation"),
            device=target_device,
        ).long(),
        meta=_move_tree(
            _mapping(_required(payload, "meta", "record.observation"), "record.observation.meta"),
            target_device,
        ),
    )
    observation.validate()
    if int(payload.get("epoch", observation.epoch)) != observation.epoch:
        raise ValueError("serialized epoch disagrees with observation_id")
    if int(payload.get("user_count", observation.user_count)) != observation.user_count:
        raise ValueError("serialized user_count disagrees with tensor shapes")
    if int(payload.get("satellite_count", observation.satellite_count)) != observation.satellite_count:
        raise ValueError("serialized satellite_count disagrees with tensor shapes")
    if edge_type.numel() != observation.edge_count:
        raise ValueError("serialized edge_type length disagrees with candidate edges")
    return observation


def target_from_record(
    record: Mapping[str, Any], device: torch.device | str
) -> IntensityFlowTarget:
    """Materialize and validate the typed target stored in a paper record."""

    target_device = torch.device(device)
    raw = record.get("next_target", record)
    if raw is None:
        raise ValueError("terminal records do not contain a next-step target")
    payload = _mapping(raw, "record.next_target")
    if int(payload.get("contract_version", -1)) != 1:
        raise ValueError("unsupported paper next-target contract version")
    target = IntensityFlowTarget(
        gamma_edge=torch.as_tensor(
            _required(payload, "gamma_edge", "record.next_target"), device=target_device
        ),
        log1p_intensity_edge=torch.as_tensor(
            _required(payload, "log1p_intensity_edge", "record.next_target"),
            device=target_device,
        ),
        feasibility_edge=torch.as_tensor(
            _required(payload, "feasibility_edge", "record.next_target"),
            device=target_device,
        ).bool(),
        persistent_edge=torch.as_tensor(
            _required(payload, "persistent_edge", "record.next_target"),
            device=target_device,
        ).bool(),
        flow_node=torch.as_tensor(
            _required(payload, "flow_node", "record.next_target"), device=target_device
        ),
    )
    return target


@dataclass(frozen=True)
class LossSummary:
    total: float
    gamma: float
    intensity: float
    flow: float
    feasibility: float
    decision_aware: float
    transitions: int
    persistent_edges: int
    sampled_model_steps: int
    sampling_opportunities: int

    @property
    def sampled_model_fraction(self) -> float:
        return self.sampled_model_steps / max(1, self.sampling_opportunities)

    def as_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["sampled_model_fraction"] = self.sampled_model_fraction
        return result


class _LossAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.gamma = 0.0
        self.intensity = 0.0
        self.flow = 0.0
        self.feasibility = 0.0
        self.decision = 0.0
        self.normalizer = 0.0
        self.transitions = 0
        self.persistent_edges = 0
        self.sampled_model_steps = 0
        self.sampling_opportunities = 0

    def add(
        self,
        loss: IntensityFlowLoss,
        decision: torch.Tensor,
        *,
        weight: float,
    ) -> None:
        self.total += float(loss.total.detach().item()) * weight
        self.gamma += float(loss.gamma.detach().item()) * weight
        self.intensity += float(loss.intensity.detach().item()) * weight
        self.flow += float(loss.flow.detach().item()) * weight
        self.feasibility += float(loss.feasibility.detach().item()) * weight
        self.decision += float(decision.detach().item()) * weight
        self.normalizer += weight
        self.transitions += 1
        self.persistent_edges += loss.persistent_edge_count

    def sample(self, use_model: bool) -> None:
        self.sampling_opportunities += 1
        self.sampled_model_steps += int(use_model)

    def merge(self, other: "_LossAccumulator") -> None:
        for name in (
            "total",
            "gamma",
            "intensity",
            "flow",
            "feasibility",
            "decision",
            "normalizer",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.transitions += other.transitions
        self.persistent_edges += other.persistent_edges
        self.sampled_model_steps += other.sampled_model_steps
        self.sampling_opportunities += other.sampling_opportunities

    def summary(self) -> LossSummary:
        if self.transitions <= 0 or self.normalizer <= 0.0:
            raise ValueError("loss summary requires at least one supervised transition")
        scale = 1.0 / self.normalizer
        return LossSummary(
            total=self.total * scale,
            gamma=self.gamma * scale,
            intensity=self.intensity * scale,
            flow=self.flow * scale,
            feasibility=self.feasibility * scale,
            decision_aware=self.decision * scale,
            transitions=self.transitions,
            persistent_edges=self.persistent_edges,
            sampled_model_steps=self.sampled_model_steps,
            sampling_opportunities=self.sampling_opportunities,
        )


@dataclass(frozen=True)
class _ActionCoupledWindow:
    """One differentiable H-step window from a continuing live episode."""

    objective: torch.Tensor
    losses: _LossAccumulator
    transitions: int


def _make_grad_scaler(enabled: bool):
    scaler_cls = getattr(torch.amp, "GradScaler", None)
    if scaler_cls is not None:
        return scaler_cls("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _detach_state(state: Any) -> Any:
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(_detach_state(value) for value in state)
    if isinstance(state, list):
        return [_detach_state(value) for value in state]
    if isinstance(state, Mapping):
        return {key: _detach_state(value) for key, value in state.items()}
    return state


def _oracle_from_target(target: IntensityFlowTarget) -> PolicyDescriptors:
    return PolicyDescriptors(
        gamma_edge=target.gamma_edge,
        intensity_edge=torch.expm1(target.log1p_intensity_edge).clamp_min(0.0),
        flow_node=target.flow_node,
    )


def _aligned_detached_prediction(
    prediction: IntensityFlowOutput,
    observation: ControlObservation,
) -> PolicyDescriptors:
    reference = observation.sim_descriptors.policy_fields
    fields = prediction.policy_descriptors
    return PolicyDescriptors(
        gamma_edge=fields.gamma_edge.detach().to(
            device=reference.gamma_edge.device, dtype=reference.gamma_edge.dtype
        ),
        intensity_edge=fields.intensity_edge.detach().to(
            device=reference.intensity_edge.device,
            dtype=reference.intensity_edge.dtype,
        ),
        flow_node=fields.flow_node.detach().to(
            device=reference.flow_node.device, dtype=reference.flow_node.dtype
        ),
    )


def _capture_rng_state() -> Dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            # Some supported Torch builds cannot serialize uint32 storages.
            # MT19937 keys fit losslessly in int64 and are cast back on restore.
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, Mapping):
        keys = torch.as_tensor(numpy_state["keys"]).cpu().numpy().astype(np.uint32)
        np.random.set_state(
            (
                str(numpy_state["algorithm"]),
                keys,
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    torch_cpu = state.get("torch_cpu")
    if isinstance(torch_cpu, torch.Tensor):
        torch.set_rng_state(torch_cpu.cpu())
    cuda_states = state.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


@dataclass(frozen=True)
class FitResult:
    best_checkpoint: Path
    last_checkpoint: Path
    best_validation: float
    best_epoch: int
    history: tuple[Mapping[str, Any], ...]


class PaperTrainer:
    """Train any paper predictor exposing ``predict_step``."""

    def __init__(
        self,
        *,
        model: nn.Module,
        config: Mapping[str, Any],
        device: torch.device | str,
        dataset_metadata: Mapping[str, Any] | None = None,
        initializer_factory: Callable[[Mapping[str, Any]], PolicyStreamInitializer]
        | None = None,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
        best_checkpoint_path: str | Path | None = None,
        last_checkpoint_path: str | Path | None = None,
    ) -> None:
        if not hasattr(model, "predict_step"):
            raise TypeError("paper model must implement predict_step(step, state, device)")
        self.config = copy.deepcopy(dict(config))
        self.settings = PaperTrainingConfig.from_config(self.config)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.dataset_metadata = copy.deepcopy(dict(dataset_metadata or {}))
        self.action_train_seeds = self._resolve_action_seeds(
            self.settings.action_coupled.train_seeds, "train"
        )
        self.action_validation_seeds = self._resolve_action_seeds(
            self.settings.action_coupled.validation_seeds, "val"
        )
        if self.settings.action_coupled.enabled:
            if not self.action_train_seeds:
                raise ValueError(
                    "action-coupled training requires train_seeds in config or in "
                    "dataset split_manifest.episode_seeds.train"
                )
            if not self.action_validation_seeds:
                raise ValueError(
                    "action-coupled validation requires validation_seeds in config or "
                    "dataset split_manifest.episode_seeds.val"
                )
            if set(self.action_train_seeds).intersection(self.action_validation_seeds):
                raise ValueError("action-coupled train and validation seeds must be disjoint")
        self.initializer_factory = initializer_factory or (
            lambda cfg: ConstantPolicyStreamInitializer.from_config(dict(cfg))
        )
        self.progress = progress
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("paper model has no trainable parameters")
        optimizer_type = (
            torch.optim.AdamW
            if self.settings.optimizer_name == "adamw"
            else torch.optim.Adam
        )
        self.optimizer = optimizer_type(
            parameters,
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        self.autocast_dtype = (
            torch.float16
            if self.settings.amp_dtype == "float16"
            else torch.bfloat16
        )
        if self.settings.amp and self.device.type == "cpu" and self.autocast_dtype == torch.float16:
            raise ValueError("CPU AMP requires paper_training amp_dtype=bfloat16")
        self.amp_enabled = self.settings.amp and self.device.type in {"cuda", "cpu"}
        scaler_enabled = (
            self.amp_enabled
            and self.device.type == "cuda"
            and self.autocast_dtype == torch.float16
        )
        self.scaler = _make_grad_scaler(scaler_enabled)
        self.sampling_generator = torch.Generator(device="cpu")
        self.sampling_generator.manual_seed(self.settings.sampling_seed)
        self.order_generator = torch.Generator(device="cpu")
        self.order_generator.manual_seed(self.settings.sampling_seed + 1)
        self.global_step = 0
        self.start_epoch = 1
        self.best_validation = math.inf
        self.best_epoch = 0
        self.history: list[Mapping[str, Any]] = []
        self._last_action_training_budget: Dict[str, Any] | None = None
        self.settings.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = (
            Path(best_checkpoint_path).expanduser()
            if best_checkpoint_path is not None
            else self.settings.save_dir / "best.pt"
        )
        self.last_checkpoint_path = (
            Path(last_checkpoint_path).expanduser()
            if last_checkpoint_path is not None
            else self.settings.save_dir / "last.pt"
        )
        if self.best_checkpoint_path == self.last_checkpoint_path:
            raise ValueError("best and last checkpoint paths must be different")

    def _resolve_action_seeds(
        self, configured: Sequence[int], split: str
    ) -> tuple[int, ...]:
        if configured:
            return tuple(int(seed) for seed in configured)
        manifest = self.dataset_metadata.get("split_manifest")
        if not isinstance(manifest, Mapping):
            return ()
        episode_seeds = manifest.get("episode_seeds")
        if not isinstance(episode_seeds, Mapping):
            return ()
        values = episode_seeds.get(split)
        if not isinstance(values, list):
            return ()
        seeds = tuple(int(value) for value in values)
        if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"dataset {split} split contains invalid episode seeds")
        return seeds

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.amp_enabled,
        )

    def _set_epoch_learning_rate(self, epoch: int) -> float:
        multiplier = self.settings.learning_rate_multiplier.value(epoch)
        learning_rate = self.settings.learning_rate * multiplier
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("scheduled learning rate must be finite and non-negative")
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def _sample_model(self, probability: float, generator: torch.Generator) -> bool:
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return bool(torch.rand((), generator=generator).item() < probability)

    def _predict(
        self, observation: ControlObservation, state: Any
    ) -> tuple[IntensityFlowOutput, Any]:
        output, next_state = self.model.predict_step(
            observation.as_model_step(), state, self.device
        )
        if not isinstance(output, IntensityFlowOutput):
            raise TypeError("paper model predict_step must return IntensityFlowOutput")
        output.validate(observation.edge_count, observation.satellite_count)
        return output, next_state

    def _losses(
        self,
        prediction: IntensityFlowOutput,
        target: IntensityFlowTarget,
        observation: ControlObservation,
        decision_weight: float,
    ) -> tuple[IntensityFlowLoss, torch.Tensor, torch.Tensor]:
        typed = intensity_flow_one_step_loss(
            prediction, target, self.settings.loss_weights
        )
        decision = typed.total.new_zeros(())
        if decision_weight > 0.0:
            hook = getattr(self.model, "decision_aware_loss_hook", None)
            if not callable(hook):
                raise ValueError(
                    "decision_aware_weight is positive but the model does not expose "
                    "decision_aware_loss_hook"
                )
            decision = hook(
                prediction,
                _oracle_from_target(target),
                observation.candidate_edge_ids,
                target.persistent_edge,
            )
            if decision.ndim != 0 or not bool(torch.isfinite(decision)):
                raise FloatingPointError("decision-aware loss must be a finite scalar")
        return typed, decision, typed.total + decision_weight * decision

    def _backward(self, objective: torch.Tensor) -> None:
        if objective.ndim != 0 or not bool(torch.isfinite(objective.detach())):
            raise FloatingPointError("training objective must be a finite scalar")
        if self.scaler.is_enabled():
            self.scaler.scale(objective).backward()
        else:
            objective.backward()

    def _optimizer_step(self) -> None:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.settings.clip_grad_norm
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise FloatingPointError("gradient norm is NaN or Inf")
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _initializer(self, cfg: Mapping[str, Any]) -> PolicyStreamInitializer:
        initializer = self.initializer_factory(cfg)
        required = ("initialize", "kind", "fingerprint")
        if any(not hasattr(initializer, name) for name in required):
            raise TypeError("initializer_factory returned an incompatible initializer")
        return initializer

    def _new_edge_descriptors(
        self,
        previous: ControlObservation,
        current: ControlObservation,
        *,
        source: str,
        initializer: PolicyStreamInitializer | None,
    ) -> PolicyDescriptors | None:
        new_edges = ~persistent_candidate_mask(previous, current)
        if not bool(new_edges.any().item()):
            return None
        if source == "teacher":
            return current.sim_descriptors.policy_fields.clone()
        if source != "configured_initializer" or initializer is None:
            raise ValueError("new candidate edges require the configured initializer")
        return initializer.initialize(current, new_edges)

    @staticmethod
    def _episode_steps(episode: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        steps = episode.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("paper episode must contain a non-empty steps list")
        if not all(isinstance(record, Mapping) for record in steps):
            raise TypeError("paper episode step records must be mappings")
        return list(steps)

    def _recorded_episode(
        self, episode: Mapping[str, Any]
    ) -> tuple[list[ControlObservation], list[IntensityFlowTarget]]:
        records = self._episode_steps(episode)
        observations = [observation_from_record(record, self.device) for record in records]
        targets = [target_from_record(record, self.device) for record in records[:-1]]
        for index, (source, current, target) in enumerate(
            zip(observations[:-1], observations[1:], targets)
        ):
            if current.episode_seed != source.episode_seed or current.epoch != source.epoch + 1:
                raise ValueError("recorded observations must be consecutive within one seed")
            target.validate(source.edge_count, source.satellite_count)
            raw_target = records[index]["next_target"]
            if raw_target["source_observation_id"] != list(source.observation_id):
                raise ValueError("serialized target source id is misaligned")
            if raw_target["target_observation_id"] != list(current.observation_id):
                raise ValueError("serialized target destination id is misaligned")
        return observations, targets

    def _train_recorded_episode(
        self,
        episode: Mapping[str, Any],
        *,
        objective_weight: float,
        model_probability: float,
        decision_weight: float,
        tbptt_steps: int,
        new_edge_source: str,
        gradient_divisor: int,
    ) -> _LossAccumulator:
        observations, targets = self._recorded_episode(episode)
        if not targets:
            return _LossAccumulator()
        initializer = (
            self._initializer(self.config)
            if new_edge_source == "configured_initializer"
            else None
        )
        accumulator = _LossAccumulator()
        state: Any = None
        current_input = observations[0]
        chunk_objective: torch.Tensor | None = None
        transition_count = len(targets)
        for index, target in enumerate(targets):
            with self._autocast():
                prediction, state = self._predict(current_input, state)
                typed, decision, combined = self._losses(
                    prediction, target, current_input, decision_weight
                )
                contribution = (
                    objective_weight
                    * combined
                    / float(transition_count)
                    / float(gradient_divisor)
                )
                chunk_objective = (
                    contribution
                    if chunk_objective is None
                    else chunk_objective + contribution
                )
            accumulator.add(typed, decision, weight=1.0)
            if index + 1 < transition_count:
                use_model = self._sample_model(
                    model_probability, self.sampling_generator
                )
                accumulator.sample(use_model)
                base_next = observations[index + 1]
                if use_model:
                    next_fields = _aligned_detached_prediction(prediction, current_input)
                    new_fields = self._new_edge_descriptors(
                        current_input,
                        base_next,
                        source=new_edge_source,
                        initializer=initializer,
                    )
                    current_input = carry_policy_stream(
                        current_input, next_fields, base_next, new_fields
                    )
                else:
                    current_input = base_next
            boundary = (index + 1) % tbptt_steps == 0 or index + 1 == transition_count
            if boundary:
                if chunk_objective is None:
                    raise RuntimeError("empty TBPTT objective")
                self._backward(chunk_objective)
                chunk_objective = None
                state = _detach_state(state)
        return accumulator

    def train_one_step_epoch(
        self, episodes: Sequence[Mapping[str, Any]], *, epoch: int
    ) -> LossSummary:
        cfg = self.settings.one_step
        if not cfg.enabled or cfg.weight is None or cfg.scheduled_sampling is None:
            raise RuntimeError("one-step objective is disabled")
        objective_weight = cfg.weight.value(epoch)
        if objective_weight <= 0.0:
            raise ValueError("train_one_step_epoch called with zero objective weight")
        model_probability = cfg.scheduled_sampling.value(epoch)
        decision_weight = self.settings.decision_aware_weight.value(epoch)
        self.model.train()
        total = _LossAccumulator()
        order = torch.randperm(len(episodes), generator=self.order_generator).tolist()
        batch_size = self.settings.batch_size_control_graphs
        initializer = (
            self._initializer(self.config)
            if cfg.new_edge_source == "configured_initializer"
            else None
        )
        pending_objective: torch.Tensor | None = None
        pending_graphs = 0
        self.optimizer.zero_grad(set_to_none=True)

        def flush_pending() -> None:
            nonlocal pending_objective, pending_graphs
            if pending_graphs == 0:
                return
            if pending_objective is None:
                raise RuntimeError("missing Intensity--Flow batch objective")
            self._backward(pending_objective / float(pending_graphs))
            self._optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)
            pending_objective = None
            pending_graphs = 0

        for episode_index in order:
            observations, targets = self._recorded_episode(episodes[episode_index])
            if not targets:
                continue
            state: Any = None
            current_input = observations[0]
            chunk_objective: torch.Tensor | None = None
            chunk_graphs = 0
            for index, target in enumerate(targets):
                with self._autocast():
                    prediction, state = self._predict(current_input, state)
                    typed, decision, combined = self._losses(
                        prediction, target, current_input, decision_weight
                    )
                    contribution = objective_weight * combined
                    chunk_objective = (
                        contribution
                        if chunk_objective is None
                        else chunk_objective + contribution
                    )
                chunk_graphs += 1
                total.add(typed, decision, weight=1.0)
                if index + 1 < len(targets):
                    use_model = self._sample_model(
                        model_probability, self.sampling_generator
                    )
                    total.sample(use_model)
                    base_next = observations[index + 1]
                    if use_model:
                        next_fields = _aligned_detached_prediction(
                            prediction, current_input
                        )
                        new_fields = self._new_edge_descriptors(
                            current_input,
                            base_next,
                            source=cfg.new_edge_source,
                            initializer=initializer,
                        )
                        current_input = carry_policy_stream(
                            current_input, next_fields, base_next, new_fields
                        )
                    else:
                        current_input = base_next
                boundary = (
                    (index + 1) % cfg.tbptt_steps == 0
                    or index + 1 == len(targets)
                )
                if boundary:
                    if chunk_objective is None or chunk_graphs <= 0:
                        raise RuntimeError("empty one-step TBPTT chunk")
                    pending_objective = (
                        chunk_objective
                        if pending_objective is None
                        else pending_objective + chunk_objective
                    )
                    pending_graphs += chunk_graphs
                    chunk_objective = None
                    chunk_graphs = 0
                    state = _detach_state(state)
                    if pending_graphs >= batch_size:
                        flush_pending()
        flush_pending()
        return total.summary()

    def _environment_for_seed(self, seed: int) -> PaperAlignedLEOEnv:
        episode_cfg = copy.deepcopy(self.config)
        episode_cfg["seed"] = int(seed)
        return PaperAlignedLEOEnv(episode_cfg, device=self.device)

    def _action_coupled_windows(
        self,
        *,
        seed: int,
        model_probability: float,
        decision_weight: float,
        sampling_generator: torch.Generator,
    ) -> Iterator[_ActionCoupledWindow]:
        """Yield H-step TBPTT windows while executing one complete episode.

        The simulator, controller stream, and detached recurrent state continue
        across yields.  This permits an optimizer update between windows while
        retaining genuine action coupling over the complete episode instead of
        repeatedly training only on its first H transitions.
        """

        cfg = self.settings.action_coupled
        if not cfg.enabled or cfg.step_weights is None:
            raise RuntimeError("action-coupled objective is disabled")
        environment = self._environment_for_seed(seed)
        initializer = self._initializer(environment.cfg)
        policy = FixedRankPolicy(environment.fixed_policy_config())
        observation = environment.reset_control()
        initial_mask = torch.ones(
            observation.edge_count,
            dtype=torch.bool,
            device=observation.candidate_edge_ids.device,
        )
        observation = cold_start_policy_stream(
            observation, initializer.initialize(observation, initial_mask)
        )
        transition_count = environment.horizon_steps - 1
        if transition_count <= 0:
            raise ValueError("action-coupled training requires horizon_steps >= 2")
        state: Any = None
        window_rows: list[
            tuple[IntensityFlowLoss, torch.Tensor, torch.Tensor]
        ] = []
        window_accumulator = _LossAccumulator()
        for transition_index in range(transition_count):
            # The current action sees only the descriptor stream staged by D0 or
            # the previous epoch; the fresh prediction below cannot affect a_t.
            action = policy.select_action(observation)
            with self._autocast():
                prediction, state = self._predict(observation, state)
            next_observation, _execution, done = environment.step_action(action)
            if done or next_observation is None:
                raise RuntimeError(
                    "environment ended before the configured supervised rollout horizon"
                )
            target = build_next_step_target(observation, next_observation)
            with self._autocast():
                typed, decision, combined = self._losses(
                    prediction, target, observation, decision_weight
                )
            window_rows.append((typed, decision, combined))

            if transition_index + 1 < transition_count:
                use_model = self._sample_model(model_probability, sampling_generator)
                window_accumulator.sample(use_model)
                if use_model:
                    next_fields = _aligned_detached_prediction(prediction, observation)
                    new_fields = self._new_edge_descriptors(
                        observation,
                        next_observation,
                        source=cfg.new_edge_source,
                        initializer=initializer,
                    )
                    observation = carry_policy_stream(
                        observation, next_fields, next_observation, new_fields
                    )
                else:
                    observation = next_observation

            boundary = (
                len(window_rows) == cfg.rollout_horizon
                or transition_index + 1 == transition_count
            )
            if boundary:
                window_length = len(window_rows)
                if window_length <= 0:
                    raise RuntimeError("empty action-coupled TBPTT window")
                step_cfg = dict(cfg.step_weights)
                if str(step_cfg.get("scheme", "")).lower() == "explicit":
                    values = step_cfg.get("values")
                    if (
                        isinstance(values, Sequence)
                        and not isinstance(values, (str, bytes))
                        and len(values) == cfg.rollout_horizon
                        and window_length < cfg.rollout_horizon
                    ):
                        step_cfg["values"] = list(values[:window_length])
                step_weights = build_multistep_weights(
                    window_length, step_cfg, device=self.device
                )
                objective = torch.stack(
                    [
                        step_weights[index] * row[2]
                        for index, row in enumerate(window_rows)
                    ]
                ).sum()
                if not bool(torch.isfinite(objective.detach())):
                    raise FloatingPointError(
                        "action-coupled rollout-window objective is NaN or Inf"
                    )
                # Give a full H-window mass H so summaries and cross-window
                # minibatches remain control-transition weighted.  A shorter
                # terminal window receives exactly its observed length.
                for index, (typed, decision, _combined) in enumerate(window_rows):
                    metric_weight = float(
                        step_weights[index].detach().item()
                    ) * window_length
                    window_accumulator.add(
                        typed,
                        decision,
                        weight=metric_weight,
                    )
                state = _detach_state(state)
                result = _ActionCoupledWindow(
                    objective=objective,
                    losses=window_accumulator,
                    transitions=window_length,
                )
                window_rows = []
                window_accumulator = _LossAccumulator()
                yield result

    def train_action_coupled_epoch(self, *, epoch: int) -> LossSummary:
        cfg = self.settings.action_coupled
        if not cfg.enabled or cfg.weight is None or cfg.scheduled_sampling is None:
            raise RuntimeError("action-coupled objective is disabled")
        objective_weight = cfg.weight.value(epoch)
        if objective_weight <= 0.0:
            raise ValueError("train_action_coupled_epoch called with zero objective weight")
        model_probability = cfg.scheduled_sampling.value(epoch)
        decision_weight = self.settings.decision_aware_weight.value(epoch)
        self.model.train()
        total = _LossAccumulator()
        scheduled_seeds = _budgeted_seed_passes(
            self.action_train_seeds,
            multiplier=cfg.budget_multiplier,
            generator=self.order_generator,
        )
        self._last_action_training_budget = {
            "reference": cfg.budget_reference,
            "allocation": cfg.budget_allocation,
            "configured_multiplier": cfg.budget_multiplier,
            "base_complete_episode_passes": len(self.action_train_seeds),
            "scheduled_complete_episode_passes": len(scheduled_seeds),
            "effective_multiplier": len(scheduled_seeds)
            / len(self.action_train_seeds),
            "validation_multiplier": cfg.validation_budget_multiplier,
        }
        sequence_batch_size = cfg.rollout_sequence_batch_size
        for start in range(0, len(scheduled_seeds), sequence_batch_size):
            seed_batch = scheduled_seeds[start : start + sequence_batch_size]
            active = [
                self._action_coupled_windows(
                    seed=seed,
                    model_probability=model_probability,
                    decision_weight=decision_weight,
                    sampling_generator=self.sampling_generator,
                )
                for seed in seed_batch
            ]
            while active:
                windows: list[_ActionCoupledWindow] = []
                still_active: list[Iterator[_ActionCoupledWindow]] = []
                for iterator in active:
                    try:
                        window = next(iterator)
                    except StopIteration:
                        continue
                    windows.append(window)
                    still_active.append(iterator)
                active = still_active
                if not windows:
                    break
                control_graphs = sum(window.transitions for window in windows)
                if control_graphs > self.settings.batch_size_control_graphs:
                    raise RuntimeError(
                        "action-coupled optimizer batch exceeded "
                        "batch_size_control_graphs"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                batch_objective = objective_weight * sum(
                    window.objective * window.transitions for window in windows
                ) / float(control_graphs)
                self._backward(batch_objective)
                self._optimizer_step()
                for window in windows:
                    total.merge(window.losses)
        return total.summary()

    def validate_one_step(
        self,
        episodes: Sequence[Mapping[str, Any]],
        *,
        model_probability: float,
        sampling_generator: torch.Generator,
    ) -> LossSummary:
        self.model.eval()
        accumulator = _LossAccumulator()
        one_cfg = self.settings.one_step
        new_edge_source = (
            one_cfg.new_edge_source if one_cfg.enabled else "teacher"
        )
        initializer = (
            self._initializer(self.config)
            if new_edge_source == "configured_initializer"
            else None
        )
        with torch.no_grad():
            for episode in episodes:
                observations, targets = self._recorded_episode(episode)
                if not targets:
                    continue
                state: Any = None
                current_input = observations[0]
                for index, target in enumerate(targets):
                    with self._autocast():
                        prediction, state = self._predict(current_input, state)
                        typed, decision, _combined = self._losses(
                            prediction, target, current_input, decision_weight=0.0
                        )
                    accumulator.add(typed, decision, weight=1.0)
                    if index + 1 < len(targets):
                        use_model = self._sample_model(
                            model_probability, sampling_generator
                        )
                        accumulator.sample(use_model)
                        base_next = observations[index + 1]
                        if use_model:
                            next_fields = _aligned_detached_prediction(
                                prediction, current_input
                            )
                            new_fields = self._new_edge_descriptors(
                                current_input,
                                base_next,
                                source=new_edge_source,
                                initializer=initializer,
                            )
                            current_input = carry_policy_stream(
                                current_input,
                                next_fields,
                                base_next,
                                new_fields,
                            )
                        else:
                            current_input = base_next
        return accumulator.summary()

    def validate_action_coupled(
        self,
        *,
        model_probability: float,
        sampling_generator: torch.Generator,
    ) -> LossSummary:
        cfg = self.settings.action_coupled
        if not cfg.enabled or not self.action_validation_seeds:
            raise ValueError("action-coupled validation seeds are not configured")
        self.model.eval()
        total = _LossAccumulator()
        with torch.no_grad():
            for seed in self.action_validation_seeds:
                for window in self._action_coupled_windows(
                    seed=seed,
                    model_probability=model_probability,
                    decision_weight=0.0,
                    sampling_generator=sampling_generator,
                ):
                    total.merge(window.losses)
        return total.summary()

    def _validation_metric(
        self,
        one_step: LossSummary | None,
        action_coupled: LossSummary | None,
    ) -> float:
        cfg = self.settings.validation
        if cfg.selection_metric == "one_step":
            if one_step is None:
                raise ValueError("one-step validation summary is unavailable")
            return one_step.total
        if cfg.selection_metric == "action_coupled":
            if action_coupled is None:
                raise ValueError("action-coupled validation summary is unavailable")
            return action_coupled.total
        numerator = 0.0
        denominator = 0.0
        if cfg.one_step_weight > 0.0:
            if one_step is None:
                raise ValueError("combined validation requires one-step validation")
            numerator += cfg.one_step_weight * one_step.total
            denominator += cfg.one_step_weight
        if cfg.action_coupled_weight > 0.0:
            if action_coupled is None:
                raise ValueError("combined validation requires action-coupled validation")
            numerator += cfg.action_coupled_weight * action_coupled.total
            denominator += cfg.action_coupled_weight
        return numerator / denominator

    def _checkpoint_state(
        self,
        *,
        epoch: int,
        validation_metric: float | None,
    ) -> Dict[str, Any]:
        training_plan = _normalized_training_config(self.config)
        immutable_training_plan = _semantic_training_plan(training_plan)
        resumable_training_fields = _resumable_training_fields(training_plan)
        resumable_training_fields.update(
            {
                "best_checkpoint_path": str(self.best_checkpoint_path.resolve()),
                "last_checkpoint_path": str(self.last_checkpoint_path.resolve()),
            }
        )
        dataset_semantics = _dataset_semantics(self.dataset_metadata)
        return {
            "trainer_schema_version": PAPER_TRAINER_SCHEMA_VERSION,
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "best_validation": float(self.best_validation),
            "best_epoch": int(self.best_epoch),
            "latest_validation": validation_metric,
            "history": copy.deepcopy(self.history),
            "training_plan": training_plan,
            "training_plan_fingerprint": _training_plan_fingerprint(training_plan),
            "training_plan_fingerprint_version": (
                TRAINING_PLAN_FINGERPRINT_VERSION
            ),
            "immutable_training_plan": immutable_training_plan,
            "immutable_training_plan_fingerprint": (
                _canonical_mapping_fingerprint(immutable_training_plan)
            ),
            "resumable_training_fields": resumable_training_fields,
            "loss_weights": asdict(self.settings.loss_weights),
            "optimizer_name": self.settings.optimizer_name,
            "batch_semantics": "control_graphs_per_optimizer_step_v2",
            "batch_size_control_graphs": self.settings.batch_size_control_graphs,
            "action_coupled_batch": {
                "window_horizon": self.settings.action_coupled.rollout_horizon,
                "sequence_batch_size": (
                    self.settings.action_coupled.rollout_sequence_batch_size
                ),
                "maximum_control_graphs_per_optimizer_step": (
                    self.settings.batch_size_control_graphs
                ),
            },
            "latest_action_training_budget": copy.deepcopy(
                self._last_action_training_budget
            ),
            "learning_rate": [
                float(group["lr"]) for group in self.optimizer.param_groups
            ],
            "amp_enabled": bool(self.amp_enabled),
            "amp_dtype": self.settings.amp_dtype,
            "scaler_state": self.scaler.state_dict(),
            "rng_state": _capture_rng_state(),
            "sampling_generator_state": self.sampling_generator.get_state(),
            "order_generator_state": self.order_generator.get_state(),
            "dataset": copy.deepcopy(self.dataset_metadata),
            "dataset_semantics": dataset_semantics,
            "dataset_semantic_fingerprint": (
                _canonical_mapping_fingerprint(dataset_semantics)
            ),
            "action_environment_fingerprint": (
                _action_environment_fingerprint(self.config)
            ),
            "action_train_seeds": list(self.action_train_seeds),
            "action_validation_seeds": list(self.action_validation_seeds),
            "model_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "paper_method": str(
                self.config.get("paper_method", getattr(self.model, "paper_method", ""))
            ),
            "trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                )
            ),
        }

    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        validation_metric: float | None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_ckpt(
            str(temporary),
            model=self.model,
            opt=self.optimizer,
            epoch=int(epoch),
            config=self.config,
            paper_method=str(
                self.config.get("paper_method", getattr(self.model, "paper_method", ""))
            ),
            trainer_state=self._checkpoint_state(
                epoch=epoch, validation_metric=validation_metric
            ),
        )
        os.replace(temporary, path)

    def resume(self, path: str | Path) -> None:
        checkpoint_path = Path(path).expanduser()
        # Validate every semantic contract before loading model/optimizer state.
        # A rejected resume therefore leaves the current trainer untouched.
        preview = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(preview, Mapping):
            raise TypeError("paper training checkpoint must contain a mapping")
        trainer_state = preview.get("trainer_state")
        if not isinstance(trainer_state, Mapping):
            raise ValueError("checkpoint does not contain paper trainer state")
        schema_version = trainer_state.get("trainer_schema_version")
        if type(schema_version) is not int or schema_version not in {
            PAPER_TRAINER_SCHEMA_VERSION,
            *PAPER_TRAINER_LEGACY_SCHEMA_VERSIONS,
        }:
            raise ValueError("unsupported paper trainer checkpoint schema")
        expected_class = f"{type(self.model).__module__}.{type(self.model).__qualname__}"
        if trainer_state.get("model_class") != expected_class:
            raise ValueError("checkpoint paper model class does not match the requested model")
        expected_method = str(
            self.config.get("paper_method", getattr(self.model, "paper_method", ""))
        )
        if trainer_state.get("paper_method") != expected_method:
            raise ValueError("checkpoint paper_method does not match the requested method")

        completed_epoch = trainer_state.get("epoch")
        if (
            isinstance(completed_epoch, bool)
            or not isinstance(completed_epoch, int)
            or completed_epoch < 0
        ):
            raise ValueError("resume checkpoint epoch must be a non-negative integer")
        current_plan = _normalized_training_config(self.config)
        _validated_resume_plans(
            trainer_state=trainer_state,
            current_plan=current_plan,
            completed_epoch=completed_epoch,
            schema_version=schema_version,
        )

        saved_model_signature = preview.get("model_signature")
        current_model_signature = model_signature_from_config(self.config)
        if not isinstance(saved_model_signature, Mapping):
            raise ValueError("resume checkpoint has no model semantic signature")
        if dict(saved_model_signature) != current_model_signature:
            raise ValueError("resume checkpoint model semantics changed")

        saved_dataset = trainer_state.get("dataset")
        if not isinstance(saved_dataset, Mapping):
            raise ValueError("resume checkpoint has no dataset metadata")
        saved_dataset_semantics = _dataset_semantics(saved_dataset)
        current_dataset_semantics = _dataset_semantics(self.dataset_metadata)
        if schema_version == PAPER_TRAINER_SCHEMA_VERSION:
            stored_dataset_semantics = trainer_state.get("dataset_semantics")
            stored_dataset_fingerprint = trainer_state.get(
                "dataset_semantic_fingerprint"
            )
            if not isinstance(stored_dataset_semantics, Mapping):
                raise ValueError("resume checkpoint has no dataset semantic metadata")
            if dict(stored_dataset_semantics) != saved_dataset_semantics:
                raise ValueError("resume checkpoint dataset metadata is inconsistent")
            if stored_dataset_fingerprint != _dataset_semantic_fingerprint(
                saved_dataset
            ):
                raise ValueError("resume checkpoint dataset fingerprint is invalid")
        if saved_dataset_semantics != current_dataset_semantics:
            raise ValueError("resume checkpoint was trained from a different dataset")

        current_environment_fingerprint = _action_environment_fingerprint(self.config)
        saved_environment_fingerprint = trainer_state.get(
            "action_environment_fingerprint"
        )
        if schema_version in PAPER_TRAINER_LEGACY_SCHEMA_VERSIONS:
            saved_config = preview.get("config")
            if not isinstance(saved_config, Mapping):
                raise ValueError(
                    "legacy resume checkpoint has no config for safe protocol migration"
                )
            saved_environment_fingerprint = _action_environment_fingerprint(
                saved_config
            )
        if saved_environment_fingerprint != current_environment_fingerprint:
            raise ValueError(
                "resume checkpoint action-environment semantics changed"
            )

        saved_train_seeds = trainer_state.get("action_train_seeds")
        saved_validation_seeds = trainer_state.get("action_validation_seeds")
        if not isinstance(saved_train_seeds, (list, tuple)):
            raise ValueError("resume checkpoint has no action-coupled train seeds")
        if not isinstance(saved_validation_seeds, (list, tuple)):
            raise ValueError("resume checkpoint has no action-coupled validation seeds")
        if tuple(saved_train_seeds) != self.action_train_seeds:
            raise ValueError("resume checkpoint action-coupled train seeds changed")
        if tuple(saved_validation_seeds) != self.action_validation_seeds:
            raise ValueError("resume checkpoint action-coupled validation seeds changed")
        saved_best_epoch = trainer_state.get("best_epoch")
        if (
            isinstance(saved_best_epoch, bool)
            or not isinstance(saved_best_epoch, int)
            or saved_best_epoch < 0
            or saved_best_epoch > completed_epoch
        ):
            raise ValueError("resume checkpoint best epoch is invalid")
        history = trainer_state.get("history", [])
        if not isinstance(history, list):
            raise TypeError("checkpoint trainer history must be a list")
        saved_budget = trainer_state.get("latest_action_training_budget")
        if saved_budget is not None and not isinstance(saved_budget, Mapping):
            raise TypeError("checkpoint action training budget must be a mapping")

        best_source: Path | None = None
        if saved_best_epoch > 0:
            candidates: list[Path] = [self.best_checkpoint_path]
            resume_fields = trainer_state.get("resumable_training_fields")
            if isinstance(resume_fields, Mapping):
                saved_best_path = resume_fields.get("best_checkpoint_path")
                if isinstance(saved_best_path, str) and saved_best_path.strip():
                    candidates.append(Path(saved_best_path).expanduser())
            candidates.append(checkpoint_path.parent / "best.pt")
            if completed_epoch == saved_best_epoch:
                candidates.append(checkpoint_path)
            seen_candidates: set[Path] = set()
            for candidate in candidates:
                resolved_candidate = candidate.resolve()
                if resolved_candidate in seen_candidates:
                    continue
                seen_candidates.add(resolved_candidate)
                if _is_compatible_best_checkpoint(
                    resolved_candidate,
                    model_signature=current_model_signature,
                    paper_method=expected_method,
                    model_class=expected_class,
                    best_epoch=saved_best_epoch,
                    training_semantics=_semantic_training_plan(current_plan),
                    dataset_semantics=current_dataset_semantics,
                ):
                    best_source = resolved_candidate
                    break
            if best_source is None:
                raise FileNotFoundError(
                    "resume could not locate the validated prior best checkpoint; "
                    "keep the old best checkpoint available when redirecting outputs"
                )

        # The preflight above passed. Apply state only now.
        load_ckpt(
            str(checkpoint_path),
            model=self.model,
            opt=self.optimizer,
            map_location=self.device,
            strict=True,
        )
        if best_source is not None:
            best_target = self.best_checkpoint_path.resolve()
            if best_source != best_target:
                best_target.parent.mkdir(parents=True, exist_ok=True)
                temporary = best_target.with_suffix(best_target.suffix + ".resume-tmp")
                shutil.copy2(best_source, temporary)
                os.replace(temporary, best_target)
        self.global_step = int(trainer_state["global_step"])
        self.start_epoch = completed_epoch + 1
        self.best_validation = float(trainer_state["best_validation"])
        self.best_epoch = saved_best_epoch
        self.history = list(history)
        self._last_action_training_budget = (
            copy.deepcopy(dict(saved_budget))
            if isinstance(saved_budget, Mapping)
            else None
        )
        scaler_state = trainer_state.get("scaler_state")
        if isinstance(scaler_state, Mapping):
            self.scaler.load_state_dict(dict(scaler_state))
        rng_state = trainer_state.get("rng_state")
        if isinstance(rng_state, Mapping):
            _restore_rng_state(rng_state)
        sampling_state = trainer_state.get("sampling_generator_state")
        if isinstance(sampling_state, torch.Tensor):
            self.sampling_generator.set_state(sampling_state.cpu())
        order_state = trainer_state.get("order_generator_state")
        if isinstance(order_state, torch.Tensor):
            self.order_generator.set_state(order_state.cpu())

    def fit(
        self,
        train_episodes: Sequence[Mapping[str, Any]],
        validation_episodes: Sequence[Mapping[str, Any]],
    ) -> FitResult:
        if self.settings.one_step.enabled and len(train_episodes) == 0:
            raise ValueError("one-step training requires a non-empty train split")
        metric = self.settings.validation.selection_metric
        if metric in {"one_step", "combined"} and len(validation_episodes) == 0:
            raise ValueError("selected validation metric requires a non-empty val split")
        best_path = self.best_checkpoint_path
        last_path = self.last_checkpoint_path

        for epoch in range(self.start_epoch, self.settings.epochs + 1):
            epoch_record: Dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": self._set_epoch_learning_rate(epoch),
                "schedule": {
                    "decision_aware_weight": self.settings.decision_aware_weight.value(epoch),
                    "one_step_weight": (
                        self.settings.one_step.weight.value(epoch)
                        if self.settings.one_step.enabled
                        and self.settings.one_step.weight is not None
                        else 0.0
                    ),
                    "one_step_model_probability": (
                        self.settings.one_step.scheduled_sampling.value(epoch)
                        if self.settings.one_step.enabled
                        and self.settings.one_step.scheduled_sampling is not None
                        else 0.0
                    ),
                    "action_coupled_weight": (
                        self.settings.action_coupled.weight.value(epoch)
                        if self.settings.action_coupled.enabled
                        and self.settings.action_coupled.weight is not None
                        else 0.0
                    ),
                    "action_coupled_model_probability": (
                        self.settings.action_coupled.scheduled_sampling.value(epoch)
                        if self.settings.action_coupled.enabled
                        and self.settings.action_coupled.scheduled_sampling is not None
                        else 0.0
                    ),
                },
            }
            one_cfg = self.settings.one_step
            if one_cfg.enabled and one_cfg.weight is not None and one_cfg.weight.value(epoch) > 0.0:
                epoch_record["train_one_step"] = self.train_one_step_epoch(
                    train_episodes, epoch=epoch
                ).as_dict()
            action_cfg = self.settings.action_coupled
            if (
                action_cfg.enabled
                and action_cfg.weight is not None
                and action_cfg.weight.value(epoch) > 0.0
            ):
                epoch_record["train_action_coupled"] = (
                    self.train_action_coupled_epoch(epoch=epoch).as_dict()
                )
                epoch_record["action_training_budget"] = copy.deepcopy(
                    self._last_action_training_budget
                )

            should_validate = (
                epoch % self.settings.validation.every_epochs == 0
                or epoch == self.settings.epochs
            )
            validation_metric: float | None = None
            if should_validate:
                validation_generator = torch.Generator(device="cpu")
                validation_generator.manual_seed(self.settings.validation.sampling_seed)
                one_summary = None
                action_summary = None
                if metric in {"one_step", "combined"} and (
                    metric == "one_step" or self.settings.validation.one_step_weight > 0.0
                ):
                    one_summary = self.validate_one_step(
                        validation_episodes,
                        model_probability=(
                            self.settings.validation.one_step_model_probability
                        ),
                        sampling_generator=validation_generator,
                    )
                    epoch_record["validation_one_step"] = one_summary.as_dict()
                if metric in {"action_coupled", "combined"} and (
                    metric == "action_coupled"
                    or self.settings.validation.action_coupled_weight > 0.0
                ):
                    action_summary = self.validate_action_coupled(
                        model_probability=(
                            self.settings.validation.action_coupled_model_probability
                        ),
                        sampling_generator=validation_generator,
                    )
                    epoch_record["validation_action_coupled"] = action_summary.as_dict()
                validation_metric = self._validation_metric(one_summary, action_summary)
                epoch_record["selection_metric"] = validation_metric
                if not math.isfinite(validation_metric):
                    raise FloatingPointError("validation selection metric is non-finite")
                if validation_metric < self.best_validation:
                    self.best_validation = validation_metric
                    self.best_epoch = epoch
                    epoch_record["is_best"] = True
                else:
                    epoch_record["is_best"] = False

            self.history.append(epoch_record)
            if should_validate and epoch_record.get("is_best"):
                self._save_checkpoint(
                    best_path,
                    epoch=epoch,
                    validation_metric=validation_metric,
                )
            self._save_checkpoint(
                last_path,
                epoch=epoch,
                validation_metric=validation_metric,
            )
            if self.progress is not None:
                self.progress(copy.deepcopy(epoch_record))

        if self.best_epoch <= 0 or not best_path.exists():
            raise RuntimeError("training completed without a validated best checkpoint")
        return FitResult(
            best_checkpoint=best_path,
            last_checkpoint=last_path,
            best_validation=self.best_validation,
            best_epoch=self.best_epoch,
            history=tuple(self.history),
        )
