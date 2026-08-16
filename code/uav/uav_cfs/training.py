"""Protocol-aligned training engine for the UAV/shared-service platform.

This module implements the two objectives used by the paper protocol without
starting either one at import time:

* typed persistent-edge ``t -> t+1`` supervision with TBPTT=1 and 32 control
  graphs per optimizer step; and
* complete action-coupled episodes partitioned into configurable H-step TBPTT
  windows.  Model-staged descriptors affect the next simulator state only via
  the fixed reassociation policy and its executed action.

The record adapter is intentionally concentrated in :class:`UAVRecordAdapter`
so the sharded data layer can evolve without weakening the typed model/loss
contract.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from os import PathLike
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .environment import UAVSharedServiceEnv
from .graph import (
    UAV_GRAPH_CONTRACT_VERSION,
    UAV_TARGET_CONTRACT_VERSION,
    UAVTarget,
    build_next_step_target,
    observation_to_graph,
    target_from_mapping,
    validate_graph,
)
from .model import (
    UAVDescriptorModel,
    UAVDescriptorOutput,
    UAVRecurrentState,
    UAV_MODEL_CONTRACT_VERSION,
    uav_structural_protocol_fingerprint,
)
from .policy import UAVFixedRankPolicy, UAVFixedRankPolicyConfig
from .protocol import UAVSharedProtocol
from .state import ServiceObservation, ServicePolicyDescriptors
from .runtime import load_cfg


UAV_TRAINER_SCHEMA_VERSION = 1
UAV_BATCH_SEMANTICS = "control_graphs_per_optimizer_step_tbptt1_v1"


def _root_mapping(source: Any) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    if isinstance(source, (str, PathLike)):
        path = Path(source).expanduser().resolve()
        loaded = load_cfg(path)
        if not isinstance(loaded, Mapping):
            raise TypeError("UAV training YAML must contain a mapping")
        return copy.deepcopy(dict(loaded))
    raise TypeError("UAV training source must be a root mapping or YAML path")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be an integer")
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return result


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be an integer")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return result


def _finite_float(
    value: Any,
    path: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if lower is not None and result < lower:
        raise ValueError(f"{path} must be >= {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{path} must be <= {upper}")
    return result


@dataclass(frozen=True)
class UAVDescriptorInitializer:
    """Explicit, target-free values for D0 and newly appearing edges."""

    eta: float
    intensity: float
    station_flow: float
    kind: str
    provenance: str

    def __post_init__(self) -> None:
        if self.kind != "configured_prior":
            raise ValueError("uav_pipeline.initializer.kind must be configured_prior")
        eta = _finite_float(self.eta, "initializer.eta", lower=0.0, upper=1.0)
        intensity = _finite_float(
            self.intensity, "initializer.intensity", lower=0.0
        )
        flow = _finite_float(
            self.station_flow,
            "initializer.station_flow",
            lower=0.0,
            upper=1.0,
        )
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise TypeError("initializer.provenance must be a non-empty string")
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "station_flow", flow)

    @classmethod
    def from_root(cls, root: Mapping[str, Any]) -> "UAVDescriptorInitializer":
        pipeline = _mapping(root.get("uav_pipeline"), "uav_pipeline")
        raw = _mapping(pipeline.get("initializer"), "uav_pipeline.initializer")
        required = {"kind", "eta", "intensity", "station_flow", "provenance"}
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - required)
        if missing or unknown:
            raise ValueError(
                "uav_pipeline.initializer schema mismatch; "
                f"missing={missing!r}, unknown={unknown!r}"
            )
        return cls(**raw)


@dataclass(frozen=True)
class UAVLossWeights:
    eta: float
    log1p_intensity: float
    station_flow: float
    feasibility: float

    def __post_init__(self) -> None:
        for name in (
            "eta",
            "log1p_intensity",
            "station_flow",
            "feasibility",
        ):
            value = _finite_float(
                getattr(self, name), f"loss_weights.{name}", lower=0.0
            )
            object.__setattr__(self, name, value)
        if sum(asdict(self).values()) <= 0.0:
            raise ValueError("at least one UAV loss weight must be positive")


@dataclass(frozen=True)
class UAVScheduledSamplingConfig:
    enabled: bool
    start_probability: float
    maximum_probability: float
    start_epoch: int
    end_epoch: int

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("scheduled_sampling.enabled must be bool")
        for name in ("start_probability", "maximum_probability"):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    f"scheduled_sampling.{name}",
                    lower=0.0,
                    upper=1.0,
                ),
            )
        object.__setattr__(
            self,
            "start_epoch",
            _positive_int(self.start_epoch, "scheduled_sampling.start_epoch"),
        )
        object.__setattr__(
            self,
            "end_epoch",
            _positive_int(self.end_epoch, "scheduled_sampling.end_epoch"),
        )
        if self.end_epoch < self.start_epoch:
            raise ValueError("scheduled_sampling.end_epoch must be >= start_epoch")

    def probability(self, epoch: int) -> float:
        if not self.enabled:
            return 0.0
        if epoch <= self.start_epoch:
            return self.start_probability
        if epoch >= self.end_epoch:
            return self.maximum_probability
        fraction = (epoch - self.start_epoch) / (
            self.end_epoch - self.start_epoch
        )
        return self.start_probability + fraction * (
            self.maximum_probability - self.start_probability
        )


@dataclass(frozen=True)
class UAVActionCoupledConfig:
    enabled: bool
    rollout_horizon: int
    model_feedback_probability: float
    sequence_batch_size: int
    train_horizon_steps: int

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("action_coupled_multistep.enabled must be bool")
        object.__setattr__(
            self,
            "rollout_horizon",
            _positive_int(
                self.rollout_horizon,
                "action_coupled_multistep.rollout_horizon",
            ),
        )
        object.__setattr__(
            self,
            "model_feedback_probability",
            _finite_float(
                self.model_feedback_probability,
                "action_coupled_multistep.model_feedback_probability",
                lower=0.0,
                upper=1.0,
            ),
        )
        object.__setattr__(
            self,
            "sequence_batch_size",
            _positive_int(
                self.sequence_batch_size,
                "action_coupled_multistep.sequence_batch_size",
            ),
        )
        object.__setattr__(
            self,
            "train_horizon_steps",
            _positive_int(
                self.train_horizon_steps,
                "action_coupled_multistep.train_horizon_steps",
            ),
        )


@dataclass(frozen=True)
class UAVTrainingConfig:
    objective: str
    epochs: int
    batch_size_control_graphs: int
    tbptt_steps: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    optimizer: str
    sampling_seed: int
    loss_weights: UAVLossWeights
    scheduled_sampling: UAVScheduledSamplingConfig
    action_coupled: UAVActionCoupledConfig

    @classmethod
    def from_source(cls, source: Any) -> "UAVTrainingConfig":
        root = _root_mapping(source)
        pipeline = _mapping(root.get("uav_pipeline"), "uav_pipeline")
        raw = _mapping(pipeline.get("training"), "uav_pipeline.training")
        dataset = _mapping(pipeline.get("dataset"), "uav_pipeline.dataset")
        objective = str(raw.get("objective", "one_step_teacher_forcing")).lower()
        allowed = {
            "one_step_teacher_forcing",
            "one_step_scheduled_sampling",
            "action_coupled_multistep",
            "hybrid",
        }
        if objective not in allowed:
            raise ValueError(
                "uav_pipeline.training.objective must be one of "
                + ", ".join(sorted(allowed))
            )
        batch = _positive_int(
            raw.get("batch_size_control_graphs", 32),
            "uav_pipeline.training.batch_size_control_graphs",
        )
        tbptt = _positive_int(
            raw.get("tbptt_steps", 1), "uav_pipeline.training.tbptt_steps"
        )
        if tbptt != 1:
            raise ValueError("the formal UAV one-step protocol requires TBPTT=1")
        weights_raw = _mapping(
            raw.get("loss_weights"), "uav_pipeline.training.loss_weights"
        )
        weights = UAVLossWeights(
            eta=weights_raw.get("eta"),
            log1p_intensity=weights_raw.get("log1p_intensity"),
            station_flow=weights_raw.get("station_flow"),
            feasibility=weights_raw.get("feasibility"),
        )
        sampling_raw = _mapping(
            raw.get("scheduled_sampling"),
            "uav_pipeline.training.scheduled_sampling",
        )
        scheduled = UAVScheduledSamplingConfig(
            enabled=sampling_raw.get("enabled"),
            start_probability=sampling_raw.get("start_probability"),
            maximum_probability=sampling_raw.get("maximum_probability"),
            start_epoch=sampling_raw.get("start_epoch"),
            end_epoch=sampling_raw.get("end_epoch"),
        )
        action_raw = _mapping(
            raw.get("action_coupled_multistep"),
            "uav_pipeline.training.action_coupled_multistep",
        )
        rollout_horizon = _positive_int(
            action_raw.get("rollout_horizon"),
            "action_coupled_multistep.rollout_horizon",
        )
        default_sequence_batch = max(1, batch // rollout_horizon)
        action = UAVActionCoupledConfig(
            enabled=action_raw.get("enabled"),
            rollout_horizon=rollout_horizon,
            model_feedback_probability=action_raw.get(
                "model_feedback_probability"
            ),
            sequence_batch_size=action_raw.get(
                "sequence_batch_size", default_sequence_batch
            ),
            train_horizon_steps=action_raw.get(
                "train_horizon_steps", dataset.get("train_horizon_steps")
            ),
        )
        if action.rollout_horizon * action.sequence_batch_size > batch:
            raise ValueError(
                "action-coupled sequence_batch_size * rollout_horizon cannot "
                "exceed batch_size_control_graphs"
            )
        if objective == "action_coupled_multistep" and not action.enabled:
            raise ValueError("action-coupled objective is selected but disabled")
        if objective == "one_step_scheduled_sampling" and not scheduled.enabled:
            raise ValueError("scheduled-sampling objective is selected but disabled")
        return cls(
            objective=objective,
            epochs=_positive_int(
                raw.get("epochs", 50), "uav_pipeline.training.epochs"
            ),
            batch_size_control_graphs=batch,
            tbptt_steps=tbptt,
            learning_rate=_finite_float(
                raw.get("learning_rate", 5e-4),
                "uav_pipeline.training.learning_rate",
                lower=0.0,
            ),
            weight_decay=_finite_float(
                raw.get("weight_decay", 1e-4),
                "uav_pipeline.training.weight_decay",
                lower=0.0,
            ),
            gradient_clip_norm=_finite_float(
                raw.get("gradient_clip_norm", 1.0),
                "uav_pipeline.training.gradient_clip_norm",
                lower=0.0,
            ),
            optimizer=str(raw.get("optimizer", "adamw")).lower(),
            sampling_seed=_nonnegative_int(
                raw.get("sampling_seed", root.get("seed", 0)),
                "uav_pipeline.training.sampling_seed",
            ),
            loss_weights=weights,
            scheduled_sampling=scheduled,
            action_coupled=action,
        )

    @property
    def uses_one_step(self) -> bool:
        return self.objective != "action_coupled_multistep"

    @property
    def uses_action_coupled(self) -> bool:
        return self.objective in {"action_coupled_multistep", "hybrid"}


@dataclass(frozen=True)
class UAVDescriptorLoss:
    total: torch.Tensor
    eta: torch.Tensor
    log1p_intensity: torch.Tensor
    station_flow: torch.Tensor
    feasibility: torch.Tensor
    persistent_edge_count: int


def uav_descriptor_one_step_loss(
    prediction: UAVDescriptorOutput,
    target: UAVTarget,
    weights: UAVLossWeights,
) -> UAVDescriptorLoss:
    """Compute typed loss only on source edges persistent at ``t+1``."""

    edge_count = int(prediction.eta_edge.numel())
    station_count = int(prediction.station_flow_node.numel())
    prediction.validate(edge_count, station_count)
    target.validate(edge_count, station_count)
    if prediction.eta_edge.device != target.eta_edge.device:
        raise ValueError("prediction and UAVTarget must share one device")
    persistent = target.persistent_edge
    if bool(persistent.any()):
        eta = F.mse_loss(
            prediction.eta_edge[persistent], target.eta_edge[persistent]
        )
        log_intensity = F.mse_loss(
            torch.log1p(prediction.intensity_edge[persistent]),
            target.log1p_intensity_edge[persistent],
        )
        feasibility = F.binary_cross_entropy_with_logits(
            prediction.feasible_start_logit_edge[persistent],
            target.feasibility_edge[persistent].to(
                dtype=prediction.feasible_start_logit_edge.dtype
            ),
        )
    else:
        # Preserve a valid autograd path in rare zero-persistence transitions.
        zero = (
            prediction.eta_edge.sum()
            + prediction.intensity_edge.sum()
            + prediction.feasible_start_logit_edge.sum()
        ) * 0.0
        eta = zero
        log_intensity = zero
        feasibility = zero
    station_flow = F.mse_loss(
        prediction.station_flow_node, target.station_flow_node
    )
    total = (
        weights.eta * eta
        + weights.log1p_intensity * log_intensity
        + weights.station_flow * station_flow
        + weights.feasibility * feasibility
    )
    if not bool(torch.isfinite(total.detach())):
        raise FloatingPointError("UAV descriptor loss is NaN or Inf")
    return UAVDescriptorLoss(
        total=total,
        eta=eta,
        log1p_intensity=log_intensity,
        station_flow=station_flow,
        feasibility=feasibility,
        persistent_edge_count=int(persistent.sum().item()),
    )


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    return value


class UAVRecordAdapter:
    """Single compatibility boundary for sharded UAV episode records.

    Required episode contract::

        {"records": [{"observation": graph, "next_target": target_or_none, ...}]}

    ``observation`` may also be a live ``ServiceObservation``.  Dataset/index
    loading stays in ``uav_shared.data``; this adapter only materializes typed
    tensors for training.
    """

    def __init__(self, protocol: UAVSharedProtocol, device: torch.device | str):
        self.protocol = protocol
        self.device = torch.device(device)

    @staticmethod
    def records(episode: Mapping[str, Any] | Sequence[Any]) -> Sequence[Any]:
        if isinstance(episode, Mapping):
            records = episode.get("records", episode.get("steps"))
            if records is None:
                raise ValueError("UAV episode requires records")
        else:
            records = episode
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("UAV episode records must be a sequence")
        return records

    def graph(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise TypeError("UAV record must be a mapping")
        raw = record.get("observation")
        if isinstance(raw, ServiceObservation):
            graph = observation_to_graph(raw, self.protocol)
        elif isinstance(raw, Mapping):
            graph = dict(raw)
        else:
            raise TypeError("record.observation must be a graph mapping")
        validate_graph(graph)
        graph = _tree_to_device(graph, self.device)
        validate_graph(graph)
        return graph

    def target(self, record: Mapping[str, Any]) -> UAVTarget | None:
        raw = record.get("next_target")
        if raw is None:
            return None
        if isinstance(raw, UAVTarget):
            target = raw
        elif isinstance(raw, Mapping):
            target = target_from_mapping(raw)
        else:
            raise TypeError("record.next_target must be UAVTarget, mapping, or None")
        return target.to(self.device)

    def transitions(
        self, episode: Mapping[str, Any] | Sequence[Any]
    ) -> list[tuple[dict[str, Any], UAVTarget]]:
        result: list[tuple[dict[str, Any], UAVTarget]] = []
        for record in self.records(episode):
            if not isinstance(record, Mapping):
                raise TypeError("each UAV episode record must be a mapping")
            target = self.target(record)
            if target is None:
                continue
            graph = self.graph(record)
            target.validate(
                int(graph["edge_index"].size(1)), int(graph["station_count"])
            )
            if not torch.equal(
                target.source_stable_candidate_id,
                graph["stable_candidate_id"],
            ):
                raise ValueError(
                    "target source ids do not match observation edge-row order"
                )
            result.append((graph, target))
        return result


def stage_prediction_on_graph(
    source_graph: Mapping[str, Any],
    next_graph: Mapping[str, Any],
    prediction: UAVDescriptorOutput,
    initializer: UAVDescriptorInitializer,
    *,
    detach: bool,
) -> dict[str, Any]:
    """Join a D_t prediction onto D_(t+1), initializing genuinely new edges."""

    validate_graph(source_graph)
    validate_graph(next_graph)
    source_ids = source_graph["stable_candidate_id"]
    next_ids = next_graph["stable_candidate_id"]
    station_count = int(next_graph["station_count"])
    prediction.validate(int(source_ids.numel()), station_count)
    if source_ids.device != prediction.eta_edge.device:
        raise ValueError("source graph and prediction must share one device")
    if next_ids.device != prediction.eta_edge.device:
        raise ValueError("next graph and prediction must share one device")

    def maybe_detach(value: torch.Tensor) -> torch.Tensor:
        return value.detach() if detach else value

    dtype = prediction.eta_edge.dtype
    eta = torch.full(
        (next_ids.numel(),),
        initializer.eta,
        dtype=dtype,
        device=prediction.eta_edge.device,
    )
    intensity = torch.full_like(eta, initializer.intensity)
    source_lookup = {
        int(stable_id): index for index, stable_id in enumerate(source_ids.tolist())
    }
    persistent_next: list[int] = []
    persistent_source: list[int] = []
    for next_index, stable_id in enumerate(next_ids.tolist()):
        source_index = source_lookup.get(int(stable_id))
        if source_index is not None:
            persistent_next.append(next_index)
            persistent_source.append(source_index)
    if persistent_next:
        destination_rows = torch.tensor(
            persistent_next, dtype=torch.long, device=eta.device
        )
        source_rows = torch.tensor(
            persistent_source, dtype=torch.long, device=eta.device
        )
        eta = eta.index_copy(
            0,
            destination_rows,
            maybe_detach(prediction.eta_edge).index_select(0, source_rows),
        )
        intensity = intensity.index_copy(
            0,
            destination_rows,
            maybe_detach(prediction.intensity_edge).index_select(0, source_rows),
        )
    station_flow = maybe_detach(prediction.station_flow_node)
    staged = dict(next_graph)
    staged["edge_attr"] = next_graph["edge_attr"].clone()
    staged["station_x"] = next_graph["station_x"].clone()
    staged["edge_attr"][:, 0] = eta
    staged["edge_attr"][:, 1] = torch.log1p(intensity)
    edge_station = next_graph["candidate_edge_ids"][:, 1]
    staged["edge_attr"][:, 2] = station_flow.index_select(0, edge_station)
    staged["station_x"][:, 2] = station_flow
    staged["policy_descriptors"] = {
        "eta_edge": eta,
        "intensity_edge": intensity,
        "station_flow_node": station_flow,
    }
    validate_graph(staged)
    return staged


def initialize_policy_observation(
    observation: ServiceObservation,
    initializer: UAVDescriptorInitializer,
) -> ServiceObservation:
    dtype = observation.uav_position.dtype
    device = observation.uav_position.device
    return observation.with_policy_descriptors(
        ServicePolicyDescriptors(
            eta_edge=torch.full(
                (observation.edge_count,), initializer.eta, dtype=dtype, device=device
            ),
            intensity_edge=torch.full(
                (observation.edge_count,),
                initializer.intensity,
                dtype=dtype,
                device=device,
            ),
            station_flow_node=torch.full(
                (observation.station_count,),
                initializer.station_flow,
                dtype=dtype,
                device=device,
            ),
        )
    )


def stage_prediction_on_observation(
    source_observation: ServiceObservation,
    next_observation: ServiceObservation,
    prediction: UAVDescriptorOutput,
    protocol: UAVSharedProtocol,
    initializer: UAVDescriptorInitializer,
    *,
    detach: bool,
) -> ServiceObservation:
    source_graph = observation_to_graph(source_observation, protocol)
    next_graph = observation_to_graph(next_observation, protocol)
    staged = stage_prediction_on_graph(
        source_graph, next_graph, prediction, initializer, detach=detach
    )
    fields = staged["policy_descriptors"]
    return next_observation.with_policy_descriptors(
        ServicePolicyDescriptors(
            eta_edge=fields["eta_edge"],
            intensity_edge=fields["intensity_edge"],
            station_flow_node=fields["station_flow_node"],
        )
    )


@dataclass(frozen=True)
class UAVLossSummary:
    total: float
    eta: float
    log1p_intensity: float
    station_flow: float
    feasibility: float
    transitions: int
    persistent_edges: int
    sampled_model_steps: int
    sampling_opportunities: int

    @property
    def sampled_model_fraction(self) -> float:
        return self.sampled_model_steps / max(1, self.sampling_opportunities)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sampled_model_fraction"] = self.sampled_model_fraction
        return result


class _LossAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.eta = 0.0
        self.log_intensity = 0.0
        self.station_flow = 0.0
        self.feasibility = 0.0
        self.transitions = 0
        self.persistent = 0
        self.sampled = 0
        self.opportunities = 0

    def add(self, loss: UAVDescriptorLoss) -> None:
        self.total += float(loss.total.detach().item())
        self.eta += float(loss.eta.detach().item())
        self.log_intensity += float(loss.log1p_intensity.detach().item())
        self.station_flow += float(loss.station_flow.detach().item())
        self.feasibility += float(loss.feasibility.detach().item())
        self.transitions += 1
        self.persistent += loss.persistent_edge_count

    def sample(self, model_feedback: bool) -> None:
        self.opportunities += 1
        self.sampled += int(model_feedback)

    def merge(self, other: "_LossAccumulator") -> None:
        for name in (
            "total",
            "eta",
            "log_intensity",
            "station_flow",
            "feasibility",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.transitions += other.transitions
        self.persistent += other.persistent
        self.sampled += other.sampled
        self.opportunities += other.opportunities

    def summary(self) -> UAVLossSummary:
        if self.transitions <= 0:
            raise ValueError("UAV loss summary requires at least one transition")
        scale = 1.0 / self.transitions
        return UAVLossSummary(
            total=self.total * scale,
            eta=self.eta * scale,
            log1p_intensity=self.log_intensity * scale,
            station_flow=self.station_flow * scale,
            feasibility=self.feasibility * scale,
            transitions=self.transitions,
            persistent_edges=self.persistent,
            sampled_model_steps=self.sampled,
            sampling_opportunities=self.opportunities,
        )


@dataclass(frozen=True)
class _ActionWindow:
    objective: torch.Tensor
    transitions: int
    losses: _LossAccumulator


@dataclass(frozen=True)
class UAVFitResult:
    best_epoch: int
    best_validation: float
    best_checkpoint: Path
    last_checkpoint: Path
    history: tuple[Mapping[str, Any], ...]


class UAVTrainer:
    """Typed trainer with automatic best/last checkpoint persistence."""

    def __init__(
        self,
        model: UAVDescriptorModel,
        config: Mapping[str, Any] | str | PathLike[str],
        *,
        device: torch.device | str,
        best_checkpoint_path: str | PathLike[str],
        last_checkpoint_path: str | PathLike[str],
        dataset_metadata: Mapping[str, Any] | None = None,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not isinstance(model, UAVDescriptorModel):
            raise TypeError("model must be UAVDescriptorModel")
        self.root = _root_mapping(config)
        self.settings = UAVTrainingConfig.from_source(self.root)
        self.protocol = model.protocol
        self.initializer = UAVDescriptorInitializer.from_root(self.root)
        self.policy_config = UAVFixedRankPolicyConfig.from_source(self.root)
        self.policy = UAVFixedRankPolicy(self.policy_config)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        if self.settings.optimizer != "adamw":
            raise ValueError("formal UAV training requires optimizer=adamw")
        if self.settings.learning_rate <= 0.0:
            raise ValueError("UAV learning rate must be positive")
        if self.settings.gradient_clip_norm <= 0.0:
            raise ValueError("UAV gradient_clip_norm must be positive")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        self.best_path = Path(best_checkpoint_path).expanduser()
        self.last_path = Path(last_checkpoint_path).expanduser()
        self.dataset_metadata = dict(dataset_metadata or {})
        self.progress = progress
        self.order_generator = torch.Generator(device="cpu")
        self.order_generator.manual_seed(self.settings.sampling_seed)
        self.sampling_generator = torch.Generator(device="cpu")
        self.sampling_generator.manual_seed(self.settings.sampling_seed + 1)
        self.global_step = 0
        self.best_validation = math.inf
        self.best_epoch = 0
        self.start_epoch = 1
        self.history: list[Mapping[str, Any]] = []

    def _sample(self, probability: float) -> bool:
        return bool(
            torch.rand((), generator=self.sampling_generator).item() < probability
        )

    def _optimizer_step(self) -> None:
        norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.settings.gradient_clip_norm
        )
        if not bool(torch.isfinite(torch.as_tensor(norm))):
            raise FloatingPointError("UAV gradient norm is NaN or Inf")
        self.optimizer.step()
        self.global_step += 1

    def train_one_step_epoch(
        self,
        episodes: Sequence[Mapping[str, Any]],
        *,
        epoch: int,
    ) -> UAVLossSummary:
        """Train with exactly 32 (configured) transition graphs per full batch."""

        if len(episodes) == 0:
            raise ValueError("UAV training split is empty")
        self.model.train()
        adapter = UAVRecordAdapter(self.protocol, self.device)
        probability = (
            0.0
            if self.settings.objective == "one_step_teacher_forcing"
            else self.settings.scheduled_sampling.probability(epoch)
        )
        accumulator = _LossAccumulator()
        order = torch.randperm(
            len(episodes), generator=self.order_generator
        ).tolist()
        pending_objective: torch.Tensor | None = None
        pending_graphs = 0
        self.optimizer.zero_grad(set_to_none=True)

        def flush() -> None:
            nonlocal pending_objective, pending_graphs
            if pending_graphs == 0:
                return
            if pending_objective is None:
                raise RuntimeError("missing UAV batch objective")
            (pending_objective / pending_graphs).backward()
            self._optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)
            pending_objective = None
            pending_graphs = 0

        for episode_index in order:
            transitions = adapter.transitions(episodes[episode_index])
            if not transitions:
                continue
            state: UAVRecurrentState | None = None
            current_graph = transitions[0][0]
            for index, (_teacher_graph, target) in enumerate(transitions):
                prediction, state = self.model.predict_step(current_graph, state)
                loss = uav_descriptor_one_step_loss(
                    prediction, target, self.settings.loss_weights
                )
                pending_objective = (
                    loss.total
                    if pending_objective is None
                    else pending_objective + loss.total
                )
                pending_graphs += 1
                accumulator.add(loss)
                # The formal one-step objective fixes TBPTT to one graph.
                state = state.detach()
                if index + 1 < len(transitions):
                    base_next = transitions[index + 1][0]
                    use_model = self._sample(probability)
                    accumulator.sample(use_model)
                    if use_model:
                        current_graph = stage_prediction_on_graph(
                            current_graph,
                            base_next,
                            prediction,
                            self.initializer,
                            detach=True,
                        )
                    else:
                        current_graph = base_next
                if pending_graphs >= self.settings.batch_size_control_graphs:
                    flush()
        flush()
        return accumulator.summary()

    def _action_coupled_windows(
        self,
        seed: int,
        *,
        feedback_probability: float,
    ) -> Iterator[_ActionWindow]:
        action_cfg = self.settings.action_coupled
        episode_protocol = replace(self.protocol, episode_seed=int(seed))
        environment = UAVSharedServiceEnv(episode_protocol, device=self.device)
        observation = initialize_policy_observation(
            environment.reset_control(), self.initializer
        )
        state: UAVRecurrentState | None = None
        window_objective: torch.Tensor | None = None
        window_transitions = 0
        window_losses = _LossAccumulator()
        # A descriptor target requires a nonterminal next observation.
        transition_limit = min(
            action_cfg.train_horizon_steps - 1,
            episode_protocol.horizon_steps - 1,
        )
        if transition_limit <= 0:
            raise ValueError("action-coupled UAV training requires at least two steps")
        for transition_index in range(transition_limit):
            action = self.policy.select_action(observation)
            prediction, state = self.model.predict_step(observation, state)
            next_observation, _execution, done = environment.step_action(action)
            if done or next_observation is None:
                raise RuntimeError("UAV environment ended before a supervised target")
            target = build_next_step_target(observation, next_observation).to(
                self.device
            )
            loss = uav_descriptor_one_step_loss(
                prediction, target, self.settings.loss_weights
            )
            window_objective = (
                loss.total
                if window_objective is None
                else window_objective + loss.total
            )
            window_transitions += 1
            window_losses.add(loss)
            if transition_index + 1 < transition_limit:
                use_model = self._sample(feedback_probability)
                window_losses.sample(use_model)
                if use_model:
                    observation = stage_prediction_on_observation(
                        observation,
                        next_observation,
                        prediction,
                        episode_protocol,
                        self.initializer,
                        detach=False,
                    )
                else:
                    observation = next_observation
            else:
                observation = next_observation
            boundary = (
                window_transitions == action_cfg.rollout_horizon
                or transition_index + 1 == transition_limit
            )
            if boundary:
                if window_objective is None:
                    raise RuntimeError("empty UAV action-coupled window")
                objective = window_objective / window_transitions
                # Truncate both recurrence and descriptor autoregression at H.
                state = state.detach()
                observation = observation.with_policy_descriptors(
                    ServicePolicyDescriptors(
                        eta_edge=observation.policy_descriptors.eta_edge.detach(),
                        intensity_edge=(
                            observation.policy_descriptors.intensity_edge.detach()
                        ),
                        station_flow_node=(
                            observation.policy_descriptors.station_flow_node.detach()
                        ),
                    )
                )
                yield _ActionWindow(
                    objective=objective,
                    transitions=window_transitions,
                    losses=window_losses,
                )
                window_objective = None
                window_transitions = 0
                window_losses = _LossAccumulator()

    def train_action_coupled_epoch(
        self, train_seeds: Sequence[int], *, epoch: int
    ) -> UAVLossSummary:
        if not train_seeds:
            raise ValueError("action-coupled UAV training requires train seeds")
        seeds = [
            _nonnegative_int(seed, "action_train_seed") for seed in train_seeds
        ]
        self.model.train()
        total = _LossAccumulator()
        feedback = self.settings.action_coupled.model_feedback_probability
        order = torch.randperm(
            len(seeds), generator=self.order_generator
        ).tolist()
        ordered = [seeds[index] for index in order]
        size = self.settings.action_coupled.sequence_batch_size
        for start in range(0, len(ordered), size):
            generators: list[Iterator[_ActionWindow]] = [
                self._action_coupled_windows(
                    seed,
                    feedback_probability=feedback,
                )
                for seed in ordered[start : start + size]
            ]
            while generators:
                active: list[Iterator[_ActionWindow]] = []
                windows: list[_ActionWindow] = []
                for generator in generators:
                    try:
                        window = next(generator)
                    except StopIteration:
                        continue
                    active.append(generator)
                    windows.append(window)
                generators = active
                if not windows:
                    break
                control_graphs = sum(window.transitions for window in windows)
                if control_graphs > self.settings.batch_size_control_graphs:
                    raise RuntimeError(
                        "action-coupled UAV optimizer batch exceeds graph budget"
                    )
                objective = sum(
                    window.objective * window.transitions for window in windows
                ) / control_graphs
                self.optimizer.zero_grad(set_to_none=True)
                objective.backward()
                self._optimizer_step()
                for window in windows:
                    total.merge(window.losses)
        return total.summary()

    @torch.no_grad()
    def validate_one_step(
        self, episodes: Sequence[Mapping[str, Any]]
    ) -> UAVLossSummary:
        if len(episodes) == 0:
            raise ValueError("UAV validation split is empty")
        self.model.eval()
        adapter = UAVRecordAdapter(self.protocol, self.device)
        total = _LossAccumulator()
        for episode in episodes:
            state: UAVRecurrentState | None = None
            for graph, target in adapter.transitions(episode):
                prediction, state = self.model.predict_step(graph, state)
                state = state.detach()
                total.add(
                    uav_descriptor_one_step_loss(
                        prediction, target, self.settings.loss_weights
                    )
                )
        return total.summary()

    def _training_fingerprint(self) -> str:
        payload = {
            "settings": asdict(self.settings),
            "initializer": asdict(self.initializer),
            "policy": asdict(self.policy_config),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _checkpoint_payload(
        self, *, epoch: int, validation: float
    ) -> dict[str, Any]:
        numpy_state = np.random.get_state()
        return {
            "uav_trainer_schema_version": UAV_TRAINER_SCHEMA_VERSION,
            "uav_model_contract_version": UAV_MODEL_CONTRACT_VERSION,
            "uav_graph_contract_version": UAV_GRAPH_CONTRACT_VERSION,
            "uav_target_contract_version": UAV_TARGET_CONTRACT_VERSION,
            "protocol_fingerprint": self.protocol.fingerprint,
            "structural_protocol_fingerprint": (
                uav_structural_protocol_fingerprint(self.protocol)
            ),
            "uav_model_fingerprint": self.model.fingerprint,
            "uav_model_signature": self.model.signature,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "best_validation": float(self.best_validation),
            "best_epoch": int(self.best_epoch),
            "latest_validation": float(validation),
            "history": copy.deepcopy(self.history),
            "training_settings": asdict(self.settings),
            "training_fingerprint": self._training_fingerprint(),
            "initializer": asdict(self.initializer),
            "policy": asdict(self.policy_config),
            "batch_semantics": UAV_BATCH_SEMANTICS,
            "batch_size_control_graphs": (
                self.settings.batch_size_control_graphs
            ),
            "training_run_seed": self.dataset_metadata.get(
                "training_run_seed"
            ),
            "dataset_metadata": copy.deepcopy(self.dataset_metadata),
            "rng_state": {
                "python": random.getstate(),
                "numpy": {
                    "bit_generator": numpy_state[0],
                    "state": torch.from_numpy(numpy_state[1].copy()),
                    "position": int(numpy_state[2]),
                    "has_gauss": int(numpy_state[3]),
                    "cached_gaussian": float(numpy_state[4]),
                },
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else []
                ),
                "order_generator": self.order_generator.get_state(),
                "sampling_generator": self.sampling_generator.get_state(),
            },
        }

    @staticmethod
    def _atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)

    def resume(self, checkpoint: str | PathLike[str]) -> None:
        path = Path(checkpoint).expanduser().resolve()
        # Keep serialized RNG byte tensors on CPU. Optimizer/model loaders move
        # their own state to the parameter device, whereas the RNG APIs require
        # CPU state tensors even when training resumes on CUDA.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError("UAV training checkpoint must be a mapping")
        if payload.get("uav_trainer_schema_version") != UAV_TRAINER_SCHEMA_VERSION:
            raise ValueError("UAV trainer checkpoint schema mismatch")
        if payload.get("uav_model_fingerprint") != self.model.fingerprint:
            raise ValueError("UAV resume model fingerprint mismatch")
        if payload.get("training_fingerprint") != self._training_fingerprint():
            raise ValueError("UAV resume training-plan fingerprint mismatch")
        expected_run_seed = self.dataset_metadata.get("training_run_seed")
        if type(expected_run_seed) is not int or expected_run_seed < 0:
            raise ValueError(
                "current UAV dataset metadata requires a non-negative "
                "training_run_seed before resume"
            )
        if payload.get("training_run_seed") != expected_run_seed:
            raise ValueError(
                "UAV resume checkpoint belongs to a different independent "
                f"training run: {payload.get('training_run_seed')!r} != "
                f"{expected_run_seed}"
            )
        saved_dataset = payload.get("dataset_metadata")
        if not isinstance(saved_dataset, Mapping):
            raise ValueError("UAV resume checkpoint lacks dataset metadata")
        for name in (
            "sha256",
            "uav_dataset_schema_version",
            "uav_graph_contract_version",
            "uav_target_contract_version",
            "dataset_kind",
        ):
            current_value = self.dataset_metadata.get(name)
            saved_value = saved_dataset.get(name)
            if current_value is None or saved_value is None:
                raise ValueError(
                    f"UAV resume dataset metadata requires {name!r}"
                )
            if saved_value != current_value:
                raise ValueError(
                    f"UAV resume dataset {name} mismatch: "
                    f"{saved_value!r} != {current_value!r}"
                )
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload["global_step"])
        self.best_validation = float(payload["best_validation"])
        self.best_epoch = int(payload["best_epoch"])
        self.start_epoch = int(payload["epoch"]) + 1
        self.history = list(payload.get("history", []))
        rng = payload.get("rng_state")
        if isinstance(rng, Mapping):
            random.setstate(rng["python"])
            numpy_state = rng["numpy"]
            if not isinstance(numpy_state, Mapping):
                raise TypeError("checkpoint numpy RNG state must be a mapping")
            np.random.set_state(
                (
                    str(numpy_state["bit_generator"]),
                    torch.as_tensor(numpy_state["state"]).cpu().numpy(),
                    int(numpy_state["position"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
            torch.set_rng_state(torch.as_tensor(rng["torch_cpu"]).cpu())
            if torch.cuda.is_available() and rng.get("torch_cuda"):
                torch.cuda.set_rng_state_all(
                    [torch.as_tensor(value).cpu() for value in rng["torch_cuda"]]
                )
            self.order_generator.set_state(
                torch.as_tensor(rng["order_generator"]).cpu()
            )
            self.sampling_generator.set_state(
                torch.as_tensor(rng["sampling_generator"]).cpu()
            )

    def fit(
        self,
        train_episodes: Sequence[Mapping[str, Any]],
        validation_episodes: Sequence[Mapping[str, Any]],
        *,
        action_train_seeds: Sequence[int] = (),
    ) -> UAVFitResult:
        for epoch in range(self.start_epoch, self.settings.epochs + 1):
            one_step = (
                self.train_one_step_epoch(train_episodes, epoch=epoch)
                if self.settings.uses_one_step
                else None
            )
            action = (
                self.train_action_coupled_epoch(action_train_seeds, epoch=epoch)
                if self.settings.uses_action_coupled
                else None
            )
            validation = self.validate_one_step(validation_episodes)
            validation_metric = validation.total
            improved = validation_metric < self.best_validation
            if improved:
                self.best_validation = validation_metric
                self.best_epoch = epoch
            row: dict[str, Any] = {
                "epoch": epoch,
                "global_step": self.global_step,
                "validation": validation.as_dict(),
                "validation_metric": validation_metric,
                "one_step": one_step.as_dict() if one_step is not None else None,
                "action_coupled": action.as_dict() if action is not None else None,
            }
            self.history.append(row)
            payload = self._checkpoint_payload(
                epoch=epoch, validation=validation_metric
            )
            self._atomic_save(self.last_path, payload)
            if improved:
                self._atomic_save(self.best_path, payload)
            if self.progress is not None:
                self.progress(row)
        if self.best_epoch <= 0:
            raise RuntimeError("UAV training produced no best checkpoint")
        return UAVFitResult(
            best_epoch=self.best_epoch,
            best_validation=self.best_validation,
            best_checkpoint=self.best_path,
            last_checkpoint=self.last_path,
            history=tuple(self.history),
        )


__all__ = [
    "UAVActionCoupledConfig",
    "UAVDescriptorInitializer",
    "UAVDescriptorLoss",
    "UAVFitResult",
    "UAVLossSummary",
    "UAVLossWeights",
    "UAVRecordAdapter",
    "UAVScheduledSamplingConfig",
    "UAVTrainer",
    "UAVTrainingConfig",
    "UAV_BATCH_SEMANTICS",
    "UAV_TRAINER_SCHEMA_VERSION",
    "initialize_policy_observation",
    "stage_prediction_on_graph",
    "stage_prediction_on_observation",
    "uav_descriptor_one_step_loss",
]
