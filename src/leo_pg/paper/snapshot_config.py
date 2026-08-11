"""Configuration adapter for the isolated Snapshot experiment pipeline.

The executable scripts intentionally share this parser so generation,
training, and evaluation cannot select different feature normalization,
feasibility margin, initial stream, loss, or method-specific score rule from
the same YAML file.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .snapshot import (
    SnapshotLossWeights,
    SnapshotMarginConfig,
    SnapshotPolicyConfig,
)
from .snapshot_data import ConstantSnapshotInitializer, SnapshotFeatureConfig
from .snapshot_models import (
    normalize_snapshot_method,
    resolved_snapshot_model_config,
)


SNAPSHOT_PIPELINE_CONFIG_VERSION = 1


def _mapping(parent: Mapping[str, Any], name: str, *, path: str) -> dict[str, Any]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{name} mapping is required")
    return copy.deepcopy(dict(value))


def _finite(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _probability(name: str, value: Any) -> float:
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _pure(value: Any, *, path: str = "root") -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _pure(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_pure(item, path=f"{path}[]") for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().contiguous().clone()
        if result.is_floating_point() and not bool(torch.isfinite(result).all()):
            raise ValueError(f"{path} contains a non-finite tensor")
        return result
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        return _finite(path, value)
    raise TypeError(f"{path} has unsupported type {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _pure(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _initial_load(value: Any) -> float | list[float]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _finite("paper_snapshot.initial_oracle_admitted_load", value.item(), nonnegative=True)
        values = value.detach().cpu().flatten().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        return _finite(
            "paper_snapshot.initial_oracle_admitted_load",
            value,
            nonnegative=True,
        )
    if not values:
        raise ValueError("paper_snapshot.initial_oracle_admitted_load cannot be empty")
    return [
        _finite(
            f"paper_snapshot.initial_oracle_admitted_load[{index}]",
            item,
            nonnegative=True,
        )
        for index, item in enumerate(values)
    ]


@dataclass(frozen=True)
class ResolvedSnapshotPipeline:
    method: str
    resolved_model_config: dict[str, Any]
    feature_config: SnapshotFeatureConfig
    margin_config: SnapshotMarginConfig
    policy_config: SnapshotPolicyConfig
    policy_provenance: str
    initializer: ConstantSnapshotInitializer
    initializer_config: dict[str, Any]
    loss_weights: SnapshotLossWeights
    configured_decision_aware_weight: float
    effective_decision_aware_weight: float
    initial_oracle_admitted_load: float | list[float]
    dataset_output_template: str

    def dataset_path(self, override: str | Path | None = None) -> Path:
        if override is not None:
            result = Path(override).expanduser()
        else:
            try:
                rendered = self.dataset_output_template.format(method=self.method)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "paper_snapshot.dataset.output_template may only use {method}"
                ) from exc
            result = Path(rendered).expanduser()
        if result.suffix.lower() != ".pt":
            raise ValueError("Snapshot dataset index path must end in .pt")
        return result

    def manifest(self) -> dict[str, Any]:
        return {
            "snapshot_pipeline_config_version": SNAPSHOT_PIPELINE_CONFIG_VERSION,
            "method": self.method,
            "feature": asdict(self.feature_config),
            "margin": asdict(self.margin_config),
            "policy": asdict(self.policy_config),
            "policy_provenance": self.policy_provenance,
            "stream_initializer": dict(self.initializer_config),
            "initializer_fingerprint": self.initializer.fingerprint,
            "loss_weights": asdict(self.loss_weights),
            "decision_aware_weight": {
                "configured": self.configured_decision_aware_weight,
                "effective": self.effective_decision_aware_weight,
                "scope": "snapshot_da_gwm_only",
            },
            "initial_oracle_admitted_load": _pure(
                self.initial_oracle_admitted_load
            ),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.manifest())


def resolve_snapshot_pipeline(
    root: Mapping[str, Any], method: str
) -> ResolvedSnapshotPipeline:
    if not isinstance(root, Mapping):
        raise TypeError("root configuration must be a mapping")
    normalized = normalize_snapshot_method(method)
    resolved = resolved_snapshot_model_config(root, normalized)
    snapshot = _mapping(resolved, "paper_snapshot", path="root")
    dataset = _mapping(snapshot, "dataset", path="paper_snapshot")
    output_template = dataset.get("output_template")
    if not isinstance(output_template, str) or not output_template.strip():
        raise ValueError("paper_snapshot.dataset.output_template is required")
    if "{method}" not in output_template:
        raise ValueError(
            "paper_snapshot.dataset.output_template must contain {method} so "
            "Snapshot-oracle datasets remain method-specific"
        )

    feature = SnapshotFeatureConfig.from_mapping(
        _mapping(snapshot, "feature", path="paper_snapshot")
    )
    margin = SnapshotMarginConfig.from_mapping(
        _mapping(snapshot, "margin", path="paper_snapshot")
    )
    policies = _mapping(snapshot, "policies", path="paper_snapshot")
    selected_policy = policies.get(normalized)
    if not isinstance(selected_policy, Mapping):
        raise ValueError(f"paper_snapshot.policies.{normalized} is required")
    selected_policy = copy.deepcopy(dict(selected_policy))
    policy = SnapshotPolicyConfig.from_mapping(selected_policy)
    provenance = selected_policy.get("provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(
            f"paper_snapshot.policies.{normalized}.provenance is required"
        )

    initializer_cfg = _mapping(
        snapshot, "stream_initializer", path="paper_snapshot"
    )
    if initializer_cfg.get("kind") != "constant_snapshot_v1":
        raise ValueError(
            "paper_snapshot.stream_initializer.kind must be constant_snapshot_v1"
        )
    required_initializer = ("gamma", "feasibility_margin", "admitted_load")
    missing = [name for name in required_initializer if name not in initializer_cfg]
    if missing:
        raise ValueError(
            "paper_snapshot.stream_initializer is missing: " + ", ".join(missing)
        )
    initializer = ConstantSnapshotInitializer(
        gamma=initializer_cfg["gamma"],
        feasibility_margin=initializer_cfg["feasibility_margin"],
        admitted_load=initializer_cfg["admitted_load"],
    )
    loss_weights = SnapshotLossWeights.from_mapping(
        _mapping(snapshot, "loss_weights", path="paper_snapshot")
    )
    configured_decision = _finite(
        "paper_snapshot.decision_aware_weight",
        snapshot.get("decision_aware_weight", 0.0),
        nonnegative=True,
    )
    # Decision-aware ranking is the defining DA-GWM objective.  The shared
    # scalar is retained in the manifest but must not be silently applied to
    # Snapshot MLP/PhysiCK/LTT-R models that expose no decision-loss hook.
    effective_decision = (
        configured_decision if normalized == "snapshot_da_gwm" else 0.0
    )
    if "initial_oracle_admitted_load" not in snapshot:
        raise ValueError("paper_snapshot.initial_oracle_admitted_load is required")
    initial_load = _initial_load(snapshot["initial_oracle_admitted_load"])

    return ResolvedSnapshotPipeline(
        method=normalized,
        resolved_model_config=resolved,
        feature_config=feature,
        margin_config=margin,
        policy_config=policy,
        policy_provenance=provenance.strip(),
        initializer=initializer,
        initializer_config={
            "kind": "constant_snapshot_v1",
            "gamma": initializer.gamma,
            "feasibility_margin": initializer.feasibility_margin,
            "admitted_load": initializer.admitted_load,
        },
        loss_weights=loss_weights,
        configured_decision_aware_weight=configured_decision,
        effective_decision_aware_weight=effective_decision,
        initial_oracle_admitted_load=initial_load,
        dataset_output_template=output_template,
    )


@dataclass(frozen=True)
class SnapshotTrainingSettings:
    epochs: int
    batch_size_control_graphs: int
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    seed: int
    device: str
    scheduled_sampling_enabled: bool
    rollin_start_probability: float
    rollin_maximum_probability: float
    rollin_start_epoch: int
    rollin_end_epoch: int
    validation_rollin_probability: float

    @classmethod
    def from_config(cls, root: Mapping[str, Any]) -> "SnapshotTrainingSettings":
        training = _mapping(root, "paper_training", path="root")
        epochs = _positive_int("paper_training.epochs", training.get("epochs"))
        batch_size = _positive_int(
            "paper_training.batch_size_control_graphs",
            training.get("batch_size_control_graphs"),
        )
        optimizer_raw = training.get("optimizer")
        if isinstance(optimizer_raw, Mapping):
            optimizer_cfg = dict(optimizer_raw)
            optimizer_name = str(optimizer_cfg.get("name", "")).strip().lower()
            learning_rate = _finite(
                "paper_training.optimizer.lr",
                optimizer_cfg.get("lr"),
            )
            weight_decay = _finite(
                "paper_training.optimizer.weight_decay",
                optimizer_cfg.get("weight_decay", 0.0),
                nonnegative=True,
            )
        else:
            optimizer_name = str(optimizer_raw).strip().lower()
            learning_rate = _finite(
                "paper_training.learning_rate",
                training.get("learning_rate"),
            )
            weight_decay = _finite(
                "paper_training.weight_decay",
                training.get("weight_decay", 0.0),
                nonnegative=True,
            )
        if optimizer_name not in {"adam", "adamw"}:
            raise ValueError("paper_training.optimizer must be adam or adamw")
        if learning_rate <= 0.0:
            raise ValueError("paper training learning rate must be positive")
        clip = _finite(
            "paper_training.gradient_clip_norm",
            training.get("gradient_clip_norm"),
        )
        if clip <= 0.0:
            raise ValueError("paper_training.gradient_clip_norm must be positive")
        schedule_name = str(
            training.get("learning_rate_schedule", "constant")
        ).strip().lower()
        if schedule_name != "constant":
            raise ValueError(
                "Snapshot pipeline currently requires learning_rate_schedule=constant"
            )
        checkpoint_selection = str(
            training.get("checkpoint_selection", "minimum_validation_loss")
        ).strip().lower()
        if checkpoint_selection != "minimum_validation_loss":
            raise ValueError(
                "Snapshot checkpoint_selection must be minimum_validation_loss"
            )
        seed_raw = root.get("seed", 0)
        if isinstance(seed_raw, bool) or int(seed_raw) != seed_raw or int(seed_raw) < 0:
            raise ValueError("seed must be a non-negative integer")
        device = str(training.get("device", "cuda")).strip().lower()
        if device not in {"cpu", "cuda"}:
            raise ValueError("paper_training.device must be cpu or cuda")
        scheduled = training.get("scheduled_sampling", {})
        if not isinstance(scheduled, Mapping):
            raise TypeError("paper_training.scheduled_sampling must be a mapping")
        enabled = scheduled.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("paper_training.scheduled_sampling.enabled must be bool")
        start = _probability(
            "paper_training.scheduled_sampling.start_probability",
            scheduled.get("start_probability", 0.0),
        )
        maximum = _probability(
            "paper_training.scheduled_sampling.maximum_probability",
            scheduled.get("maximum_probability", start),
        )
        start_epoch = int(scheduled.get("start_epoch", 0))
        end_epoch = int(scheduled.get("end_epoch", start_epoch))
        if start_epoch < 0 or end_epoch < start_epoch:
            raise ValueError("scheduled-sampling epoch bounds are invalid")
        validation_probability = _probability(
            "paper_training.snapshot_validation_rollin_probability",
            training.get("snapshot_validation_rollin_probability", 0.0),
        )
        return cls(
            epochs=epochs,
            batch_size_control_graphs=batch_size,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=clip,
            seed=int(seed_raw),
            device=device,
            scheduled_sampling_enabled=enabled,
            rollin_start_probability=start,
            rollin_maximum_probability=maximum,
            rollin_start_epoch=start_epoch,
            rollin_end_epoch=end_epoch,
            validation_rollin_probability=validation_probability,
        )

    def train_rollin_probability(self, epoch: int) -> float:
        if not self.scheduled_sampling_enabled:
            return 0.0
        if epoch <= self.rollin_start_epoch:
            return self.rollin_start_probability
        if epoch >= self.rollin_end_epoch:
            return self.rollin_maximum_probability
        span = max(1, self.rollin_end_epoch - self.rollin_start_epoch)
        alpha = (epoch - self.rollin_start_epoch) / span
        return self.rollin_start_probability + alpha * (
            self.rollin_maximum_probability - self.rollin_start_probability
        )

    def manifest(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.manifest())


__all__ = [
    "SNAPSHOT_PIPELINE_CONFIG_VERSION",
    "ResolvedSnapshotPipeline",
    "SnapshotTrainingSettings",
    "canonical_sha256",
    "resolve_snapshot_pipeline",
]
