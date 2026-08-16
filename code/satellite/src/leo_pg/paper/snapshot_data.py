"""Scientifically isolated data contract for the paper Snapshot condition.

Snapshot is not an alias for the Intensity--Flow feature table.  This module
rebuilds every model tensor from an allow-list:

* scaled position and velocity;
* current elevation and distance geometry;
* current Snapshot SINR cue and instantaneous-occupancy feasibility margin;
* current admitted users divided by fixed capacity; and
* association/dwell history.

Neither integrated intensity nor the raw/EMA flow state is read when building
``SnapshotControlInput``, its serialized form, or ``SnapshotTarget``.  The
simulator's EMA-load feasibility predicate remains authoritative for action
execution, but it is not a real-valued Snapshot model/controller feature.

The formal generator is :func:`generate_snapshot_oracle_episode`: it advances
the environment with the Snapshot oracle controller.  Converting an existing
Intensity--Flow episode is exposed separately as
:func:`snapshot_episode_from_paper_episode` and labelled diagnostic because it
inherits the Intensity--Flow behaviour policy and induced state distribution;
such conversions are not valid factorial training data.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch.utils.data import Dataset

from ..sim.paper_environment import PAPER_PROTOCOL_VERSION, PaperAlignedLEOEnv
from ..sim.state import ControlObservation, ExecutionResult, ServingAction
from .dataset import build_split_manifest
from .snapshot import (
    SnapshotFixedRankController,
    SnapshotMarginConfig,
    SnapshotOutput,
    SnapshotPolicyConfig,
    SnapshotTarget,
    instantaneous_admitted_load_snapshot,
    simulator_snapshot_output,
)


SNAPSHOT_FEATURE_CONTRACT_VERSION = 1
SNAPSHOT_TARGET_CONTRACT_VERSION = 1
SNAPSHOT_DATASET_SCHEMA_VERSION = 1
SNAPSHOT_DATASET_KIND = "leo_pg.paper.snapshot_oracle_action_coupled"
SNAPSHOT_DIAGNOSTIC_DATASET_KIND = "leo_pg.paper.snapshot_from_if_diagnostic"
SNAPSHOT_SHARD_VERSION = 1
SNAPSHOT_SPLITS = ("train", "val", "test")

SNAPSHOT_NODE_FEATURE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "snapshot_instantaneous_admitted_load_normalized",
)
SNAPSHOT_EDGE_FEATURE_NAMES = (
    "elevation_normalized",
    "log_distance",
    "snapshot_gamma_normalized",
    "snapshot_feasibility_margin",
    "snapshot_destination_instantaneous_admitted_load_normalized",
    "is_current_association",
    "dwell_normalized",
)


def _finite_real(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _cpu(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.detach().cpu().contiguous().clone()
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _json_safe(value: Any, *, path: str = "root") -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, path=f"{path}[]") for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _cpu(value, path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if callable(value):
        fingerprint = getattr(value, "__protocol_fingerprint__", None)
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError(
                f"{path} callable requires a stable __protocol_fingerprint__"
            )
        return {
            "callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__qualname__)}",
            "protocol_fingerprint": fingerprint.strip(),
        }
    raise TypeError(f"{path} has unsupported type {type(value).__name__}")


@dataclass(frozen=True)
class SnapshotFeatureConfig:
    """Explicit normalization of the two raw Snapshot input domains."""

    gamma_scale: float
    admitted_load_scale: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "gamma_scale", _finite_real("gamma_scale", self.gamma_scale, positive=True)
        )
        object.__setattr__(
            self,
            "admitted_load_scale",
            _finite_real(
                "admitted_load_scale", self.admitted_load_scale, positive=True
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SnapshotFeatureConfig":
        if not isinstance(value, Mapping):
            raise TypeError("Snapshot feature config must be a mapping")
        missing = [name for name in ("gamma_scale", "admitted_load_scale") if name not in value]
        if missing:
            raise ValueError("Snapshot feature config is missing: " + ", ".join(missing))
        return cls(
            gamma_scale=value["gamma_scale"],
            admitted_load_scale=value["admitted_load_scale"],
        )


@dataclass(frozen=True)
class SnapshotControlInput:
    """Complete controller/model input with no Intensity--Flow descriptor channel."""

    observation_id: tuple[int, int]
    node_x: torch.Tensor
    edge_index: torch.Tensor
    edge_z: torch.Tensor
    edge_type: torch.Tensor
    candidate_edge_ids: torch.Tensor
    feasible_edge: torch.Tensor
    current_serving: torch.Tensor
    hold_steps: torch.Tensor
    user_order: torch.Tensor
    descriptors: SnapshotOutput
    initialized_edge: torch.Tensor
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def episode_seed(self) -> int:
        return int(self.observation_id[0])

    @property
    def epoch(self) -> int:
        return int(self.observation_id[1])

    @property
    def user_count(self) -> int:
        return int(self.current_serving.numel())

    @property
    def satellite_count(self) -> int:
        return int(self.node_x.size(0) - self.user_count)

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.size(1))

    def validate(self) -> None:
        if (
            not isinstance(self.observation_id, tuple)
            or len(self.observation_id) != 2
            or self.episode_seed < 0
            or self.epoch < 0
        ):
            raise ValueError("observation_id must be a non-negative (seed, epoch)")
        n = self.user_count + self.satellite_count
        e = self.edge_count
        if self.user_count <= 0 or self.satellite_count <= 0:
            raise ValueError("Snapshot input requires users and satellites")
        if self.node_x.shape != (n, len(SNAPSHOT_NODE_FEATURE_NAMES)):
            raise ValueError("Snapshot node_x does not match its seven-field contract")
        if self.edge_index.shape != (2, e) or self.edge_index.dtype != torch.long:
            raise ValueError("edge_index must have shape [2,E] and dtype long")
        if self.edge_z.shape != (e, len(SNAPSHOT_EDGE_FEATURE_NAMES)):
            raise ValueError("Snapshot edge_z does not match its seven-field contract")
        if self.edge_type.shape != (e,) or self.edge_type.dtype != torch.long:
            raise ValueError("edge_type must have shape [E] and dtype long")
        if bool((self.edge_type != 0).any()):
            raise ValueError("Snapshot candidate edge_type must be zero")
        if self.candidate_edge_ids.shape != (e, 2) or self.candidate_edge_ids.dtype != torch.long:
            raise ValueError("candidate_edge_ids must have shape [E,2] and dtype long")
        if self.feasible_edge.shape != (e,) or self.feasible_edge.dtype != torch.bool:
            raise ValueError("feasible_edge must have shape [E] and dtype bool")
        if self.initialized_edge.shape != (e,) or self.initialized_edge.dtype != torch.bool:
            raise ValueError("initialized_edge must have shape [E] and dtype bool")
        if self.current_serving.shape != (self.user_count,) or self.current_serving.dtype != torch.long:
            raise ValueError("current_serving must be a long user vector")
        if self.hold_steps.shape != (self.user_count,) or self.hold_steps.dtype != torch.long:
            raise ValueError("hold_steps must be a long user vector")
        if self.user_order.shape != (self.user_count,) or self.user_order.dtype != torch.long:
            raise ValueError("user_order must be a long user vector")
        tensors = (
            self.node_x,
            self.edge_index,
            self.edge_z,
            self.edge_type,
            self.candidate_edge_ids,
            self.feasible_edge,
            self.current_serving,
            self.hold_steps,
            self.user_order,
            self.initialized_edge,
            self.descriptors.gamma_edge,
            self.descriptors.feasibility_margin_edge,
            self.descriptors.admitted_load_node,
        )
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all Snapshot input tensors must share one device")
        if not bool(torch.isfinite(self.node_x).all()) or not bool(torch.isfinite(self.edge_z).all()):
            raise ValueError("Snapshot graph contains NaN or Inf")
        if e:
            users, satellites = self.candidate_edge_ids.unbind(dim=1)
            if not torch.equal(self.edge_index[0], users) or not torch.equal(
                self.edge_index[1] - self.user_count, satellites
            ):
                raise ValueError("candidate ids and edge_index disagree")
        self.descriptors.validate(e, self.satellite_count)
        names = tuple(self.meta.get("node_feature_names", ())) + tuple(
            self.meta.get("edge_feature_names", ())
        )
        forbidden = ("intensity", "ema", "policy_flow")
        if any(any(token in str(name).lower() for token in forbidden) for name in names):
            raise ValueError("Snapshot feature metadata contains an Intensity--Flow field")

    def as_model_step(self) -> dict[str, Any]:
        self.validate()
        return {
            "t": self.epoch,
            "node_x": self.node_x,
            "edge_index": self.edge_index,
            "edge_z": self.edge_z,
            "edge_type": self.edge_type,
            "meta": {
                **dict(self.meta),
                "K_users": self.user_count,
                "S_sats": self.satellite_count,
                "candidate_edge_ids": self.candidate_edge_ids,
                "observation_id": self.observation_id,
            },
        }

    def with_descriptors(
        self,
        descriptors: SnapshotOutput,
        *,
        initialized_edge: torch.Tensor | None = None,
    ) -> "SnapshotControlInput":
        descriptors.validate(self.edge_count, self.satellite_count)
        feature = SnapshotFeatureConfig.from_mapping(self.meta["normalization"])
        node_x = self.node_x.clone()
        edge_z = self.edge_z.clone()
        node_x[self.user_count :, 6] = (
            descriptors.admitted_load_node / feature.admitted_load_scale
        )
        if self.edge_count:
            destination = self.candidate_edge_ids[:, 1]
            edge_z[:, 2] = descriptors.gamma_edge / feature.gamma_scale
            edge_z[:, 3] = descriptors.feasibility_margin_edge
            edge_z[:, 4] = descriptors.admitted_load_node.index_select(
                0, destination
            ) / feature.admitted_load_scale
        result = SnapshotControlInput(
            observation_id=self.observation_id,
            node_x=node_x,
            edge_index=self.edge_index.clone(),
            edge_z=edge_z,
            edge_type=self.edge_type.clone(),
            candidate_edge_ids=self.candidate_edge_ids.clone(),
            feasible_edge=self.feasible_edge.clone(),
            current_serving=self.current_serving.clone(),
            hold_steps=self.hold_steps.clone(),
            user_order=self.user_order.clone(),
            descriptors=descriptors.clone(),
            initialized_edge=(
                self.initialized_edge.clone()
                if initialized_edge is None
                else initialized_edge.clone()
            ),
            meta=dict(self.meta),
        )
        result.validate()
        return result


def snapshot_control_from_observation(
    observation: ControlObservation,
    descriptors: SnapshotOutput,
    feature_config: SnapshotFeatureConfig,
    *,
    initialized_edge: torch.Tensor | None = None,
) -> SnapshotControlInput:
    """Build the isolated Snapshot graph from an allow-list of safe columns."""

    if not isinstance(observation, ControlObservation):
        raise TypeError("observation must be a ControlObservation")
    if not isinstance(feature_config, SnapshotFeatureConfig):
        raise TypeError("feature_config must be SnapshotFeatureConfig")
    observation.validate()
    descriptors.validate(observation.edge_count, observation.satellite_count)
    if observation.node_x.size(1) < 6 or observation.edge_features.size(1) < 7:
        raise ValueError("simulator observation lacks geometry/history source columns")
    device = observation.node_x.device
    dtype = observation.node_x.dtype
    node_x = torch.zeros(
        (observation.user_count + observation.satellite_count, 7),
        dtype=dtype,
        device=device,
    )
    node_x[:, :6] = observation.node_x[:, :6]
    node_x[observation.user_count :, 6] = (
        descriptors.admitted_load_node / feature_config.admitted_load_scale
    )
    edge_z = torch.zeros(
        (observation.edge_count, 7), dtype=observation.edge_features.dtype, device=device
    )
    if observation.edge_count:
        destination = observation.candidate_edge_ids[:, 1]
        edge_z[:, 0] = observation.edge_features[:, 0]
        edge_z[:, 1] = observation.edge_features[:, 1]
        edge_z[:, 2] = descriptors.gamma_edge / feature_config.gamma_scale
        edge_z[:, 3] = descriptors.feasibility_margin_edge
        edge_z[:, 4] = descriptors.admitted_load_node.index_select(
            0, destination
        ) / feature_config.admitted_load_scale
        # Recompute association instead of copying a descriptor-bearing row.
        edge_z[:, 5] = (
            observation.current_serving[observation.candidate_edge_ids[:, 0]]
            == destination
        ).to(edge_z.dtype)
        # Column six is the simulator's descriptor-independent dwell cue.
        edge_z[:, 6] = observation.edge_features[:, 6]
    if initialized_edge is None:
        initialized_edge = torch.zeros(
            observation.edge_count, dtype=torch.bool, device=device
        )
    safe_meta = {
        "snapshot_feature_contract_version": SNAPSHOT_FEATURE_CONTRACT_VERSION,
        "node_feature_names": SNAPSHOT_NODE_FEATURE_NAMES,
        "edge_feature_names": SNAPSHOT_EDGE_FEATURE_NAMES,
        "normalization": asdict(feature_config),
        "protocol_fingerprint": observation.meta.get("protocol_fingerprint"),
        "capacity_users": observation.meta.get("capacity_users"),
        "hard_feasibility_mask": observation.meta.get("hard_feasibility_mask"),
        "scientific_isolation": "no_integrated_intensity_no_raw_or_ema_flow",
    }
    result = SnapshotControlInput(
        observation_id=observation.observation_id,
        node_x=node_x,
        edge_index=observation.candidate_edge_index.clone(),
        edge_z=edge_z,
        edge_type=torch.zeros(
            observation.edge_count, dtype=torch.long, device=device
        ),
        candidate_edge_ids=observation.candidate_edge_ids.clone(),
        feasible_edge=observation.sim_descriptors.feasible_edge.clone(),
        current_serving=observation.current_serving.clone(),
        hold_steps=observation.hold_steps.clone(),
        user_order=observation.user_order.clone(),
        descriptors=descriptors.clone(),
        initialized_edge=initialized_edge.clone(),
        meta=safe_meta,
    )
    result.validate()
    return result


def build_snapshot_target(
    source: SnapshotControlInput,
    current: SnapshotControlInput,
) -> SnapshotTarget:
    """Join the isolated ``t+1`` Snapshot descriptors onto candidates at ``t``."""

    source.validate()
    current.validate()
    if source.episode_seed != current.episode_seed or current.epoch != source.epoch + 1:
        raise ValueError("Snapshot targets require consecutive observations in one episode")
    if source.user_count != current.user_count or source.satellite_count != current.satellite_count:
        raise ValueError("Snapshot node identities must remain stable")
    lookup = {
        (int(user), int(satellite)): index
        for index, (user, satellite) in enumerate(current.candidate_edge_ids.tolist())
    }
    gamma = torch.zeros_like(source.descriptors.gamma_edge)
    margin = torch.zeros_like(source.descriptors.feasibility_margin_edge)
    persistent = torch.zeros(
        source.edge_count, dtype=torch.bool, device=source.candidate_edge_ids.device
    )
    for index, pair in enumerate(source.candidate_edge_ids.tolist()):
        target_index = lookup.get((int(pair[0]), int(pair[1])))
        if target_index is not None:
            persistent[index] = True
            gamma[index] = current.descriptors.gamma_edge[target_index]
            margin[index] = current.descriptors.feasibility_margin_edge[target_index]
    target = SnapshotTarget(
        gamma_edge=gamma,
        feasibility_margin_edge=margin,
        admitted_load_node=current.descriptors.admitted_load_node.clone(),
        persistent_edge=persistent,
    )
    target.validate(source.edge_count, source.satellite_count)
    return target


@dataclass(frozen=True)
class SnapshotInitializationRequest:
    """Shape-only initializer request; it contains no simulator descriptor values."""

    observation_id: tuple[int, int]
    candidate_edge_ids: torch.Tensor
    satellite_count: int
    dtype: torch.dtype
    device: torch.device

    @property
    def edge_count(self) -> int:
        return int(self.candidate_edge_ids.size(0))

    @classmethod
    def from_observation(cls, observation: ControlObservation) -> "SnapshotInitializationRequest":
        observation.validate()
        return cls(
            observation_id=observation.observation_id,
            candidate_edge_ids=observation.candidate_edge_ids.clone(),
            satellite_count=observation.satellite_count,
            dtype=observation.node_x.dtype,
            device=observation.node_x.device,
        )

    @classmethod
    def from_control(cls, control: SnapshotControlInput) -> "SnapshotInitializationRequest":
        control.validate()
        return cls(
            observation_id=control.observation_id,
            candidate_edge_ids=control.candidate_edge_ids.clone(),
            satellite_count=control.satellite_count,
            dtype=control.node_x.dtype,
            device=control.node_x.device,
        )


class SnapshotStreamInitializer(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def initialize(
        self,
        request: SnapshotInitializationRequest,
        missing_edge: torch.Tensor,
    ) -> SnapshotOutput: ...


class ConstantSnapshotInitializer:
    """Explicit constant D0/new-edge initializer; no constants are implicit."""

    def __init__(self, *, gamma: float, feasibility_margin: float, admitted_load: float) -> None:
        self.gamma = _finite_real("gamma", gamma)
        self.feasibility_margin = _finite_real("feasibility_margin", feasibility_margin)
        self.admitted_load = _finite_real("admitted_load", admitted_load)
        if self.admitted_load < 0.0:
            raise ValueError("admitted_load must be non-negative")
        payload = json.dumps(
            {
                "kind": "constant_snapshot_v1",
                "gamma": self.gamma,
                "feasibility_margin": self.feasibility_margin,
                "admitted_load": self.admitted_load,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._fingerprint = hashlib.sha256(payload.encode()).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def initialize(
        self,
        request: SnapshotInitializationRequest,
        missing_edge: torch.Tensor,
    ) -> SnapshotOutput:
        if missing_edge.shape != (request.edge_count,) or missing_edge.dtype != torch.bool:
            raise ValueError("missing_edge must be a bool vector in current candidate order")
        if missing_edge.device != request.device:
            raise ValueError("missing_edge and initialization request must share a device")
        result = SnapshotOutput(
            gamma_edge=torch.full(
                (request.edge_count,), self.gamma, dtype=request.dtype, device=request.device
            ),
            feasibility_margin_edge=torch.full(
                (request.edge_count,),
                self.feasibility_margin,
                dtype=request.dtype,
                device=request.device,
            ),
            admitted_load_node=torch.full(
                (request.satellite_count,),
                self.admitted_load,
                dtype=request.dtype,
                device=request.device,
            ),
        )
        result.validate(request.edge_count, request.satellite_count)
        return result


def persistent_snapshot_mask(
    previous: SnapshotControlInput,
    current: SnapshotControlInput,
) -> torch.Tensor:
    previous.validate()
    current.validate()
    if previous.episode_seed != current.episode_seed or current.epoch != previous.epoch + 1:
        raise ValueError("Snapshot persistence requires consecutive observations")
    previous_ids = {
        (int(user), int(satellite))
        for user, satellite in previous.candidate_edge_ids.tolist()
    }
    return torch.tensor(
        [
            (int(user), int(satellite)) in previous_ids
            for user, satellite in current.candidate_edge_ids.tolist()
        ],
        dtype=torch.bool,
        device=current.candidate_edge_ids.device,
    )


def carry_snapshot_stream(
    previous: SnapshotControlInput,
    prediction: SnapshotOutput,
    current_template: SnapshotControlInput,
    initializer: SnapshotStreamInitializer | None,
) -> SnapshotControlInput:
    """Stage a source-aligned prediction on persistent candidates at ``t+1``."""

    previous.validate()
    current_template.validate()
    prediction.validate(previous.edge_count, previous.satellite_count)
    persistent = persistent_snapshot_mask(previous, current_template)
    gamma = torch.empty_like(current_template.descriptors.gamma_edge)
    margin = torch.empty_like(current_template.descriptors.feasibility_margin_edge)
    previous_lookup = {
        (int(user), int(satellite)): index
        for index, (user, satellite) in enumerate(previous.candidate_edge_ids.tolist())
    }
    for index, pair in enumerate(current_template.candidate_edge_ids.tolist()):
        if bool(persistent[index]):
            source_index = previous_lookup[(int(pair[0]), int(pair[1]))]
            gamma[index] = prediction.gamma_edge[source_index]
            margin[index] = prediction.feasibility_margin_edge[source_index]
    missing = ~persistent
    if bool(missing.any()):
        if initializer is None:
            raise ValueError("new Snapshot candidates require an explicit initializer")
        initialized = initializer.initialize(
            SnapshotInitializationRequest.from_control(current_template), missing
        )
        initialized.validate(current_template.edge_count, current_template.satellite_count)
        gamma[missing] = initialized.gamma_edge[missing]
        margin[missing] = initialized.feasibility_margin_edge[missing]
    staged = SnapshotOutput(
        gamma_edge=gamma,
        feasibility_margin_edge=margin,
        admitted_load_node=prediction.admitted_load_node.clone(),
    )
    return current_template.with_descriptors(staged, initialized_edge=missing)


def admitted_counts_from_execution(
    execution: ExecutionResult,
    satellite_count: int,
) -> torch.Tensor:
    execution.validate(execution.requested_serving.numel(), satellite_count)
    admitted = execution.executed_serving[execution.executed_serving >= 0]
    return torch.bincount(admitted, minlength=satellite_count).to(
        dtype=torch.float32, device=execution.executed_serving.device
    )


def instantaneous_load_after_execution(
    environment: PaperAlignedLEOEnv,
    execution: ExecutionResult,
) -> torch.Tensor:
    """Return current admitted occupancy, independent of the flow/EMA state."""

    counts = admitted_counts_from_execution(execution, environment.S)
    return instantaneous_admitted_load_snapshot(
        counts,
        capacity_users=environment.capacity_users,
        clip_bounds=(0.0, 1.0),
    )


def _initial_load(
    value: Real | Sequence[Real] | torch.Tensor,
    *,
    satellite_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, Real) and not isinstance(value, bool):
        scalar = _finite_real("initial_admitted_load", value)
        result = torch.full((satellite_count,), scalar, dtype=dtype, device=device)
    else:
        result = torch.as_tensor(value, dtype=dtype, device=device)
        if result.shape != (satellite_count,):
            raise ValueError(
                f"initial_admitted_load must have shape [{satellite_count}]"
            )
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError("initial_admitted_load must be finite and non-negative")
    return result.clone()


def resolve_initial_admitted_load(
    value: Real | Sequence[Real] | torch.Tensor,
    *,
    satellite_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Public validator for the explicitly configured epoch-zero Snapshot load."""

    return _initial_load(
        value,
        satellite_count=satellite_count,
        dtype=dtype,
        device=device,
    )


def serialize_snapshot_output(value: SnapshotOutput) -> dict[str, torch.Tensor]:
    return {
        "gamma_edge": _cpu(value.gamma_edge, "snapshot.gamma_edge"),
        "feasibility_margin_edge": _cpu(
            value.feasibility_margin_edge, "snapshot.feasibility_margin_edge"
        ),
        "admitted_load_node": _cpu(
            value.admitted_load_node, "snapshot.admitted_load_node"
        ),
    }


def serialize_snapshot_control(value: SnapshotControlInput) -> dict[str, Any]:
    value.validate()
    return {
        "observation_id": list(value.observation_id),
        "node_x": _cpu(value.node_x, "snapshot_control.node_x"),
        "edge_index": _cpu(value.edge_index, "snapshot_control.edge_index"),
        "edge_z": _cpu(value.edge_z, "snapshot_control.edge_z"),
        "edge_type": _cpu(value.edge_type, "snapshot_control.edge_type"),
        "candidate_edge_ids": _cpu(
            value.candidate_edge_ids, "snapshot_control.candidate_edge_ids"
        ),
        "feasible_edge": _cpu(value.feasible_edge, "snapshot_control.feasible_edge"),
        "current_serving": _cpu(
            value.current_serving, "snapshot_control.current_serving"
        ),
        "hold_steps": _cpu(value.hold_steps, "snapshot_control.hold_steps"),
        "user_order": _cpu(value.user_order, "snapshot_control.user_order"),
        "descriptors": serialize_snapshot_output(value.descriptors),
        "initialized_edge": _cpu(
            value.initialized_edge, "snapshot_control.initialized_edge"
        ),
        "meta": _json_safe(dict(value.meta), path="snapshot_control.meta"),
    }


def serialize_snapshot_target(
    value: SnapshotTarget,
    *,
    source_id: tuple[int, int],
    target_id: tuple[int, int],
) -> dict[str, Any]:
    return {
        "contract_version": SNAPSHOT_TARGET_CONTRACT_VERSION,
        "source_observation_id": list(source_id),
        "target_observation_id": list(target_id),
        "gamma_edge": _cpu(value.gamma_edge, "snapshot_target.gamma_edge"),
        "feasibility_margin_edge": _cpu(
            value.feasibility_margin_edge, "snapshot_target.feasibility_margin_edge"
        ),
        "admitted_load_node": _cpu(
            value.admitted_load_node, "snapshot_target.admitted_load_node"
        ),
        "persistent_edge": _cpu(
            value.persistent_edge, "snapshot_target.persistent_edge"
        ),
    }


def snapshot_control_from_record(
    record: Mapping[str, Any], device: torch.device | str
) -> SnapshotControlInput:
    raw = record.get("snapshot_observation", record)
    if not isinstance(raw, Mapping):
        raise TypeError("snapshot observation record must be a mapping")
    target_device = torch.device(device)
    desc = raw.get("descriptors")
    if not isinstance(desc, Mapping):
        raise ValueError("snapshot observation requires descriptors")
    observation_id = raw.get("observation_id")
    if not isinstance(observation_id, Sequence) or len(observation_id) != 2:
        raise ValueError("snapshot observation_id must contain seed and epoch")
    output = SnapshotOutput(
        gamma_edge=torch.as_tensor(desc["gamma_edge"], device=target_device),
        feasibility_margin_edge=torch.as_tensor(
            desc["feasibility_margin_edge"], device=target_device
        ),
        admitted_load_node=torch.as_tensor(
            desc["admitted_load_node"], device=target_device
        ),
    )
    result = SnapshotControlInput(
        observation_id=(int(observation_id[0]), int(observation_id[1])),
        node_x=torch.as_tensor(raw["node_x"], device=target_device),
        edge_index=torch.as_tensor(raw["edge_index"], device=target_device).long(),
        edge_z=torch.as_tensor(raw["edge_z"], device=target_device),
        edge_type=torch.as_tensor(raw["edge_type"], device=target_device).long(),
        candidate_edge_ids=torch.as_tensor(
            raw["candidate_edge_ids"], device=target_device
        ).long(),
        feasible_edge=torch.as_tensor(
            raw["feasible_edge"], device=target_device
        ).bool(),
        current_serving=torch.as_tensor(
            raw["current_serving"], device=target_device
        ).long(),
        hold_steps=torch.as_tensor(raw["hold_steps"], device=target_device).long(),
        user_order=torch.as_tensor(raw["user_order"], device=target_device).long(),
        descriptors=output,
        initialized_edge=torch.as_tensor(
            raw["initialized_edge"], device=target_device
        ).bool(),
        meta=dict(raw.get("meta", {})),
    )
    result.validate()
    return result


def snapshot_target_from_record(
    record: Mapping[str, Any], device: torch.device | str
) -> SnapshotTarget:
    raw = record.get("next_target", record)
    if not isinstance(raw, Mapping):
        raise ValueError("terminal records do not contain a Snapshot target")
    if int(raw.get("contract_version", -1)) != SNAPSHOT_TARGET_CONTRACT_VERSION:
        raise ValueError("unsupported Snapshot target contract")
    target_device = torch.device(device)
    result = SnapshotTarget(
        gamma_edge=torch.as_tensor(raw["gamma_edge"], device=target_device),
        feasibility_margin_edge=torch.as_tensor(
            raw["feasibility_margin_edge"], device=target_device
        ),
        admitted_load_node=torch.as_tensor(
            raw["admitted_load_node"], device=target_device
        ),
        persistent_edge=torch.as_tensor(
            raw["persistent_edge"], device=target_device
        ).bool(),
    )
    return result


def _serialize_action(action: ServingAction) -> dict[str, Any]:
    return {
        "observation_id": list(action.observation_id),
        "requested_serving": _cpu(action.requested_serving, "action.requested_serving"),
    }


def _serialize_snapshot_execution(execution: ExecutionResult) -> dict[str, Any]:
    """Serialize outcomes while deliberately omitting raw/EMA flow tensors."""

    return {
        "observation_id": list(execution.observation_id),
        "requested_serving": _cpu(
            execution.requested_serving, "execution.requested_serving"
        ),
        "executed_serving": _cpu(
            execution.executed_serving, "execution.executed_serving"
        ),
        "admitted": _cpu(execution.admitted, "execution.admitted"),
        "failure_reason": _cpu(execution.failure_reason, "execution.failure_reason"),
        "handover_attempted": _cpu(
            execution.handover_attempted, "execution.handover_attempted"
        ),
        "handover_executed": _cpu(
            execution.handover_executed, "execution.handover_executed"
        ),
    }


def generate_snapshot_oracle_episode(
    cfg: Mapping[str, Any],
    *,
    episode_id: int,
    seed: int,
    margin_config: SnapshotMarginConfig,
    feature_config: SnapshotFeatureConfig,
    policy_config: SnapshotPolicyConfig,
    initial_admitted_load: Real | Sequence[Real] | torch.Tensor,
    horizon_steps: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Generate formal factorial data under the Snapshot oracle controller."""

    if isinstance(episode_id, bool) or int(episode_id) != episode_id or episode_id < 0:
        raise ValueError("episode_id must be a non-negative integer")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    episode_cfg = copy.deepcopy(dict(cfg))
    episode_cfg["seed"] = int(seed)
    if horizon_steps is not None:
        if isinstance(horizon_steps, bool) or int(horizon_steps) != horizon_steps or horizon_steps <= 0:
            raise ValueError("horizon_steps must be a positive integer")
        protocol = episode_cfg.setdefault("paper_protocol", {})
        if not isinstance(protocol, dict):
            raise TypeError("paper_protocol must be a mapping")
        protocol["horizon_steps"] = int(horizon_steps)
    environment = PaperAlignedLEOEnv(episode_cfg, device=device)
    if policy_config.hard_feasibility_mask != environment.hard_feasibility_mask:
        raise ValueError("Snapshot policy hard mask must match the environment protocol")
    if policy_config.min_dwell_steps != environment.min_dwell_steps:
        raise ValueError("Snapshot policy dwell must match the environment protocol")
    if policy_config.hysteresis != environment.hysteresis:
        raise ValueError("Snapshot policy hysteresis must match the environment protocol")
    controller = SnapshotFixedRankController(policy_config)
    observation = environment.reset_control()
    admitted_load = _initial_load(
        initial_admitted_load,
        satellite_count=environment.S,
        dtype=environment.flow.dtype,
        device=environment.device,
    )
    snapshot = simulator_snapshot_output(observation, admitted_load, margin_config)
    control = snapshot_control_from_observation(observation, snapshot, feature_config)
    records: list[dict[str, Any]] = []

    while True:
        action = controller.select_action(control, control.descriptors)
        next_observation, execution, done = environment.step_action(action)
        next_control = None
        target = None
        if next_observation is not None:
            admitted_load = instantaneous_load_after_execution(environment, execution)
            next_snapshot = simulator_snapshot_output(
                next_observation, admitted_load, margin_config
            )
            next_control = snapshot_control_from_observation(
                next_observation, next_snapshot, feature_config
            )
            typed_target = build_snapshot_target(control, next_control)
            target = serialize_snapshot_target(
                typed_target,
                source_id=control.observation_id,
                target_id=next_control.observation_id,
            )
        records.append(
            {
                "step_index": len(records),
                "snapshot_observation": serialize_snapshot_control(control),
                "action": _serialize_action(action),
                "execution": _serialize_snapshot_execution(execution),
                "next_target": target,
                "done": bool(done),
            }
        )
        if done:
            break
        if next_control is None:
            raise RuntimeError("non-terminal Snapshot transition omitted t+1")
        control = next_control

    return {
        "episode_id": int(episode_id),
        "seed": int(seed),
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "protocol_fingerprint": environment.protocol_fingerprint,
        "snapshot_feature_contract_version": SNAPSHOT_FEATURE_CONTRACT_VERSION,
        "snapshot_target_contract_version": SNAPSHOT_TARGET_CONTRACT_VERSION,
        "horizon_steps": len(records),
        "behavior_policy": {
            "name": "snapshot_fixed_rank_oracle_v1",
            "descriptor_source": "simulator_snapshot_oracle",
            "config": _json_safe(asdict(policy_config), path="behavior_policy.config"),
        },
        "initial_admitted_load": _cpu(
            _initial_load(
                initial_admitted_load,
                satellite_count=environment.S,
                dtype=environment.flow.dtype,
                device=environment.device,
            ),
            "initial_admitted_load",
        ),
        "steps": records,
        "terminal": {
            "epoch": int(environment.epoch),
            "serving": _cpu(environment.current_serving, "terminal.serving"),
        },
    }


def _execution_from_if_record(
    record: Mapping[str, Any], *, user_count: int, satellite_count: int, device: torch.device
) -> ExecutionResult:
    raw = record.get("execution")
    if not isinstance(raw, Mapping):
        raise ValueError("Intensity--Flow diagnostic record requires execution")
    observation_id = raw.get("observation_id")
    if not isinstance(observation_id, Sequence) or len(observation_id) != 2:
        raise ValueError("diagnostic execution observation_id is invalid")
    value = ExecutionResult(
        observation_id=(int(observation_id[0]), int(observation_id[1])),
        requested_serving=torch.as_tensor(raw["requested_serving"], device=device).long(),
        executed_serving=torch.as_tensor(raw["executed_serving"], device=device).long(),
        admitted=torch.as_tensor(raw["admitted"], device=device).bool(),
        failure_reason=torch.as_tensor(raw["failure_reason"], device=device).long(),
        handover_attempted=torch.as_tensor(
            raw["handover_attempted"], device=device
        ).bool(),
        handover_executed=torch.as_tensor(
            raw["handover_executed"], device=device
        ).bool(),
        flow_before=torch.as_tensor(raw["flow_before"], device=device),
        flow_after=torch.as_tensor(raw["flow_after"], device=device),
    )
    value.validate(user_count, satellite_count)
    return value


def snapshot_episode_from_paper_episode(
    episode: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    margin_config: SnapshotMarginConfig,
    feature_config: SnapshotFeatureConfig,
    initial_admitted_load: Real | Sequence[Real] | torch.Tensor,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Diagnostic-only conversion of an Intensity--Flow behaviour trajectory.

    The returned record is safe to inspect with Snapshot models, but its
    ``dataset_kind`` and provenance deliberately prevent it from being accepted
    by :class:`SnapshotEpisodeDataset` for formal factorial training.
    """

    # Local import avoids making the isolated Snapshot module depend on the IF
    # trainer during normal generation/evaluation.
    from .training import observation_from_record

    records = episode.get("steps")
    if not isinstance(records, list) or not records:
        raise ValueError("Intensity--Flow episode requires a non-empty steps list")
    target_device = torch.device(device)
    observations = [observation_from_record(record, target_device) for record in records]
    environment_cfg = copy.deepcopy(dict(cfg))
    environment_cfg["seed"] = int(episode.get("seed", observations[0].episode_seed))
    environment = PaperAlignedLEOEnv(environment_cfg, device=target_device)
    admitted_load = _initial_load(
        initial_admitted_load,
        satellite_count=observations[0].satellite_count,
        dtype=observations[0].node_x.dtype,
        device=target_device,
    )
    controls: list[SnapshotControlInput] = []
    executions: list[ExecutionResult] = []
    for index, (record, observation) in enumerate(zip(records, observations)):
        snapshot = simulator_snapshot_output(observation, admitted_load, margin_config)
        controls.append(
            snapshot_control_from_observation(observation, snapshot, feature_config)
        )
        execution = _execution_from_if_record(
            record,
            user_count=observation.user_count,
            satellite_count=observation.satellite_count,
            device=target_device,
        )
        executions.append(execution)
        if index + 1 < len(records):
            admitted_load = instantaneous_load_after_execution(environment, execution)
    converted: list[dict[str, Any]] = []
    for index, (record, control, execution) in enumerate(
        zip(records, controls, executions)
    ):
        target_payload = None
        if index + 1 < len(controls):
            target_payload = serialize_snapshot_target(
                build_snapshot_target(control, controls[index + 1]),
                source_id=control.observation_id,
                target_id=controls[index + 1].observation_id,
            )
        action = record.get("action")
        converted.append(
            {
                "step_index": index,
                "snapshot_observation": serialize_snapshot_control(control),
                "action": _json_safe(action, path="diagnostic.action"),
                "execution": _serialize_snapshot_execution(execution),
                "next_target": target_payload,
                "done": index == len(controls) - 1,
            }
        )
    result = {
        "dataset_kind": SNAPSHOT_DIAGNOSTIC_DATASET_KIND,
        "formal_factorial_training": False,
        "diagnostic_warning": (
            "inherits Intensity--Flow behavior actions and induced state distribution"
        ),
        "episode_id": int(episode.get("episode_id", 0)),
        "seed": observations[0].episode_seed,
        "protocol_version": int(episode.get("protocol_version", PAPER_PROTOCOL_VERSION)),
        "protocol_fingerprint": str(episode.get("protocol_fingerprint", "")),
        "snapshot_feature_contract_version": SNAPSHOT_FEATURE_CONTRACT_VERSION,
        "snapshot_target_contract_version": SNAPSHOT_TARGET_CONTRACT_VERSION,
        "horizon_steps": len(converted),
        "behavior_policy": {
            "name": "inherited_intensity_flow_behavior_diagnostic",
            "source": _json_safe(episode.get("behavior_policy"), path="behavior_policy"),
        },
        "steps": converted,
    }
    validate_snapshot_episode(result)
    return result


def validate_snapshot_episode(episode: Mapping[str, Any]) -> None:
    if not isinstance(episode, Mapping):
        raise TypeError("Snapshot episode must be a mapping")
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Snapshot episode requires a non-empty steps list")
    seed = int(episode.get("seed", -1))
    if seed < 0 or int(episode.get("horizon_steps", -1)) != len(steps):
        raise ValueError("Snapshot episode seed/horizon metadata is invalid")
    for index, record in enumerate(steps):
        if not isinstance(record, Mapping) or int(record.get("step_index", -1)) != index:
            raise ValueError("Snapshot step indices must be contiguous")
        control = snapshot_control_from_record(record, "cpu")
        if control.observation_id != (seed, index):
            raise ValueError("Snapshot record observation id is misaligned")
        done = record.get("done")
        if done is not (index == len(steps) - 1):
            raise ValueError("only the final Snapshot record may be terminal")
        if done and record.get("next_target") is not None:
            raise ValueError("terminal Snapshot record must not have a target")
        if not done:
            target = snapshot_target_from_record(record, "cpu")
            target.validate(control.edge_count, control.satellite_count)


def generate_sharded_snapshot_dataset(
    path: str | Path,
    cfg: Mapping[str, Any],
    *,
    episode_count: int,
    margin_config: SnapshotMarginConfig,
    feature_config: SnapshotFeatureConfig,
    policy_config: SnapshotPolicyConfig,
    initial_admitted_load: Real | Sequence[Real] | torch.Tensor,
    horizon_steps: int | None = None,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    split_counts: Sequence[int] | None = None,
    split_seed: int | None = None,
    base_seed: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Write one formal Snapshot-oracle episode per restricted ``.pt`` shard."""

    if isinstance(episode_count, bool) or int(episode_count) != episode_count or episode_count <= 0:
        raise ValueError("episode_count must be a positive integer")
    resolved_seed = int(cfg.get("seed", 7) if base_seed is None else base_seed)
    resolved_split_seed = resolved_seed if split_seed is None else int(split_seed)
    identities = [
        {"episode_id": index, "seed": resolved_seed + index}
        for index in range(int(episode_count))
    ]
    split_manifest = build_split_manifest(
        identities, ratios=split_ratios, counts=split_counts, seed=resolved_split_seed
    )
    split_by_index = {
        int(index): split
        for split in SNAPSHOT_SPLITS
        for index in split_manifest["episode_indices"][split]
    }
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output.parent / f"{output.stem}_episodes"
    shard_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for index in range(int(episode_count)):
        episode = generate_snapshot_oracle_episode(
            cfg,
            episode_id=index,
            seed=resolved_seed + index,
            margin_config=margin_config,
            feature_config=feature_config,
            policy_config=policy_config,
            initial_admitted_load=initial_admitted_load,
            horizon_steps=horizon_steps,
            device=device,
        )
        validate_snapshot_episode(episode)
        payload = {
            "snapshot_shard_version": SNAPSHOT_SHARD_VERSION,
            "dataset_kind": SNAPSHOT_DATASET_KIND,
            "episode": episode,
        }
        filename = f"episode_{index:06d}.pt"
        shard_path = shard_dir / filename
        temporary = shard_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(shard_path)
        digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        entries.append(
            {
                "episode_index": index,
                "episode_id": index,
                "seed": resolved_seed + index,
                "split": split_by_index[index],
                "relative_path": f"{shard_dir.name}/{filename}",
                "size_bytes": shard_path.stat().st_size,
                "sha256": digest,
            }
        )
    index_payload = {
        "snapshot_dataset_schema_version": SNAPSHOT_DATASET_SCHEMA_VERSION,
        "dataset_kind": SNAPSHOT_DATASET_KIND,
        "paper_protocol_version": PAPER_PROTOCOL_VERSION,
        "snapshot_feature_contract_version": SNAPSHOT_FEATURE_CONTRACT_VERSION,
        "snapshot_target_contract_version": SNAPSHOT_TARGET_CONTRACT_VERSION,
        "feature_contract": {
            "node_feature_names": list(SNAPSHOT_NODE_FEATURE_NAMES),
            "edge_feature_names": list(SNAPSHOT_EDGE_FEATURE_NAMES),
            "normalization": asdict(feature_config),
            "scientific_isolation": "no_integrated_intensity_no_raw_or_ema_flow",
        },
        "target_contract": {
            "alignment": "persistent_source_candidates_t_to_snapshot_t_plus_1",
            "load_domain": "current_admitted_users_divided_by_fixed_capacity",
            "memory": "no_flow_state_no_mean_flow_feedback_no_ema",
        },
        "source_config": _json_safe(cfg, path="source_config"),
        "snapshot_protocol": {
            "margin": asdict(margin_config),
            "policy": asdict(policy_config),
            "initial_admitted_load": _json_safe(
                initial_admitted_load, path="initial_admitted_load"
            ),
        },
        "generation": {
            "episode_count": int(episode_count),
            "base_seed": resolved_seed,
            "split_seed": resolved_split_seed,
            "behavior_policy": "snapshot_fixed_rank_oracle_v1",
        },
        "split_manifest": split_manifest,
        "episode_shards": entries,
    }
    temporary_index = output.with_suffix(output.suffix + ".tmp")
    torch.save(index_payload, temporary_index)
    temporary_index.replace(output)
    return index_payload


class SnapshotEpisodeDataset(Dataset):
    """Lazy loader that rejects diagnostic/Intensity--Flow data by default."""

    def __init__(self, path: str | Path, split: str = "train") -> None:
        if split not in SNAPSHOT_SPLITS:
            raise ValueError("split must be train, val, or test")
        self.index_path = Path(path).expanduser()
        self.payload = torch.load(
            self.index_path, map_location="cpu", weights_only=True
        )
        if not isinstance(self.payload, dict) or self.payload.get("dataset_kind") != SNAPSHOT_DATASET_KIND:
            raise ValueError(
                "formal Snapshot training requires a snapshot-oracle action-coupled dataset"
            )
        indices = self.payload.get("split_manifest", {}).get("episode_indices", {}).get(split)
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"Snapshot split {split!r} is empty")
        entries = self.payload.get("episode_shards")
        if not isinstance(entries, list):
            raise ValueError("Snapshot dataset index requires episode_shards")
        self.entries = [entries[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        relative = PurePosixPath(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid Snapshot shard relative path")
        root = self.index_path.parent.resolve()
        path = root.joinpath(*relative.parts).resolve()
        path.relative_to(root)
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError("Snapshot shard size mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Snapshot shard digest mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("dataset_kind") != SNAPSHOT_DATASET_KIND:
            raise ValueError("Snapshot shard kind mismatch")
        episode = payload.get("episode")
        validate_snapshot_episode(episode)
        return episode


__all__ = [
    "SNAPSHOT_DATASET_KIND",
    "SNAPSHOT_EDGE_FEATURE_NAMES",
    "SNAPSHOT_FEATURE_CONTRACT_VERSION",
    "SNAPSHOT_NODE_FEATURE_NAMES",
    "SNAPSHOT_TARGET_CONTRACT_VERSION",
    "ConstantSnapshotInitializer",
    "SnapshotControlInput",
    "SnapshotEpisodeDataset",
    "SnapshotFeatureConfig",
    "SnapshotInitializationRequest",
    "SnapshotStreamInitializer",
    "admitted_counts_from_execution",
    "build_snapshot_target",
    "carry_snapshot_stream",
    "generate_sharded_snapshot_dataset",
    "generate_snapshot_oracle_episode",
    "instantaneous_load_after_execution",
    "persistent_snapshot_mask",
    "resolve_initial_admitted_load",
    "serialize_snapshot_control",
    "snapshot_control_from_observation",
    "snapshot_control_from_record",
    "snapshot_episode_from_paper_episode",
    "snapshot_target_from_record",
    "validate_snapshot_episode",
]
