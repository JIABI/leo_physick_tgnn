"""Conventional Snapshot interface used by the paper control baselines.

The Snapshot condition retains the current edge-quality cue and replaces the
two slow Intensity--Flow fields with instantaneous quantities:

* ``gamma_edge`` is the current simulator SINR cue;
* ``feasibility_margin_edge`` is a signed, current-epoch proxy constructed
  only from current SINR and current admitted occupancy; and
* ``admitted_load_node`` is admitted users divided by fixed capacity, with no
  mean-flow state or EMA input.

No function in this module evaluates look-ahead geometry, a Cox hazard, an
integral, or an EMA.  Simulator authority remains separate from model output:
``simulator_snapshot_output`` reads the immutable simulator descriptor channel,
whereas a predicted :class:`SnapshotOutput` is consumed only by the Snapshot
controller.

The manuscript specifies that Snapshot score weights are tuned independently
for each comparison cell.  Accordingly, :class:`SnapshotScoreWeights` has no
defaults and :meth:`SnapshotPolicyConfig.from_mapping` rejects missing keys.
The values printed in a results table must be supplied by an experiment config;
this module does not present them as universal or recovered defaults.

``SnapshotHead`` deliberately follows the same ``forward(embedding, step)`` and
``output.validate(edge_count, satellite_count)`` contracts as
``IntensityFlowHead``.  It can therefore be injected as the readout of the
paper LTT-R or DA-GWM backbone without changing their temporal/spatial state
contract.  :func:`snapshot_one_step_loss` supplies the corresponding training
contract.  :class:`SnapshotDecisionAwareRankingLoss` has the same four-argument
hook shape as the DA-GWM decision-aware loss and can be installed alongside the
Snapshot head.  The dedicated ``snapshot_data``, ``snapshot_models``,
``snapshot_training`` and ``snapshot_evaluation`` modules install this branch
without routing Snapshot tensors through ``PolicyDescriptors``.

The paper does not state a unique dimensional formula for combining a dB SINR
slack and a normalized-load slack into one feasibility margin.  This module
uses the dimensionless Snapshot definition

``min((gamma-gamma_min)/gamma_scale, (load_max-load)/load_scale)``.

It is non-negative when current SINR and instantaneous occupancy satisfy the
Snapshot thresholds.  The simulator may still reject an action using its own
authoritative EMA-load gate.  Both scales and both thresholds are explicit in
:class:`SnapshotMarginConfig`, so the choice is fingerprintable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..control.policy import normalized_ordinal_rank
from ..sim.state import ControlObservation, ServingAction


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _dropout(value: Real) -> float:
    result = _finite_real("dropout", value)
    if not 0.0 <= result < 1.0:
        raise ValueError("dropout must lie in [0, 1)")
    return result


def _finite_vector(name: str, value: torch.Tensor, length: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1 or value.numel() != length:
        raise ValueError(
            f"{name} must have shape [{length}], got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")


def _same_device(name: str, tensors: tuple[torch.Tensor, ...]) -> None:
    if tensors and len({tensor.device for tensor in tensors}) != 1:
        raise ValueError(f"{name} tensors must share one device")


def _candidate_ids(
    name: str,
    value: torch.Tensor,
    *,
    edge_count: int,
    user_count: int | None = None,
    satellite_count: int | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (edge_count, 2) or value.dtype != torch.long:
        raise ValueError(f"{name} must have shape [{edge_count},2] and dtype long")
    if edge_count:
        users, satellites = value.unbind(dim=1)
        if int(users.min()) < 0 or int(satellites.min()) < 0:
            raise ValueError(f"{name} contains a negative local id")
        if user_count is not None and int(users.max()) >= user_count:
            raise ValueError(f"{name} contains an out-of-range user id")
        if satellite_count is not None and int(satellites.max()) >= satellite_count:
            raise ValueError(f"{name} contains an out-of-range satellite id")
        if torch.unique(value, dim=0).size(0) != edge_count:
            raise ValueError(f"{name} contains duplicate (user, satellite) pairs")


@dataclass(frozen=True)
class SnapshotMarginConfig:
    """Explicit normalization for the current simulator feasibility gates."""

    gamma_min: float
    load_max: float
    gamma_scale: float
    load_scale: float

    def __post_init__(self) -> None:
        gamma_min = _finite_real("gamma_min", self.gamma_min)
        load_max = _finite_real("load_max", self.load_max)
        gamma_scale = _finite_real("gamma_scale", self.gamma_scale)
        load_scale = _finite_real("load_scale", self.load_scale)
        if load_max < 0.0:
            raise ValueError("load_max must be non-negative")
        if gamma_scale <= 0.0 or load_scale <= 0.0:
            raise ValueError("gamma_scale and load_scale must be positive")
        object.__setattr__(self, "gamma_min", gamma_min)
        object.__setattr__(self, "load_max", load_max)
        object.__setattr__(self, "gamma_scale", gamma_scale)
        object.__setattr__(self, "load_scale", load_scale)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SnapshotMarginConfig":
        if not isinstance(config, Mapping):
            raise TypeError("snapshot margin config must be a mapping")
        required = ("gamma_min", "load_max", "gamma_scale", "load_scale")
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(
                "snapshot margin config is missing: " + ", ".join(missing)
            )
        return cls(**{name: config[name] for name in required})


@dataclass(frozen=True)
class SnapshotOutput:
    """Three real-valued descriptors exposed to the Snapshot score rule."""

    gamma_edge: torch.Tensor
    feasibility_margin_edge: torch.Tensor
    admitted_load_node: torch.Tensor

    def validate(self, edge_count: int, satellite_count: int) -> None:
        _finite_vector("gamma_edge", self.gamma_edge, edge_count)
        _finite_vector(
            "feasibility_margin_edge",
            self.feasibility_margin_edge,
            edge_count,
        )
        _finite_vector(
            "admitted_load_node",
            self.admitted_load_node,
            satellite_count,
        )
        _same_device(
            "SnapshotOutput",
            (
                self.gamma_edge,
                self.feasibility_margin_edge,
                self.admitted_load_node,
            ),
        )
        if bool((self.admitted_load_node < 0).any()):
            raise ValueError("admitted_load_node must be non-negative")

    def clone(self) -> "SnapshotOutput":
        return SnapshotOutput(
            gamma_edge=self.gamma_edge.clone(),
            feasibility_margin_edge=self.feasibility_margin_edge.clone(),
            admitted_load_node=self.admitted_load_node.clone(),
        )

    def detach(self) -> "SnapshotOutput":
        return SnapshotOutput(
            gamma_edge=self.gamma_edge.detach(),
            feasibility_margin_edge=self.feasibility_margin_edge.detach(),
            admitted_load_node=self.admitted_load_node.detach(),
        )

    def to(self, device: torch.device | str) -> "SnapshotOutput":
        return SnapshotOutput(
            gamma_edge=self.gamma_edge.to(device),
            feasibility_margin_edge=self.feasibility_margin_edge.to(device),
            admitted_load_node=self.admitted_load_node.to(device),
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "gamma_edge": self.gamma_edge,
            "feasibility_margin_edge": self.feasibility_margin_edge,
            "admitted_load_node": self.admitted_load_node,
        }


class SnapshotStepPredictor(Protocol):
    """Streaming prediction contract shared by Snapshot paper backbones."""

    def predict_step(
        self,
        step: Mapping[str, Any],
        state: torch.Tensor | None,
        device: torch.device | str,
    ) -> tuple[SnapshotOutput, torch.Tensor]:
        """Return next-epoch Snapshot descriptors and episode-local state."""


class SnapshotControllerInput(Protocol):
    """Descriptor-isolated view consumed by the Snapshot controller.

    The protocol intentionally has no ``sim_descriptors`` or
    ``policy_descriptors`` attribute.  Its concrete implementation is
    :class:`leo_pg.paper.snapshot_data.SnapshotControlInput`.
    """

    observation_id: tuple[int, int]
    candidate_edge_ids: torch.Tensor
    feasible_edge: torch.Tensor
    current_serving: torch.Tensor
    hold_steps: torch.Tensor
    user_order: torch.Tensor

    @property
    def user_count(self) -> int: ...

    @property
    def satellite_count(self) -> int: ...

    @property
    def edge_count(self) -> int: ...

    def validate(self) -> None: ...


def feasibility_margin_from_gates(
    gamma_sim_edge: torch.Tensor,
    gate_load_node: torch.Tensor,
    candidate_edge_ids: torch.Tensor,
    config: SnapshotMarginConfig,
) -> torch.Tensor:
    """Build the signed current-epoch Snapshot margin.

    The load column of ``candidate_edge_ids`` uses local satellite ids.  A
    result greater than or equal to zero is equivalent to the instantaneous
    Snapshot proxy ``gamma >= gamma_min and occupancy <= load_max``.  It does
    not claim equivalence to the simulator's separate EMA-load gate.
    """

    if not isinstance(config, SnapshotMarginConfig):
        raise TypeError("config must be a SnapshotMarginConfig")
    if not isinstance(candidate_edge_ids, torch.Tensor):
        raise TypeError("candidate_edge_ids must be a torch.Tensor")
    if not isinstance(gate_load_node, torch.Tensor):
        raise TypeError("gate_load_node must be a torch.Tensor")
    edge_count = int(candidate_edge_ids.size(0))
    satellite_count = int(gate_load_node.numel())
    _finite_vector("gamma_sim_edge", gamma_sim_edge, edge_count)
    _finite_vector("gate_load_node", gate_load_node, satellite_count)
    _candidate_ids(
        "candidate_edge_ids",
        candidate_edge_ids,
        edge_count=edge_count,
        satellite_count=satellite_count,
    )
    _same_device(
        "feasibility margin",
        (gamma_sim_edge, gate_load_node, candidate_edge_ids),
    )
    if bool((gate_load_node < 0).any()):
        raise ValueError("gate_load_node must be non-negative")
    destination = candidate_edge_ids[:, 1]
    gamma_slack = (gamma_sim_edge - config.gamma_min) / config.gamma_scale
    load_slack = (
        config.load_max - gate_load_node.index_select(0, destination)
    ) / config.load_scale
    margin = torch.minimum(gamma_slack, load_slack)
    if not bool(torch.isfinite(margin).all()):
        raise FloatingPointError("snapshot feasibility margin is NaN or Inf")
    return margin


def instantaneous_admitted_load_snapshot(
    admitted_users: torch.Tensor,
    *,
    capacity_users: int,
    clip_bounds: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Return current admitted occupancy without any congestion-state memory.

    Snapshot uses the current admitted-user count normalised by the fixed
    per-satellite capacity.  It deliberately does not read ``L(t)``, the
    mean-flow coupling map or the EMA update.
    """

    if not isinstance(admitted_users, torch.Tensor):
        raise TypeError("admitted_users must be a torch.Tensor")
    if admitted_users.ndim != 1 or admitted_users.numel() == 0:
        raise ValueError("admitted_users must be a non-empty vector")
    if not admitted_users.is_floating_point():
        raise TypeError("admitted_users must have a floating-point dtype")
    if not bool(torch.isfinite(admitted_users).all()) or bool(
        (admitted_users < 0).any()
    ):
        raise ValueError("admitted_users must be finite and non-negative")
    capacity = _positive_int("capacity_users", capacity_users)
    if not isinstance(clip_bounds, tuple) or len(clip_bounds) != 2:
        raise TypeError("clip_bounds must be a (lower, upper) tuple")
    lower = _finite_real("clip_bounds[0]", clip_bounds[0])
    upper = _finite_real("clip_bounds[1]", clip_bounds[1])
    if not 0.0 <= lower < upper:
        raise ValueError("clip bounds must satisfy 0 <= lower < upper")
    return (admitted_users / float(capacity)).clamp(lower, upper)


def simulator_snapshot_output(
    observation: ControlObservation,
    admitted_load_node: torch.Tensor,
    margin_config: SnapshotMarginConfig,
) -> SnapshotOutput:
    """Construct an oracle Snapshot descriptor copy from simulator authority.

    The Snapshot margin and load score both use ``admitted_load_node``, the
    current occupancy fraction without EMA memory.  The simulator's own
    feasibility mask remains available only as an authoritative execution
    predicate; it is not copied into either real-valued Snapshot descriptor.
    """

    if not isinstance(observation, ControlObservation):
        raise TypeError("observation must be a ControlObservation")
    observation.validate()
    _finite_vector(
        "admitted_load_node",
        admitted_load_node,
        observation.satellite_count,
    )
    fields = observation.sim_descriptors.policy_fields
    _same_device(
        "simulator Snapshot",
        (
            observation.candidate_edge_ids,
            fields.gamma_edge,
            admitted_load_node,
        ),
    )
    margin = feasibility_margin_from_gates(
        fields.gamma_edge,
        admitted_load_node,
        observation.candidate_edge_ids,
        margin_config,
    )
    output = SnapshotOutput(
        gamma_edge=fields.gamma_edge.clone(),
        feasibility_margin_edge=margin,
        admitted_load_node=admitted_load_node.clone(),
    )
    output.validate(observation.edge_count, observation.satellite_count)
    return output


def _head_graph(
    step: Mapping[str, Any],
    *,
    device: torch.device,
    node_count: int,
    edge_in_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if not isinstance(step, Mapping):
        raise TypeError("step must be a mapping")
    meta = step.get("meta")
    if not isinstance(meta, Mapping) or "K_users" not in meta:
        raise ValueError("SnapshotHead requires step.meta.K_users")
    user_count = _positive_int("step.meta.K_users", meta["K_users"])
    satellite_count = _positive_int(
        "step.meta.S_sats",
        meta.get("S_sats", node_count - user_count),
    )
    if user_count + satellite_count != node_count:
        raise ValueError("step K_users/S_sats do not match node embeddings")
    if "edge_index" not in step or "edge_z" not in step:
        raise ValueError("SnapshotHead requires step.edge_index and step.edge_z")
    edge_index = torch.as_tensor(step["edge_index"], device=device)
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("step.edge_index must have shape [2,E]")
    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("step.edge_index must have an integer dtype")
    edge_index = edge_index.long()
    edge_z = torch.as_tensor(step["edge_z"], device=device).float()
    if edge_z.shape != (edge_index.size(1), edge_in_dim):
        raise ValueError(
            f"step.edge_z must have shape [{edge_index.size(1)},{edge_in_dim}]"
        )
    if not bool(torch.isfinite(edge_z).all()):
        raise ValueError("step.edge_z contains NaN or Inf")

    edge_type = step.get("edge_type")
    if edge_type is not None:
        edge_type_tensor = torch.as_tensor(edge_type, device=device)
        if edge_type_tensor.ndim != 1 or edge_type_tensor.numel() != edge_index.size(1):
            raise ValueError("step.edge_type must have one entry per edge")
        candidate = edge_type_tensor == 0
        edge_index = edge_index[:, candidate]
        edge_z = edge_z[candidate]
    if edge_index.numel():
        source, destination = edge_index
        if (
            int(source.min()) < 0
            or int(source.max()) >= user_count
            or int(destination.min()) < user_count
            or int(destination.max()) >= node_count
        ):
            raise ValueError("Snapshot candidate edges must run from users to satellites")
    return edge_index, edge_z, user_count, satellite_count


class SnapshotHead(nn.Module):
    """Read current-cue, current-margin, and instantaneous-load descriptors."""

    def __init__(
        self,
        in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        load_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_dim = _positive_int("in_dim", in_dim)
        self.edge_in_dim = _positive_int("edge_in_dim", edge_in_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.dropout = _dropout(dropout)
        if not isinstance(load_bounds, tuple) or len(load_bounds) != 2:
            raise TypeError("load_bounds must be a (lower, upper) tuple")
        lower = _finite_real("load_bounds[0]", load_bounds[0])
        upper = _finite_real("load_bounds[1]", load_bounds[1])
        if not 0.0 <= lower < upper:
            raise ValueError("load_bounds must satisfy 0 <= lower < upper")
        self.load_lower = lower
        self.load_span = upper - lower

        self.edge_trunk = nn.Sequential(
            nn.Linear(2 * self.in_dim + self.edge_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.gamma_head = nn.Linear(self.hidden_dim, 1)
        self.feasibility_margin_head = nn.Linear(self.hidden_dim, 1)
        self.load_head = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        step: Mapping[str, Any] | None = None,
    ) -> SnapshotOutput:
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("embedding must be a torch.Tensor")
        if embedding.ndim != 2 or embedding.size(1) != self.in_dim:
            raise ValueError(
                f"embedding must have shape [N,{self.in_dim}], "
                f"got {tuple(embedding.shape)}"
            )
        if not embedding.is_floating_point() or not bool(
            torch.isfinite(embedding).all()
        ):
            raise ValueError("embedding must be a finite floating-point tensor")
        if step is None:
            raise ValueError("SnapshotHead requires a graph step mapping")
        edge_index, edge_z, user_count, satellite_count = _head_graph(
            step,
            device=embedding.device,
            node_count=int(embedding.size(0)),
            edge_in_dim=self.edge_in_dim,
        )
        source, destination = edge_index
        edge_hidden = self.edge_trunk(
            torch.cat(
                (embedding[source], embedding[destination], edge_z),
                dim=-1,
            )
        )
        gamma = self.gamma_head(edge_hidden).squeeze(-1)
        margin = self.feasibility_margin_head(edge_hidden).squeeze(-1)
        normalized_load = torch.sigmoid(
            self.load_head(embedding[user_count:]).squeeze(-1)
        )
        admitted_load = self.load_lower + self.load_span * normalized_load
        output = SnapshotOutput(
            gamma_edge=gamma,
            feasibility_margin_edge=margin,
            admitted_load_node=admitted_load,
        )
        output.validate(int(edge_index.size(1)), satellite_count)
        return output


@dataclass(frozen=True)
class SnapshotTarget:
    """Next-epoch Snapshot targets aligned to the source candidate order."""

    gamma_edge: torch.Tensor
    feasibility_margin_edge: torch.Tensor
    admitted_load_node: torch.Tensor
    persistent_edge: torch.Tensor

    def validate(self, edge_count: int, satellite_count: int) -> None:
        _finite_vector("gamma_edge", self.gamma_edge, edge_count)
        _finite_vector(
            "feasibility_margin_edge",
            self.feasibility_margin_edge,
            edge_count,
        )
        _finite_vector(
            "admitted_load_node",
            self.admitted_load_node,
            satellite_count,
        )
        if not isinstance(self.persistent_edge, torch.Tensor):
            raise TypeError("persistent_edge must be a torch.Tensor")
        if self.persistent_edge.shape != (edge_count,):
            raise ValueError(f"persistent_edge must have shape [{edge_count}]")
        if self.persistent_edge.dtype != torch.bool:
            raise ValueError("persistent_edge must have dtype bool")
        _same_device(
            "SnapshotTarget",
            (
                self.gamma_edge,
                self.feasibility_margin_edge,
                self.admitted_load_node,
                self.persistent_edge,
            ),
        )
        if bool((self.admitted_load_node < 0).any()):
            raise ValueError("admitted_load_node must be non-negative")

    def as_output(self) -> SnapshotOutput:
        """Expose target values to a ranking-loss hook without the join mask."""

        return SnapshotOutput(
            gamma_edge=self.gamma_edge,
            feasibility_margin_edge=self.feasibility_margin_edge,
            admitted_load_node=self.admitted_load_node,
        )


def build_snapshot_next_step_target(
    source_observation: ControlObservation,
    next_observation: ControlObservation,
    next_snapshot: SnapshotOutput,
) -> SnapshotTarget:
    """Join Snapshot values at ``t+1`` onto candidate ids observed at ``t``."""

    if not isinstance(source_observation, ControlObservation) or not isinstance(
        next_observation, ControlObservation
    ):
        raise TypeError("source_observation and next_observation must be observations")
    if not isinstance(next_snapshot, SnapshotOutput):
        raise TypeError("next_snapshot must be a SnapshotOutput")
    source_observation.validate()
    next_observation.validate()
    if source_observation.episode_seed != next_observation.episode_seed:
        raise ValueError("source and next observations must belong to one episode")
    if next_observation.epoch != source_observation.epoch + 1:
        raise ValueError("Snapshot target construction requires consecutive epochs")
    if (
        source_observation.user_count != next_observation.user_count
        or source_observation.satellite_count != next_observation.satellite_count
    ):
        raise ValueError("node identities must be stable across a one-step target")
    next_snapshot.validate(
        next_observation.edge_count,
        next_observation.satellite_count,
    )
    _candidate_ids(
        "source candidate_edge_ids",
        source_observation.candidate_edge_ids,
        edge_count=source_observation.edge_count,
        user_count=source_observation.user_count,
        satellite_count=source_observation.satellite_count,
    )
    _candidate_ids(
        "next candidate_edge_ids",
        next_observation.candidate_edge_ids,
        edge_count=next_observation.edge_count,
        user_count=next_observation.user_count,
        satellite_count=next_observation.satellite_count,
    )
    _same_device(
        "Snapshot target join",
        (
            source_observation.candidate_edge_ids,
            next_observation.candidate_edge_ids,
            next_snapshot.gamma_edge,
            next_snapshot.feasibility_margin_edge,
            next_snapshot.admitted_load_node,
        ),
    )

    edge_count = source_observation.edge_count
    gamma = torch.zeros(
        edge_count,
        dtype=next_snapshot.gamma_edge.dtype,
        device=next_snapshot.gamma_edge.device,
    )
    margin = torch.zeros_like(gamma)
    persistent = torch.zeros(
        edge_count,
        dtype=torch.bool,
        device=next_snapshot.gamma_edge.device,
    )
    next_lookup = {
        (int(user), int(satellite)): index
        for index, (user, satellite) in enumerate(
            next_observation.candidate_edge_ids.tolist()
        )
    }
    for source_index, (user, satellite) in enumerate(
        source_observation.candidate_edge_ids.tolist()
    ):
        next_index = next_lookup.get((int(user), int(satellite)))
        if next_index is None:
            continue
        persistent[source_index] = True
        gamma[source_index] = next_snapshot.gamma_edge[next_index]
        margin[source_index] = next_snapshot.feasibility_margin_edge[next_index]

    target = SnapshotTarget(
        gamma_edge=gamma,
        feasibility_margin_edge=margin,
        admitted_load_node=next_snapshot.admitted_load_node.clone(),
        persistent_edge=persistent,
    )
    target.validate(edge_count, source_observation.satellite_count)
    return target


@dataclass(frozen=True)
class SnapshotLossWeights:
    """Explicit one-step weights for the three Snapshot descriptor domains."""

    gamma: float
    feasibility_margin: float
    admitted_load: float

    def __post_init__(self) -> None:
        values = {
            "gamma": self.gamma,
            "feasibility_margin": self.feasibility_margin,
            "admitted_load": self.admitted_load,
        }
        normalized = {name: _finite_real(name, value) for name, value in values.items()}
        if any(value < 0 for value in normalized.values()):
            raise ValueError("Snapshot loss weights must be non-negative")
        if sum(normalized.values()) <= 0:
            raise ValueError("at least one Snapshot loss weight must be positive")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SnapshotLossWeights":
        if not isinstance(config, Mapping):
            raise TypeError("Snapshot loss-weight config must be a mapping")
        required = ("gamma", "feasibility_margin", "admitted_load")
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError("Snapshot loss weights are missing: " + ", ".join(missing))
        return cls(**{name: config[name] for name in required})


@dataclass(frozen=True)
class SnapshotLoss:
    total: torch.Tensor
    gamma: torch.Tensor
    feasibility_margin: torch.Tensor
    admitted_load: torch.Tensor
    persistent_edge_count: int


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.sum() * 0.0


def snapshot_one_step_loss(
    prediction: SnapshotOutput,
    target: SnapshotTarget,
    weights: SnapshotLossWeights,
) -> SnapshotLoss:
    """Compute persistent-edge MSE plus node-domain instantaneous-load MSE."""

    if not isinstance(prediction, SnapshotOutput):
        raise TypeError("prediction must be a SnapshotOutput")
    if not isinstance(target, SnapshotTarget):
        raise TypeError("target must be a SnapshotTarget")
    if not isinstance(weights, SnapshotLossWeights):
        raise TypeError("weights must be SnapshotLossWeights")
    edge_count = int(prediction.gamma_edge.numel())
    satellite_count = int(prediction.admitted_load_node.numel())
    prediction.validate(edge_count, satellite_count)
    target.validate(edge_count, satellite_count)
    _same_device(
        "Snapshot loss",
        (
            prediction.gamma_edge,
            prediction.feasibility_margin_edge,
            prediction.admitted_load_node,
            target.gamma_edge,
            target.feasibility_margin_edge,
            target.admitted_load_node,
            target.persistent_edge,
        ),
    )
    persistent = target.persistent_edge
    gamma_loss = _masked_mean(
        (prediction.gamma_edge - target.gamma_edge).square(),
        persistent,
    )
    margin_loss = _masked_mean(
        (
            prediction.feasibility_margin_edge
            - target.feasibility_margin_edge
        ).square(),
        persistent,
    )
    load_loss = F.mse_loss(
        prediction.admitted_load_node,
        target.admitted_load_node,
    )
    total = (
        weights.gamma * gamma_loss
        + weights.feasibility_margin * margin_loss
        + weights.admitted_load * load_loss
    )
    components = (total, gamma_loss, margin_loss, load_loss)
    if not all(bool(torch.isfinite(value)) for value in components):
        raise FloatingPointError("Snapshot loss contains NaN or Inf")
    return SnapshotLoss(
        total=total,
        gamma=gamma_loss,
        feasibility_margin=margin_loss,
        admitted_load=load_loss,
        persistent_edge_count=int(persistent.sum().item()),
    )


@dataclass(frozen=True)
class SnapshotScoreWeights:
    """Explicit Snapshot score weights; intentionally no paper-value defaults."""

    gamma: float
    feasibility: float
    load: float

    def __post_init__(self) -> None:
        values = {
            "gamma": self.gamma,
            "feasibility": self.feasibility,
            "load": self.load,
        }
        normalized = {name: _finite_real(name, value) for name, value in values.items()}
        if any(value < 0 for value in normalized.values()):
            raise ValueError("Snapshot score weights must be non-negative")
        if sum(normalized.values()) <= 0:
            raise ValueError("at least one Snapshot score weight must be positive")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SnapshotScoreWeights":
        if not isinstance(config, Mapping):
            raise TypeError("Snapshot score-weight config must be a mapping")
        required = ("gamma", "feasibility", "load")
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(
                "Snapshot score weights are missing: " + ", ".join(missing)
            )
        return cls(**{name: config[name] for name in required})


# Supplementary Information factorial search.  These are reported experiment
# settings, not fallbacks for an omitted configuration.
SNAPSHOT_SI_SCORE_GRID = {
    "gamma": (0.5, 0.75, 1.0, 1.25, 1.5),
    "feasibility": (0.2, 0.4, 0.6, 0.8),
    "load": (0.2, 0.4, 0.6, 0.8),
}
SNAPSHOT_SI_SELECTED_SCORE_WEIGHTS = {
    "snapshot_mlp": SnapshotScoreWeights(
        gamma=1.0, feasibility=0.6, load=0.8
    ),
    "snapshot_physick": SnapshotScoreWeights(
        gamma=1.0, feasibility=0.4, load=0.8
    ),
}


def snapshot_si_policy_config(
    method: str,
    *,
    hard_feasibility_mask: bool,
    min_dwell_steps: int,
    hysteresis: float | None,
) -> "SnapshotPolicyConfig":
    """Return a policy config only for a method with SI-selected weights."""

    normalized = str(method).strip().lower().replace("-", "_")
    weights = SNAPSHOT_SI_SELECTED_SCORE_WEIGHTS.get(normalized)
    if weights is None:
        choices = ", ".join(sorted(SNAPSHOT_SI_SELECTED_SCORE_WEIGHTS))
        raise ValueError(
            f"no SI-selected Snapshot score weights for {method!r}; expected {choices}"
        )
    return SnapshotPolicyConfig(
        weights=weights,
        hard_feasibility_mask=hard_feasibility_mask,
        min_dwell_steps=min_dwell_steps,
        hysteresis=hysteresis,
    )


@dataclass(frozen=True)
class SnapshotPolicyConfig:
    """Fixed controller settings for one explicitly weighted Snapshot cell."""

    weights: SnapshotScoreWeights
    hard_feasibility_mask: bool = False
    min_dwell_steps: int = 10
    hysteresis: float | None = 1.0 / 6.0

    def __post_init__(self) -> None:
        if not isinstance(self.weights, SnapshotScoreWeights):
            raise TypeError("weights must be a SnapshotScoreWeights instance")
        if not isinstance(self.hard_feasibility_mask, bool):
            raise TypeError("hard_feasibility_mask must be bool")
        if isinstance(self.min_dwell_steps, bool):
            raise TypeError("min_dwell_steps must be an integer")
        try:
            dwell = int(self.min_dwell_steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("min_dwell_steps must be an integer") from exc
        if dwell != self.min_dwell_steps or dwell < 0:
            raise ValueError("min_dwell_steps must be a non-negative integer")
        object.__setattr__(self, "min_dwell_steps", dwell)
        if self.hysteresis is not None:
            hysteresis = _finite_real("hysteresis", self.hysteresis)
            if hysteresis < 0:
                raise ValueError("hysteresis must be non-negative")
            object.__setattr__(self, "hysteresis", hysteresis)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SnapshotPolicyConfig":
        """Parse a config while requiring all three score-weight keys."""

        if not isinstance(config, Mapping):
            raise TypeError("Snapshot policy config must be a mapping")
        weight_config = config.get("score_weights")
        if not isinstance(weight_config, Mapping):
            raise ValueError(
                "Snapshot policy config requires a score_weights mapping with "
                "gamma, feasibility, and load"
            )
        return cls(
            weights=SnapshotScoreWeights.from_mapping(weight_config),
            hard_feasibility_mask=config.get("hard_feasibility_mask", False),
            min_dwell_steps=config.get("min_dwell_steps", 10),
            hysteresis=config.get("hysteresis", 1.0 / 6.0),
        )


@dataclass(frozen=True)
class SnapshotCandidateScores:
    gamma_rank: torch.Tensor
    feasibility_rank: torch.Tensor
    load_rank: torch.Tensor
    total: torch.Tensor
    eligible: torch.Tensor


def score_snapshot_candidates(
    observation: SnapshotControllerInput,
    descriptors: SnapshotOutput,
    config: SnapshotPolicyConfig,
) -> SnapshotCandidateScores:
    """Rank Snapshot descriptors per user under the copied authoritative mask."""

    required = (
        "validate",
        "candidate_edge_ids",
        "feasible_edge",
        "current_serving",
        "hold_steps",
        "user_order",
    )
    if any(not hasattr(observation, name) for name in required):
        raise TypeError("observation must implement SnapshotControllerInput")
    if not isinstance(descriptors, SnapshotOutput):
        raise TypeError("descriptors must be a SnapshotOutput")
    if not isinstance(config, SnapshotPolicyConfig):
        raise TypeError("config must be a SnapshotPolicyConfig")
    observation.validate()
    descriptors.validate(observation.edge_count, observation.satellite_count)
    _same_device(
        "Snapshot controller",
        (
            observation.candidate_edge_ids,
            observation.feasible_edge,
            observation.current_serving,
            observation.hold_steps,
            observation.user_order,
            descriptors.gamma_edge,
            descriptors.feasibility_margin_edge,
            descriptors.admitted_load_node,
        ),
    )

    edge_count = observation.edge_count
    dtype = descriptors.gamma_edge.dtype
    device = descriptors.gamma_edge.device
    gamma_rank = torch.zeros(edge_count, dtype=dtype, device=device)
    feasibility_rank = torch.zeros_like(gamma_rank)
    load_rank = torch.zeros_like(gamma_rank)
    total = torch.full_like(gamma_rank, -torch.inf)
    eligible = torch.zeros(edge_count, dtype=torch.bool, device=device)
    edge_users = observation.candidate_edge_ids[:, 0]
    edge_satellites = observation.candidate_edge_ids[:, 1]

    for user in observation.user_order.tolist():
        user_edges = torch.nonzero(edge_users == int(user), as_tuple=False).flatten()
        if user_edges.numel() == 0:
            continue
        satellites = edge_satellites.index_select(0, user_edges)
        if torch.unique(satellites).numel() != satellites.numel():
            raise ValueError(f"user {user} has duplicate candidate satellite ids")
        ranked_edges = user_edges
        if config.hard_feasibility_mask:
            feasible = observation.feasible_edge.index_select(0, ranked_edges)
            ranked_edges = ranked_edges[feasible]
        if ranked_edges.numel() == 0:
            continue
        satellites = edge_satellites.index_select(0, ranked_edges)
        user_gamma_rank = normalized_ordinal_rank(
            descriptors.gamma_edge.index_select(0, ranked_edges),
            satellites,
            higher_is_better=True,
        )
        user_feasibility_rank = normalized_ordinal_rank(
            descriptors.feasibility_margin_edge.index_select(0, ranked_edges),
            satellites,
            higher_is_better=True,
        )
        user_load_rank = normalized_ordinal_rank(
            descriptors.admitted_load_node.index_select(0, satellites),
            satellites,
            higher_is_better=False,
        )
        user_total = (
            config.weights.gamma * user_gamma_rank
            + config.weights.feasibility * user_feasibility_rank
            + config.weights.load * user_load_rank
        )
        gamma_rank[ranked_edges] = user_gamma_rank
        feasibility_rank[ranked_edges] = user_feasibility_rank
        load_rank[ranked_edges] = user_load_rank
        total[ranked_edges] = user_total
        eligible[ranked_edges] = True

    return SnapshotCandidateScores(
        gamma_rank=gamma_rank,
        feasibility_rank=feasibility_rank,
        load_rank=load_rank,
        total=total,
        eligible=eligible,
    )


def _best_snapshot_edge(
    edge_indices: torch.Tensor,
    scores: torch.Tensor,
    satellite_ids: torch.Tensor,
) -> int:
    if edge_indices.numel() == 0:
        raise ValueError("cannot choose from an empty edge set")
    satellite_order = torch.argsort(
        satellite_ids.index_select(0, edge_indices),
        stable=True,
    )
    by_satellite = edge_indices.index_select(0, satellite_order)
    score_order = torch.argsort(
        scores.index_select(0, by_satellite),
        descending=True,
        stable=True,
    )
    return int(by_satellite[score_order[0]].item())


class SnapshotFixedRankController:
    """Snapshot score rule with the shared dwell and hysteresis constraints."""

    def __init__(self, config: SnapshotPolicyConfig) -> None:
        if not isinstance(config, SnapshotPolicyConfig):
            raise TypeError("config must be a SnapshotPolicyConfig")
        self.config = config

    def score(
        self,
        observation: SnapshotControllerInput,
        descriptors: SnapshotOutput,
    ) -> SnapshotCandidateScores:
        return score_snapshot_candidates(observation, descriptors, self.config)

    def select_action(
        self,
        observation: SnapshotControllerInput,
        descriptors: SnapshotOutput,
    ) -> ServingAction:
        scores = self.score(observation, descriptors)
        requested = observation.current_serving.clone()
        edge_users = observation.candidate_edge_ids[:, 0]
        edge_satellites = observation.candidate_edge_ids[:, 1]

        for user in observation.user_order.tolist():
            user_edges = torch.nonzero(
                edge_users == int(user), as_tuple=False
            ).flatten()
            ranked_edges = user_edges[scores.eligible.index_select(0, user_edges)]
            current = int(observation.current_serving[user].item())
            if ranked_edges.numel() == 0:
                continue
            best_edge = _best_snapshot_edge(
                ranked_edges,
                scores.total,
                edge_satellites,
            )
            best_satellite = int(edge_satellites[best_edge].item())
            if current < 0:
                requested[user] = best_satellite
                continue
            current_edges = user_edges[
                edge_satellites.index_select(0, user_edges) == current
            ]
            if current_edges.numel() != 1 or not bool(
                scores.eligible[current_edges[0]].item()
            ):
                continue
            if best_satellite == current:
                continue
            if int(observation.hold_steps[user].item()) < self.config.min_dwell_steps:
                continue
            current_edge = int(current_edges[0].item())
            hysteresis = self.config.hysteresis
            if hysteresis is None:
                hysteresis = 1.0 / float(ranked_edges.numel())
            best_score = float(scores.total[best_edge].item())
            required_score = float(scores.total[current_edge].item()) + hysteresis
            if best_score >= required_score or math.isclose(
                best_score,
                required_score,
                rel_tol=1e-7,
                abs_tol=1e-8,
            ):
                requested[user] = best_satellite

        action = ServingAction(
            observation_id=observation.observation_id,
            requested_serving=requested,
        )
        action.validate(observation.user_count, observation.satellite_count)
        return action

    def __call__(
        self,
        observation: SnapshotControllerInput,
        descriptors: SnapshotOutput,
    ) -> ServingAction:
        return self.select_action(observation, descriptors)


def select_snapshot_action(
    observation: SnapshotControllerInput,
    descriptors: SnapshotOutput,
    config: SnapshotPolicyConfig,
) -> ServingAction:
    """Functional entry point for the explicitly configured Snapshot policy."""

    return SnapshotFixedRankController(config).select_action(
        observation,
        descriptors,
    )


class SnapshotDecisionAwareRankingLoss(nn.Module):
    """DA-GWM-compatible differentiable surrogate for Snapshot score order.

    The score weights and descriptor scales are all constructor arguments.
    Thus a DA-GWM Snapshot cell uses the same validated weight configuration as
    its fixed controller and does not inherit Intensity--Flow defaults.
    """

    def __init__(
        self,
        *,
        weights: SnapshotScoreWeights,
        gamma_scale: float,
        feasibility_scale: float,
        load_scale: float,
        soft_rank_temperature: float,
        pair_margin: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(weights, SnapshotScoreWeights):
            raise TypeError("weights must be SnapshotScoreWeights")
        self.weights = weights
        self.gamma_scale = _finite_real("gamma_scale", gamma_scale)
        self.feasibility_scale = _finite_real(
            "feasibility_scale", feasibility_scale
        )
        self.load_scale = _finite_real("load_scale", load_scale)
        self.soft_rank_temperature = _finite_real(
            "soft_rank_temperature", soft_rank_temperature
        )
        self.pair_margin = _finite_real("pair_margin", pair_margin)
        if (
            self.gamma_scale <= 0
            or self.feasibility_scale <= 0
            or self.load_scale <= 0
            or self.soft_rank_temperature <= 0
        ):
            raise ValueError("descriptor scales and temperature must be positive")
        if self.pair_margin < 0:
            raise ValueError("pair_margin must be non-negative")

    def _soft_rank(
        self,
        values: torch.Tensor,
        *,
        higher_is_better: bool,
    ) -> torch.Tensor:
        signed = values if higher_is_better else -values
        pairwise = torch.sigmoid(
            (signed[:, None] - signed[None, :])
            / self.soft_rank_temperature
        )
        return (pairwise.sum(dim=1) + 0.5) / float(values.numel())

    def forward(
        self,
        prediction: SnapshotOutput,
        oracle: SnapshotOutput | SnapshotTarget,
        candidate_edge_ids: torch.Tensor,
        persistent_edge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(prediction, SnapshotOutput):
            raise TypeError("prediction must be a SnapshotOutput")
        if not isinstance(oracle, (SnapshotOutput, SnapshotTarget)):
            raise TypeError("oracle must be a SnapshotOutput or SnapshotTarget")
        if not isinstance(candidate_edge_ids, torch.Tensor):
            raise TypeError("candidate_edge_ids must be a torch.Tensor")
        edge_count = int(candidate_edge_ids.size(0))
        satellite_count = int(oracle.admitted_load_node.numel())
        prediction.validate(edge_count, satellite_count)
        oracle.validate(edge_count, satellite_count)
        if isinstance(oracle, SnapshotTarget):
            if persistent_edge is None:
                persistent_edge = oracle.persistent_edge
            oracle_output = oracle.as_output()
        else:
            oracle_output = oracle
        _candidate_ids(
            "candidate_edge_ids",
            candidate_edge_ids,
            edge_count=edge_count,
            satellite_count=satellite_count,
        )
        _same_device(
            "Snapshot decision-aware loss",
            (
                candidate_edge_ids,
                prediction.gamma_edge,
                prediction.feasibility_margin_edge,
                prediction.admitted_load_node,
                oracle_output.gamma_edge,
                oracle_output.feasibility_margin_edge,
                oracle_output.admitted_load_node,
            ),
        )
        if persistent_edge is None:
            persistent = torch.ones(
                edge_count,
                dtype=torch.bool,
                device=candidate_edge_ids.device,
            )
        else:
            if not isinstance(persistent_edge, torch.Tensor):
                raise TypeError("persistent_edge must be a torch.Tensor or None")
            if (
                persistent_edge.shape != (edge_count,)
                or persistent_edge.dtype != torch.bool
            ):
                raise ValueError("persistent_edge must be bool with shape [E]")
            if persistent_edge.device != candidate_edge_ids.device:
                raise ValueError("persistent_edge must share the candidate device")
            persistent = persistent_edge

        losses: list[torch.Tensor] = []
        edge_users = candidate_edge_ids[:, 0]
        edge_satellites = candidate_edge_ids[:, 1]
        for user in torch.unique(edge_users).tolist():
            edges = torch.nonzero(
                (edge_users == int(user)) & persistent,
                as_tuple=False,
            ).flatten()
            if edges.numel() < 2:
                continue
            satellites = edge_satellites.index_select(0, edges)
            oracle_score = (
                self.weights.gamma
                * normalized_ordinal_rank(
                    oracle_output.gamma_edge.index_select(0, edges),
                    satellites,
                    higher_is_better=True,
                )
                + self.weights.feasibility
                * normalized_ordinal_rank(
                    oracle_output.feasibility_margin_edge.index_select(0, edges),
                    satellites,
                    higher_is_better=True,
                )
                + self.weights.load
                * normalized_ordinal_rank(
                    oracle_output.admitted_load_node.index_select(0, satellites),
                    satellites,
                    higher_is_better=False,
                )
            ).detach()
            predicted_score = (
                self.weights.gamma
                * self._soft_rank(
                    prediction.gamma_edge.index_select(0, edges)
                    / self.gamma_scale,
                    higher_is_better=True,
                )
                + self.weights.feasibility
                * self._soft_rank(
                    prediction.feasibility_margin_edge.index_select(0, edges)
                    / self.feasibility_scale,
                    higher_is_better=True,
                )
                + self.weights.load
                * self._soft_rank(
                    prediction.admitted_load_node.index_select(0, satellites)
                    / self.load_scale,
                    higher_is_better=False,
                )
            )
            oracle_difference = oracle_score[:, None] - oracle_score[None, :]
            ordered = oracle_difference > 0
            if not bool(ordered.any()):
                continue
            predicted_difference = (
                predicted_score[:, None] - predicted_score[None, :]
            )
            violation = self.pair_margin - predicted_difference[ordered]
            losses.append(
                F.softplus(violation / self.soft_rank_temperature)
                * self.soft_rank_temperature
            )

        if not losses:
            return prediction.gamma_edge.sum() * 0.0
        result = torch.cat(losses).mean()
        if not bool(torch.isfinite(result)):
            raise FloatingPointError(
                "Snapshot decision-aware ranking loss is NaN or Inf"
            )
        return result
