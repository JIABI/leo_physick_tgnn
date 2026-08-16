from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Mapping, Tuple

import torch


ObservationId = Tuple[int, int]
PAPER_FEATURE_CONTRACT_VERSION = 2
PAPER_NODE_FEATURE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "policy_flow",
)
PAPER_EDGE_FEATURE_NAMES = (
    "elevation_normalized",
    "log_distance",
    "policy_gamma_normalized",
    "policy_destination_flow",
    "policy_log1p_intensity",
    "is_current_association",
    "dwell_normalized",
)


def _finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _vector(name: str, value: torch.Tensor, length: int) -> None:
    if value.ndim != 1 or value.numel() != length:
        raise ValueError(f"{name} must have shape [{length}], got {tuple(value.shape)}")


@dataclass
class PolicyDescriptors:
    """The three real-valued fields exposed to the fixed score controller.

    ``gamma_edge`` and ``intensity_edge`` follow the stable candidate-edge order
    in :class:`ControlObservation`; ``flow_node`` follows local satellite ids.
    The auxiliary feasibility prediction is intentionally absent from this
    contract because the manuscript assigns feasibility authority to the
    simulator.
    """

    gamma_edge: torch.Tensor
    intensity_edge: torch.Tensor
    flow_node: torch.Tensor

    def validate(self, edge_count: int, satellite_count: int) -> None:
        _vector("gamma_edge", self.gamma_edge, edge_count)
        _vector("intensity_edge", self.intensity_edge, edge_count)
        _vector("flow_node", self.flow_node, satellite_count)
        _finite("gamma_edge", self.gamma_edge)
        _finite("intensity_edge", self.intensity_edge)
        _finite("flow_node", self.flow_node)
        if torch.any(self.intensity_edge < 0):
            raise ValueError("intensity_edge must be non-negative")

    def clone(self) -> "PolicyDescriptors":
        return PolicyDescriptors(
            gamma_edge=self.gamma_edge.clone(),
            intensity_edge=self.intensity_edge.clone(),
            flow_node=self.flow_node.clone(),
        )

    def detach(self) -> "PolicyDescriptors":
        return PolicyDescriptors(
            gamma_edge=self.gamma_edge.detach(),
            intensity_edge=self.intensity_edge.detach(),
            flow_node=self.flow_node.detach(),
        )

    def to(self, device: torch.device | str) -> "PolicyDescriptors":
        return PolicyDescriptors(
            gamma_edge=self.gamma_edge.to(device),
            intensity_edge=self.intensity_edge.to(device),
            flow_node=self.flow_node.to(device),
        )

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "gamma_edge": self.gamma_edge,
            "intensity_edge": self.intensity_edge,
            "flow_node": self.flow_node,
        }


@dataclass
class SimulatorDescriptors:
    """Simulator-authoritative descriptors and feasibility predicate."""

    policy_fields: PolicyDescriptors
    feasible_edge: torch.Tensor

    def validate(self, edge_count: int, satellite_count: int) -> None:
        self.policy_fields.validate(edge_count, satellite_count)
        _vector("feasible_edge", self.feasible_edge, edge_count)
        if self.feasible_edge.dtype != torch.bool:
            raise ValueError("feasible_edge must have dtype bool")

    def clone(self) -> "SimulatorDescriptors":
        return SimulatorDescriptors(
            policy_fields=self.policy_fields.clone(),
            feasible_edge=self.feasible_edge.clone(),
        )


@dataclass
class ControlObservation:
    """Immutable-by-convention decision-epoch observation.

    The simulator and policy descriptor channels are stored separately. Calling
    :meth:`with_policy_descriptors` replaces only the decision-channel copy;
    geometry, history, and simulator authority are cloned unchanged.
    """

    observation_id: ObservationId
    node_x: torch.Tensor
    candidate_edge_index: torch.Tensor
    candidate_edge_ids: torch.Tensor
    edge_features: torch.Tensor
    elevation_deg: torch.Tensor
    sim_descriptors: SimulatorDescriptors
    policy_descriptors: PolicyDescriptors
    current_serving: torch.Tensor
    hold_steps: torch.Tensor
    user_order: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)

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
        return int(self.candidate_edge_index.size(1))

    def validate(self) -> None:
        if (
            not isinstance(self.observation_id, tuple)
            or len(self.observation_id) != 2
            or self.episode_seed < 0
            or self.epoch < 0
        ):
            raise ValueError("observation_id must be a non-negative (episode_seed, epoch) tuple")
        if self.node_x.ndim != 2 or self.node_x.size(1) < 3:
            raise ValueError("node_x must have shape [N, feature_dim>=3]")
        _finite("node_x", self.node_x)
        K, S, E = self.user_count, self.satellite_count, self.edge_count
        if K <= 0 or S <= 0:
            raise ValueError("an observation requires at least one user and one satellite")
        if self.candidate_edge_index.shape != (2, E):
            raise ValueError("candidate_edge_index must have shape [2,E]")
        if self.candidate_edge_index.dtype != torch.long:
            raise ValueError("candidate_edge_index must have dtype long")
        if self.candidate_edge_ids.shape != (E, 2) or self.candidate_edge_ids.dtype != torch.long:
            raise ValueError("candidate_edge_ids must have shape [E,2] and dtype long")
        if E:
            src, dst = self.candidate_edge_index
            local_u, local_s = self.candidate_edge_ids.unbind(dim=1)
            if (
                int(src.min()) < 0
                or int(src.max()) >= K
                or int(dst.min()) < K
                or int(dst.max()) >= K + S
            ):
                raise ValueError("candidate edges must run from user nodes to satellite nodes")
            if not torch.equal(src, local_u) or not torch.equal(dst - K, local_s):
                raise ValueError("candidate_edge_ids do not match candidate_edge_index")
        if self.edge_features.ndim != 2 or self.edge_features.size(0) != E:
            raise ValueError("edge_features must have shape [E,F]")
        _finite("edge_features", self.edge_features)
        _vector("elevation_deg", self.elevation_deg, E)
        _finite("elevation_deg", self.elevation_deg)
        _vector("current_serving", self.current_serving, K)
        _vector("hold_steps", self.hold_steps, K)
        _vector("user_order", self.user_order, K)
        if self.current_serving.dtype != torch.long or self.hold_steps.dtype != torch.long:
            raise ValueError("current_serving and hold_steps must have dtype long")
        if self.user_order.dtype != torch.long:
            raise ValueError("user_order must have dtype long")
        if torch.any(self.current_serving < -1) or torch.any(self.current_serving >= S):
            raise ValueError("current_serving contains an invalid local satellite id")
        if torch.any(self.hold_steps < 0):
            raise ValueError("hold_steps must be non-negative")
        if not torch.equal(torch.sort(self.user_order).values, torch.arange(K, device=self.user_order.device)):
            raise ValueError("user_order must be a permutation of [0,K)")
        self.sim_descriptors.validate(E, S)
        self.policy_descriptors.validate(E, S)
        initialized_edge = self.meta.get("policy_initialized_edge")
        if initialized_edge is not None:
            _vector("meta.policy_initialized_edge", initialized_edge, E)
            if initialized_edge.dtype != torch.bool:
                raise ValueError("meta.policy_initialized_edge must have dtype bool")
            if initialized_edge.device != self.candidate_edge_index.device:
                raise ValueError("meta.policy_initialized_edge must share the graph device")
    def with_policy_descriptors(self, descriptors: PolicyDescriptors) -> "ControlObservation":
        descriptors.validate(self.edge_count, self.satellite_count)
        observation = ControlObservation(
            observation_id=self.observation_id,
            node_x=self.node_x.clone(),
            candidate_edge_index=self.candidate_edge_index.clone(),
            candidate_edge_ids=self.candidate_edge_ids.clone(),
            edge_features=self.edge_features.clone(),
            elevation_deg=self.elevation_deg.clone(),
            sim_descriptors=self.sim_descriptors.clone(),
            policy_descriptors=descriptors.clone(),
            current_serving=self.current_serving.clone(),
            hold_steps=self.hold_steps.clone(),
            user_order=self.user_order.clone(),
            meta={
                name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in self.meta.items()
            },
        )
        observation.validate()
        return observation

    def as_model_step(self) -> Dict[str, Any]:
        """Materialize a TGN graph from the current policy-input copy.

        Paper-protocol observations carry immutable geometry/history plus two
        descriptor channels. For those observations, this method writes the
        policy-facing gamma, flow, and intensity fields into fresh feature
        tensors. It never changes simulator descriptors and never exposes a
        supervision target. Generic observations without the paper feature
        metadata retain their stored feature tensors unchanged.
        """
        node_x = self.node_x.clone()
        edge_z = self.edge_features.clone()
        if self.meta.get("feature_contract_version") == PAPER_FEATURE_CONTRACT_VERSION:
            node_flow_column = int(self.meta["node_flow_column"])
            edge_gamma_column = int(self.meta["edge_gamma_column"])
            edge_flow_column = int(self.meta["edge_flow_column"])
            edge_intensity_column = int(self.meta["edge_intensity_column"])
            gamma_scale = float(self.meta["gamma_feature_scale"])
            if gamma_scale <= 0:
                raise ValueError("gamma_feature_scale must be positive")
            if node_x.size(1) != len(PAPER_NODE_FEATURE_NAMES):
                raise ValueError("node feature width does not match the paper contract")
            if edge_z.size(1) != len(PAPER_EDGE_FEATURE_NAMES):
                raise ValueError("edge feature width does not match the paper contract")
            node_x[self.user_count :, node_flow_column] = self.policy_descriptors.flow_node
            if self.edge_count:
                destination = self.candidate_edge_ids[:, 1]
                edge_z[:, edge_gamma_column] = self.policy_descriptors.gamma_edge / gamma_scale
                edge_z[:, edge_flow_column] = self.policy_descriptors.flow_node[destination]
                edge_z[:, edge_intensity_column] = torch.log1p(
                    self.policy_descriptors.intensity_edge
                )
        return {
            "t": self.epoch,
            "node_x": node_x,
            "edge_index": self.candidate_edge_index,
            "edge_z": edge_z,
            "edge_type": torch.zeros(
                self.edge_count,
                dtype=torch.long,
                device=self.candidate_edge_index.device,
            ),
            "meta": {
                **self.meta,
                "K_users": self.user_count,
                "S_sats": self.satellite_count,
                "candidate_edge_ids": self.candidate_edge_ids,
                "observation_id": self.observation_id,
            },
        }


@dataclass
class ServingAction:
    observation_id: ObservationId
    requested_serving: torch.Tensor

    def validate(self, user_count: int, satellite_count: int) -> None:
        _vector("requested_serving", self.requested_serving, user_count)
        if self.requested_serving.dtype != torch.long:
            raise ValueError("requested_serving must have dtype long")
        if torch.any(self.requested_serving < -1) or torch.any(
            self.requested_serving >= satellite_count
        ):
            raise ValueError("requested_serving contains an invalid local satellite id")


class FailureReason(IntEnum):
    NONE = 0
    ABSTAIN = 1
    NOT_CANDIDATE = 2
    INFEASIBLE = 3
    CAPACITY = 4


@dataclass
class ExecutionResult:
    observation_id: ObservationId
    requested_serving: torch.Tensor
    executed_serving: torch.Tensor
    admitted: torch.Tensor
    failure_reason: torch.Tensor
    handover_attempted: torch.Tensor
    handover_executed: torch.Tensor
    flow_before: torch.Tensor
    flow_after: torch.Tensor

    def validate(self, user_count: int, satellite_count: int) -> None:
        for name, value in (
            ("requested_serving", self.requested_serving),
            ("executed_serving", self.executed_serving),
            ("admitted", self.admitted),
            ("failure_reason", self.failure_reason),
            ("handover_attempted", self.handover_attempted),
            ("handover_executed", self.handover_executed),
        ):
            _vector(name, value, user_count)
        if self.requested_serving.dtype != torch.long or self.executed_serving.dtype != torch.long:
            raise ValueError("serving tensors must have dtype long")
        if self.failure_reason.dtype != torch.long:
            raise ValueError("failure_reason must have dtype long")
        for name, value in (
            ("admitted", self.admitted),
            ("handover_attempted", self.handover_attempted),
            ("handover_executed", self.handover_executed),
        ):
            if value.dtype != torch.bool:
                raise ValueError(f"{name} must have dtype bool")
        _vector("flow_before", self.flow_before, satellite_count)
        _vector("flow_after", self.flow_after, satellite_count)
        _finite("flow_before", self.flow_before)
        _finite("flow_after", self.flow_after)


def clone_tensor_mapping(values: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clone a tensor mapping when recording trace data without storage aliasing."""
    return {name: value.clone() for name, value in values.items()}
