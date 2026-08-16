"""Typed graph and next-step target contract for the UAV/shared platform.

The allocation graph is a bipartite UAV--station graph. Candidate identities
are local physical identities, not row positions: ``stable_candidate_id`` is
always ``uav_id * station_count + station_id``.  This makes the ``t -> t+1``
join well defined even when Top-3 reachability changes the candidate rows.

Only the policy-facing eta, intensity and station-flow streams are written into
model features.  ``hard_feasible_start_edge`` is carried next to the graph for
the simulator-authoritative controller mask; it is deliberately absent from
``edge_attr`` and therefore cannot leak into the learned descriptor predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .protocol import UAVSharedProtocol
from .state import ServiceObservation, UAVPhase


UAV_GRAPH_CONTRACT_VERSION = 1
UAV_TARGET_CONTRACT_VERSION = 1

UAV_UAV_FEATURE_NAMES = (
    "position_x_normalized",
    "position_y_normalized",
    "energy_fraction",
    "phase_idle",
    "phase_travelling",
    "phase_queued",
    "phase_in_service",
    "phase_depleted",
    "has_current_station",
)

UAV_STATION_FEATURE_NAMES = (
    "position_x_normalized",
    "position_y_normalized",
    "policy_station_flow",
    "current_target_share",
    "queued_share",
    "in_service_share",
)

UAV_EDGE_FEATURE_NAMES = (
    "policy_eta",
    "policy_log1p_intensity",
    "policy_destination_station_flow",
    "distance_normalized",
    "candidate_rank_normalized",
    "is_current_station",
)


def stable_candidate_ids(
    candidate_edge_ids: torch.Tensor,
    station_count: int,
) -> torch.Tensor:
    """Return collision-free physical ids for local ``(UAV, station)`` pairs."""

    if type(station_count) is not int or station_count <= 0:
        raise ValueError("station_count must be a positive integer")
    if (
        not isinstance(candidate_edge_ids, torch.Tensor)
        or candidate_edge_ids.ndim != 2
        or candidate_edge_ids.size(1) != 2
        or candidate_edge_ids.dtype != torch.long
    ):
        raise ValueError("candidate_edge_ids must have shape [E,2] and dtype long")
    if candidate_edge_ids.numel() == 0:
        return torch.empty(
            candidate_edge_ids.size(0),
            dtype=torch.long,
            device=candidate_edge_ids.device,
        )
    if bool((candidate_edge_ids < 0).any()):
        raise ValueError("candidate_edge_ids cannot contain negative ids")
    result = candidate_edge_ids[:, 0] * station_count + candidate_edge_ids[:, 1]
    if torch.unique(result).numel() != result.numel():
        raise ValueError("candidate_edge_ids contain duplicate physical edges")
    return result


@dataclass(frozen=True)
class UAVTarget:
    """Next-epoch descriptor targets aligned to candidate rows at epoch ``t``."""

    eta_edge: torch.Tensor
    log1p_intensity_edge: torch.Tensor
    feasibility_edge: torch.Tensor
    persistent_edge: torch.Tensor
    station_flow_node: torch.Tensor
    source_stable_candidate_id: torch.Tensor

    def validate(self, edge_count: int, station_count: int) -> None:
        if type(edge_count) is not int or edge_count < 0:
            raise ValueError("edge_count must be a non-negative integer")
        if type(station_count) is not int or station_count <= 0:
            raise ValueError("station_count must be a positive integer")
        for name, value in (
            ("eta_edge", self.eta_edge),
            ("log1p_intensity_edge", self.log1p_intensity_edge),
            ("feasibility_edge", self.feasibility_edge),
            ("persistent_edge", self.persistent_edge),
            ("source_stable_candidate_id", self.source_stable_candidate_id),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != (edge_count,):
                raise ValueError(f"{name} must have shape [{edge_count}]")
        if self.station_flow_node.shape != (station_count,):
            raise ValueError(
                f"station_flow_node must have shape [{station_count}]"
            )
        if self.feasibility_edge.dtype != torch.bool:
            raise ValueError("feasibility_edge must have dtype bool")
        if self.persistent_edge.dtype != torch.bool:
            raise ValueError("persistent_edge must have dtype bool")
        if self.source_stable_candidate_id.dtype != torch.long:
            raise ValueError("source_stable_candidate_id must have dtype long")
        for name, value in (
            ("eta_edge", self.eta_edge),
            ("log1p_intensity_edge", self.log1p_intensity_edge),
            ("station_flow_node", self.station_flow_node),
        ):
            if not value.dtype.is_floating_point:
                raise ValueError(f"{name} must have floating dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        if bool((self.eta_edge < 0.0).any()) or bool((self.eta_edge > 1.0).any()):
            raise ValueError("eta_edge must lie in [0,1]")
        if bool((self.log1p_intensity_edge < 0.0).any()):
            raise ValueError("log1p_intensity_edge must be non-negative")
        if bool((self.station_flow_node < 0.0).any()) or bool(
            (self.station_flow_node > 1.0).any()
        ):
            raise ValueError("station_flow_node must lie in [0,1]")
        tensors = (
            self.eta_edge,
            self.log1p_intensity_edge,
            self.feasibility_edge,
            self.persistent_edge,
            self.station_flow_node,
            self.source_stable_candidate_id,
        )
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all UAVTarget tensors must share one device")
        if torch.unique(self.source_stable_candidate_id).numel() != edge_count:
            raise ValueError("source_stable_candidate_id must be unique")

    def as_dict(self) -> dict[str, torch.Tensor | int]:
        return {
            "contract_version": UAV_TARGET_CONTRACT_VERSION,
            "eta_edge": self.eta_edge,
            "log1p_intensity_edge": self.log1p_intensity_edge,
            "feasibility_edge": self.feasibility_edge,
            "persistent_edge": self.persistent_edge,
            "station_flow_node": self.station_flow_node,
            "source_stable_candidate_id": self.source_stable_candidate_id,
        }

    def to(self, device: torch.device | str) -> "UAVTarget":
        return UAVTarget(
            eta_edge=self.eta_edge.to(device),
            log1p_intensity_edge=self.log1p_intensity_edge.to(device),
            feasibility_edge=self.feasibility_edge.to(device),
            persistent_edge=self.persistent_edge.to(device),
            station_flow_node=self.station_flow_node.to(device),
            source_stable_candidate_id=self.source_stable_candidate_id.to(device),
        )


def _one_hot_phase(phase: torch.Tensor) -> torch.Tensor:
    classes = len(UAVPhase)
    return torch.nn.functional.one_hot(phase, num_classes=classes).to(
        dtype=torch.float32
    )


def observation_to_graph(
    observation: ServiceObservation,
    protocol: UAVSharedProtocol,
) -> dict[str, Any]:
    """Materialize the descriptor-isolated bipartite model input.

    ``edge_index`` uses local node ids: row zero indexes ``uav_x`` and row one
    indexes ``station_x``.  Consumers that require one homogeneous graph can
    offset the second row by ``uav_count`` without changing candidate identity.
    """

    if not isinstance(observation, ServiceObservation):
        raise TypeError("observation must be a ServiceObservation")
    if not isinstance(protocol, UAVSharedProtocol):
        raise TypeError("protocol must be a UAVSharedProtocol")
    observation.validate()
    if observation.episode_seed != protocol.episode_seed:
        raise ValueError("observation seed does not match the protocol")
    if observation.user_count != protocol.uav_count:
        raise ValueError("observation UAV count does not match the protocol")
    if observation.station_count != protocol.station_count:
        raise ValueError("observation station count does not match the protocol")

    device = observation.uav_position.device
    dtype = observation.uav_position.dtype
    users = observation.user_count
    stations = observation.station_count
    edges = observation.edge_count
    exact = protocol.exact
    estimated = protocol.estimated
    policy = observation.policy_descriptors
    sim = observation.sim_descriptors
    edge_ids = observation.candidate_edge_ids
    stable_ids = stable_candidate_ids(edge_ids, stations)

    phase_features = _one_hot_phase(observation.phase).to(device=device, dtype=dtype)
    uav_x = torch.cat(
        (
            (observation.uav_position[:, 0:1] / estimated.region_width_m).clamp(0, 1),
            (observation.uav_position[:, 1:2] / estimated.region_height_m).clamp(0, 1),
            observation.energy_fraction[:, None].to(dtype=dtype),
            phase_features,
            (observation.current_station >= 0).to(dtype=dtype)[:, None],
        ),
        dim=1,
    )

    current_target_count = torch.zeros(stations, dtype=dtype, device=device)
    valid_target = observation.current_station >= 0
    if bool(valid_target.any()):
        current_target_count.scatter_add_(
            0,
            observation.current_station[valid_target],
            torch.ones(int(valid_target.sum().item()), dtype=dtype, device=device),
        )
    queued_count = torch.zeros_like(current_target_count)
    queued = valid_target & (observation.phase == int(UAVPhase.QUEUED))
    if bool(queued.any()):
        queued_count.scatter_add_(
            0,
            observation.current_station[queued],
            torch.ones(int(queued.sum().item()), dtype=dtype, device=device),
        )
    service_count = torch.zeros_like(current_target_count)
    in_service = valid_target & (observation.phase == int(UAVPhase.IN_SERVICE))
    if bool(in_service.any()):
        service_count.scatter_add_(
            0,
            observation.current_station[in_service],
            torch.ones(int(in_service.sum().item()), dtype=dtype, device=device),
        )
    station_x = torch.stack(
        (
            (observation.station_position[:, 0] / estimated.region_width_m).clamp(0, 1),
            (observation.station_position[:, 1] / estimated.region_height_m).clamp(0, 1),
            policy.station_flow_node.to(dtype=dtype),
            (current_target_count / max(users, 1)).clamp(0, 1),
            (queued_count / exact.fifo_queue_capacity).clamp(0, 1),
            (service_count / exact.active_slots_per_station).clamp(0, 1),
        ),
        dim=1,
    )

    if edges:
        edge_users = edge_ids[:, 0]
        edge_stations = edge_ids[:, 1]
        distance = torch.linalg.vector_norm(
            observation.station_position[edge_stations]
            - observation.uav_position[edge_users],
            dim=1,
        )
        rank_denominator = max(exact.candidate_topk - 1, 1)
        edge_attr = torch.stack(
            (
                policy.eta_edge.to(dtype=dtype),
                torch.log1p(policy.intensity_edge).to(dtype=dtype),
                policy.station_flow_node[edge_stations].to(dtype=dtype),
                (distance / estimated.maximum_reachable_distance_m).clamp(0, 1),
                (observation.candidate_rank.to(dtype=dtype) / rank_denominator).clamp(0, 1),
                (observation.current_station[edge_users] == edge_stations).to(dtype=dtype),
            ),
            dim=1,
        )
    else:
        edge_attr = torch.empty(
            (0, len(UAV_EDGE_FEATURE_NAMES)), dtype=dtype, device=device
        )

    result: dict[str, Any] = {
        "graph_contract_version": UAV_GRAPH_CONTRACT_VERSION,
        "observation_id": [observation.episode_seed, observation.epoch],
        "t": observation.epoch,
        "uav_count": users,
        "station_count": stations,
        "uav_x": uav_x,
        "station_x": station_x,
        "edge_index": edge_ids.transpose(0, 1).contiguous(),
        "edge_attr": edge_attr,
        "candidate_edge_ids": edge_ids.clone(),
        "stable_candidate_id": stable_ids,
        "candidate_rank": observation.candidate_rank.clone(),
        "hard_feasible_start_edge": sim.feasible_start_edge.clone(),
        "current_station": observation.current_station.clone(),
        "user_order": observation.user_order.clone(),
        "policy_descriptors": {
            "eta_edge": policy.eta_edge.clone(),
            "intensity_edge": policy.intensity_edge.clone(),
            "station_flow_node": policy.station_flow_node.clone(),
        },
        "sim_descriptors": {
            "eta_edge": sim.policy_fields.eta_edge.clone(),
            "intensity_edge": sim.policy_fields.intensity_edge.clone(),
            "station_flow_node": sim.policy_fields.station_flow_node.clone(),
            "reachable_edge": sim.reachable_edge.clone(),
            "feasible_start_edge": sim.feasible_start_edge.clone(),
            "service_start_eta_s": sim.service_start_eta_s.clone(),
            "arrival_energy_fraction": sim.arrival_energy_fraction.clone(),
        },
        "meta": {
            "uav_feature_names": list(UAV_UAV_FEATURE_NAMES),
            "station_feature_names": list(UAV_STATION_FEATURE_NAMES),
            "edge_feature_names": list(UAV_EDGE_FEATURE_NAMES),
            "edge_index_semantics": "local_uav_to_local_station",
            "stable_candidate_id_semantics": "uav_id_times_station_count_plus_station_id",
            "hard_feasibility_role": "simulator_authoritative_controller_mask_only",
            "protocol_fingerprint": protocol.fingerprint,
        },
    }
    validate_graph(result)
    return result


def validate_graph(graph: Mapping[str, Any]) -> None:
    """Validate a materialized graph without importing serialized dataclasses."""

    if not isinstance(graph, Mapping):
        raise TypeError("UAV graph must be a mapping")
    if graph.get("graph_contract_version") != UAV_GRAPH_CONTRACT_VERSION:
        raise ValueError("UAV graph contract version mismatch")
    users = graph.get("uav_count")
    stations = graph.get("station_count")
    if type(users) is not int or users <= 0:
        raise ValueError("uav_count must be a positive integer")
    if type(stations) is not int or stations <= 0:
        raise ValueError("station_count must be a positive integer")
    uav_x = graph.get("uav_x")
    station_x = graph.get("station_x")
    edge_index = graph.get("edge_index")
    edge_attr = graph.get("edge_attr")
    candidate_ids = graph.get("candidate_edge_ids")
    stable_ids = graph.get("stable_candidate_id")
    if not isinstance(uav_x, torch.Tensor) or uav_x.shape != (
        users,
        len(UAV_UAV_FEATURE_NAMES),
    ):
        raise ValueError("uav_x does not match the UAV feature contract")
    if not isinstance(station_x, torch.Tensor) or station_x.shape != (
        stations,
        len(UAV_STATION_FEATURE_NAMES),
    ):
        raise ValueError("station_x does not match the station feature contract")
    if not isinstance(candidate_ids, torch.Tensor) or (
        candidate_ids.ndim != 2 or candidate_ids.size(1) != 2
    ):
        raise ValueError("candidate_edge_ids must have shape [E,2]")
    edge_count = candidate_ids.size(0)
    if candidate_ids.dtype != torch.long:
        raise ValueError("candidate_edge_ids must have dtype long")
    if edge_count:
        edge_users, edge_stations = candidate_ids.unbind(dim=1)
        if int(edge_users.min()) < 0 or int(edge_users.max()) >= users:
            raise ValueError("candidate_edge_ids contain an out-of-range UAV id")
        if int(edge_stations.min()) < 0 or int(edge_stations.max()) >= stations:
            raise ValueError("candidate_edge_ids contain an out-of-range station id")
    if not isinstance(edge_index, torch.Tensor) or edge_index.shape != (2, edge_count):
        raise ValueError("edge_index must have shape [2,E]")
    if edge_index.dtype != torch.long or not torch.equal(
        edge_index, candidate_ids.transpose(0, 1)
    ):
        raise ValueError("edge_index must be the transposed local candidate ids")
    if not isinstance(edge_attr, torch.Tensor) or edge_attr.shape != (
        edge_count,
        len(UAV_EDGE_FEATURE_NAMES),
    ):
        raise ValueError("edge_attr does not match the edge feature contract")
    expected_stable = stable_candidate_ids(candidate_ids, stations)
    if not isinstance(stable_ids, torch.Tensor) or not torch.equal(
        stable_ids, expected_stable
    ):
        raise ValueError("stable_candidate_id does not match physical candidate ids")
    for name, shape, dtype in (
        ("candidate_rank", (edge_count,), torch.long),
        ("hard_feasible_start_edge", (edge_count,), torch.bool),
        ("current_station", (users,), torch.long),
        ("user_order", (users,), torch.long),
    ):
        value = graph.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    expected_order = torch.arange(users, device=graph["user_order"].device)
    if not torch.equal(torch.sort(graph["user_order"]).values, expected_order):
        raise ValueError("user_order must be a permutation of [0,U)")
    for name, value in (
        ("uav_x", uav_x),
        ("station_x", station_x),
        ("edge_attr", edge_attr),
    ):
        if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite floating tensor")


def build_next_step_target(
    source_observation: ServiceObservation,
    next_observation: ServiceObservation,
) -> UAVTarget:
    """Join simulator descriptors at ``t+1`` onto physical edges from ``t``."""

    source_observation.validate()
    next_observation.validate()
    if source_observation.episode_seed != next_observation.episode_seed:
        raise ValueError("source and next observations must belong to one episode")
    if next_observation.epoch != source_observation.epoch + 1:
        raise ValueError("target construction requires consecutive observations")
    if source_observation.user_count != next_observation.user_count:
        raise ValueError("UAV identities must remain stable across a target")
    if source_observation.station_count != next_observation.station_count:
        raise ValueError("station identities must remain stable across a target")

    stations = source_observation.station_count
    source_ids = stable_candidate_ids(
        source_observation.candidate_edge_ids, stations
    )
    next_ids = stable_candidate_ids(next_observation.candidate_edge_ids, stations)
    source_fields = source_observation.sim_descriptors.policy_fields
    next_fields = next_observation.sim_descriptors.policy_fields
    device = source_ids.device
    source_float = source_fields.eta_edge
    tensors = (
        source_ids,
        next_ids,
        source_fields.eta_edge,
        source_fields.intensity_edge,
        source_fields.station_flow_node,
        next_fields.eta_edge,
        next_fields.intensity_edge,
        next_fields.station_flow_node,
        next_observation.sim_descriptors.feasible_start_edge,
    )
    if len({value.device for value in tensors}) != 1:
        raise ValueError("source and next target tensors must share one device")

    edge_count = source_ids.numel()
    eta = torch.zeros(edge_count, dtype=source_float.dtype, device=device)
    log_intensity = torch.zeros_like(eta)
    feasibility = torch.zeros(edge_count, dtype=torch.bool, device=device)
    persistent = torch.zeros_like(feasibility)
    next_lookup = {int(value): index for index, value in enumerate(next_ids.tolist())}
    for source_index, stable_id in enumerate(source_ids.tolist()):
        next_index = next_lookup.get(int(stable_id))
        if next_index is None:
            continue
        persistent[source_index] = True
        eta[source_index] = next_fields.eta_edge[next_index]
        log_intensity[source_index] = torch.log1p(
            next_fields.intensity_edge[next_index]
        )
        feasibility[source_index] = (
            next_observation.sim_descriptors.feasible_start_edge[next_index]
        )
    target = UAVTarget(
        eta_edge=eta,
        log1p_intensity_edge=log_intensity,
        feasibility_edge=feasibility,
        persistent_edge=persistent,
        station_flow_node=next_fields.station_flow_node.clone(),
        source_stable_candidate_id=source_ids.clone(),
    )
    target.validate(edge_count, stations)
    return target


def target_from_mapping(value: Mapping[str, Any]) -> UAVTarget:
    """Rehydrate a typed target from a ``weights_only=True`` payload mapping."""

    if not isinstance(value, Mapping):
        raise TypeError("target payload must be a mapping")
    if value.get("contract_version") != UAV_TARGET_CONTRACT_VERSION:
        raise ValueError("UAV target contract version mismatch")
    target = UAVTarget(
        eta_edge=value["eta_edge"],
        log1p_intensity_edge=value["log1p_intensity_edge"],
        feasibility_edge=value["feasibility_edge"],
        persistent_edge=value["persistent_edge"],
        station_flow_node=value["station_flow_node"],
        source_stable_candidate_id=value["source_stable_candidate_id"],
    )
    target.validate(target.eta_edge.numel(), target.station_flow_node.numel())
    return target


__all__ = [
    "UAV_EDGE_FEATURE_NAMES",
    "UAV_GRAPH_CONTRACT_VERSION",
    "UAV_STATION_FEATURE_NAMES",
    "UAV_TARGET_CONTRACT_VERSION",
    "UAV_UAV_FEATURE_NAMES",
    "UAVTarget",
    "build_next_step_target",
    "observation_to_graph",
    "stable_candidate_ids",
    "target_from_mapping",
    "validate_graph",
]
