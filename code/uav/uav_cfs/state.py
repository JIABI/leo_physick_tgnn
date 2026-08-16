from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, TYPE_CHECKING

import torch

from .protocol import UAV_SHARED_SNAPSHOT_SCHEMA_VERSION

if TYPE_CHECKING:
    from .protocol import UAVSharedProtocol


ObservationId = tuple[int, int]


def _finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")


def _bool_vector(name: str, value: torch.Tensor, length: int) -> None:
    _shape(name, value, (length,))
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must have dtype bool")


def _long_vector(name: str, value: torch.Tensor, length: int) -> None:
    _shape(name, value, (length,))
    if value.dtype != torch.long:
        raise ValueError(f"{name} must have dtype long")


def _float_vector(name: str, value: torch.Tensor, length: int) -> None:
    _float_tensor(name, value, (length,))


def _float_tensor(
    name: str,
    value: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    _shape(name, value, expected)
    if not value.dtype.is_floating_point:
        raise ValueError(f"{name} must have floating dtype")
    _finite(name, value)


class UAVPhase(IntEnum):
    IDLE = 0
    TRAVELLING = 1
    QUEUED = 2
    IN_SERVICE = 3
    DEPLETED = 4


class ServiceFailureReason(IntEnum):
    NONE = 0
    NOT_CANDIDATE = 1
    ENERGY_DEPLETED = 2
    RESERVE_VIOLATION = 3
    QUEUE_FULL = 4


@dataclass
class ServicePolicyDescriptors:
    """The only three streams visible to the reassociation policy."""

    eta_edge: torch.Tensor
    intensity_edge: torch.Tensor
    station_flow_node: torch.Tensor

    def validate(self, edge_count: int, station_count: int) -> None:
        _float_vector("eta_edge", self.eta_edge, edge_count)
        _float_vector("intensity_edge", self.intensity_edge, edge_count)
        _float_vector("station_flow_node", self.station_flow_node, station_count)
        if torch.any((self.eta_edge < 0.0) | (self.eta_edge > 1.0)):
            raise ValueError("eta_edge must be in [0,1]")
        if torch.any((self.intensity_edge < 0.0) | (self.intensity_edge > 3.0)):
            raise ValueError("intensity_edge must be in [0,3]")
        if torch.any(
            (self.station_flow_node < 0.0) | (self.station_flow_node > 1.0)
        ):
            raise ValueError("station_flow_node must be in [0,1]")

    def clone(self) -> "ServicePolicyDescriptors":
        return ServicePolicyDescriptors(
            eta_edge=self.eta_edge.clone(),
            intensity_edge=self.intensity_edge.clone(),
            station_flow_node=self.station_flow_node.clone(),
        )

    def to(self, device: torch.device | str) -> "ServicePolicyDescriptors":
        return ServicePolicyDescriptors(
            eta_edge=self.eta_edge.to(device),
            intensity_edge=self.intensity_edge.to(device),
            station_flow_node=self.station_flow_node.to(device),
        )

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "eta_edge": self.eta_edge,
            "intensity_edge": self.intensity_edge,
            "station_flow_node": self.station_flow_node,
        }


@dataclass
class ServiceSimulatorDescriptors:
    """Simulator-authoritative copy and current feasible-start predicate."""

    policy_fields: ServicePolicyDescriptors
    reachable_edge: torch.Tensor
    feasible_start_edge: torch.Tensor
    service_start_eta_s: torch.Tensor
    arrival_energy_fraction: torch.Tensor

    def validate(self, edge_count: int, station_count: int) -> None:
        self.policy_fields.validate(edge_count, station_count)
        _bool_vector("reachable_edge", self.reachable_edge, edge_count)
        _bool_vector("feasible_start_edge", self.feasible_start_edge, edge_count)
        _float_vector("service_start_eta_s", self.service_start_eta_s, edge_count)
        _float_vector(
            "arrival_energy_fraction", self.arrival_energy_fraction, edge_count
        )
        if torch.any(~self.reachable_edge):
            raise ValueError("candidate edges must already satisfy reachability")
        if torch.any(self.service_start_eta_s < 0.0):
            raise ValueError("service_start_eta_s must be non-negative")
        if torch.any(
            (self.arrival_energy_fraction < 0.0)
            | (self.arrival_energy_fraction > 1.0)
        ):
            raise ValueError("arrival_energy_fraction must be in [0,1]")

    def clone(self) -> "ServiceSimulatorDescriptors":
        return ServiceSimulatorDescriptors(
            policy_fields=self.policy_fields.clone(),
            reachable_edge=self.reachable_edge.clone(),
            feasible_start_edge=self.feasible_start_edge.clone(),
            service_start_eta_s=self.service_start_eta_s.clone(),
            arrival_energy_fraction=self.arrival_energy_fraction.clone(),
        )


@dataclass
class ServiceObservation:
    """Decision-epoch observation with separated policy/simulator channels."""

    observation_id: ObservationId
    uav_position: torch.Tensor
    station_position: torch.Tensor
    energy_fraction: torch.Tensor
    phase: torch.Tensor
    current_station: torch.Tensor
    candidate_edge_ids: torch.Tensor
    candidate_rank: torch.Tensor
    sim_descriptors: ServiceSimulatorDescriptors
    policy_descriptors: ServicePolicyDescriptors
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
        return int(self.uav_position.size(0))

    @property
    def station_count(self) -> int:
        return int(self.station_position.size(0))

    @property
    def edge_count(self) -> int:
        return int(self.candidate_edge_ids.size(0))

    def validate(self) -> None:
        if (
            not isinstance(self.observation_id, tuple)
            or len(self.observation_id) != 2
            or self.episode_seed < 0
            or self.epoch < 0
        ):
            raise ValueError("observation_id must be a non-negative (seed, epoch) tuple")
        users = self.user_count
        stations = self.station_count
        edges = self.edge_count
        _float_tensor("uav_position", self.uav_position, (users, 2))
        _float_tensor("station_position", self.station_position, (stations, 2))
        _float_vector("energy_fraction", self.energy_fraction, users)
        if torch.any((self.energy_fraction < 0.0) | (self.energy_fraction > 1.0)):
            raise ValueError("energy_fraction must be in [0,1]")
        _long_vector("phase", self.phase, users)
        _long_vector("current_station", self.current_station, users)
        if torch.any(self.phase < int(UAVPhase.IDLE)) or torch.any(
            self.phase > int(UAVPhase.DEPLETED)
        ):
            raise ValueError("phase contains an invalid UAVPhase value")
        if torch.any(self.current_station < -1) or torch.any(
            self.current_station >= stations
        ):
            raise ValueError("current_station contains an invalid station id")
        if self.candidate_edge_ids.shape != (edges, 2):
            raise ValueError("candidate_edge_ids must have shape [E,2]")
        if self.candidate_edge_ids.dtype != torch.long:
            raise ValueError("candidate_edge_ids must have dtype long")
        _long_vector("candidate_rank", self.candidate_rank, edges)
        if edges:
            uav, station = self.candidate_edge_ids.unbind(dim=1)
            if int(uav.min()) < 0 or int(uav.max()) >= users:
                raise ValueError("candidate edges contain an invalid UAV id")
            if int(station.min()) < 0 or int(station.max()) >= stations:
                raise ValueError("candidate edges contain an invalid station id")
            if torch.any(self.candidate_rank < 0):
                raise ValueError("candidate_rank must be non-negative")
        _long_vector("user_order", self.user_order, users)
        if not torch.equal(
            torch.sort(self.user_order).values,
            torch.arange(users, device=self.user_order.device),
        ):
            raise ValueError("user_order must be a permutation of [0,U)")
        self.sim_descriptors.validate(edges, stations)
        self.policy_descriptors.validate(edges, stations)

    def clone(self) -> "ServiceObservation":
        return ServiceObservation(
            observation_id=self.observation_id,
            uav_position=self.uav_position.clone(),
            station_position=self.station_position.clone(),
            energy_fraction=self.energy_fraction.clone(),
            phase=self.phase.clone(),
            current_station=self.current_station.clone(),
            candidate_edge_ids=self.candidate_edge_ids.clone(),
            candidate_rank=self.candidate_rank.clone(),
            sim_descriptors=self.sim_descriptors.clone(),
            policy_descriptors=self.policy_descriptors.clone(),
            user_order=self.user_order.clone(),
            meta={
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in self.meta.items()
            },
        )

    def with_policy_descriptors(
        self, descriptors: ServicePolicyDescriptors
    ) -> "ServiceObservation":
        descriptors.validate(self.edge_count, self.station_count)
        observation = self.clone()
        observation.policy_descriptors = descriptors.clone()
        observation.validate()
        return observation


@dataclass
class ReassociationAction:
    """Requested station per UAV; ``-1`` explicitly means keep current state."""

    observation_id: ObservationId
    requested_station: torch.Tensor

    def validate(self, user_count: int, station_count: int) -> None:
        _long_vector("requested_station", self.requested_station, user_count)
        if torch.any(self.requested_station < -1) or torch.any(
            self.requested_station >= station_count
        ):
            raise ValueError("requested_station contains an invalid station id")


@dataclass
class UAVSharedState:
    episode_seed: int
    epoch: int
    uav_position: torch.Tensor
    heading_rad: torch.Tensor
    energy_units: torch.Tensor
    target_station: torch.Tensor
    phase: torch.Tensor
    station_position: torch.Tensor
    active_uav: torch.Tensor
    service_remaining_s: torch.Tensor
    fifo_queues: tuple[tuple[int, ...], ...]
    station_flow: torch.Tensor
    service_visit_count: torch.Tensor
    missions_completed: torch.Tensor
    failed_service_starts: torch.Tensor
    reassociations: torch.Tensor

    def validate(self, protocol: "UAVSharedProtocol") -> None:
        users = protocol.uav_count
        stations = protocol.station_count
        slots = protocol.exact.active_slots_per_station
        if type(self.episode_seed) is not int or type(self.epoch) is not int:
            raise TypeError("state seed and epoch must be strict integers")
        if self.episode_seed != protocol.episode_seed:
            raise ValueError(
                "state episode_seed does not match the environment protocol"
            )
        if not 0 <= self.epoch <= protocol.horizon_steps:
            raise ValueError(
                "state epoch must lie within the closed interval [0, horizon_steps]"
            )
        _float_tensor("uav_position", self.uav_position, (users, 2))
        _float_tensor("station_position", self.station_position, (stations, 2))
        _float_vector("heading_rad", self.heading_rad, users)
        _float_vector("energy_units", self.energy_units, users)
        if torch.any(self.energy_units < 0.0) or torch.any(
            self.energy_units > protocol.estimated.nominal_battery_units + 1e-6
        ):
            raise ValueError("energy_units lies outside the battery budget")
        _long_vector("target_station", self.target_station, users)
        _long_vector("phase", self.phase, users)
        if torch.any(self.target_station < -1) or torch.any(
            self.target_station >= stations
        ):
            raise ValueError("target_station contains an invalid station id")
        if torch.any(self.phase < int(UAVPhase.IDLE)) or torch.any(
            self.phase > int(UAVPhase.DEPLETED)
        ):
            raise ValueError("phase contains an invalid UAVPhase value")
        _shape("active_uav", self.active_uav, (stations, slots))
        if self.active_uav.dtype != torch.long:
            raise ValueError("active_uav must have dtype long")
        if torch.any(self.active_uav < -1) or torch.any(self.active_uav >= users):
            raise ValueError("active_uav contains an invalid UAV id")
        _float_tensor(
            "service_remaining_s",
            self.service_remaining_s,
            (stations, slots),
        )
        occupied = self.active_uav >= 0
        if torch.any(self.service_remaining_s[occupied] <= 0.0):
            raise ValueError("occupied service slots must have positive remaining time")
        if torch.any(self.service_remaining_s[~occupied] != 0.0):
            raise ValueError("empty service slots must have zero remaining time")
        if len(self.fifo_queues) != stations:
            raise ValueError("fifo_queues must contain one queue per station")
        queued_users: list[int] = []
        for station, queue in enumerate(self.fifo_queues):
            if len(queue) > protocol.exact.fifo_queue_capacity:
                raise ValueError("a FIFO queue exceeds the manuscript capacity")
            if len(queue) != len(set(queue)):
                raise ValueError("a FIFO queue contains a duplicate UAV")
            for user in queue:
                if user < 0 or user >= users:
                    raise ValueError("a FIFO queue contains an invalid UAV id")
                if int(self.target_station[user].item()) != station:
                    raise ValueError("queued UAV target does not match its queue")
            queued_users.extend(queue)
        active_users = self.active_uav[occupied].tolist()
        if len(active_users) != len(set(active_users)):
            raise ValueError("a UAV occupies more than one service slot")
        if set(active_users).intersection(queued_users):
            raise ValueError("a UAV cannot be both queued and in service")
        if len(queued_users) != len(set(queued_users)):
            raise ValueError("a UAV appears in more than one FIFO queue")
        active_set = set(active_users)
        queue_set = set(queued_users)
        for user in range(users):
            phase = UAVPhase(int(self.phase[user].item()))
            target = int(self.target_station[user].item())
            if phase is UAVPhase.IN_SERVICE:
                if user not in active_set or target < 0:
                    raise ValueError("IN_SERVICE UAV is not in its target slot")
                station_ids = torch.nonzero(
                    self.active_uav == user, as_tuple=False
                )[:, 0]
                if station_ids.numel() != 1 or int(station_ids[0].item()) != target:
                    raise ValueError("IN_SERVICE UAV target does not match its slot")
            elif phase is UAVPhase.QUEUED:
                if user not in queue_set or target < 0:
                    raise ValueError("QUEUED UAV is not in its target FIFO queue")
            elif phase is UAVPhase.TRAVELLING:
                if target < 0 or user in active_set or user in queue_set:
                    raise ValueError("TRAVELLING UAV has inconsistent resource state")
            else:
                if target != -1 or user in active_set or user in queue_set:
                    raise ValueError("IDLE/DEPLETED UAV must have no station resource")
            if phase is UAVPhase.DEPLETED and self.energy_units[user] > 1e-6:
                raise ValueError("DEPLETED UAV must have zero energy")
        _float_vector("station_flow", self.station_flow, stations)
        if torch.any((self.station_flow < 0.0) | (self.station_flow > 1.0)):
            raise ValueError("station_flow must be in [0,1]")
        for name, value in (
            ("service_visit_count", self.service_visit_count),
            ("missions_completed", self.missions_completed),
            ("failed_service_starts", self.failed_service_starts),
            ("reassociations", self.reassociations),
        ):
            _long_vector(name, value, users)
            if torch.any(value < 0):
                raise ValueError(f"{name} must be non-negative")

    def clone(self) -> "UAVSharedState":
        return UAVSharedState(
            episode_seed=self.episode_seed,
            epoch=self.epoch,
            uav_position=self.uav_position.clone(),
            heading_rad=self.heading_rad.clone(),
            energy_units=self.energy_units.clone(),
            target_station=self.target_station.clone(),
            phase=self.phase.clone(),
            station_position=self.station_position.clone(),
            active_uav=self.active_uav.clone(),
            service_remaining_s=self.service_remaining_s.clone(),
            fifo_queues=tuple(tuple(queue) for queue in self.fifo_queues),
            station_flow=self.station_flow.clone(),
            service_visit_count=self.service_visit_count.clone(),
            missions_completed=self.missions_completed.clone(),
            failed_service_starts=self.failed_service_starts.clone(),
            reassociations=self.reassociations.clone(),
        )

    def to(self, device: torch.device | str) -> "UAVSharedState":
        state = self.clone()
        for name, value in vars(state).items():
            if isinstance(value, torch.Tensor):
                setattr(state, name, value.to(device))
        return state


@dataclass
class UAVSharedSnapshot:
    protocol_fingerprint: str
    state: UAVSharedState
    schema_version: int = UAV_SHARED_SNAPSHOT_SCHEMA_VERSION

    def clone(self) -> "UAVSharedSnapshot":
        return UAVSharedSnapshot(
            protocol_fingerprint=self.protocol_fingerprint,
            state=self.state.clone(),
            schema_version=self.schema_version,
        )


@dataclass
class ServiceExecutionResult:
    observation_id: ObservationId
    requested_station: torch.Tensor
    target_before: torch.Tensor
    target_after: torch.Tensor
    assignment_attempted: torch.Tensor
    assignment_accepted: torch.Tensor
    reassociation_executed: torch.Tensor
    failure_reason: torch.Tensor
    queue_admitted: torch.Tensor
    service_started: torch.Tensor
    service_completed: torch.Tensor
    energy_depleted: torch.Tensor
    energy_before: torch.Tensor
    energy_after: torch.Tensor
    station_flow_before: torch.Tensor
    station_flow_after: torch.Tensor
    queue_length_before: torch.Tensor
    queue_length_after: torch.Tensor
    active_slots_before: torch.Tensor
    active_slots_after: torch.Tensor

    def validate(self, user_count: int, station_count: int) -> None:
        for name, value in (
            ("requested_station", self.requested_station),
            ("target_before", self.target_before),
            ("target_after", self.target_after),
            ("failure_reason", self.failure_reason),
        ):
            _long_vector(name, value, user_count)
        for name, value in (
            ("assignment_attempted", self.assignment_attempted),
            ("assignment_accepted", self.assignment_accepted),
            ("reassociation_executed", self.reassociation_executed),
            ("queue_admitted", self.queue_admitted),
            ("service_started", self.service_started),
            ("service_completed", self.service_completed),
            ("energy_depleted", self.energy_depleted),
        ):
            _bool_vector(name, value, user_count)
        _float_vector("energy_before", self.energy_before, user_count)
        _float_vector("energy_after", self.energy_after, user_count)
        _float_vector("station_flow_before", self.station_flow_before, station_count)
        _float_vector("station_flow_after", self.station_flow_after, station_count)
        for name, value in (
            ("queue_length_before", self.queue_length_before),
            ("queue_length_after", self.queue_length_after),
            ("active_slots_before", self.active_slots_before),
            ("active_slots_after", self.active_slots_after),
        ):
            _long_vector(name, value, station_count)
