from __future__ import annotations

import math
import hashlib
from typing import Iterable

import torch

from .protocol import (
    UAV_SHARED_PROTOCOL_VERSION,
    UAV_SHARED_SNAPSHOT_SCHEMA_VERSION,
    UAVSharedProtocol,
)
from .randomness import keyed_order, keyed_signed_uniform, keyed_uniform
from .perturbations import UAVStructuralPerturbation
from .state import (
    ReassociationAction,
    ServiceExecutionResult,
    ServiceFailureReason,
    ServiceObservation,
    ServicePolicyDescriptors,
    ServiceSimulatorDescriptors,
    UAVPhase,
    UAVSharedSnapshot,
    UAVSharedState,
)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class UAVSharedServiceEnv:
    """Action-coupled UAV/shared-station simulator for the paper's platform two.

    The environment maintains two descriptor channels.  ``observe`` exposes a
    simulator-authoritative descriptor copy and an initially identical policy
    copy.  Replacing the policy copy cannot mutate queues, service predicates,
    energy, or station state.  Only :meth:`step_action` changes the simulator,
    so a descriptor intervention reaches epoch ``t+1`` solely through the
    executed reassociation action at epoch ``t``.

    The transition order is fixed and fingerprinted by the package protocol:

    1. validate and execute accepted reassociation releases;
    2. advance existing service and refill freed slots from FIFO heads;
    3. advance bounded-turn travel and finite-energy dynamics;
    4. admit simultaneous arrivals in a keyed, order-independent order;
    5. update the slow station-flow carrier and advance the epoch.
    """

    def __init__(
        self,
        protocol: UAVSharedProtocol,
        device: torch.device | str = "cpu",
        perturbation: UAVStructuralPerturbation | None = None,
    ) -> None:
        if not isinstance(protocol, UAVSharedProtocol):
            raise TypeError("protocol must be a fully resolved UAVSharedProtocol")
        self.protocol = protocol
        self.perturbation = perturbation or UAVStructuralPerturbation()
        if (
            self.perturbation.kind == "station_memory_shift"
            and math.isclose(
                float(self.perturbation.station_flow_memory),
                protocol.estimated.flow_memory,
            )
        ):
            raise ValueError("station-memory shift must change the nominal memory")
        if (
            self.perturbation.kind == "threshold_tightening"
            and float(self.perturbation.feasible_start_reserve_fraction)
            <= protocol.exact.feasible_start_reserve_fraction
        ):
            raise ValueError(
                "threshold tightening must exceed the nominal reserve fraction"
            )
        self.device = torch.device(device)
        self.state: UAVSharedState | None = None
        self._pending_observation: ServiceObservation | None = None

    @property
    def protocol_fingerprint(self) -> str:
        payload = (
            self.protocol.fingerprint + ":" + self.perturbation.fingerprint
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def new_instance(self) -> "UAVSharedServiceEnv":
        """Construct an empty environment with identical evaluation semantics."""

        return type(self)(
            self.protocol,
            device=self.device,
            perturbation=self.perturbation,
        )

    def _reserve_fraction(self) -> float:
        override = self.perturbation.feasible_start_reserve_fraction
        return (
            self.protocol.exact.feasible_start_reserve_fraction
            if override is None
            else override
        )

    @property
    def done(self) -> bool:
        return self.state is not None and self.state.epoch >= self.protocol.horizon_steps

    def _seeded_points(
        self,
        count: int,
        *,
        seed: int,
        stream: str,
        margin_fraction: float,
    ) -> torch.Tensor:
        dynamics = self.protocol.estimated
        x_margin = dynamics.region_width_m * margin_fraction
        y_margin = dynamics.region_height_m * margin_fraction
        x_span = dynamics.region_width_m - 2.0 * x_margin
        y_span = dynamics.region_height_m - 2.0 * y_margin
        values = [
            (
                x_margin + x_span * keyed_uniform(seed, 0, stream, index, "x"),
                y_margin + y_span * keyed_uniform(seed, 0, stream, index, "y"),
            )
            for index in range(count)
        ]
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _initial_state(self) -> UAVSharedState:
        protocol = self.protocol
        exact = protocol.exact
        dynamics = protocol.estimated
        users = protocol.uav_count
        stations = protocol.station_count
        seed = protocol.episode_seed
        uav_position = self._seeded_points(
            users,
            seed=seed,
            stream="initial-uav-position",
            margin_fraction=0.0,
        )
        station_position = self._seeded_points(
            stations,
            seed=seed,
            stream="station-position",
            margin_fraction=dynamics.station_boundary_margin_fraction,
        )
        headings = torch.tensor(
            [
                2.0 * math.pi * keyed_uniform(seed, 0, "initial-heading", user)
                - math.pi
                for user in range(users)
            ],
            dtype=torch.float32,
            device=self.device,
        )
        energy_span = (
            dynamics.initial_energy_fraction_max
            - dynamics.initial_energy_fraction_min
        )
        energy = torch.tensor(
            [
                dynamics.nominal_battery_units
                * (
                    dynamics.initial_energy_fraction_min
                    + energy_span
                    * keyed_uniform(seed, 0, "initial-energy", user)
                )
                for user in range(users)
            ],
            dtype=torch.float32,
            device=self.device,
        )
        state = UAVSharedState(
            episode_seed=seed,
            epoch=0,
            uav_position=uav_position,
            heading_rad=headings,
            energy_units=energy,
            target_station=torch.full(
                (users,), -1, dtype=torch.long, device=self.device
            ),
            phase=torch.full(
                (users,), int(UAVPhase.IDLE), dtype=torch.long, device=self.device
            ),
            station_position=station_position,
            active_uav=torch.full(
                (stations, exact.active_slots_per_station),
                -1,
                dtype=torch.long,
                device=self.device,
            ),
            service_remaining_s=torch.zeros(
                (stations, exact.active_slots_per_station),
                dtype=torch.float32,
                device=self.device,
            ),
            fifo_queues=tuple(() for _ in range(stations)),
            station_flow=torch.zeros(
                stations, dtype=torch.float32, device=self.device
            ),
            service_visit_count=torch.zeros(
                users, dtype=torch.long, device=self.device
            ),
            missions_completed=torch.zeros(
                users, dtype=torch.long, device=self.device
            ),
            failed_service_starts=torch.zeros(
                users, dtype=torch.long, device=self.device
            ),
            reassociations=torch.zeros(
                users, dtype=torch.long, device=self.device
            ),
        )
        state.validate(protocol)
        return state

    def reset_control(self) -> ServiceObservation:
        self.state = self._initial_state()
        self._pending_observation = None
        return self.observe()

    def _require_state(self) -> UAVSharedState:
        if self.state is None:
            raise RuntimeError("call reset_control() before using the environment")
        return self.state

    def _distance_and_arrival_energy(
        self, state: UAVSharedState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dynamics = self.protocol.estimated
        displacement = (
            state.station_position.unsqueeze(0) - state.uav_position.unsqueeze(1)
        )
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        travel_time = distance / dynamics.max_speed_m_s
        travel_cost = (
            distance * dynamics.travel_energy_units_per_m
            + travel_time * dynamics.idle_energy_units_per_s
        )
        arrival_energy = state.energy_units[:, None] - travel_cost
        return distance, travel_time, arrival_energy

    def _eta_all_stations(
        self,
        distance: torch.Tensor,
        arrival_energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        exact = self.protocol.exact
        dynamics = self.protocol.estimated
        reserve = (
            self._reserve_fraction()
            * dynamics.nominal_battery_units
        )
        travel_utility = (
            1.0 - distance / dynamics.maximum_reachable_distance_m
        ).clamp(*exact.eta_clip)
        arrival_margin = (
            (arrival_energy - reserve)
            / max(dynamics.nominal_battery_units - reserve, 1e-12)
        ).clamp(*exact.eta_clip)
        eta = (
            dynamics.eta_travel_weight * travel_utility
            + dynamics.eta_arrival_energy_weight * arrival_margin
        ).clamp(*exact.eta_clip)
        return eta, arrival_margin

    def _reachable_mask(
        self,
        state: UAVSharedState,
        distance: torch.Tensor,
        arrival_energy: torch.Tensor,
    ) -> torch.Tensor:
        exact = self.protocol.exact
        dynamics = self.protocol.estimated
        reserve = (
            self._reserve_fraction()
            * dynamics.nominal_battery_units
        )
        live = state.phase != int(UAVPhase.DEPLETED)
        return (
            live[:, None]
            & (distance <= dynamics.maximum_reachable_distance_m)
            & (arrival_energy >= reserve)
        )

    def _candidate_edges(
        self,
        eta_all: torch.Tensor,
        reachable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        users, stations = reachable.shape
        edge_ids: list[tuple[int, int]] = []
        ranks: list[int] = []
        topk = self.protocol.exact.candidate_topk
        for user in range(users):
            candidates = [
                station
                for station in range(stations)
                if bool(reachable[user, station].item())
            ]
            candidates.sort(
                key=lambda station: (
                    -float(eta_all[user, station].item()),
                    station,
                )
            )
            for rank, station in enumerate(candidates[:topk]):
                edge_ids.append((user, station))
                ranks.append(rank)
        if not edge_ids:
            return (
                torch.empty((0, 2), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
            )
        return (
            torch.tensor(edge_ids, dtype=torch.long, device=self.device),
            torch.tensor(ranks, dtype=torch.long, device=self.device),
        )

    def _station_wait_eta(self, state: UAVSharedState) -> torch.Tensor:
        """Frozen-decision ETA for a new arrival joining behind current work."""

        dynamics = self.protocol.estimated
        compression = self.protocol.capacity_compression
        wait = torch.zeros(
            self.protocol.station_count, dtype=torch.float32, device=self.device
        )
        for station in range(self.protocol.station_count):
            occupied = state.active_uav[station] >= 0
            if bool((~occupied).any().item()):
                wait[station] = 0.0
                continue
            availability = sorted(
                float(value)
                for value in (
                    state.service_remaining_s[station] / compression
                ).tolist()
            )
            for _queued_user in state.fifo_queues[station]:
                earliest = availability.pop(0)
                availability.append(
                    earliest + dynamics.nominal_service_duration_s / compression
                )
                availability.sort()
            wait[station] = availability[0]
        return wait

    def _descriptors(
        self,
        state: UAVSharedState,
        edge_ids: torch.Tensor,
        eta_all: torch.Tensor,
        arrival_margin_all: torch.Tensor,
        travel_time_all: torch.Tensor,
    ) -> ServiceSimulatorDescriptors:
        exact = self.protocol.exact
        dynamics = self.protocol.estimated
        edges = int(edge_ids.size(0))
        if edges == 0:
            empty_float = torch.empty(0, dtype=torch.float32, device=self.device)
            empty_bool = torch.empty(0, dtype=torch.bool, device=self.device)
            fields = ServicePolicyDescriptors(
                eta_edge=empty_float.clone(),
                intensity_edge=empty_float.clone(),
                station_flow_node=state.station_flow.clone(),
            )
            return ServiceSimulatorDescriptors(
                policy_fields=fields,
                reachable_edge=empty_bool,
                feasible_start_edge=empty_bool.clone(),
                service_start_eta_s=empty_float.clone(),
                arrival_energy_fraction=empty_float.clone(),
            )
        users, stations = edge_ids.unbind(dim=1)
        wait_station = self._station_wait_eta(state)
        service_start_eta = travel_time_all[users, stations] + wait_station[stations]
        arrival_margin = arrival_margin_all[users, stations]
        arrival_fraction = (
            arrival_margin * (1.0 - self._reserve_fraction())
            + self._reserve_fraction()
        ).clamp(0.0, 1.0)
        has_free_slot = (state.active_uav < 0).any(dim=1)
        queue_has_room = torch.tensor(
            [
                len(queue) < exact.fifo_queue_capacity
                for queue in state.fifo_queues
            ],
            dtype=torch.bool,
            device=self.device,
        )
        feasible_start = has_free_slot[stations] | queue_has_room[stations]
        start_pressure = (
            service_start_eta / exact.intensity_lookahead_s
        ).clamp(0.0, 1.0)
        energy_pressure = (1.0 - arrival_margin).clamp(0.0, 1.0)
        raw_intensity = (
            dynamics.intensity_start_eta_weight * start_pressure
            + dynamics.intensity_energy_margin_weight * energy_pressure
            + dynamics.intensity_station_flow_weight * state.station_flow[stations]
        )
        intensity = (
            exact.intensity_clip[1] * raw_intensity
        ).clamp(*exact.intensity_clip)
        intensity = torch.where(
            feasible_start,
            intensity,
            torch.full_like(intensity, exact.intensity_clip[1]),
        )
        fields = ServicePolicyDescriptors(
            eta_edge=eta_all[users, stations].clamp(*exact.eta_clip),
            intensity_edge=intensity,
            station_flow_node=state.station_flow.clone(),
        )
        return ServiceSimulatorDescriptors(
            policy_fields=fields,
            reachable_edge=torch.ones(edges, dtype=torch.bool, device=self.device),
            feasible_start_edge=feasible_start.bool(),
            service_start_eta_s=service_start_eta,
            arrival_energy_fraction=arrival_fraction,
        )

    def observe(self) -> ServiceObservation:
        state = self._require_state()
        if state.epoch >= self.protocol.horizon_steps:
            raise RuntimeError("episode is finished")
        if self._pending_observation is not None:
            return self._pending_observation.clone()
        distance, travel_time, arrival_energy = self._distance_and_arrival_energy(state)
        eta_all, arrival_margin_all = self._eta_all_stations(
            distance, arrival_energy
        )
        reachable = self._reachable_mask(state, distance, arrival_energy)
        edge_ids, candidate_rank = self._candidate_edges(eta_all, reachable)
        sim_descriptors = self._descriptors(
            state,
            edge_ids,
            eta_all,
            arrival_margin_all,
            travel_time,
        )
        user_order = torch.tensor(
            keyed_order(
                range(self.protocol.uav_count),
                state.episode_seed,
                state.epoch,
                "decision-user-order",
            ),
            dtype=torch.long,
            device=self.device,
        )
        observation = ServiceObservation(
            observation_id=(state.episode_seed, state.epoch),
            uav_position=state.uav_position.clone(),
            station_position=state.station_position.clone(),
            energy_fraction=(
                state.energy_units / self.protocol.estimated.nominal_battery_units
            ).clamp(0.0, 1.0),
            phase=state.phase.clone(),
            current_station=state.target_station.clone(),
            candidate_edge_ids=edge_ids,
            candidate_rank=candidate_rank,
            sim_descriptors=sim_descriptors,
            policy_descriptors=sim_descriptors.policy_fields.clone(),
            user_order=user_order,
            meta={
                "protocol_version": UAV_SHARED_PROTOCOL_VERSION,
                "protocol_fingerprint": self.protocol_fingerprint,
                "structural_perturbation": self.perturbation.manifest(),
                "descriptor_semantics": {
                    "eta_edge": "normalized travel/arrival-energy local utility",
                    "intensity_edge": (
                        "bounded service-pressure score combining start delay, "
                        "energy margin and the station Flow state"
                    ),
                    "station_flow_node": (
                        "slow queue/occupancy/inbound station-pressure carrier"
                    ),
                },
                "candidate_topk": self.protocol.exact.candidate_topk,
                "reserve_fraction": (
                    self.protocol.exact.feasible_start_reserve_fraction
                ),
                "intensity_lookahead_s": (
                    self.protocol.exact.intensity_lookahead_s
                ),
                "fifo_queue_capacity": (
                    self.protocol.exact.fifo_queue_capacity
                ),
                "active_slots_per_station": (
                    self.protocol.exact.active_slots_per_station
                ),
                "capacity_compression": self.protocol.capacity_compression,
                "simulator_authority": (
                    "reachability, queue admission, service start, motion, energy"
                ),
            },
        )
        observation.validate()
        self._pending_observation = observation
        return observation.clone()

    def _queue_lengths(self, state: UAVSharedState) -> torch.Tensor:
        return torch.tensor(
            [len(queue) for queue in state.fifo_queues],
            dtype=torch.long,
            device=self.device,
        )

    def _active_counts(self, state: UAVSharedState) -> torch.Tensor:
        return (state.active_uav >= 0).sum(dim=1).to(torch.long)

    def _detach_user(
        self,
        state: UAVSharedState,
        queues: list[list[int]],
        user: int,
    ) -> None:
        active_location = torch.nonzero(
            state.active_uav == user, as_tuple=False
        )
        for station, slot in active_location.tolist():
            state.active_uav[station, slot] = -1
            state.service_remaining_s[station, slot] = 0.0
        for queue in queues:
            while user in queue:
                queue.remove(user)

    def _service_duration(
        self,
        state: UAVSharedState,
        user: int,
        station: int,
    ) -> float:
        dynamics = self.protocol.estimated
        visit = int(state.service_visit_count[user].item())
        duration = (
            dynamics.nominal_service_duration_s
            + dynamics.service_duration_jitter_s
            * keyed_signed_uniform(
                state.episode_seed,
                state.epoch,
                "service-duration",
                user,
                station,
                visit,
            )
        )
        duration *= self.perturbation.service_time_multiplier
        return max(duration, 1e-6)

    def _start_service(
        self,
        state: UAVSharedState,
        user: int,
        station: int,
        service_started: torch.Tensor,
    ) -> bool:
        free = torch.nonzero(state.active_uav[station] < 0, as_tuple=False)
        if free.numel() == 0:
            return False
        slot = int(free[0, 0].item())
        state.active_uav[station, slot] = user
        state.service_remaining_s[station, slot] = self._service_duration(
            state, user, station
        )
        state.service_visit_count[user] += 1
        state.target_station[user] = station
        state.phase[user] = int(UAVPhase.IN_SERVICE)
        service_started[user] = True
        return True

    def _backfill_fifo(
        self,
        state: UAVSharedState,
        queues: list[list[int]],
        service_started: torch.Tensor,
    ) -> None:
        for station, queue in enumerate(queues):
            while queue and bool((state.active_uav[station] < 0).any().item()):
                if self.perturbation.queue_law == "fifo":
                    user = queue.pop(0)
                else:
                    if self.perturbation.queue_law == "lowest_energy_first":
                        key = lambda candidate: (
                            float(state.energy_units[candidate].item()),
                            candidate,
                        )
                    else:
                        key = lambda candidate: (
                            -float(state.energy_units[candidate].item()),
                            candidate,
                        )
                    selected = min(queue, key=key)
                    queue.remove(selected)
                    user = selected
                self._start_service(
                    state, user, station, service_started
                )

    def _advance_service(
        self,
        state: UAVSharedState,
        queues: list[list[int]],
        service_started: torch.Tensor,
        service_completed: torch.Tensor,
    ) -> None:
        dynamics = self.protocol.estimated
        progress = self.protocol.exact.control_dt_s * self.protocol.capacity_compression
        for station in range(self.protocol.station_count):
            for slot in range(self.protocol.exact.active_slots_per_station):
                user = int(state.active_uav[station, slot].item())
                if user < 0:
                    continue
                remaining = float(state.service_remaining_s[station, slot].item())
                served = min(remaining, progress)
                state.energy_units[user] = min(
                    dynamics.nominal_battery_units,
                    float(state.energy_units[user].item())
                    + dynamics.service_energy_units_per_s * served,
                )
                remaining -= progress
                if remaining <= 1e-7:
                    state.active_uav[station, slot] = -1
                    state.service_remaining_s[station, slot] = 0.0
                    state.target_station[user] = -1
                    state.phase[user] = int(UAVPhase.IDLE)
                    state.missions_completed[user] += 1
                    service_completed[user] = True
                else:
                    state.service_remaining_s[station, slot] = remaining
        self._backfill_fifo(state, queues, service_started)

    def _deplete_user(
        self,
        state: UAVSharedState,
        queues: list[list[int]],
        user: int,
        energy_depleted: torch.Tensor,
    ) -> None:
        self._detach_user(state, queues, user)
        state.energy_units[user] = 0.0
        state.target_station[user] = -1
        state.phase[user] = int(UAVPhase.DEPLETED)
        energy_depleted[user] = True

    def _advance_motion_and_energy(
        self,
        state: UAVSharedState,
        queues: list[list[int]],
        energy_depleted: torch.Tensor,
        service_completed: torch.Tensor,
    ) -> list[int]:
        exact = self.protocol.exact
        dynamics = self.protocol.estimated
        arrivals: list[int] = []
        for user in range(self.protocol.uav_count):
            # Service completion is committed at the epoch boundary.  Charging
            # and idle drain must not both be charged for that same interval.
            if bool(service_completed[user].item()):
                continue
            phase = UAVPhase(int(state.phase[user].item()))
            if phase is UAVPhase.DEPLETED or phase is UAVPhase.IN_SERVICE:
                continue
            distance_moved = 0.0
            if phase is UAVPhase.TRAVELLING:
                station = int(state.target_station[user].item())
                displacement = (
                    state.station_position[station] - state.uav_position[user]
                )
                distance = float(torch.linalg.vector_norm(displacement).item())
                if distance > dynamics.docking_radius_m:
                    desired = math.atan2(
                        float(displacement[1].item()),
                        float(displacement[0].item()),
                    )
                    desired += dynamics.heading_jitter_rad * keyed_signed_uniform(
                        state.episode_seed,
                        state.epoch,
                        "heading-jitter",
                        user,
                        station,
                    )
                    current = float(state.heading_rad[user].item())
                    maximum_turn = (
                        dynamics.max_turn_rate_rad_s * exact.control_dt_s
                    )
                    turn = max(
                        -maximum_turn,
                        min(maximum_turn, _wrap_angle(desired - current)),
                    )
                    heading = _wrap_angle(current + turn)
                    state.heading_rad[user] = heading
                    intended_distance = min(
                        dynamics.max_speed_m_s * exact.control_dt_s,
                        distance,
                    )
                    previous_position = state.uav_position[user].clone()
                    delta = torch.tensor(
                        [math.cos(heading), math.sin(heading)],
                        dtype=state.uav_position.dtype,
                        device=self.device,
                    )
                    state.uav_position[user] += intended_distance * delta
                    state.uav_position[user, 0].clamp_(
                        0.0, dynamics.region_width_m
                    )
                    state.uav_position[user, 1].clamp_(
                        0.0, dynamics.region_height_m
                    )
                    distance_moved = float(
                        torch.linalg.vector_norm(
                            state.uav_position[user] - previous_position
                        ).item()
                    )
            energy_cost = dynamics.idle_energy_units_per_s * exact.control_dt_s
            energy_cost += dynamics.travel_energy_units_per_m * distance_moved
            state.energy_units[user] -= energy_cost
            if float(state.energy_units[user].item()) <= 0.0:
                self._deplete_user(state, queues, user, energy_depleted)
                continue
            if phase is UAVPhase.TRAVELLING:
                station = int(state.target_station[user].item())
                remaining = torch.linalg.vector_norm(
                    state.station_position[station] - state.uav_position[user]
                )
                if float(remaining.item()) <= dynamics.docking_radius_m:
                    state.uav_position[user] = state.station_position[station]
                    arrivals.append(user)
        return arrivals

    def _admit_arrivals(
        self,
        state: UAVSharedState,
        queues: list[list[int]],
        arrivals: Iterable[int],
        queue_admitted: torch.Tensor,
        service_started: torch.Tensor,
        failure_reason: torch.Tensor,
    ) -> None:
        exact = self.protocol.exact
        dynamics = self.protocol.estimated
        reserve = (
            self._reserve_fraction()
            * dynamics.nominal_battery_units
        )
        ordered = keyed_order(
            arrivals,
            state.episode_seed,
            state.epoch,
            "simultaneous-arrival-order",
        )
        for user in ordered:
            station = int(state.target_station[user].item())
            if float(state.energy_units[user].item()) < reserve:
                state.failed_service_starts[user] += 1
                state.target_station[user] = -1
                state.phase[user] = int(UAVPhase.IDLE)
                failure_reason[user] = int(ServiceFailureReason.RESERVE_VIOLATION)
                continue
            if self._start_service(
                state, user, station, service_started
            ):
                continue
            if len(queues[station]) < exact.fifo_queue_capacity:
                queues[station].append(user)
                state.phase[user] = int(UAVPhase.QUEUED)
                queue_admitted[user] = True
                continue
            state.failed_service_starts[user] += 1
            state.target_station[user] = -1
            state.phase[user] = int(UAVPhase.IDLE)
            failure_reason[user] = int(ServiceFailureReason.QUEUE_FULL)

    def _raw_station_pressure(
        self,
        state: UAVSharedState,
        queues: list[list[int]],
    ) -> torch.Tensor:
        exact = self.protocol.exact
        dynamics = self.protocol.estimated
        queue_pressure = torch.tensor(
            [len(queue) / exact.fifo_queue_capacity for queue in queues],
            dtype=torch.float32,
            device=self.device,
        )
        occupancy = (state.active_uav >= 0).sum(dim=1).to(torch.float32)
        occupancy /= exact.active_slots_per_station
        inbound = torch.zeros(
            self.protocol.station_count, dtype=torch.float32, device=self.device
        )
        travelling = state.phase == int(UAVPhase.TRAVELLING)
        if bool(travelling.any().item()):
            inbound.scatter_add_(
                0,
                state.target_station[travelling],
                torch.ones(
                    int(travelling.sum().item()),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        nominal_share = max(
            1.0,
            self.protocol.uav_count / self.protocol.station_count,
        )
        inbound = (inbound / nominal_share).clamp(0.0, 1.0)
        return (
            dynamics.flow_queue_weight * queue_pressure
            + dynamics.flow_occupancy_weight * occupancy
            + dynamics.flow_inbound_weight * inbound
        ).clamp(*exact.station_flow_clip)

    def _update_station_flow(
        self,
        state: UAVSharedState,
        queues: list[list[int]],
    ) -> None:
        memory = self.perturbation.station_flow_memory
        if memory is None:
            memory = self.protocol.estimated.flow_memory
        raw = self._raw_station_pressure(state, queues)
        state.station_flow = (
            memory * state.station_flow + (1.0 - memory) * raw
        ).clamp(*self.protocol.exact.station_flow_clip)

    def step_action(
        self,
        action: ReassociationAction,
    ) -> tuple[ServiceObservation | None, ServiceExecutionResult, bool]:
        state = self._require_state()
        if state.epoch >= self.protocol.horizon_steps:
            raise RuntimeError("episode is finished")
        observation = self._pending_observation or self.observe()
        if action.observation_id != observation.observation_id:
            raise ValueError(
                f"stale action for {action.observation_id}; "
                f"expected {observation.observation_id}"
            )
        action.validate(self.protocol.uav_count, self.protocol.station_count)
        users = self.protocol.uav_count
        target_before = state.target_station.clone()
        energy_before = state.energy_units.clone()
        station_flow_before = state.station_flow.clone()
        queue_length_before = self._queue_lengths(state)
        active_slots_before = self._active_counts(state)
        assignment_attempted = torch.zeros(
            users, dtype=torch.bool, device=self.device
        )
        assignment_accepted = torch.zeros_like(assignment_attempted)
        reassociation_executed = torch.zeros_like(assignment_attempted)
        queue_admitted = torch.zeros_like(assignment_attempted)
        service_started = torch.zeros_like(assignment_attempted)
        service_completed = torch.zeros_like(assignment_attempted)
        energy_depleted = torch.zeros_like(assignment_attempted)
        failure_reason = torch.full(
            (users,),
            int(ServiceFailureReason.NONE),
            dtype=torch.long,
            device=self.device,
        )
        edge_lookup = {
            (int(user), int(station))
            for user, station in observation.candidate_edge_ids.tolist()
        }
        queues = [list(queue) for queue in state.fifo_queues]
        for user in range(users):
            requested = int(action.requested_station[user].item())
            previous = int(target_before[user].item())
            if requested < 0:
                assignment_accepted[user] = True
                continue
            if requested == previous and previous >= 0:
                assignment_accepted[user] = True
                continue
            assignment_attempted[user] = True
            if UAVPhase(int(state.phase[user].item())) is UAVPhase.DEPLETED:
                failure_reason[user] = int(ServiceFailureReason.ENERGY_DEPLETED)
                continue
            if (user, requested) not in edge_lookup:
                failure_reason[user] = int(ServiceFailureReason.NOT_CANDIDATE)
                continue
            self._detach_user(state, queues, user)
            state.target_station[user] = requested
            state.phase[user] = int(UAVPhase.TRAVELLING)
            assignment_accepted[user] = True
            if previous >= 0 and previous != requested:
                reassociation_executed[user] = True
                state.reassociations[user] += 1
        self._advance_service(
            state,
            queues,
            service_started,
            service_completed,
        )
        arrivals = self._advance_motion_and_energy(
            state, queues, energy_depleted, service_completed
        )
        self._admit_arrivals(
            state,
            queues,
            arrivals,
            queue_admitted,
            service_started,
            failure_reason,
        )
        state.fifo_queues = tuple(tuple(queue) for queue in queues)
        self._update_station_flow(state, queues)
        state.epoch += 1
        state.validate(self.protocol)
        result = ServiceExecutionResult(
            observation_id=observation.observation_id,
            requested_station=action.requested_station.clone(),
            target_before=target_before,
            target_after=state.target_station.clone(),
            assignment_attempted=assignment_attempted,
            assignment_accepted=assignment_accepted,
            reassociation_executed=reassociation_executed,
            failure_reason=failure_reason,
            queue_admitted=queue_admitted,
            service_started=service_started,
            service_completed=service_completed,
            energy_depleted=energy_depleted,
            energy_before=energy_before,
            energy_after=state.energy_units.clone(),
            station_flow_before=station_flow_before,
            station_flow_after=state.station_flow.clone(),
            queue_length_before=queue_length_before,
            queue_length_after=self._queue_lengths(state),
            active_slots_before=active_slots_before,
            active_slots_after=self._active_counts(state),
        )
        result.validate(users, self.protocol.station_count)
        self._pending_observation = None
        done = state.epoch >= self.protocol.horizon_steps
        next_observation = None if done else self.observe()
        return next_observation, result, done

    def snapshot(self) -> UAVSharedSnapshot:
        state = self._require_state()
        return UAVSharedSnapshot(
            protocol_fingerprint=self.protocol_fingerprint,
            state=state.clone(),
        )

    def restore(self, snapshot: UAVSharedSnapshot) -> ServiceObservation | None:
        if snapshot.schema_version != UAV_SHARED_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                "snapshot schema mismatch: "
                f"{snapshot.schema_version} != {UAV_SHARED_SNAPSHOT_SCHEMA_VERSION}"
            )
        if snapshot.protocol_fingerprint != self.protocol_fingerprint:
            raise ValueError("snapshot protocol fingerprint does not match environment")
        restored = snapshot.state.to(self.device)
        restored.validate(self.protocol)
        self.state = restored
        self._pending_observation = None
        return None if self.done else self.observe()

    def fork(self) -> "UAVSharedServiceEnv":
        clone = self.new_instance()
        clone.restore(self.snapshot())
        return clone
