from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch

from uav_cfs import (
    EstimatedDynamicsParameters,
    ReassociationAction,
    ServiceFailureReason,
    ServicePolicyDescriptors,
    UAVPhase,
    UAVSharedProtocol,
    UAVSharedServiceEnv,
    resolve_uav_shared_protocol,
)
from uav_cfs.randomness import keyed_order


def _protocol(**estimate_overrides: float) -> UAVSharedProtocol:
    source = Path(__file__).parents[1] / "configs" / "model_mlp.yaml"
    base = resolve_uav_shared_protocol(source)
    estimates = replace(base.estimated, **estimate_overrides)
    return replace(base, episode_seed=31, estimated=estimates)


def _set_geometry(
    env: UAVSharedServiceEnv,
    *,
    uav_xy: tuple[float, float] = (500.0, 500.0),
    station_xy: tuple[float, float] = (500.0, 500.0),
) -> None:
    snapshot = env.snapshot()
    state = snapshot.state
    state.uav_position[:] = torch.tensor(uav_xy, dtype=torch.float32)
    state.station_position[:] = torch.tensor(station_xy, dtype=torch.float32)
    state.energy_units.fill_(env.protocol.estimated.nominal_battery_units)
    state.heading_rad.zero_()
    env.restore(snapshot)


def _request_all(observation, station: int) -> ReassociationAction:
    return ReassociationAction(
        observation_id=observation.observation_id,
        requested_station=torch.full(
            (observation.user_count,), station, dtype=torch.long
        ),
    )


def _keep_all(observation) -> ReassociationAction:
    return ReassociationAction(
        observation_id=observation.observation_id,
        requested_station=torch.full(
            (observation.user_count,), -1, dtype=torch.long
        ),
    )


def test_reachable_candidates_are_top3_with_stable_station_ties() -> None:
    env = UAVSharedServiceEnv(_protocol())
    env.reset_control()
    _set_geometry(env)

    observation = env.observe()
    first_user = observation.candidate_edge_ids[:, 0] == 0
    assert observation.candidate_edge_ids[first_user, 1].tolist() == [0, 1, 2]
    assert observation.candidate_rank[first_user].tolist() == [0, 1, 2]
    assert bool(observation.sim_descriptors.reachable_edge.all())
    assert torch.all((observation.policy_descriptors.eta_edge >= 0.0))
    assert torch.all((observation.policy_descriptors.eta_edge <= 1.0))
    assert torch.all((observation.policy_descriptors.intensity_edge >= 0.0))
    assert torch.all((observation.policy_descriptors.intensity_edge <= 3.0))


def test_energy_filter_removes_station_that_would_break_arrival_reserve() -> None:
    env = UAVSharedServiceEnv(
        _protocol(
            maximum_reachable_distance_m=1_000.0,
            travel_energy_units_per_m=0.2,
            idle_energy_units_per_s=0.0,
        )
    )
    env.reset_control()
    snapshot = env.snapshot()
    state = snapshot.state
    state.uav_position[:] = torch.tensor([100.0, 100.0])
    state.station_position[:] = torch.tensor([900.0, 100.0])
    state.station_position[0] = torch.tensor([110.0, 100.0])
    state.energy_units.fill_(20.0)
    env.restore(snapshot)

    observation = env.observe()
    first_user_stations = observation.candidate_edge_ids[
        observation.candidate_edge_ids[:, 0] == 0, 1
    ].tolist()
    assert first_user_stations == [0]


def test_fifo_slots_queue_capacity_and_service_start_order() -> None:
    env = UAVSharedServiceEnv(
        _protocol(
            nominal_service_duration_s=1.0,
            service_duration_jitter_s=0.0,
            docking_radius_m=2.0,
            service_energy_units_per_s=0.0,
            flow_memory=0.0,
        )
    )
    env.reset_control()
    _set_geometry(env)
    observation = env.observe()
    next_observation, result, done = env.step_action(
        _request_all(observation, 0)
    )

    assert not done
    assert next_observation is not None
    expected_order = keyed_order(
        range(24), 31, 0, "simultaneous-arrival-order"
    )
    state = env.snapshot().state
    assert state.active_uav[0].tolist() == list(expected_order[:2])
    assert state.fifo_queues[0] == expected_order[2:10]
    assert int(result.service_started.sum().item()) == 2
    assert int(result.queue_admitted.sum().item()) == 8
    assert int(
        (result.failure_reason == int(ServiceFailureReason.QUEUE_FULL)).sum().item()
    ) == 14

    queue_before_release = state.fifo_queues[0]
    completing_users = state.active_uav[0].clone()
    energy_before_completion = state.energy_units[completing_users].clone()
    _, second, _ = env.step_action(_keep_all(next_observation))
    state = env.snapshot().state
    assert state.active_uav[0].tolist() == list(queue_before_release[:2])
    assert state.fifo_queues[0] == queue_before_release[2:]
    assert int(second.service_completed.sum().item()) == 2
    assert int(second.service_started.sum().item()) == 2
    assert torch.equal(
        state.energy_units[completing_users], energy_before_completion
    )


def test_reassociation_releases_old_slot_and_uses_new_station() -> None:
    env = UAVSharedServiceEnv(
        _protocol(
            nominal_service_duration_s=10.0,
            service_duration_jitter_s=0.0,
            docking_radius_m=2.0,
        )
    )
    env.reset_control()
    _set_geometry(env)
    observation = env.observe()
    next_observation, _, _ = env.step_action(_request_all(observation, 0))
    assert next_observation is not None
    before = env.snapshot().state
    moving_user = int(before.active_uav[0, 0].item())
    old_fifo_head = before.fifo_queues[0][0]
    request = torch.full((24,), -1, dtype=torch.long)
    request[moving_user] = 1
    _, result, _ = env.step_action(
        ReassociationAction(next_observation.observation_id, request)
    )
    after = env.snapshot().state

    assert bool(result.reassociation_executed[moving_user].item())
    assert int(after.target_station[moving_user].item()) == 1
    assert int(after.phase[moving_user].item()) == int(UAVPhase.IN_SERVICE)
    assert moving_user in after.active_uav[1].tolist()
    assert old_fifo_head in after.active_uav[0].tolist()
    assert moving_user not in after.fifo_queues[0]


def test_policy_descriptor_overwrite_cannot_bypass_full_fifo_authority() -> None:
    env = UAVSharedServiceEnv(
        _protocol(
            nominal_service_duration_s=10.0,
            service_duration_jitter_s=0.0,
            docking_radius_m=2.0,
        )
    )
    env.reset_control()
    _set_geometry(env)
    observation = env.observe()
    next_observation, first, _ = env.step_action(_request_all(observation, 0))
    assert next_observation is not None
    rejected = torch.nonzero(
        first.failure_reason == int(ServiceFailureReason.QUEUE_FULL),
        as_tuple=False,
    )[0, 0]
    edge = torch.nonzero(
        (next_observation.candidate_edge_ids[:, 0] == rejected)
        & (next_observation.candidate_edge_ids[:, 1] == 0),
        as_tuple=False,
    )[0, 0]
    assert not bool(
        next_observation.sim_descriptors.feasible_start_edge[edge].item()
    )
    overwritten = next_observation.with_policy_descriptors(
        ServicePolicyDescriptors(
            eta_edge=torch.ones_like(next_observation.policy_descriptors.eta_edge),
            intensity_edge=torch.zeros_like(
                next_observation.policy_descriptors.intensity_edge
            ),
            station_flow_node=torch.zeros_like(
                next_observation.policy_descriptors.station_flow_node
            ),
        )
    )
    assert bool(overwritten.policy_descriptors.eta_edge[edge].item())
    request = torch.full((24,), -1, dtype=torch.long)
    request[rejected] = 0
    _, second, _ = env.step_action(
        ReassociationAction(overwritten.observation_id, request)
    )
    assert int(second.failure_reason[rejected].item()) == int(
        ServiceFailureReason.QUEUE_FULL
    )


def test_motion_is_turn_bounded_and_consumes_finite_energy() -> None:
    turn_rate = 0.1
    env = UAVSharedServiceEnv(
        _protocol(
            max_speed_m_s=10.0,
            max_turn_rate_rad_s=turn_rate,
            heading_jitter_rad=0.0,
            docking_radius_m=1.0,
            travel_energy_units_per_m=0.2,
            idle_energy_units_per_s=0.5,
        )
    )
    env.reset_control()
    snapshot = env.snapshot()
    state = snapshot.state
    state.uav_position[:] = torch.tensor([500.0, 500.0])
    state.station_position[:] = torch.tensor([900.0, 900.0])
    state.station_position[0] = torch.tensor([600.0, 500.0])
    state.heading_rad.fill_(math.pi)
    state.energy_units.fill_(100.0)
    env.restore(snapshot)
    observation = env.observe()
    request = torch.full((24,), -1, dtype=torch.long)
    request[0] = 0
    before = env.snapshot().state
    env.step_action(ReassociationAction(observation.observation_id, request))
    after = env.snapshot().state

    angle_delta = math.atan2(
        math.sin(float(after.heading_rad[0] - before.heading_rad[0])),
        math.cos(float(after.heading_rad[0] - before.heading_rad[0])),
    )
    displacement = torch.linalg.vector_norm(
        after.uav_position[0] - before.uav_position[0]
    )
    assert abs(angle_delta) <= turn_rate + 1e-6
    assert float(displacement.item()) <= 10.0 + 1e-6
    assert float(after.energy_units[0].item()) < float(before.energy_units[0].item())


def test_snapshot_fork_replays_same_action_and_diverges_only_after_actions_differ() -> None:
    env = UAVSharedServiceEnv(
        _protocol(
            docking_radius_m=2.0,
            nominal_service_duration_s=5.0,
            service_duration_jitter_s=0.0,
            flow_memory=0.5,
        )
    )
    env.reset_control()
    _set_geometry(env)

    same_a = env.fork()
    same_b = env.fork()
    obs_a = same_a.observe()
    obs_b = same_b.observe()
    same_a.step_action(_request_all(obs_a, 0))
    same_b.step_action(_request_all(obs_b, 0))
    state_a = same_a.snapshot().state
    state_b = same_b.snapshot().state
    assert torch.equal(state_a.active_uav, state_b.active_uav)
    assert state_a.fifo_queues == state_b.fifo_queues
    assert torch.equal(state_a.energy_units, state_b.energy_units)
    assert torch.equal(state_a.station_flow, state_b.station_flow)

    left = env.fork()
    right = env.fork()
    left.step_action(_request_all(left.observe(), 0))
    right.step_action(_request_all(right.observe(), 1))
    left_state = left.snapshot().state
    right_state = right.snapshot().state
    assert not torch.equal(left_state.active_uav, right_state.active_uav)
    assert not torch.equal(left_state.station_flow, right_state.station_flow)


def test_snapshot_restore_rejects_a_different_estimated_protocol() -> None:
    source = UAVSharedServiceEnv(_protocol())
    source.reset_control()
    incompatible = UAVSharedServiceEnv(_protocol(max_speed_m_s=30.0))
    with pytest.raises(ValueError, match="fingerprint"):
        incompatible.restore(source.snapshot())
