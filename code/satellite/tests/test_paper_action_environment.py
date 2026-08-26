import torch

from leo_pg.control import FixedRankPolicy
from leo_pg.sim.paper_environment import PaperAlignedLEOEnv
from leo_pg.sim.state import FailureReason, ServingAction


def _cfg(*, horizon_steps: int = 3, capacity: int = 2, hard_mask: bool = True):
    return {
        "seed": 17,
        "T": horizon_steps,
        "K_users": 2,
        "S_sats": 2,
        "ephemeris": {
            "mode": "debug",
            "initial_user_pos": [[1.0, 0.0, 0.0], [1.0, 0.01, 0.0]],
            "initial_satellite_pos": [[2.0, 0.0, 0.0], [2.0, 0.1, 0.0]],
            "initial_user_vel": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "initial_satellite_vel": [[0.0, 0.01, 0.0], [0.0, -0.01, 0.0]],
        },
        "paper_protocol": {
            "horizon_steps": horizon_steps,
            "dt_ctrl": 0.1,
            "dt_phy": 0.001,
            "candidates": {"minimum_elevation_deg": 0.0, "topk": 2},
            "feasibility": {
                "gamma_min_db": -30.0,
                "flow_max": 0.95,
                "capacity_users": capacity,
                "admission_order": "keyed_random",
            },
            "phy": {
                "noise_power": 1.0,
                "reference_power": 10.0,
                "distance_reference": 1.0,
                "pathloss_exponent": 2.0,
                "interference_eta": 0.6,
                "shadow_sigma": 0.1,
            },
            "flow": {
                "arrival_rate": 1.0,
                "service_rate": 0.0,
                "feedback_strength": 0.0,
                "ema_factor": 0.9,
                "phi": "zero",
                "clip_lower": 0.0,
                "clip_upper": 1.0,
            },
            "intensity": {
                "baseline_hazard": 0.1,
                "beta_gamma": 0.8,
                "beta_flow": 0.9,
                "beta_elevation": 0.2,
                "gamma_covariate_scale_db": 10.0,
                "elevation_covariate_scale_deg": 90.0,
                "horizon": 1.0,
                "num_intervals": 8,
            },
            "policy": {
                "hard_feasibility_mask": hard_mask,
                "min_dwell_steps": 10,
                "hysteresis": 0.5,
            },
        },
    }


def _action(observation, serving):
    return ServingAction(
        observation_id=observation.observation_id,
        requested_serving=torch.tensor(serving, dtype=torch.long),
    )


def test_reset_observes_without_hidden_assignment():
    env = PaperAlignedLEOEnv(_cfg())
    observation = env.reset_control()
    assert observation.current_serving.tolist() == [-1, -1]
    assert torch.equal(env.flow, torch.zeros(2))
    assert observation.observation_id == (17, 0)
    assert observation.edge_count == 4


def test_committed_action_updates_real_flow_and_advances_once():
    env = PaperAlignedLEOEnv(_cfg())
    observation = env.reset_control()
    next_observation, execution, done = env.step_action(_action(observation, [0, 0]))
    assert not done and next_observation is not None
    assert execution.executed_serving.tolist() == [0, 0]
    # instantaneous flow is 0.2 on sat0; rho=.9 makes the next carrier .02
    assert torch.allclose(execution.flow_after, torch.tensor([0.02, 0.0]))
    assert next_observation.observation_id == (17, 1)
    assert next_observation.current_serving.tolist() == [0, 0]
    assert next_observation.hold_steps.tolist() == [1, 1]


def test_different_actions_change_state_but_not_exogenous_geometry():
    concentrated = PaperAlignedLEOEnv(_cfg())
    balanced = PaperAlignedLEOEnv(_cfg())
    obs_a = concentrated.reset_control()
    obs_b = balanced.reset_control()
    assert torch.equal(obs_a.node_x[:, :6], obs_b.node_x[:, :6])
    next_a, _, _ = concentrated.step_action(_action(obs_a, [0, 0]))
    next_b, _, _ = balanced.step_action(_action(obs_b, [0, 1]))
    assert next_a is not None and next_b is not None
    assert torch.equal(next_a.node_x[:, :6], next_b.node_x[:, :6])
    assert not torch.equal(next_a.sim_descriptors.policy_fields.flow_node,
                           next_b.sim_descriptors.policy_fields.flow_node)
    assert not torch.equal(next_a.sim_descriptors.policy_fields.gamma_edge,
                           next_b.sim_descriptors.policy_fields.gamma_edge)


def test_simulator_authority_rejects_infeasible_requested_action():
    env = PaperAlignedLEOEnv(_cfg())
    env.reset_control()
    env.flow[0] = 0.96
    observation = env.observe()
    assert not torch.any(
        observation.sim_descriptors.feasible_edge[
            observation.candidate_edge_ids[:, 1] == 0
        ]
    )
    _, execution, _ = env.step_action(_action(observation, [0, 1]))
    assert execution.executed_serving.tolist() == [-1, 1]
    # The primary protocol materializes only the post-mask graph, so a request
    # for a rejected geometry edge is no longer a candidate at policy time.
    assert execution.failure_reason[0].item() == int(FailureReason.NOT_CANDIDATE)


def test_empty_authorized_graph_yields_null_proposal_and_executor_abstention():
    env = PaperAlignedLEOEnv(_cfg())
    first = env.reset_control()
    next_observation, _, _ = env.step_action(_action(first, [0, 1]))
    assert next_observation is not None
    env.flow.fill_(1.0)
    empty = env.observe()
    assert empty.edge_count == 0

    action = FixedRankPolicy(env.fixed_policy_config())(empty)
    assert action.requested_serving.tolist() == [-1, -1]
    _, execution, _ = env.step_action(action)

    assert execution.executed_serving.tolist() == [-1, -1]
    assert torch.all(execution.failure_reason == int(FailureReason.ABSTAIN))
    assert not torch.any(execution.handover_attempted)


def test_frozen_visibility_subset_is_shared_before_per_user_topk():
    cfg = _cfg()
    cfg["paper_protocol"]["visibility"] = {
        "source": "frozen_clipped_distribution",
        "mean": 1.0,
        "standard_deviation": 0.1,
        "minimum": 1,
        "maximum": 1,
        "episode_index": 0,
    }
    env = PaperAlignedLEOEnv(cfg)
    observation = env.reset_control()

    assert env.regional_visibility_count == 1
    assert env.regional_satellite_ids is not None
    selected = int(env.regional_satellite_ids.item())
    assert torch.equal(
        observation.meta["regional_satellite_ids"],
        env.regional_satellite_ids,
    )
    assert observation.candidate_edge_ids.tolist() == [[0, selected], [1, selected]]


def test_full_geometric_protocol_can_rank_then_reject_an_infeasible_argmax():
    env = PaperAlignedLEOEnv(_cfg(hard_mask=False))
    env.reset_control()
    env.flow[0] = 0.96
    observation = env.observe()
    sat0 = observation.candidate_edge_ids[:, 1] == 0
    observation.policy_descriptors.gamma_edge = torch.where(sat0, 100.0, -100.0)
    observation.policy_descriptors.intensity_edge = torch.where(sat0, 0.0, 100.0)
    observation.policy_descriptors.flow_node = torch.tensor([0.0, 100.0])
    action = FixedRankPolicy(env.fixed_policy_config())(observation)
    assert action.requested_serving.tolist() == [0, 0]
    _, execution, _ = env.step_action(action)
    assert execution.executed_serving.tolist() == [-1, -1]
    assert torch.all(execution.failure_reason == int(FailureReason.INFEASIBLE))


def test_stale_or_repeated_action_is_rejected():
    env = PaperAlignedLEOEnv(_cfg())
    observation = env.reset_control()
    next_observation, _, _ = env.step_action(_action(observation, [0, 1]))
    assert next_observation is not None
    stale = _action(observation, [0, 1])
    try:
        env.step_action(stale)
    except ValueError as error:
        assert "stale action" in str(error)
    else:  # pragma: no cover
        raise AssertionError("stale action was accepted")


def test_hard_mask_configuration_rejects_string_booleans():
    cfg = _cfg()
    cfg["paper_protocol"]["policy"]["hard_feasibility_mask"] = "false"
    try:
        PaperAlignedLEOEnv(cfg)
    except TypeError as error:
        assert "must be a boolean" in str(error)
    else:  # pragma: no cover
        raise AssertionError("string boolean was silently accepted")
