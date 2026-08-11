import torch
import pytest

from leo_pg.eval.closed_loop import (
    CallableDescriptorProvider,
    CallablePolicyStreamInitializer,
    ConstantPolicyStreamInitializer,
    PolicyInitializerKind,
    run_closed_loop,
)
from leo_pg.control import FixedRankPolicy
from leo_pg.control import FixedRankPolicyConfig
from leo_pg.eval.substitution import SubstitutionMode
from leo_pg.sim.paper_environment import PaperAlignedLEOEnv
from leo_pg.sim.state import PolicyDescriptors


def _cfg(*, horizon_steps: int = 3, min_dwell_steps: int = 10):
    return {
        "seed": 29,
        "K_users": 2,
        "S_sats": 2,
        "ephemeris": {
            "mode": "debug",
            "initial_user_pos": [[1.0, 0.0, 0.0], [1.0, 0.01, 0.0]],
            "initial_satellite_pos": [[2.0, 0.0, 0.0], [2.0, 0.12, 0.0]],
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
                "capacity_users": 2,
                "admission_order": "keyed_random",
            },
            "phy": {
                "noise_power": 1.0,
                "reference_power": 10.0,
                "distance_reference": 1.0,
                "pathloss_exponent": 2.0,
                "interference_eta": 0.6,
                "shadow_sigma": 0.0,
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
                "hard_feasibility_mask": True,
                "gamma_weight": 1.0,
                "load_weight": 0.4,
                "intensity_weight": 0.6,
                "min_dwell_steps": min_dwell_steps,
                "hysteresis": 0.5,
            },
        },
    }


def _prefer_satellite(satellite: int):
    def predict(observation):
        destination = observation.candidate_edge_ids[:, 1]
        preferred = destination == satellite
        return PolicyDescriptors(
            gamma_edge=torch.where(preferred, 10.0, -10.0),
            intensity_edge=torch.where(preferred, 0.0, 10.0),
            flow_node=torch.tensor(
                [10.0, 0.0] if satellite == 1 else [0.0, 10.0],
                dtype=observation.node_x.dtype,
                device=observation.node_x.device,
            ),
        )

    return CallableDescriptorProvider(predict, fingerprint=f"prefer-satellite-{satellite}")


def _zero_initializer():
    return ConstantPolicyStreamInitializer(gamma=0.0, intensity=0.0, flow=0.0)


def _assert_same_trace(left, right):
    assert left.action_count == right.action_count
    for left_step, right_step in zip(left.records, right.records):
        assert left_step.observation_id == right_step.observation_id
        assert torch.equal(left_step.candidate_edge_ids, right_step.candidate_edge_ids)
        assert torch.equal(
            left_step.action.requested_serving,
            right_step.action.requested_serving,
        )
        assert torch.equal(
            left_step.execution.executed_serving,
            right_step.execution.executed_serving,
        )
        assert torch.equal(left_step.execution.flow_after, right_step.execution.flow_after)


class _PerfectNextProvider:
    """One-step oracle used only to test temporal staging and identity."""

    def __init__(self, cfg):
        self.environment = PaperAlignedLEOEnv(cfg)
        self.policy = FixedRankPolicy(self.environment.fixed_policy_config())
        self.observation = None
        self.fingerprint = "perfect-next-oracle-v1"

    def reset(self):
        self.observation = self.environment.reset_control()

    def predict_next(self, policy_observation):
        assert self.observation is not None
        action = self.policy(policy_observation)
        next_observation, _, done = self.environment.step_action(action)
        assert not done and next_observation is not None
        next_lookup = {
            tuple(edge): index
            for index, edge in enumerate(next_observation.candidate_edge_ids.tolist())
        }
        fields = next_observation.sim_descriptors.policy_fields
        gamma = torch.zeros(policy_observation.edge_count)
        intensity = torch.zeros(policy_observation.edge_count)
        for source_index, edge in enumerate(policy_observation.candidate_edge_ids.tolist()):
            target_index = next_lookup.get(tuple(edge))
            if target_index is not None:
                gamma[source_index] = fields.gamma_edge[target_index]
                intensity[source_index] = fields.intensity_edge[target_index]
        self.observation = next_observation
        return PolicyDescriptors(gamma, intensity, fields.flow_node.clone())


def test_oracle_substitution_is_identity_for_an_oracle_prediction():
    cfg = _cfg()
    initializer = CallablePolicyStreamInitializer(
        lambda observation, edge_mask: observation.sim_descriptors.policy_fields,
        fingerprint="explicit-oracle-warm-start-v1",
        kind=PolicyInitializerKind.ORACLE_WARM_START,
    )
    oracle = run_closed_loop(
        PaperAlignedLEOEnv(cfg),
        mode=SubstitutionMode.ORACLE,
    )
    model = run_closed_loop(
        PaperAlignedLEOEnv(cfg),
        mode=SubstitutionMode.MODEL,
        descriptor_provider=_PerfectNextProvider(cfg),
        policy_initializer=initializer,
        allow_oracle_warm_start=True,
    )
    _assert_same_trace(oracle, model)
    assert torch.equal(oracle.final_serving, model.final_serving)
    assert torch.equal(oracle.final_flow, model.final_flow)


def test_descriptor_divergence_reaches_state_only_through_executed_action():
    sat0 = run_closed_loop(
        PaperAlignedLEOEnv(_cfg(min_dwell_steps=0)),
        mode=SubstitutionMode.MODEL,
        descriptor_provider=_prefer_satellite(0),
        policy_initializer=_zero_initializer(),
    )
    sat1 = run_closed_loop(
        PaperAlignedLEOEnv(_cfg(min_dwell_steps=0)),
        mode=SubstitutionMode.MODEL,
        descriptor_provider=_prefer_satellite(1),
        policy_initializer=_zero_initializer(),
    )

    first0, first1 = sat0.records[0], sat1.records[0]
    # Before an action is committed, simulator state and hard gate are exactly
    # identical; only the policy-facing prediction differs.
    assert torch.equal(first0.candidate_edge_ids, first1.candidate_edge_ids)
    assert torch.equal(
        first0.sim_descriptors.policy_fields.gamma_edge,
        first1.sim_descriptors.policy_fields.gamma_edge,
    )
    assert torch.equal(first0.feasible_edge, first1.feasible_edge)
    # The t0 cold-start action is shared. The t0 forecast is staged, so the
    # first action divergence appears at t1 rather than being used one step too
    # early at t0.
    assert torch.equal(first0.action.requested_serving, first1.action.requested_serving)
    assert not torch.equal(
        sat0.records[1].action.requested_serving,
        sat1.records[1].action.requested_serving,
    )
    # The different t1 action then changes the simulator carrier at t2.
    assert not torch.equal(
        sat0.records[2].sim_descriptors.policy_fields.flow_node,
        sat1.records[2].sim_descriptors.policy_fields.flow_node,
    )
    assert not torch.equal(
        sat0.records[1].model_input_descriptors.flow_node,
        sat1.records[1].model_input_descriptors.flow_node,
    )
    assert not torch.equal(sat0.final_flow, sat1.final_flow)


def test_runner_commits_exactly_one_action_per_protocol_epoch():
    result = run_closed_loop(
        PaperAlignedLEOEnv(_cfg(horizon_steps=4)),
        mode="oracle",
    )
    assert result.action_count == 4
    assert [record.observation_id[1] for record in result.records] == [0, 1, 2, 3]
    assert all(record.execution.observation_id == record.observation_id for record in result.records)


def test_partial_oracle_mode_keeps_simulator_authority_unchanged():
    result = run_closed_loop(
        PaperAlignedLEOEnv(_cfg(horizon_steps=1)),
        mode=SubstitutionMode.INTENSITY_FLOW,
        descriptor_provider=_prefer_satellite(1),
        policy_initializer=_zero_initializer(),
    )
    record = result.records[0]
    assert record.model_descriptors is not None
    assert torch.equal(
        record.policy_descriptors.intensity_edge,
        record.sim_descriptors.policy_fields.intensity_edge,
    )
    assert torch.equal(
        record.policy_descriptors.flow_node,
        record.sim_descriptors.policy_fields.flow_node,
    )
    assert torch.equal(
        record.policy_descriptors.gamma_edge,
        record.model_descriptors.gamma_edge,
    )
    assert torch.equal(record.feasible_edge, record.sim_descriptors.feasible_edge)


def test_partial_oracle_copy_is_not_reused_as_the_model_input():
    seen = []

    def predict(observation):
        seen.append(observation.policy_descriptors.clone())
        return PolicyDescriptors(
            gamma_edge=torch.zeros(observation.edge_count),
            intensity_edge=torch.zeros(observation.edge_count),
            flow_node=torch.zeros(observation.satellite_count),
        )

    initializer = ConstantPolicyStreamInitializer(
        gamma=-3.0,
        intensity=4.0,
        flow=0.25,
    )
    result = run_closed_loop(
        PaperAlignedLEOEnv(_cfg(horizon_steps=2)),
        mode=SubstitutionMode.INTENSITY_FLOW,
        descriptor_provider=CallableDescriptorProvider(
            predict,
            fingerprint="record-model-input-v1",
        ),
        policy_initializer=initializer,
    )

    assert len(seen) == 1
    assert torch.equal(seen[0].gamma_edge, torch.full((4,), -3.0))
    assert torch.equal(seen[0].intensity_edge, torch.full((4,), 4.0))
    assert torch.equal(seen[0].flow_node, torch.full((2,), 0.25))
    assert not torch.equal(
        result.records[0].policy_descriptors.intensity_edge,
        seen[0].intensity_edge,
    )


def test_model_rollout_requires_explicit_non_missing_d0():
    with pytest.raises(ValueError, match="requires a policy_stream_initializer"):
        run_closed_loop(
            PaperAlignedLEOEnv(_cfg(horizon_steps=1)),
            mode="model",
            descriptor_provider=_prefer_satellite(0),
        )


def test_main_protocol_rejects_an_unlabelled_oracle_warm_start():
    initializer = CallablePolicyStreamInitializer(
        lambda observation, edge_mask: observation.sim_descriptors.policy_fields,
        fingerprint="oracle-warm-start-v1",
        kind="oracle_warm_start",
    )
    with pytest.raises(ValueError, match="no-teacher-warm-start"):
        run_closed_loop(
            PaperAlignedLEOEnv(_cfg(horizon_steps=1)),
            mode="model",
            descriptor_provider=_prefer_satellite(0),
            policy_initializer=initializer,
        )


def test_paper_runner_rejects_a_silent_policy_override():
    mismatched = FixedRankPolicy(
        FixedRankPolicyConfig(
            hard_feasibility_mask=False,
            min_dwell_steps=0,
            hysteresis=0.0,
        )
    )
    with pytest.raises(ValueError, match="must match environment"):
        run_closed_loop(
            PaperAlignedLEOEnv(_cfg(horizon_steps=1)),
            mode="oracle",
            policy=mismatched,
        )


def test_result_records_protocol_and_initializer_identity():
    result = run_closed_loop(
        PaperAlignedLEOEnv(_cfg(horizon_steps=1)),
        mode="oracle",
    )
    assert len(result.protocol_fingerprint) == 64
    assert result.provider_fingerprint == "oracle"
    assert result.initializer_kind == "simulator_oracle"
    assert result.initializer_fingerprint == "oracle"
    assert result.hard_feasibility_mask is True
    assert result.initial_policy_strategy == "simulator_oracle"
    assert result.new_edge_strategy == "simulator_oracle"
