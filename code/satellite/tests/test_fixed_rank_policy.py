import pytest
import torch

from leo_pg.control.policy import (
    FixedRankPolicy,
    FixedRankPolicyConfig,
    normalized_ordinal_rank,
)
from leo_pg.sim.state import (
    ControlObservation,
    PolicyDescriptors,
    SimulatorDescriptors,
)


def _observation(
    *,
    user_count=1,
    satellite_count=3,
    edges=((0, 0), (0, 1), (0, 2)),
    gamma=(0.0, 1.0, 2.0),
    intensity=(2.0, 1.0, 0.0),
    flow=(2.0, 1.0, 0.0),
    feasible=(True, True, True),
    current=(-1,),
    hold=(0,),
):
    edge_ids = torch.tensor(edges, dtype=torch.long).reshape(-1, 2)
    if edge_ids.numel():
        edge_index = torch.stack(
            (edge_ids[:, 0], edge_ids[:, 1] + user_count),
            dim=0,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    policy_fields = PolicyDescriptors(
        gamma_edge=torch.tensor(gamma, dtype=torch.float32),
        intensity_edge=torch.tensor(intensity, dtype=torch.float32),
        flow_node=torch.tensor(flow, dtype=torch.float32),
    )
    observation = ControlObservation(
        observation_id=(17, 4),
        node_x=torch.zeros(user_count + satellite_count, 3),
        candidate_edge_index=edge_index,
        candidate_edge_ids=edge_ids,
        edge_features=torch.zeros(len(edges), 1),
        elevation_deg=torch.full((len(edges),), 30.0),
        sim_descriptors=SimulatorDescriptors(
            policy_fields=policy_fields.clone(),
            feasible_edge=torch.tensor(feasible, dtype=torch.bool),
        ),
        policy_descriptors=policy_fields,
        current_serving=torch.tensor(current, dtype=torch.long),
        hold_steps=torch.tensor(hold, dtype=torch.long),
        user_order=torch.arange(user_count, dtype=torch.long),
    )
    observation.validate()
    return observation


def test_normalized_ordinal_rank_is_order_independent_and_uses_satellite_ties():
    ranks = normalized_ordinal_rank(
        torch.tensor([2.0, 2.0, 1.0]),
        torch.tensor([2, 0, 1]),
        higher_is_better=True,
    )
    assert ranks.tolist() == pytest.approx([2.0 / 3.0, 1.0, 1.0 / 3.0])


def test_explicit_hard_mask_excludes_infeasible_highest_score():
    observation = _observation(
        gamma=(0.0, 100.0, 5.0),
        intensity=(2.0, 0.0, 1.0),
        flow=(2.0, 0.0, 1.0),
        feasible=(True, False, True),
    )
    policy = FixedRankPolicy(FixedRankPolicyConfig(hard_feasibility_mask=True))
    scores = policy.score(observation)
    action = policy(observation)

    assert scores.eligible.tolist() == [True, False, True]
    assert torch.isneginf(scores.total[1])
    assert action.requested_serving.tolist() == [2]


def test_empty_feasible_set_preserves_current_association():
    observation = _observation(
        edges=((0, 0), (0, 1)),
        gamma=(2.0, 1.0),
        intensity=(0.0, 1.0),
        feasible=(False, False),
        current=(2,),
    )
    policy = FixedRankPolicy(FixedRankPolicyConfig(hard_feasibility_mask=True))
    assert policy(observation).requested_serving.tolist() == [2]


@pytest.mark.parametrize(
    ("hold", "intensity", "flow", "expected"),
    [
        (9, (1.0, 0.0), (1.0, 0.0), 0),
        # Candidate 1 leads by 0.4, below the default 1/k = 0.5 margin.
        (10, (0.0, 1.0), (1.0, 0.0), 0),
        (10, (1.0, 0.0), (1.0, 0.0), 1),
    ],
)
def test_dwell_and_per_user_hysteresis(hold, intensity, flow, expected):
    observation = _observation(
        satellite_count=2,
        edges=((0, 0), (0, 1)),
        gamma=(0.0, 1.0),
        intensity=intensity,
        flow=flow,
        feasible=(True, True),
        current=(0,),
        hold=(hold,),
    )
    policy = FixedRankPolicy(FixedRankPolicyConfig(hysteresis=0.5))
    assert policy(observation).requested_serving.tolist() == [expected]


def test_infeasible_current_does_not_bypass_dwell_or_hysteresis():
    observation = _observation(
        satellite_count=2,
        edges=((0, 0), (0, 1)),
        gamma=(100.0, 0.0),
        intensity=(0.0, 10.0),
        flow=(0.0, 10.0),
        feasible=(False, True),
        current=(0,),
        hold=(0,),
    )
    policy = FixedRankPolicy(FixedRankPolicyConfig(hard_feasibility_mask=True))
    assert policy(observation).requested_serving.tolist() == [0]


def test_unavailable_current_is_not_a_hidden_stability_constraint_exception():
    observation = _observation(
        satellite_count=3,
        edges=((0, 0), (0, 1)),
        gamma=(0.0, 1.0),
        intensity=(1.0, 0.0),
        flow=(1.0, 0.0, 2.0),
        feasible=(True, True),
        current=(2,),
        hold=(0,),
    )
    assert FixedRankPolicy()(observation).requested_serving.tolist() == [2]


def test_equal_composite_score_uses_smaller_satellite_id():
    observation = _observation(
        satellite_count=2,
        edges=((0, 1), (0, 0)),
        gamma=(0.0, 1.0),
        intensity=(0.0, 1.0),
        flow=(1.0, 0.0),
        feasible=(True, True),
    )
    # Sat 0 wins gamma while sat 1 wins load and intensity. With weights
    # (1,.4,.6), their composite scores tie exactly.
    scores = FixedRankPolicy().score(observation)
    assert scores.total[0].item() == pytest.approx(scores.total[1].item())
    assert FixedRankPolicy()(observation).requested_serving.tolist() == [0]


def test_full_geometric_ranking_is_the_standalone_default():
    observation = _observation(
        gamma=(0.0, 100.0, 5.0),
        intensity=(2.0, 0.0, 1.0),
        flow=(2.0, 0.0, 1.0),
        feasible=(True, False, True),
    )
    policy = FixedRankPolicy()
    assert policy(observation).requested_serving.tolist() == [1]


def test_nonpolicy_metadata_cannot_change_the_fixed_candidate_set():
    observation = _observation(
        gamma=(0.0, 100.0, 5.0),
        intensity=(2.0, 0.0, 1.0),
        flow=(2.0, 0.0, 1.0),
    )
    observation.meta["policy_available_edge"] = torch.tensor([True, False, True])
    action = FixedRankPolicy()(observation)
    assert action.requested_serving.tolist() == [1]
