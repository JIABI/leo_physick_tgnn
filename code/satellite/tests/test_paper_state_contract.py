import pytest
import torch

from leo_pg.sim.state import (
    ControlObservation,
    PAPER_EDGE_FEATURE_NAMES,
    PAPER_FEATURE_CONTRACT_VERSION,
    PAPER_NODE_FEATURE_NAMES,
    PolicyDescriptors,
    SimulatorDescriptors,
)


def _observation() -> ControlObservation:
    policy = PolicyDescriptors(
        gamma_edge=torch.tensor([2.0]),
        intensity_edge=torch.tensor([0.4]),
        flow_node=torch.tensor([0.2]),
    )
    observation = ControlObservation(
        observation_id=(11, 3),
        node_x=torch.zeros(2, 7),
        candidate_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        candidate_edge_ids=torch.tensor([[0, 0]], dtype=torch.long),
        edge_features=torch.zeros(1, 6),
        elevation_deg=torch.tensor([30.0]),
        sim_descriptors=SimulatorDescriptors(policy.clone(), torch.tensor([True])),
        policy_descriptors=policy.clone(),
        current_serving=torch.tensor([0]),
        hold_steps=torch.tensor([10]),
        user_order=torch.tensor([0]),
    )
    observation.validate()
    return observation


def test_policy_replacement_has_no_alias_or_simulator_writeback():
    original = _observation()
    replacement = PolicyDescriptors(
        gamma_edge=torch.tensor([-2.0]),
        intensity_edge=torch.tensor([1.4]),
        flow_node=torch.tensor([0.8]),
    )
    replaced = original.with_policy_descriptors(replacement)
    replacement.gamma_edge[0] = 999.0
    replaced.policy_descriptors.flow_node[0] = 0.9
    assert original.policy_descriptors.gamma_edge.item() == 2.0
    assert original.sim_descriptors.policy_fields.gamma_edge.item() == 2.0
    assert replaced.policy_descriptors.gamma_edge.item() == -2.0
    assert torch.allclose(
        replaced.sim_descriptors.policy_fields.flow_node,
        torch.tensor([0.2]),
    )


def test_model_step_never_contains_supervision_target():
    step = _observation().as_model_step()
    assert "y" not in step
    assert step["meta"]["observation_id"] == (11, 3)


def test_model_step_materializes_policy_copy_in_paper_feature_contract():
    observation = _observation()
    observation.node_x = torch.zeros((2, len(PAPER_NODE_FEATURE_NAMES)))
    observation.edge_features = torch.zeros((1, len(PAPER_EDGE_FEATURE_NAMES)))
    observation.meta.update(
        {
            "feature_contract_version": PAPER_FEATURE_CONTRACT_VERSION,
            "node_flow_column": 6,
            "edge_gamma_column": 2,
            "edge_flow_column": 3,
            "edge_intensity_column": 4,
            "gamma_feature_scale": 10.0,
        }
    )
    replacement = PolicyDescriptors(
        gamma_edge=torch.tensor([20.0]),
        intensity_edge=torch.expm1(torch.tensor([2.0])),
        flow_node=torch.tensor([0.75]),
    )
    step = observation.with_policy_descriptors(replacement).as_model_step()
    assert step["node_x"][1, 6].item() == pytest.approx(0.75)
    assert step["edge_z"][0, 2].item() == pytest.approx(2.0)
    assert step["edge_z"][0, 3].item() == pytest.approx(0.75)
    assert step["edge_z"][0, 4].item() == pytest.approx(2.0)
    assert torch.equal(observation.node_x, torch.zeros_like(observation.node_x))
    assert torch.equal(observation.edge_features, torch.zeros_like(observation.edge_features))
