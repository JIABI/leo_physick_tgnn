import pytest
import torch

from leo_pg.models.heads import IntensityFlowOutput
from leo_pg.sim.state import ControlObservation, PolicyDescriptors, SimulatorDescriptors
from leo_pg.train.intensity_flow import (
    IntensityFlowLossWeights,
    build_next_step_target,
    intensity_flow_one_step_loss,
)


def _observation(epoch, edge_ids, gamma, intensity, feasible, flow):
    edge_ids = torch.tensor(edge_ids, dtype=torch.long).reshape(-1, 2)
    fields = PolicyDescriptors(
        gamma_edge=torch.tensor(gamma, dtype=torch.float32),
        intensity_edge=torch.tensor(intensity, dtype=torch.float32),
        flow_node=torch.tensor(flow, dtype=torch.float32),
    )
    observation = ControlObservation(
        observation_id=(5, epoch),
        node_x=torch.zeros(4, 3),
        candidate_edge_index=torch.stack((edge_ids[:, 0], edge_ids[:, 1] + 2)),
        candidate_edge_ids=edge_ids,
        edge_features=torch.zeros(len(edge_ids), 1),
        elevation_deg=torch.full((len(edge_ids),), 30.0),
        sim_descriptors=SimulatorDescriptors(
            policy_fields=fields.clone(),
            feasible_edge=torch.tensor(feasible, dtype=torch.bool),
        ),
        policy_descriptors=fields.clone(),
        current_serving=torch.tensor([0, 1]),
        hold_steps=torch.tensor([10, 10]),
        user_order=torch.tensor([0, 1]),
    )
    observation.validate()
    return observation


def test_next_step_target_joins_by_edge_identity_and_masks_churn():
    source = _observation(
        0,
        [(0, 0), (0, 1), (1, 1)],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [True, True, True],
        [0.0, 0.0],
    )
    next_observation = _observation(
        1,
        [(1, 1), (0, 0), (1, 0)],
        [30.0, 10.0, 40.0],
        [3.0, 1.0, 4.0],
        [False, True, True],
        [0.2, 0.8],
    )
    target = build_next_step_target(source, next_observation)
    assert target.persistent_edge.tolist() == [True, False, True]
    assert target.gamma_edge.tolist() == [10.0, 0.0, 30.0]
    assert target.log1p_intensity_edge.tolist() == pytest.approx(
        [torch.log(torch.tensor(2.0)).item(), 0.0, torch.log(torch.tensor(4.0)).item()]
    )
    assert target.feasibility_edge.tolist() == [True, False, False]
    assert torch.equal(target.flow_node, torch.tensor([0.2, 0.8]))


def test_multitask_loss_uses_log_intensity_and_shared_persistent_mask():
    source = _observation(
        0,
        [(0, 0), (0, 1)],
        [0.0, 0.0],
        [0.0, 0.0],
        [True, True],
        [0.0, 0.0],
    )
    next_observation = _observation(
        1,
        [(0, 0), (1, 1)],
        [2.0, 99.0],
        [3.0, 99.0],
        [True, False],
        [0.25, 0.75],
    )
    target = build_next_step_target(source, next_observation)
    prediction = IntensityFlowOutput(
        policy_descriptors=PolicyDescriptors(
            gamma_edge=torch.tensor([1.0, 1000.0], requires_grad=True),
            intensity_edge=torch.tensor([1.0, 1000.0], requires_grad=True),
            flow_node=torch.tensor([0.0, 1.0], requires_grad=True),
        ),
        feasibility_logit=torch.tensor([0.0, -1000.0], requires_grad=True),
    )
    loss = intensity_flow_one_step_loss(
        prediction,
        target,
        IntensityFlowLossWeights(
            gamma=1.0,
            intensity=1.0,
            flow=1.0,
            feasibility=1.0,
        ),
    )
    assert loss.persistent_edge_count == 1
    assert loss.gamma.item() == pytest.approx(1.0)
    assert loss.intensity.item() == pytest.approx(
        (torch.log(torch.tensor(2.0)) - torch.log(torch.tensor(4.0))).square().item()
    )
    assert loss.flow.item() == pytest.approx(0.0625)
    assert loss.feasibility.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    loss.total.backward()
    assert prediction.policy_descriptors.gamma_edge.grad[1].item() == 0.0
    assert prediction.feasibility_logit.grad[1].item() == 0.0


def test_no_persistent_edges_yields_zero_edge_losses_but_keeps_flow_loss():
    source = _observation(
        0, [(0, 0)], [0.0], [0.0], [True], [0.0, 0.0]
    )
    next_observation = _observation(
        1, [(1, 1)], [1.0], [1.0], [True], [0.5, 0.5]
    )
    target = build_next_step_target(source, next_observation)
    prediction = IntensityFlowOutput(
        policy_descriptors=PolicyDescriptors(
            gamma_edge=torch.tensor([10.0], requires_grad=True),
            intensity_edge=torch.tensor([10.0], requires_grad=True),
            flow_node=torch.zeros(2, requires_grad=True),
        ),
        feasibility_logit=torch.tensor([10.0], requires_grad=True),
    )
    loss = intensity_flow_one_step_loss(
        prediction,
        target,
        IntensityFlowLossWeights(1.0, 1.0, 1.0, 1.0),
    )
    assert loss.persistent_edge_count == 0
    assert loss.gamma.item() == 0.0
    assert loss.intensity.item() == 0.0
    assert loss.feasibility.item() == 0.0
    assert loss.flow.item() == pytest.approx(0.25)
