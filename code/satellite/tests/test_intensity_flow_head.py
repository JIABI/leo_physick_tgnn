import pytest
import torch

from leo_pg.models.heads.intensity_flow import IntensityFlowHead, IntensityFlowOutput
from leo_pg.models.registry import build_model
from leo_pg.sim.state import PolicyDescriptors
from leo_pg.train.checkpoint import load_ckpt, save_ckpt


def _step(*, include_target: bool = False):
    step = {
        "t": 0,
        "node_x": torch.randn(5, 7),
        # The final edge is satellite-to-satellite and must not appear in the
        # candidate-edge descriptor output.
        "edge_index": torch.tensor([[0, 0, 1, 2], [2, 3, 4, 3]]),
        "edge_z": torch.randn(4, 7),
        "edge_type": torch.tensor([0, 0, 0, 1]),
        "meta": {"K_users": 2, "S_sats": 3},
    }
    if include_target:
        step["y"] = torch.randn(5, 1)
    return step


def _cfg(*, head_dropout: float = 0.0):
    return {
        "model": {
            "node_in_dim": 7,
            "edge_in_dim": 7,
            "mem_dim": 8,
            "msg_dim": 8,
            "emb_dim": 8,
            "message_type": "mlp",
            "aggregator": "sum",
            "dropout": 0.0,
            "use_edge_type": True,
            "edge_type_vocab": 8,
        },
        "head": {
            "type": "intensity_flow",
            "hidden_dim": 10,
            "dropout": head_dropout,
        },
    }


def test_intensity_flow_head_returns_typed_bounded_contract():
    torch.manual_seed(3)
    head = IntensityFlowHead(in_dim=8, edge_in_dim=7, hidden_dim=10, dropout=0.0)
    with torch.no_grad():
        head.gamma_head.weight.zero_()
        head.gamma_head.bias.fill_(-2.5)
        head.intensity_head.weight.zero_()
        head.intensity_head.bias.zero_()
        head.feasibility_head.weight.zero_()
        head.feasibility_head.bias.fill_(-3.0)

    output = head(torch.randn(5, 8), _step())

    assert isinstance(output, IntensityFlowOutput)
    assert isinstance(output.policy_descriptors, PolicyDescriptors)
    output.validate(edge_count=3, satellite_count=3)
    assert output.policy_descriptors.gamma_edge.shape == (3,)
    assert torch.equal(output.policy_descriptors.gamma_edge, torch.full((3,), -2.5))
    assert torch.allclose(
        output.policy_descriptors.intensity_edge,
        torch.full((3,), torch.log(torch.tensor(2.0))),
    )
    assert torch.all((0.0 <= output.policy_descriptors.flow_node) & (output.policy_descriptors.flow_node <= 1.0))
    assert output.policy_descriptors.flow_node.shape == (3,)
    assert torch.equal(output.feasibility_logit, torch.full((3,), -3.0))
    assert "feasibility_logit" not in output.policy_descriptors.as_dict()


def test_intensity_flow_edge_predictions_use_current_edge_descriptors():
    head = IntensityFlowHead(in_dim=8, edge_in_dim=7, hidden_dim=1)
    with torch.no_grad():
        edge_linear = head.edge_trunk[0]
        edge_linear.weight.zero_()
        edge_linear.bias.zero_()
        edge_linear.weight[0, 16] = 1.0  # first edge_z coordinate
        head.gamma_head.weight.fill_(1.0)
        head.gamma_head.bias.zero_()

    embedding = torch.zeros(5, 8)
    low = _step()
    low["edge_z"].zero_()
    high = _step()
    high["edge_z"].zero_()
    high["edge_z"][:, 0] = 2.0

    gamma_low = head(embedding, low).policy_descriptors.gamma_edge
    gamma_high = head(embedding, high).policy_descriptors.gamma_edge
    assert torch.equal(gamma_low, torch.zeros(3))
    assert torch.equal(gamma_high, torch.full((3,), 2.0))


def test_intensity_flow_head_is_differentiable_and_supports_empty_candidates():
    head = IntensityFlowHead(in_dim=8, edge_in_dim=7, hidden_dim=10)
    embedding = torch.randn(5, 8, requires_grad=True)
    output = head(embedding, _step())
    values = output.as_dict()
    loss = sum(value.sum() for value in values.values())
    loss.backward()
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )

    empty_step = _step()
    empty_step["edge_type"] = torch.ones(4, dtype=torch.long)
    empty = head(torch.randn(5, 8), empty_step)
    empty.validate(edge_count=0, satellite_count=3)


def test_tgn_predict_step_does_not_require_supervision_target():
    model = build_model(_cfg())
    prediction, memory = model.predict_step(_step(include_target=False), None, torch.device("cpu"))
    assert isinstance(prediction, IntensityFlowOutput)
    assert prediction.policy_descriptors.gamma_edge.shape == (3,)
    assert prediction.policy_descriptors.flow_node.shape == (3,)
    assert memory.shape == (5, 8)

    supervised_step = _step(include_target=True)
    supervised_prediction, target, _ = model.forward_step(
        supervised_step,
        None,
        torch.device("cpu"),
    )
    assert isinstance(supervised_prediction, IntensityFlowOutput)
    assert target.shape == (5, 1)


def test_intensity_flow_head_rejects_non_candidate_endpoints():
    head = IntensityFlowHead(in_dim=8, edge_in_dim=7)
    step = _step()
    step["edge_index"][1, 0] = 1
    with pytest.raises(ValueError, match="candidate edges"):
        head(torch.randn(5, 8), step)


def test_checkpoint_signature_covers_intensity_flow_semantics(tmp_path):
    path = tmp_path / "intensity_flow.pt"
    saved_cfg = _cfg(head_dropout=0.0)
    save_ckpt(str(path), model=build_model(saved_cfg), config=saved_cfg)
    load_ckpt(str(path), model=build_model(saved_cfg), map_location="cpu")

    changed_cfg = _cfg(head_dropout=0.25)
    with pytest.raises(RuntimeError, match="model signature"):
        load_ckpt(str(path), model=build_model(changed_cfg), map_location="cpu")
