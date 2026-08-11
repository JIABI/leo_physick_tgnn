import torch

from leo_pg.kernels.physick.projection import project_onto_l1_ball


def test_l1_projection_is_signed_and_bounded():
    weights = torch.tensor([[1.0, -1.0, 0.2], [0.1, -0.2, 0.3]], requires_grad=True)
    projected = project_onto_l1_ball(weights, radius=1.0)
    assert torch.all(projected.abs().sum(dim=-1) <= 1.0 + 1e-6)
    assert torch.equal(projected[1], weights[1])
    assert projected[0, 0] > 0
    assert projected[0, 1] < 0

    projected.square().sum().backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_l1_projection_zero_radius():
    weights = torch.randn(4, 16)
    assert torch.equal(project_onto_l1_ball(weights, radius=0.0), torch.zeros_like(weights))
