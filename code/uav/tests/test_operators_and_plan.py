from __future__ import annotations

import torch

from uav_cfs.experiments import UAVAblation, paper_task_matrix
from uav_cfs.operators import build_message_operator, project_onto_l1_ball
from uav_cfs.perturbations import UAVStructuralPerturbation


def _operator(kind: str, gain_control: bool = True):
    return build_message_operator(
        kind,
        input_dim=384,
        output_dim=128,
        dropout=0.1,
        kan_knots=16,
        physick_latent_dim=128,
        physick_descriptor_dim=16,
        physick_kernel_count=16,
        physick_kernel_hidden_dim=128,
        physick_coefficient_hidden_dim=128,
        physick_projection_radius=1.0,
        physick_operating_clip=5.0,
        physick_gain_control=gain_control,
    )


def test_all_operator_variants_preserve_message_shape() -> None:
    edge_state = torch.randn(7, 384)
    for kind in ("mlp", "kan", "physick"):
        assert _operator(kind)(edge_state).shape == (7, 128)


def test_signed_l1_projection_enforces_radius() -> None:
    values = torch.tensor([[4.0, -3.0, 2.0]])
    projected = project_onto_l1_ball(values, 1.0)
    assert float(projected.abs().sum()) <= 1.0 + 1e-6


def test_structural_perturbations_require_explicit_magnitudes() -> None:
    try:
        UAVStructuralPerturbation(kind="service_time_inflation")
    except ValueError:
        pass
    else:
        raise AssertionError("service-time inflation accepted a hidden default")


def test_paper_matrix_contains_all_nested_ablation_rows() -> None:
    names = {task.ablation for task in paper_task_matrix() if task.family == "component_removal"}
    assert names == {item.value for item in UAVAblation}

