from pathlib import Path

import pytest
import torch
import yaml

from leo_pg.kernels.interface import MessageFunction
from leo_pg.paper.extended_experiments import (
    CheckpointLoss,
    EXP3_ARMS,
    EXP5Variant,
    aba_ratio_of_rate_ratios,
    build_exp5_operator,
    coefficient_diagnostics,
    decision_aware_combined_loss,
    evaluate_equivalence_gate,
    expansive_jacobian_fraction,
    near_tie_cascade,
    operator_jump_p95,
    paired_ratio_equivalence,
    select_checkpoint_pair,
)


def _checkpoint(identifier, epoch, loss):
    return CheckpointLoss(identifier, epoch, loss, {"flow": loss, "stage": loss})


def test_minimum_and_loss_matched_checkpoint_selection_are_deterministic():
    reference = [_checkpoint("r0", 0, 1.00), _checkpoint("r1", 1, 1.02)]
    comparison = [_checkpoint("c0", 0, 1.04), _checkpoint("c1", 1, 1.01)]

    minimum = select_checkpoint_pair(reference, comparison, "minimum_validation_loss")
    assert (minimum.reference.checkpoint_id, minimum.comparison.checkpoint_id) == (
        "r0",
        "c1",
    )

    matched = select_checkpoint_pair(reference, comparison, "loss_matched_0.025")
    assert (matched.reference.checkpoint_id, matched.comparison.checkpoint_id) == (
        "r1",
        "c1",
    )
    assert matched.aggregate_loss_ratio == pytest.approx(1.01 / 1.02)


def test_selection_records_cannot_carry_controller_outcomes():
    with pytest.raises(TypeError):
        CheckpointLoss("r0", 0, 1.0, {"flow": 1.0}, aba_rate=0.1)


def test_paired_log_ratio_tost_and_field_gate():
    passed = paired_ratio_equivalence(
        [1.0] * 20, [1.02] * 20, band=(0.95, 1.05)
    )
    failed = paired_ratio_equivalence(
        [1.0] * 20, [1.12] * 20, band=(0.95, 1.05)
    )
    assert passed.equivalent
    assert passed.geometric_mean_ratio == pytest.approx(1.02)
    assert not failed.equivalent

    gate = evaluate_equivalence_gate(
        [1.0] * 20,
        [1.02] * 20,
        {"flow": [2.0] * 20, "stage": [3.0] * 20},
        {"flow": [2.04] * 20, "stage": [3.12] * 20},
    )
    assert gate.passed
    assert set(gate.fields) == {"flow", "stage"}


def test_near_tie_cascade_uses_one_support_denominator_and_enforces_nesting():
    result = near_tie_cascade(
        [True, True, True, False],
        [True, True, False, False],
        [True, False, False, False],
        [True, False, False, False],
    )
    assert result.supported_count == 3
    assert result.ranking_disagreement_rate == pytest.approx(2 / 3)
    assert result.proposal_disagreement_rate == pytest.approx(1 / 3)
    assert result.executed_disagreement_rate == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="without proposal"):
        near_tie_cascade([True], [True], [False], [True])


def test_exp3_arm_registry_and_decision_aware_loss():
    assert [(arm.arm_id, arm.operator.value, arm.objective.value) for arm in EXP3_ARMS] == [
        ("A1", "mlp", "predictive"),
        ("A2", "physick", "predictive"),
        ("B1", "mlp", "decision_aware"),
        ("B2", "physick", "decision_aware"),
    ]
    predicted = torch.tensor([0.1, 0.3, -0.2, 0.0], requires_grad=True)
    oracle = torch.tensor([3.0, 2.0, 1.0, 5.0])
    breakdown = decision_aware_combined_loss(
        torch.tensor(2.0),
        predicted,
        oracle,
        group_ids=torch.tensor([0, 0, 0, 1]),
    )
    assert breakdown.ordered_pair_count == 3
    assert breakdown.group_count == 1
    assert breakdown.total > breakdown.field
    breakdown.total.backward()
    assert torch.isfinite(predicted.grad).all()


def test_decision_aware_loss_has_exact_zero_ranking_terms_without_pairs():
    predicted = torch.tensor([0.2, 0.1], requires_grad=True)
    result = decision_aware_combined_loss(
        torch.tensor(1.5),
        predicted,
        torch.tensor([1.0, 1.0]),
        group_ids=torch.tensor([0, 1]),
    )
    assert result.pairwise_logistic.item() == 0.0
    assert result.listmle_top5.item() == 0.0
    assert result.margin.item() == 0.0
    assert result.total.item() == pytest.approx(1.5)


def test_exp3_aba_ratio_of_rate_ratios_uses_all_four_corrected_counts():
    observed = aba_ratio_of_rate_ratios(9, 4, 19, 4)
    expected = ((4 + 0.5) / (19 + 0.5)) / ((4 + 0.5) / (9 + 0.5))
    assert observed == pytest.approx(expected)
    assert aba_ratio_of_rate_ratios(0, 0, 0, 0) == pytest.approx(1.0)


@pytest.mark.parametrize("variant", list(EXP5Variant))
def test_exp5_factory_builds_every_registered_operator(variant):
    torch.manual_seed(7)
    operator = build_exp5_operator(
        variant,
        mem_dim=3,
        edge_dim=2,
        msg_dim=4,
        num_kernels=3,
        hidden_dim=8,
        descriptor_dim=3,
        dropout=0.0,
        projection_radius=1.25,
        use_edge_type=False,
    )
    inputs = (torch.randn(5, 3), torch.randn(5, 3), torch.randn(5, 2))
    output = operator(*inputs)
    assert output.shape == (5, 4)
    assert torch.isfinite(output).all()

    diagnostics = coefficient_diagnostics(operator, *inputs)
    if variant is EXP5Variant.MLP:
        assert not diagnostics.available
    else:
        assert diagnostics.available
    if variant is EXP5Variant.PROJECTED_PHYSICK:
        _, effective = operator.coefficient_tensors(*inputs)
        assert torch.all(effective.abs().sum(dim=-1) <= 1.25 + 1e-6)
    if variant is EXP5Variant.GENERIC_DYNAMIC_MIXTURE:
        _, effective = operator.coefficient_tensors(*inputs)
        assert torch.allclose(effective.sum(dim=-1), torch.ones(5))


class _LinearEdgeMessage(MessageFunction):
    def forward(self, mem_src, mem_dst, z_ij, edge_type=None):
        return 2.0 * z_ij[:, :1]


def test_exp5_jump_and_jacobian_diagnostics_have_operational_definitions():
    operator = _LinearEdgeMessage()
    base = (torch.zeros(4, 1), torch.zeros(4, 1), torch.zeros(4, 1))
    perturbed = (torch.zeros(4, 1), torch.zeros(4, 1), torch.full((4, 1), 0.1))
    assert operator_jump_p95(operator, base, perturbed) == pytest.approx(0.2)
    assert expansive_jacobian_fraction(operator, *base, threshold=1.0) == 1.0
    assert expansive_jacobian_fraction(operator, *base, threshold=2.0) == 0.0


def test_extended_yaml_files_match_the_registered_protocol():
    config_root = Path(__file__).resolve().parents[3] / "configs" / "extended"
    loaded = {
        name: yaml.safe_load((config_root / f"{name}.yaml").read_text())
        for name in ("exp1", "exp2", "exp3", "exp5")
    }
    assert loaded["exp1"]["checkpoint_selection"]["outcome_blind"] is True
    assert loaded["exp1"]["aba_windows_s"] == [2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
    assert loaded["exp2"]["cells"]["C8"] == {
        "descriptor": "D1",
        "staging": "Stage1",
        "score_map": "S1",
    }
    assert loaded["exp3"]["decision_aware_objective"]["margin"] == 0.05
    assert loaded["exp5"]["operator"]["projected_variant_radius"] == 1.25


def test_extended_protocol_is_available_from_the_stable_paper_api():
    import leo_pg.paper as paper

    assert paper.EXP2_CELLS[0].cell_id == "C1"
    assert paper.EXP3_ARMS[-1].arm_id == "B2"
    assert paper.EXP5Variant.PROJECTED_PHYSICK.value == "projected_physick"
    assert callable(paper.select_checkpoint_pair)
