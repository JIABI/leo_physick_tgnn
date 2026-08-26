import pytest
import torch

from leo_pg.paper.crossover import (
    ConstantIntensityFlowCrossoverInitializer,
    ConstantSnapshotCrossoverInitializer,
    CrossoverCell,
    CrossoverDecisionContext,
    CrossoverEpochInput,
    CrossoverPhase,
    CrossoverRunner,
    CrossoverRuntime,
    DescriptorFrameOrigin,
    DescriptorCode,
    EXP2_CELLS,
    ExplicitCrossoverScoreAdapter,
    IntensityFlowDescriptorFrame,
    MissingScoreBindingError,
    ScoreMapCode,
    SnapshotDescriptorFrame,
    StagingCode,
    crossover_cell,
    documented_reference_bindings,
    register_exp2_runs,
    stage_crossover_descriptors,
    validate_exp2_cells,
)
from leo_pg.paper.snapshot import SnapshotOutput
from leo_pg.sim.state import PolicyDescriptors


def test_exp2_registry_is_the_frozen_complete_eight_cell_design():
    observed = {
        cell.cell_id: (
            cell.descriptor.value,
            cell.staging.value,
            cell.score_map.value,
        )
        for cell in EXP2_CELLS
    }
    assert observed == {
        "C1": ("D0", "Stage0", "S0"),
        "C2": ("D0", "Stage0", "S1"),
        "C3": ("D0", "Stage1", "S0"),
        "C4": ("D0", "Stage1", "S1"),
        "C5": ("D1", "Stage0", "S0"),
        "C6": ("D1", "Stage0", "S1"),
        "C7": ("D1", "Stage1", "S0"),
        "C8": ("D1", "Stage1", "S1"),
    }
    runs = register_exp2_runs()
    assert len(runs) == 16
    assert len({run.run_spec_id for run in runs}) == 16
    assert all(run.matched_seed_blocks == 10 for run in runs)
    assert all(run.held_out_episodes_per_block == 30 for run in runs)


def test_exp2_registry_rejects_a_semantically_duplicated_cell():
    duplicated = EXP2_CELLS[:-1] + (
        CrossoverCell("C8", DescriptorCode.D0, StagingCode.STAGE0, ScoreMapCode.S0),
    )
    with pytest.raises(ValueError, match="complete"):
        validate_exp2_cells(duplicated)


def test_crossover_runtime_keeps_the_three_factors_separate():
    runtime = CrossoverRuntime(
        descriptor_builders={
            DescriptorCode.D0: lambda context: f"snapshot({context})",
            DescriptorCode.D1: lambda context: f"if({context})",
        },
        staging_functions={
            StagingCode.STAGE0: lambda value, history: f"current:{value}",
            StagingCode.STAGE1: lambda value, history: f"staged:{history}:{value}",
        },
        score_functions={
            ScoreMapCode.S0: lambda value: f"static:{value}",
            ScoreMapCode.S1: lambda value: f"hazard:{value}",
        },
    )
    assert runtime.evaluate("C1", "x", "h") == "static:current:snapshot(x)"
    assert runtime.evaluate(crossover_cell("c8"), "x", "h") == "hazard:staged:h:if(x)"


def test_crossover_runtime_requires_every_registered_factor_level():
    with pytest.raises(ValueError, match="descriptor_builders"):
        CrossoverRuntime(
            descriptor_builders={DescriptorCode.D0: lambda value: value},
            staging_functions={level: lambda value, history: value for level in StagingCode},
            score_functions={level: lambda value: value for level in ScoreMapCode},
        )


def _typed_epoch() -> CrossoverEpochInput:
    context = CrossoverDecisionContext(
        observation_id=(7, 1),
        candidate_edge_ids=torch.tensor(
            [[0, 1], [0, 2], [1, 0], [1, 2]], dtype=torch.long
        ),
        feasible_edge=torch.tensor([True, True, True, False]),
        current_serving=torch.tensor([1, 0], dtype=torch.long),
        hold_steps=torch.tensor([10, 10], dtype=torch.long),
        user_order=torch.tensor([0, 1], dtype=torch.long),
        satellite_count=3,
    )
    current_ids = context.candidate_edge_ids.clone()
    source_ids = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 2]], dtype=torch.long
    )
    return CrossoverEpochInput(
        phase=CrossoverPhase.ORACLE_DRIVEN,
        context=context,
        current_snapshot=SnapshotDescriptorFrame(
            candidate_edge_ids=current_ids,
            descriptors=SnapshotOutput(
                gamma_edge=torch.tensor([1.0, 3.0, 2.0, 8.0]),
                feasibility_margin_edge=torch.tensor([0.2, -0.1, 0.4, 0.9]),
                admitted_load_node=torch.tensor([0.1, 0.2, 0.4]),
            ),
            produced_at=(7, 1),
            intended_for=(7, 1),
            origin=DescriptorFrameOrigin.CURRENT_EPOCH,
            source_id="oracle_snapshot_epoch_1",
        ),
        current_intensity_flow=IntensityFlowDescriptorFrame(
            candidate_edge_ids=current_ids,
            descriptors=PolicyDescriptors(
                gamma_edge=torch.tensor([1.0, 3.0, 2.0, 8.0]),
                intensity_edge=torch.tensor([0.2, 0.6, 0.1, 0.9]),
                flow_node=torch.tensor([0.1, 0.2, 0.4]),
            ),
            produced_at=(7, 1),
            intended_for=(7, 1),
            origin=DescriptorFrameOrigin.CURRENT_EPOCH,
            source_id="oracle_intensity_flow_epoch_1",
        ),
        causal_snapshot=SnapshotDescriptorFrame(
            candidate_edge_ids=source_ids,
            descriptors=SnapshotOutput(
                gamma_edge=torch.tensor([10.0, 20.0, 30.0, 40.0]),
                feasibility_margin_edge=torch.tensor([1.0, 2.0, 3.0, 4.0]),
                admitted_load_node=torch.tensor([0.3, 0.2, 0.1]),
            ),
            produced_at=(7, 0),
            intended_for=(7, 1),
            origin=DescriptorFrameOrigin.CAUSAL_PREDICTION,
            source_id="frozen_snapshot_prediction_epoch_0_to_1",
        ),
        causal_intensity_flow=IntensityFlowDescriptorFrame(
            candidate_edge_ids=source_ids,
            descriptors=PolicyDescriptors(
                gamma_edge=torch.tensor([10.0, 20.0, 30.0, 40.0]),
                intensity_edge=torch.tensor([1.0, 2.0, 3.0, 4.0]),
                flow_node=torch.tensor([0.3, 0.2, 0.1]),
            ),
            produced_at=(7, 0),
            intended_for=(7, 1),
            origin=DescriptorFrameOrigin.CAUSAL_PREDICTION,
            source_id="frozen_intensity_flow_prediction_epoch_0_to_1",
        ),
        snapshot_initializer=ConstantSnapshotCrossoverInitializer(
            gamma=-3.0,
            feasibility_margin=0.0,
            admitted_load=0.5,
        ),
        intensity_flow_initializer=ConstantIntensityFlowCrossoverInitializer(
            gamma=-3.0,
            intensity=0.1,
            flow=0.5,
        ),
    )


def test_stage1_joins_persistent_identities_and_initializes_only_new_edges():
    epoch = _typed_epoch()
    snapshot = stage_crossover_descriptors(epoch, "C3")
    assert isinstance(snapshot.values, SnapshotOutput)
    assert snapshot.values.gamma_edge.tolist() == [20.0, -3.0, 30.0, 40.0]
    assert snapshot.values.feasibility_margin_edge.tolist() == [2.0, 0.0, 3.0, 4.0]
    assert snapshot.values.admitted_load_node.tolist() == pytest.approx([0.3, 0.2, 0.1])
    assert snapshot.initialized_edge.tolist() == [False, True, False, False]

    intensity_flow = stage_crossover_descriptors(epoch, "C8")
    assert isinstance(intensity_flow.values, PolicyDescriptors)
    assert intensity_flow.values.gamma_edge.tolist() == [20.0, -3.0, 30.0, 40.0]
    assert intensity_flow.values.intensity_edge.tolist() == pytest.approx(
        [2.0, 0.1, 3.0, 4.0]
    )
    assert intensity_flow.values.flow_node.tolist() == pytest.approx([0.3, 0.2, 0.1])


def test_documented_native_bindings_execute_exact_rank_equations_after_hard_mask():
    epoch = _typed_epoch()
    runner = CrossoverRunner()

    snapshot = runner.run_epoch("C1", epoch)
    assert snapshot.scores.binding_id == "D0_S0_snapshot_static_equation_v1"
    assert snapshot.scores.total.tolist()[:3] == pytest.approx([1.7, 1.6, 2.2])
    assert torch.isneginf(snapshot.scores.total[3])
    assert snapshot.action.requested_serving.tolist() == [1, 0]

    intensity_flow = runner.run_epoch("C6", epoch)
    assert intensity_flow.scores.binding_id == "D1_S1_intensity_flow_hazard_equation_v1"
    assert intensity_flow.scores.eligible.tolist() == [True, True, True, False]
    assert torch.isneginf(intensity_flow.scores.total[3])


def test_s1_x_d0_fails_closed_until_author_supplies_field_mapping():
    epoch = _typed_epoch()
    with pytest.raises(MissingScoreBindingError, match="v4 EXP2 source"):
        CrossoverRunner().run_epoch("C2", epoch)


def test_explicit_adapter_makes_the_unrecorded_mapping_auditable_and_executable():
    epoch = _typed_epoch()

    def author_d0_s1(context, staged):
        assert isinstance(staged.values, SnapshotOutput)
        return staged.values.gamma_edge.clone()

    adapter = ExplicitCrossoverScoreAdapter(
        descriptor_code=DescriptorCode.D0,
        score_map_code=ScoreMapCode.S1,
        binding_id="AUTHOR_EXP2_D0_S1_MAPPING_V1",
        evidence_basis="author-supplied dated EXP2 field-definition record",
        score_function=author_d0_s1,
    )
    result = CrossoverRunner(documented_reference_bindings(adapter)).run_epoch(
        "C2", epoch
    )
    assert result.scores.binding_id == "AUTHOR_EXP2_D0_S1_MAPPING_V1"
    assert result.action.requested_serving.tolist() == [2, 0]
    assert torch.isneginf(result.scores.total[3])


def test_all_eight_cells_run_only_when_both_cross_domain_maps_are_explicit():
    epoch = _typed_epoch()

    def gamma_only(context, staged):
        return staged.values.gamma_edge.to(dtype=torch.float32).clone()

    adapters = (
        ExplicitCrossoverScoreAdapter(
            descriptor_code=DescriptorCode.D0,
            score_map_code=ScoreMapCode.S1,
            binding_id="AUTHOR_D0_S1",
            evidence_basis="author field ledger",
            score_function=gamma_only,
        ),
        ExplicitCrossoverScoreAdapter(
            descriptor_code=DescriptorCode.D1,
            score_map_code=ScoreMapCode.S0,
            binding_id="AUTHOR_D1_S0",
            evidence_basis="author field ledger",
            score_function=gamma_only,
        ),
    )
    registry = documented_reference_bindings(*adapters)
    assert registry.missing_pairs == ()
    results = [CrossoverRunner(registry).run_epoch(cell, epoch) for cell in EXP2_CELLS]
    assert [result.cell.cell_id for result in results] == [f"C{i}" for i in range(1, 9)]
    assert all(result.action.observation_id == (7, 1) for result in results)
