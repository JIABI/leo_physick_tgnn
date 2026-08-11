import pytest
import torch

from leo_pg.eval.substitution import (
    SubstitutionMode,
    apply_substitution,
    carry_policy_stream,
    compose_policy_descriptors,
)
from leo_pg.sim.state import (
    ControlObservation,
    PolicyDescriptors,
    SimulatorDescriptors,
)


def _descriptors(offset):
    return PolicyDescriptors(
        gamma_edge=torch.tensor([offset + 1.0, offset + 2.0]),
        intensity_edge=torch.tensor([offset + 3.0, offset + 4.0]),
        flow_node=torch.tensor([offset + 5.0, offset + 6.0]),
    )


def _observation():
    oracle = _descriptors(0.0)
    observation = ControlObservation(
        observation_id=(11, 2),
        node_x=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        candidate_edge_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        candidate_edge_ids=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        edge_features=torch.tensor([[1.0], [2.0]]),
        elevation_deg=torch.tensor([30.0, 40.0]),
        sim_descriptors=SimulatorDescriptors(
            policy_fields=oracle,
            feasible_edge=torch.tensor([True, False]),
        ),
        policy_descriptors=_descriptors(20.0),
        current_serving=torch.tensor([0, 1], dtype=torch.long),
        hold_steps=torch.tensor([10, 12], dtype=torch.long),
        user_order=torch.tensor([1, 0], dtype=torch.long),
        meta={"tag": "kept"},
    )
    observation.validate()
    return observation


@pytest.mark.parametrize(
    ("mode", "expected_offsets"),
    [
        ("model", (100.0, 100.0, 100.0)),
        ("oracle", (0.0, 0.0, 0.0)),
        ("gamma", (0.0, 100.0, 100.0)),
        ("intensity", (100.0, 0.0, 100.0)),
        ("flow", (100.0, 100.0, 0.0)),
        ("intensity_flow", (100.0, 0.0, 0.0)),
    ],
)
def test_substitution_modes_are_partial_oracle_replacements(mode, expected_offsets):
    observation = _observation()
    result = apply_substitution(observation, _descriptors(100.0), mode)
    expected = PolicyDescriptors(
        gamma_edge=_descriptors(expected_offsets[0]).gamma_edge,
        intensity_edge=_descriptors(expected_offsets[1]).intensity_edge,
        flow_node=_descriptors(expected_offsets[2]).flow_node,
    )

    assert torch.equal(result.policy_descriptors.gamma_edge, expected.gamma_edge)
    assert torch.equal(result.policy_descriptors.intensity_edge, expected.intensity_edge)
    assert torch.equal(result.policy_descriptors.flow_node, expected.flow_node)


def test_substitution_changes_only_policy_copy_and_does_not_alias_sources():
    observation = _observation()
    model = _descriptors(100.0)
    original_node_x = observation.node_x.clone()
    original_sim = observation.sim_descriptors.clone()
    result = apply_substitution(
        observation,
        model,
        SubstitutionMode.INTENSITY_FLOW,
    )

    for result_value in result.policy_descriptors.as_dict().values():
        for source_value in (
            *model.as_dict().values(),
            *observation.sim_descriptors.policy_fields.as_dict().values(),
            *observation.policy_descriptors.as_dict().values(),
        ):
            assert result_value.data_ptr() != source_value.data_ptr()

    result.policy_descriptors.gamma_edge.add_(1000.0)
    result.policy_descriptors.intensity_edge.add_(1000.0)
    result.policy_descriptors.flow_node.add_(1000.0)
    assert torch.equal(observation.node_x, original_node_x)
    assert torch.equal(
        observation.sim_descriptors.policy_fields.gamma_edge,
        original_sim.policy_fields.gamma_edge,
    )
    assert torch.equal(
        observation.sim_descriptors.policy_fields.intensity_edge,
        original_sim.policy_fields.intensity_edge,
    )
    assert torch.equal(
        observation.sim_descriptors.policy_fields.flow_node,
        original_sim.policy_fields.flow_node,
    )
    assert torch.equal(result.candidate_edge_ids, observation.candidate_edge_ids)
    assert torch.equal(result.current_serving, observation.current_serving)
    assert result.meta == observation.meta


def test_oracle_mode_does_not_require_model_descriptors():
    observation = _observation()
    result = apply_substitution(observation, mode="oracle")
    assert torch.equal(
        result.policy_descriptors.gamma_edge,
        observation.sim_descriptors.policy_fields.gamma_edge,
    )


def test_partial_mode_requires_model_descriptors():
    with pytest.raises(ValueError, match="requires model descriptors"):
        apply_substitution(_observation(), mode="flow")


def test_unknown_mode_and_incompatible_model_fail_explicitly():
    observation = _observation()
    with pytest.raises(ValueError, match="unknown substitution mode"):
        apply_substitution(observation, _descriptors(100.0), "not-a-mode")

    bad_model = PolicyDescriptors(
        gamma_edge=torch.tensor([1.0]),
        intensity_edge=torch.tensor([1.0]),
        flow_node=torch.tensor([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match=r"gamma_edge must have shape \[2\]"):
        apply_substitution(observation, bad_model, "model")


def test_compose_returns_fresh_tensors_even_for_full_oracle():
    oracle = _descriptors(0.0)
    composed = compose_policy_descriptors(oracle, None, "oracle")
    for result_value, source_value in zip(
        composed.as_dict().values(),
        oracle.as_dict().values(),
    ):
        assert torch.equal(result_value, source_value)
        assert result_value.data_ptr() != source_value.data_ptr()


def test_policy_stream_carry_aligns_persistent_edges_and_requires_new_edge_values():
    previous = _observation()
    current = _observation()
    current.observation_id = (previous.episode_seed, previous.epoch + 1)
    # Preserve edge (1,1), replace edge (0,0) with a newly appearing (0,1).
    current.candidate_edge_ids = torch.tensor([[0, 1], [1, 1]], dtype=torch.long)
    current.candidate_edge_index = torch.tensor([[0, 1], [3, 3]], dtype=torch.long)
    previous_policy = PolicyDescriptors(
        gamma_edge=torch.tensor([100.0, 200.0]),
        intensity_edge=torch.tensor([10.0, 20.0]),
        flow_node=torch.tensor([0.7, 0.8]),
    )
    with pytest.raises(ValueError, match="require explicit initialized descriptors"):
        carry_policy_stream(previous, previous_policy, current)
    initializer = PolicyDescriptors(
        gamma_edge=torch.tensor([-5.0, -6.0]),
        intensity_edge=torch.tensor([3.0, 4.0]),
        flow_node=torch.tensor([0.1, 0.2]),
    )
    carried = carry_policy_stream(previous, previous_policy, current, initializer)
    # The explicit initializer supplies the new edge; the persistent edge is
    # joined by identity rather than incoming row position.
    assert carried.policy_descriptors.gamma_edge.tolist() == [-5.0, 200.0]
    assert carried.policy_descriptors.intensity_edge.tolist() == [3.0, 20.0]
    assert carried.meta["policy_initialized_edge"].tolist() == [True, False]
    assert torch.equal(carried.policy_descriptors.flow_node, torch.tensor([0.7, 0.8]))
    assert torch.equal(
        carried.sim_descriptors.policy_fields.gamma_edge,
        current.sim_descriptors.policy_fields.gamma_edge,
    )
