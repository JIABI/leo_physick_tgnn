from __future__ import annotations

from enum import Enum

import torch

from leo_pg.sim.state import ControlObservation, PolicyDescriptors


class SubstitutionMode(str, Enum):
    """Descriptor source used by a policy-facing counterfactual.

    Partial modes describe *oracle substitutions into a model baseline*. For
    example, ``INTENSITY_FLOW`` takes gamma from the model while replacing the
    intensity and flow fields with their simulator-oracle counterparts.
    """

    MODEL = "model"
    ORACLE = "oracle"
    GAMMA = "gamma"
    INTENSITY = "intensity"
    FLOW = "flow"
    INTENSITY_FLOW = "intensity_flow"

    @classmethod
    def parse(cls, value: "SubstitutionMode | str") -> "SubstitutionMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"unknown substitution mode {value!r}; expected one of {choices}") from exc


def _check_compatible(
    oracle: PolicyDescriptors,
    model: PolicyDescriptors,
) -> None:
    oracle_fields = oracle.as_dict()
    model_fields = model.as_dict()
    for name in oracle_fields:
        oracle_value = oracle_fields[name]
        model_value = model_fields[name]
        if oracle_value.shape != model_value.shape:
            raise ValueError(
                f"model {name} shape {tuple(model_value.shape)} does not match "
                f"oracle shape {tuple(oracle_value.shape)}"
            )
        if oracle_value.device != model_value.device:
            raise ValueError(f"model and oracle {name} must be on the same device")
        if oracle_value.dtype != model_value.dtype:
            raise ValueError(f"model and oracle {name} must have the same dtype")


def compose_policy_descriptors(
    oracle_descriptors: PolicyDescriptors,
    model_descriptors: PolicyDescriptors | None,
    mode: SubstitutionMode | str = SubstitutionMode.MODEL,
) -> PolicyDescriptors:
    """Compose an independent policy descriptor copy for one eval mode.

    ``model`` uses all predicted fields and ``oracle`` uses all simulator
    fields. The four partial modes start from the model fields and replace the
    named field(s) with oracle values. Every returned tensor is cloned.
    """

    selected_mode = SubstitutionMode.parse(mode)
    if selected_mode is SubstitutionMode.ORACLE:
        return oracle_descriptors.clone()
    if model_descriptors is None:
        raise ValueError(f"mode {selected_mode.value!r} requires model descriptors")
    _check_compatible(oracle_descriptors, model_descriptors)

    use_oracle_gamma = selected_mode in {
        SubstitutionMode.GAMMA,
    }
    use_oracle_intensity = selected_mode in {
        SubstitutionMode.INTENSITY,
        SubstitutionMode.INTENSITY_FLOW,
    }
    use_oracle_flow = selected_mode in {
        SubstitutionMode.FLOW,
        SubstitutionMode.INTENSITY_FLOW,
    }
    return PolicyDescriptors(
        gamma_edge=(
            oracle_descriptors.gamma_edge
            if use_oracle_gamma
            else model_descriptors.gamma_edge
        ).clone(),
        intensity_edge=(
            oracle_descriptors.intensity_edge
            if use_oracle_intensity
            else model_descriptors.intensity_edge
        ).clone(),
        flow_node=(
            oracle_descriptors.flow_node
            if use_oracle_flow
            else model_descriptors.flow_node
        ).clone(),
    )


def apply_substitution(
    observation: ControlObservation,
    model_descriptors: PolicyDescriptors | None = None,
    mode: SubstitutionMode | str = SubstitutionMode.MODEL,
) -> ControlObservation:
    """Return a policy-only observation copy for oracle substitution.

    Simulator descriptors, topology, geometry, and discrete history are never
    sourced from model predictions. ``ControlObservation`` performs a further
    defensive clone while replacing the policy descriptor copy, preventing
    storage aliasing with the simulator or caller-owned prediction tensors.
    """

    observation.validate()
    oracle_descriptors = observation.sim_descriptors.policy_fields
    if model_descriptors is not None:
        model_descriptors.validate(
            observation.edge_count,
            observation.satellite_count,
        )
    composed = compose_policy_descriptors(
        oracle_descriptors,
        model_descriptors,
        mode,
    )
    return observation.with_policy_descriptors(composed)


def substitute_policy_descriptors(
    observation: ControlObservation,
    model_descriptors: PolicyDescriptors | None = None,
    mode: SubstitutionMode | str = SubstitutionMode.MODEL,
) -> ControlObservation:
    """Named alias for :func:`apply_substitution`."""

    return apply_substitution(observation, model_descriptors, mode)


def replace_policy_descriptors(
    observation: ControlObservation,
    model_descriptors: PolicyDescriptors | None = None,
    mode: SubstitutionMode | str = SubstitutionMode.MODEL,
) -> ControlObservation:
    """Protocol-language alias for :func:`apply_substitution`."""

    return apply_substitution(observation, model_descriptors, mode)


def persistent_candidate_mask(
    previous_observation: ControlObservation,
    current_observation: ControlObservation,
) -> torch.Tensor:
    """Mark current candidate edges that also existed at the previous epoch."""

    previous_observation.validate()
    current_observation.validate()
    if previous_observation.episode_seed != current_observation.episode_seed:
        raise ValueError("candidate persistence can only be computed within one episode")
    if current_observation.epoch != previous_observation.epoch + 1:
        raise ValueError("candidate persistence requires consecutive epochs")
    if (
        previous_observation.user_count != current_observation.user_count
        or previous_observation.satellite_count != current_observation.satellite_count
    ):
        raise ValueError("node identities must remain stable across candidate persistence")
    previous_ids = {
        (int(user), int(satellite))
        for user, satellite in previous_observation.candidate_edge_ids.tolist()
    }
    return torch.tensor(
        [
            (int(user), int(satellite)) in previous_ids
            for user, satellite in current_observation.candidate_edge_ids.tolist()
        ],
        dtype=torch.bool,
        device=current_observation.candidate_edge_ids.device,
    )


def carry_policy_stream(
    previous_observation: ControlObservation,
    previous_policy_descriptors: PolicyDescriptors,
    current_observation: ControlObservation,
    new_edge_descriptors: PolicyDescriptors | None = None,
) -> ControlObservation:
    """Carry the recursively used decision stream to the next model input.

    Edge fields are joined by stable local ``(user, satellite)`` ids. Persistent
    edges inherit the previous policy copy. Newly appearing edges must be
    supplied by an explicit initializer in the current candidate-edge order;
    missing values are never represented by a semantic zero or by changing the
    controller's candidate set. The node-domain flow carrier is inherited from
    the staged next-step prediction for every satellite. The returned
    observation retains its untouched simulator channel and geometry/history.
    """

    previous_observation.validate()
    current_observation.validate()
    previous_policy_descriptors.validate(
        previous_observation.edge_count,
        previous_observation.satellite_count,
    )
    if previous_observation.episode_seed != current_observation.episode_seed:
        raise ValueError("policy stream can only be carried within one episode")
    if current_observation.epoch != previous_observation.epoch + 1:
        raise ValueError("policy stream carry requires consecutive epochs")
    if (
        previous_observation.user_count != current_observation.user_count
        or previous_observation.satellite_count != current_observation.satellite_count
    ):
        raise ValueError("node identities must remain stable across policy stream carry")

    carried = PolicyDescriptors(
        gamma_edge=torch.empty_like(
            current_observation.sim_descriptors.policy_fields.gamma_edge
        ),
        intensity_edge=torch.empty_like(
            current_observation.sim_descriptors.policy_fields.intensity_edge
        ),
        flow_node=previous_policy_descriptors.flow_node.clone(),
    )
    previous_lookup = {
        (int(user), int(satellite)): index
        for index, (user, satellite) in enumerate(
            previous_observation.candidate_edge_ids.tolist()
        )
    }
    persistent = persistent_candidate_mask(previous_observation, current_observation)
    for current_index, edge_id in enumerate(current_observation.candidate_edge_ids.tolist()):
        previous_index = previous_lookup.get((int(edge_id[0]), int(edge_id[1])))
        if previous_index is None:
            continue
        carried.gamma_edge[current_index] = previous_policy_descriptors.gamma_edge[
            previous_index
        ]
        carried.intensity_edge[current_index] = previous_policy_descriptors.intensity_edge[
            previous_index
        ]
    new_edges = ~persistent
    if bool(new_edges.any().item()):
        if new_edge_descriptors is None:
            missing_ids = current_observation.candidate_edge_ids[new_edges].tolist()
            raise ValueError(
                "new candidate edges require explicit initialized descriptors; "
                f"missing edge ids: {missing_ids[:8]}"
            )
        new_edge_descriptors.validate(
            current_observation.edge_count,
            current_observation.satellite_count,
        )
        carried.gamma_edge[new_edges] = new_edge_descriptors.gamma_edge[new_edges]
        carried.intensity_edge[new_edges] = new_edge_descriptors.intensity_edge[new_edges]
    result = current_observation.with_policy_descriptors(carried)
    result.meta["policy_initialized_edge"] = new_edges
    result.validate()
    return result


def cold_start_policy_stream(
    observation: ControlObservation,
    initial_descriptors: PolicyDescriptors | None = None,
) -> ControlObservation:
    """Install an explicit epoch-zero descriptor prediction.

    The caller decides whether this comes from a learned initializer, recorded
    context, or an explicitly labelled oracle warm start. Missing values are
    rejected because zero is a valid, policy-relevant Intensity--Flow value.
    """

    observation.validate()
    if initial_descriptors is None:
        raise ValueError(
            "explicit initial policy descriptors are required for a model rollout; "
            "missing predictions cannot be encoded as semantic zeros"
        )
    initial_descriptors.validate(
        observation.edge_count,
        observation.satellite_count,
    )
    result = observation.with_policy_descriptors(initial_descriptors)
    result.meta["policy_initialized_edge"] = torch.ones(
        observation.edge_count,
        dtype=torch.bool,
        device=observation.candidate_edge_ids.device,
    )
    return result
