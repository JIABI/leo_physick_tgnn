"""Action-coupled descriptor-substitution evaluation for the UAV platform.

The evaluator implements the paper timeline literally::

    D_t -> a_t -> simulator transition -> D^oracle_{t+1}
         -> model prediction used for D^model_{t+1}

Oracle replacement is applied only to the policy-facing copy at the current
decision.  The recursively carried model stream is never repaired with oracle
values.  Edge fields are carried with the stable local identity
``uav_id * station_count + station_id``; a fingerprinted initializer is
mandatory for ``D_0`` and every newly appearing candidate edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
import torch.nn as nn

from .environment import UAVSharedServiceEnv
from .graph import observation_to_graph
from .protocol import UAV_SHARED_PROTOCOL_VERSION, UAVSharedProtocol
from .state import (
    ReassociationAction,
    ServiceExecutionResult,
    ServiceObservation,
    ServicePolicyDescriptors,
)


UAV_EVALUATION_CONTRACT_VERSION = 1


class UAVSubstitutionMode(str, Enum):
    """Descriptor source visible to the current policy evaluation."""

    MODEL = "model"
    ORACLE = "oracle"
    ETA = "eta"
    INTENSITY = "intensity"
    FLOW = "flow"
    INTENSITY_FLOW = "intensity_flow"

    @classmethod
    def parse(cls, value: "UAVSubstitutionMode | str") -> "UAVSubstitutionMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unknown UAV substitution mode {value!r}; expected {choices}"
            ) from exc


class UAVDescriptorProvider(Protocol):
    """Stateful, frozen one-step descriptor predictor."""

    @property
    def fingerprint(self) -> str: ...

    def reset(self) -> None: ...

    def predict_next(
        self, observation: ServiceObservation
    ) -> ServicePolicyDescriptors: ...


class UAVPolicyStreamInitializer(Protocol):
    """Explicit source for ``D_0`` and candidate edges without a prior id."""

    @property
    def fingerprint(self) -> str: ...

    @property
    def kind(self) -> str: ...

    def initialize(
        self,
        observation: ServiceObservation,
        edge_mask: torch.Tensor,
    ) -> ServicePolicyDescriptors: ...


class UAVReassociationPolicy(Protocol):
    """Policy contract. Simulator transition authority is deliberately absent."""

    def select_action(
        self,
        observation: ServiceObservation,
        descriptors: ServicePolicyDescriptors | None = None,
    ) -> ReassociationAction: ...


def _finite_scalar(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class ConstantUAVPolicyStreamInitializer:
    """Fingerprint-stable prior; it never copies simulator descriptor values."""

    kind = "configured_prior"

    def __init__(
        self,
        *,
        eta: float,
        intensity: float,
        station_flow: float,
    ) -> None:
        self.eta = _finite_scalar("eta", eta)
        self.intensity = _finite_scalar("intensity", intensity)
        self.station_flow = _finite_scalar("station_flow", station_flow)
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("initializer eta must be in [0,1]")
        if not 0.0 <= self.intensity <= 3.0:
            raise ValueError("initializer intensity must be in [0,3]")
        if not 0.0 <= self.station_flow <= 1.0:
            raise ValueError("initializer station_flow must be in [0,1]")
        payload = {
            "contract": "constant_uav_policy_stream_v1",
            "eta": self.eta,
            "intensity": self.intensity,
            "station_flow": self.station_flow,
        }
        self._fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def initialize(
        self,
        observation: ServiceObservation,
        edge_mask: torch.Tensor,
    ) -> ServicePolicyDescriptors:
        observation.validate()
        _validate_edge_mask(observation, edge_mask)
        reference = observation.sim_descriptors.policy_fields
        return ServicePolicyDescriptors(
            eta_edge=torch.full_like(reference.eta_edge, self.eta),
            intensity_edge=torch.full_like(
                reference.intensity_edge, self.intensity
            ),
            station_flow_node=torch.full_like(
                reference.station_flow_node, self.station_flow
            ),
        )


def _model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    digest.update(type(model).__module__.encode())
    digest.update(type(model).__qualname__.encode())
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _output_descriptors(value: Any) -> ServicePolicyDescriptors:
    if isinstance(value, ServicePolicyDescriptors):
        return value
    candidate = getattr(value, "policy_descriptors", None)
    if isinstance(candidate, ServicePolicyDescriptors):
        return candidate
    try:
        return ServicePolicyDescriptors(
            eta_edge=value.eta_edge,
            intensity_edge=value.intensity_edge,
            station_flow_node=value.station_flow_node,
        )
    except AttributeError as exc:
        raise TypeError(
            "UAV model output must be ServicePolicyDescriptors or expose "
            "eta_edge, intensity_edge, and station_flow_node"
        ) from exc


class UAVModelProvider:
    """Adapter for ``model.predict_step(observation, state)``.

    The model receives the observation whose *policy* channel is the recursively
    carried model stream.  No target is passed.  A returned feasibility logit is
    allowed for training provenance but is intentionally not policy-visible;
    feasibility remains simulator-authoritative.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str = "cpu",
        fingerprint: str | None = None,
        protocol: UAVSharedProtocol | None = None,
    ) -> None:
        if not hasattr(model, "predict_step"):
            raise TypeError("UAV model must implement predict_step(observation, state)")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._fingerprint = str(fingerprint or _model_fingerprint(model)).strip()
        if not self._fingerprint:
            raise ValueError("provider fingerprint cannot be empty")
        self._state: Any = None
        self.protocol = protocol

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def reset(self) -> None:
        self._state = None

    @torch.no_grad()
    def predict_next(
        self, observation: ServiceObservation
    ) -> ServicePolicyDescriptors:
        observation.validate()
        self.model.eval()
        model_input: ServiceObservation | Mapping[str, Any] = observation
        protocol = self.protocol
        if protocol is None:
            model_protocol = getattr(self.model, "protocol", None)
            if isinstance(model_protocol, UAVSharedProtocol):
                protocol = replace(
                    model_protocol,
                    episode_seed=observation.episode_seed,
                )
        if protocol is not None:
            if protocol.episode_seed != observation.episode_seed:
                raise ValueError("provider protocol seed differs from the observation")
            model_input = observation_to_graph(observation, protocol)
        returned = self.model.predict_step(model_input, self._state)
        if not isinstance(returned, tuple) or len(returned) != 2:
            raise TypeError("UAV predict_step must return (output, next_state)")
        output, self._state = returned
        predicted = _output_descriptors(output)
        oracle = observation.sim_descriptors.policy_fields
        aligned = ServicePolicyDescriptors(
            eta_edge=predicted.eta_edge.detach().to(
                device=oracle.eta_edge.device, dtype=oracle.eta_edge.dtype
            ),
            intensity_edge=predicted.intensity_edge.detach().to(
                device=oracle.intensity_edge.device,
                dtype=oracle.intensity_edge.dtype,
            ),
            station_flow_node=predicted.station_flow_node.detach().to(
                device=oracle.station_flow_node.device,
                dtype=oracle.station_flow_node.dtype,
            ),
        )
        aligned.validate(observation.edge_count, observation.station_count)
        return aligned.clone()


def stable_candidate_edge_ids(observation: ServiceObservation) -> torch.Tensor:
    """Return stable scalar ids and reject duplicate candidate pairs."""

    observation.validate()
    pairs = observation.candidate_edge_ids
    ids = pairs[:, 0] * observation.station_count + pairs[:, 1]
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("candidate graph contains duplicate (uav,station) ids")
    return ids


def persistent_candidate_mask(
    previous: ServiceObservation,
    current: ServiceObservation,
) -> torch.Tensor:
    previous.validate()
    current.validate()
    if previous.episode_seed != current.episode_seed:
        raise ValueError("candidate persistence cannot cross episode seeds")
    if current.epoch != previous.epoch + 1:
        raise ValueError("candidate persistence requires consecutive epochs")
    if (
        previous.user_count != current.user_count
        or previous.station_count != current.station_count
    ):
        raise ValueError("stable node domains changed inside an episode")
    previous_ids = set(stable_candidate_edge_ids(previous).tolist())
    return torch.tensor(
        [int(value) in previous_ids for value in stable_candidate_edge_ids(current)],
        dtype=torch.bool,
        device=current.candidate_edge_ids.device,
    )


def _validate_edge_mask(
    observation: ServiceObservation,
    edge_mask: torch.Tensor,
) -> None:
    if not isinstance(edge_mask, torch.Tensor):
        raise TypeError("edge_mask must be a tensor")
    if edge_mask.dtype != torch.bool or edge_mask.shape != (
        observation.edge_count,
    ):
        raise ValueError("edge_mask must be bool with shape [edge_count]")
    if edge_mask.device != observation.candidate_edge_ids.device:
        raise ValueError("edge_mask must share the candidate graph device")


def carry_model_policy_stream(
    previous: ServiceObservation,
    staged_prediction: ServicePolicyDescriptors,
    current_oracle: ServiceObservation,
    initializer: UAVPolicyStreamInitializer,
) -> tuple[ServiceObservation, torch.Tensor]:
    """Join a prediction made on ``D_t`` into the candidate graph at ``t+1``."""

    previous.validate()
    current_oracle.validate()
    staged_prediction.validate(previous.edge_count, previous.station_count)
    persistent = persistent_candidate_mask(previous, current_oracle)
    new_edge = ~persistent
    initialized = initializer.initialize(current_oracle, new_edge.clone())
    initialized.validate(current_oracle.edge_count, current_oracle.station_count)

    previous_ids = stable_candidate_edge_ids(previous).tolist()
    previous_lookup = {int(value): index for index, value in enumerate(previous_ids)}
    current_ids = stable_candidate_edge_ids(current_oracle).tolist()
    eta = initialized.eta_edge.clone()
    intensity = initialized.intensity_edge.clone()
    for current_index, edge_id in enumerate(current_ids):
        previous_index = previous_lookup.get(int(edge_id))
        if previous_index is None:
            continue
        eta[current_index] = staged_prediction.eta_edge[previous_index]
        intensity[current_index] = staged_prediction.intensity_edge[previous_index]

    # Station ids are stable and dense, so node fields need no candidate join.
    carried = ServicePolicyDescriptors(
        eta_edge=eta,
        intensity_edge=intensity,
        station_flow_node=staged_prediction.station_flow_node.clone(),
    )
    carried.validate(current_oracle.edge_count, current_oracle.station_count)
    return current_oracle.with_policy_descriptors(carried), new_edge


def compose_policy_descriptors(
    oracle: ServicePolicyDescriptors,
    model: ServicePolicyDescriptors | None,
    mode: UAVSubstitutionMode | str,
) -> ServicePolicyDescriptors:
    """Create an independent current-policy copy for one substitution mode."""

    selected = UAVSubstitutionMode.parse(mode)
    if selected is UAVSubstitutionMode.ORACLE:
        return oracle.clone()
    if model is None:
        raise ValueError(f"mode {selected.value} requires model descriptors")
    oracle_fields = oracle.as_dict()
    model_fields = model.as_dict()
    for name, oracle_value in oracle_fields.items():
        model_value = model_fields[name]
        if (
            oracle_value.shape != model_value.shape
            or oracle_value.device != model_value.device
            or oracle_value.dtype != model_value.dtype
        ):
            raise ValueError(f"oracle/model {name} tensors are incompatible")
    return ServicePolicyDescriptors(
        eta_edge=(
            oracle.eta_edge
            if selected is UAVSubstitutionMode.ETA
            else model.eta_edge
        ).clone(),
        intensity_edge=(
            oracle.intensity_edge
            if selected
            in {UAVSubstitutionMode.INTENSITY, UAVSubstitutionMode.INTENSITY_FLOW}
            else model.intensity_edge
        ).clone(),
        station_flow_node=(
            oracle.station_flow_node
            if selected
            in {UAVSubstitutionMode.FLOW, UAVSubstitutionMode.INTENSITY_FLOW}
            else model.station_flow_node
        ).clone(),
    )


def apply_policy_substitution(
    model_observation: ServiceObservation,
    oracle_observation: ServiceObservation,
    mode: UAVSubstitutionMode | str,
) -> ServiceObservation:
    """Return a policy-only intervention without mutating either source."""

    model_observation.validate()
    oracle_observation.validate()
    if model_observation.observation_id != oracle_observation.observation_id:
        raise ValueError("model/oracle observations identify different decisions")
    if not torch.equal(
        model_observation.candidate_edge_ids,
        oracle_observation.candidate_edge_ids,
    ):
        raise ValueError("model/oracle observations have different candidate order")
    descriptors = compose_policy_descriptors(
        oracle_observation.sim_descriptors.policy_fields,
        model_observation.policy_descriptors,
        mode,
    )
    # with_policy_descriptors preserves a defensive simulator-authority copy.
    return oracle_observation.with_policy_descriptors(descriptors)


def _clone_action(value: ReassociationAction) -> ReassociationAction:
    return ReassociationAction(
        observation_id=value.observation_id,
        requested_station=value.requested_station.clone(),
    )


def _clone_execution(value: ServiceExecutionResult) -> ServiceExecutionResult:
    kwargs = {
        name: (item.clone() if isinstance(item, torch.Tensor) else item)
        for name, item in vars(value).items()
    }
    return ServiceExecutionResult(**kwargs)


@dataclass(frozen=True)
class UAVClosedLoopStep:
    observation_id: tuple[int, int]
    candidate_edge_ids: torch.Tensor
    candidate_rank: torch.Tensor
    feasible_start_edge: torch.Tensor
    simulator_descriptors: ServicePolicyDescriptors
    model_input_descriptors: ServicePolicyDescriptors | None
    policy_descriptors: ServicePolicyDescriptors
    initialized_edge: torch.Tensor
    next_model_prediction: ServicePolicyDescriptors | None
    phase_before: torch.Tensor
    current_station_before: torch.Tensor
    action: ReassociationAction
    execution: ServiceExecutionResult


@dataclass(frozen=True)
class UAVClosedLoopResult:
    mode: UAVSubstitutionMode
    episode_seed: int
    protocol_version: int
    protocol_fingerprint: str
    paired_fingerprint: str
    provider_fingerprint: str
    initializer_kind: str
    initializer_fingerprint: str
    policy_config: Mapping[str, Any]
    records: tuple[UAVClosedLoopStep, ...]
    final_energy_fraction: torch.Tensor
    final_phase: torch.Tensor
    final_station_flow: torch.Tensor
    final_missions_completed: torch.Tensor
    final_failed_service_starts: torch.Tensor
    final_reassociations: torch.Tensor
    evaluation_contract_version: int = UAV_EVALUATION_CONTRACT_VERSION

    @property
    def action_count(self) -> int:
        return len(self.records)


def _policy_manifest(policy: UAVReassociationPolicy) -> Mapping[str, Any]:
    config = getattr(policy, "config", None)
    if config is None:
        return {"policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}"}
    if is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, Mapping):
        value = dict(config)
    else:
        value = {"repr": repr(config)}
    return {
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "config": value,
    }


def _validate_fingerprint(name: str, value: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} fingerprint cannot be empty")
    return result


def run_uav_closed_loop(
    environment: UAVSharedServiceEnv,
    *,
    mode: UAVSubstitutionMode | str,
    policy: UAVReassociationPolicy,
    descriptor_provider: UAVDescriptorProvider | None = None,
    stream_initializer: UAVPolicyStreamInitializer | None = None,
    paired_fingerprint: str = "standalone",
    reset_environment: bool = True,
) -> UAVClosedLoopResult:
    """Run one staged, action-coupled condition without oracle feedback repair."""

    selected = UAVSubstitutionMode.parse(mode)
    requires_model = selected is not UAVSubstitutionMode.ORACLE
    if requires_model:
        if descriptor_provider is None or stream_initializer is None:
            raise ValueError(
                f"{selected.value} requires a descriptor provider and initializer"
            )
        descriptor_provider.reset()
    elif descriptor_provider is not None or stream_initializer is not None:
        raise ValueError("oracle mode must not instantiate a model stream")

    oracle_observation = (
        environment.reset_control()
        if reset_environment
        else environment.observe()
    )
    if requires_model:
        assert stream_initializer is not None
        d0_mask = torch.ones(
            oracle_observation.edge_count,
            dtype=torch.bool,
            device=oracle_observation.candidate_edge_ids.device,
        )
        d0 = stream_initializer.initialize(oracle_observation, d0_mask.clone())
        d0.validate(oracle_observation.edge_count, oracle_observation.station_count)
        model_observation: ServiceObservation | None = (
            oracle_observation.with_policy_descriptors(d0)
        )
        initialized_edge = d0_mask
    else:
        model_observation = None
        initialized_edge = torch.zeros(
            oracle_observation.edge_count,
            dtype=torch.bool,
            device=oracle_observation.candidate_edge_ids.device,
        )

    records: list[UAVClosedLoopStep] = []
    while True:
        has_next = oracle_observation.epoch + 1 < environment.protocol.horizon_steps
        prediction: ServicePolicyDescriptors | None = None
        if requires_model and has_next:
            assert descriptor_provider is not None and model_observation is not None
            # Prediction is staged before a_t and sees only the recursive model stream.
            prediction = descriptor_provider.predict_next(model_observation)

        if requires_model:
            assert model_observation is not None
            policy_observation = apply_policy_substitution(
                model_observation, oracle_observation, selected
            )
        else:
            policy_observation = oracle_observation.with_policy_descriptors(
                oracle_observation.sim_descriptors.policy_fields
            )
        action = policy.select_action(
            policy_observation,
            policy_observation.policy_descriptors,
        )
        if not isinstance(action, ReassociationAction):
            raise TypeError("UAV policy must return ReassociationAction")
        next_oracle, execution, done = environment.step_action(action)
        records.append(
            UAVClosedLoopStep(
                observation_id=oracle_observation.observation_id,
                candidate_edge_ids=oracle_observation.candidate_edge_ids.clone(),
                candidate_rank=oracle_observation.candidate_rank.clone(),
                feasible_start_edge=(
                    oracle_observation.sim_descriptors.feasible_start_edge.clone()
                ),
                simulator_descriptors=(
                    oracle_observation.sim_descriptors.policy_fields.clone()
                ),
                model_input_descriptors=(
                    None
                    if model_observation is None
                    else model_observation.policy_descriptors.clone()
                ),
                policy_descriptors=policy_observation.policy_descriptors.clone(),
                initialized_edge=initialized_edge.clone(),
                next_model_prediction=(
                    None if prediction is None else prediction.clone()
                ),
                phase_before=oracle_observation.phase.clone(),
                current_station_before=oracle_observation.current_station.clone(),
                action=_clone_action(action),
                execution=_clone_execution(execution),
            )
        )
        if done:
            break
        if next_oracle is None:
            raise RuntimeError("environment omitted a non-terminal observation")
        if requires_model:
            if prediction is None or model_observation is None:
                raise RuntimeError("staged model prediction missing at t+1 carry")
            assert stream_initializer is not None
            model_observation, initialized_edge = carry_model_policy_stream(
                model_observation,
                prediction,
                next_oracle,
                stream_initializer,
            )
        else:
            initialized_edge = torch.zeros(
                next_oracle.edge_count,
                dtype=torch.bool,
                device=next_oracle.candidate_edge_ids.device,
            )
        oracle_observation = next_oracle

    final = environment.snapshot().state
    nominal_energy = environment.protocol.estimated.nominal_battery_units
    policy_manifest = _policy_manifest(policy)
    return UAVClosedLoopResult(
        mode=selected,
        episode_seed=final.episode_seed,
        protocol_version=UAV_SHARED_PROTOCOL_VERSION,
        protocol_fingerprint=environment.protocol_fingerprint,
        paired_fingerprint=_validate_fingerprint(
            "paired evaluation", paired_fingerprint
        ),
        provider_fingerprint=(
            "uav_oracle"
            if descriptor_provider is None
            else _validate_fingerprint("provider", descriptor_provider.fingerprint)
        ),
        initializer_kind=(
            "oracle"
            if stream_initializer is None
            else str(stream_initializer.kind)
        ),
        initializer_fingerprint=(
            "uav_oracle"
            if stream_initializer is None
            else _validate_fingerprint(
                "initializer", stream_initializer.fingerprint
            )
        ),
        policy_config=policy_manifest,
        records=tuple(records),
        final_energy_fraction=(final.energy_units / nominal_energy).clone(),
        final_phase=final.phase.clone(),
        final_station_flow=final.station_flow.clone(),
        final_missions_completed=final.missions_completed.clone(),
        final_failed_service_starts=final.failed_service_starts.clone(),
        final_reassociations=final.reassociations.clone(),
    )


def _canonical_json(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical_json(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def run_uav_paired_conditions(
    environment: UAVSharedServiceEnv,
    *,
    modes: Sequence[UAVSubstitutionMode | str],
    policy_factory: Callable[[], UAVReassociationPolicy],
    descriptor_provider_factory: Callable[[], UAVDescriptorProvider] | None,
    stream_initializer_factory: Callable[[], UAVPolicyStreamInitializer] | None,
) -> dict[UAVSubstitutionMode, UAVClosedLoopResult]:
    """Run all conditions from one identical snapshot and common keyed seed."""

    selected = tuple(UAVSubstitutionMode.parse(mode) for mode in modes)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("paired modes must be non-empty and unique")
    if any(mode is not UAVSubstitutionMode.ORACLE for mode in selected):
        if descriptor_provider_factory is None or stream_initializer_factory is None:
            raise ValueError("non-oracle paired modes require provider/initializer factories")

    environment.reset_control()
    common_snapshot = environment.snapshot()
    components: dict[
        UAVSubstitutionMode,
        tuple[
            UAVReassociationPolicy,
            UAVDescriptorProvider | None,
            UAVPolicyStreamInitializer | None,
        ],
    ] = {}
    provider_ids: set[str] = set()
    initializer_ids: set[str] = set()
    policy_manifests: list[Mapping[str, Any]] = []
    for mode in selected:
        policy = policy_factory()
        policy_manifests.append(_policy_manifest(policy))
        if mode is UAVSubstitutionMode.ORACLE:
            components[mode] = (policy, None, None)
            continue
        assert descriptor_provider_factory is not None
        assert stream_initializer_factory is not None
        provider = descriptor_provider_factory()
        initializer = stream_initializer_factory()
        provider_ids.add(_validate_fingerprint("provider", provider.fingerprint))
        initializer_ids.add(
            _validate_fingerprint("initializer", initializer.fingerprint)
        )
        components[mode] = (policy, provider, initializer)
    if len(provider_ids) > 1:
        raise ValueError("paired conditions use different frozen model fingerprints")
    if len(initializer_ids) > 1:
        raise ValueError("paired conditions use different initializer fingerprints")
    canonical_policies = {
        json.dumps(_canonical_json(value), sort_keys=True, separators=(",", ":"))
        for value in policy_manifests
    }
    if len(canonical_policies) != 1:
        raise ValueError("paired conditions use different policy configurations")
    fingerprint_payload = {
        "contract": UAV_EVALUATION_CONTRACT_VERSION,
        "episode_seed": common_snapshot.state.episode_seed,
        "protocol_fingerprint": environment.protocol_fingerprint,
        "modes": [mode.value for mode in selected],
        "provider_fingerprint": next(iter(provider_ids), "uav_oracle"),
        "initializer_fingerprint": next(iter(initializer_ids), "uav_oracle"),
        "policy": _canonical_json(policy_manifests[0]),
    }
    paired_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()

    results: dict[UAVSubstitutionMode, UAVClosedLoopResult] = {}
    for mode in selected:
        policy, provider, initializer = components[mode]
        condition_env = environment.new_instance()
        condition_env.restore(common_snapshot)
        # Preserve the exact common snapshot; the condition runner is told not
        # to perform its normal standalone reset.
        result = run_uav_closed_loop(
            condition_env,
            mode=mode,
            policy=policy,
            descriptor_provider=provider,
            stream_initializer=initializer,
            paired_fingerprint=paired_fingerprint,
            reset_environment=False,
        )
        if result.protocol_fingerprint != environment.protocol_fingerprint:
            raise RuntimeError("paired condition changed the simulator protocol")
        results[mode] = result
    horizons = {result.action_count for result in results.values()}
    seeds = {result.episode_seed for result in results.values()}
    fingerprints = {result.paired_fingerprint for result in results.values()}
    if len(horizons) != 1 or len(seeds) != 1 or len(fingerprints) != 1:
        raise RuntimeError("paired conditions do not share seed/horizon/fingerprint")
    return results


__all__ = [
    "ConstantUAVPolicyStreamInitializer",
    "UAVClosedLoopResult",
    "UAVClosedLoopStep",
    "UAVDescriptorProvider",
    "UAVModelProvider",
    "UAVPolicyStreamInitializer",
    "UAVSubstitutionMode",
    "UAV_EVALUATION_CONTRACT_VERSION",
    "apply_policy_substitution",
    "carry_model_policy_stream",
    "compose_policy_descriptors",
    "persistent_candidate_mask",
    "run_uav_closed_loop",
    "run_uav_paired_conditions",
    "stable_candidate_edge_ids",
]
