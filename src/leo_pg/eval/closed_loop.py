from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Callable, Dict, Iterable, Protocol

import torch

from leo_pg.control.policy import FixedRankPolicy, FixedRankPolicyConfig
from leo_pg.models.heads.intensity_flow import IntensityFlowOutput
from leo_pg.sim.paper_environment import (
    PAPER_PROTOCOL_VERSION,
    PaperAlignedLEOEnv,
    protocol_fingerprint,
)
from leo_pg.sim.state import (
    ControlObservation,
    ExecutionResult,
    PolicyDescriptors,
    ServingAction,
    SimulatorDescriptors,
)

from .substitution import (
    SubstitutionMode,
    apply_substitution,
    carry_policy_stream,
    cold_start_policy_stream,
    persistent_candidate_mask,
)


class DescriptorProvider(Protocol):
    """Stateful one-step source of model-facing Intensity--Flow descriptors."""

    def reset(self) -> None:
        """Reset episode-local state such as recurrent memory."""

    def predict_next(self, observation: ControlObservation) -> PolicyDescriptors:
        """Predict ``t+1`` fields in the epoch-``t`` candidate-edge order."""

    @property
    def fingerprint(self) -> str:
        """Stable identity of the frozen predictor/checkpoint."""


class PolicyInitializerKind(str, Enum):
    """Auditable source of epoch-zero and newly appearing edge descriptors."""

    LEARNED = "learned"
    CONFIGURED_PRIOR = "configured_prior"
    RECORDED_CONTEXT = "recorded_context"
    ORACLE_WARM_START = "oracle_warm_start"

    @classmethod
    def parse(cls, value: "PolicyInitializerKind | str") -> "PolicyInitializerKind":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unknown policy initializer kind {value!r}; expected one of {choices}"
            ) from exc


class PolicyStreamInitializer(Protocol):
    """Explicit source for descriptors that have no staged prior prediction."""

    @property
    def kind(self) -> PolicyInitializerKind:
        """Provenance category recorded in the closed-loop result."""

    @property
    def fingerprint(self) -> str:
        """Stable identity of the initializer parameters/checkpoint."""

    def initialize(
        self,
        observation: ControlObservation,
        edge_mask: torch.Tensor,
    ) -> PolicyDescriptors:
        """Return complete descriptors in ``observation`` candidate order."""


class CallablePolicyStreamInitializer:
    """Adapt an explicitly identified initializer callable to the core API."""

    def __init__(
        self,
        function: Callable[[ControlObservation, torch.Tensor], PolicyDescriptors],
        *,
        fingerprint: str,
        kind: PolicyInitializerKind | str,
    ) -> None:
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("a non-empty initializer fingerprint is required")
        self._function = function
        self._fingerprint = fingerprint.strip()
        self._kind = PolicyInitializerKind.parse(kind)

    @property
    def kind(self) -> PolicyInitializerKind:
        return self._kind

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def initialize(
        self,
        observation: ControlObservation,
        edge_mask: torch.Tensor,
    ) -> PolicyDescriptors:
        observation.validate()
        if edge_mask.shape != (observation.edge_count,) or edge_mask.dtype != torch.bool:
            raise ValueError("initializer edge_mask must be bool with shape [edge_count]")
        if edge_mask.device != observation.candidate_edge_ids.device:
            raise ValueError("initializer edge_mask must share the observation graph device")
        descriptors = self._function(observation, edge_mask.clone())
        if not isinstance(descriptors, PolicyDescriptors):
            raise TypeError("policy initializer must return PolicyDescriptors")
        descriptors.validate(observation.edge_count, observation.satellite_count)
        return descriptors.clone()


class ConstantPolicyStreamInitializer:
    """No-teacher-warm-start prior with explicit, fingerprinted constants."""

    def __init__(self, *, gamma: float, intensity: float, flow: float) -> None:
        values = (float(gamma), float(intensity), float(flow))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("initializer constants must be finite")
        if values[1] < 0:
            raise ValueError("initializer intensity must be non-negative")
        self.gamma, self.intensity, self.flow = values
        payload = (
            "constant-policy-stream-v1|"
            f"gamma={self.gamma:.17g}|intensity={self.intensity:.17g}|"
            f"flow={self.flow:.17g}"
        )
        self._fingerprint = hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_config(cls, cfg: Dict[str, object]) -> "ConstantPolicyStreamInitializer":
        protocol = cfg.get("paper_protocol")
        if not isinstance(protocol, dict):
            raise ValueError("paper_protocol mapping is required")
        initializer = protocol.get("policy_initializer")
        if not isinstance(initializer, dict):
            raise ValueError("paper_protocol.policy_initializer mapping is required")
        if str(initializer.get("kind", "")).lower() != "configured_prior":
            raise ValueError(
                "ConstantPolicyStreamInitializer requires policy_initializer.kind="
                "'configured_prior'"
            )
        required = ("gamma", "intensity", "flow")
        missing = [name for name in required if name not in initializer]
        if missing:
            raise ValueError(
                "policy_initializer is missing required fields: " + ", ".join(missing)
            )
        return cls(
            gamma=float(initializer["gamma"]),
            intensity=float(initializer["intensity"]),
            flow=float(initializer["flow"]),
        )

    @property
    def kind(self) -> PolicyInitializerKind:
        return PolicyInitializerKind.CONFIGURED_PRIOR

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def initialize(
        self,
        observation: ControlObservation,
        edge_mask: torch.Tensor,
    ) -> PolicyDescriptors:
        observation.validate()
        if edge_mask.shape != (observation.edge_count,) or edge_mask.dtype != torch.bool:
            raise ValueError("initializer edge_mask must be bool with shape [edge_count]")
        if edge_mask.device != observation.candidate_edge_ids.device:
            raise ValueError("initializer edge_mask must share the observation graph device")
        reference = observation.sim_descriptors.policy_fields
        return PolicyDescriptors(
            gamma_edge=torch.full_like(reference.gamma_edge, self.gamma),
            intensity_edge=torch.full_like(reference.intensity_edge, self.intensity),
            flow_node=torch.full_like(reference.flow_node, self.flow),
        )


class CallableDescriptorProvider:
    """Adapt a pure prediction callable to the provider contract."""

    def __init__(
        self,
        function: Callable[[ControlObservation], PolicyDescriptors],
        fingerprint: str,
        reset: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("a non-empty callable provider fingerprint is required")
        self._function = function
        self._fingerprint = fingerprint.strip()
        self._reset = reset

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def reset(self) -> None:
        if self._reset is not None:
            self._reset()

    def predict_next(self, observation: ControlObservation) -> PolicyDescriptors:
        prediction = self._function(observation)
        if not isinstance(prediction, PolicyDescriptors):
            raise TypeError("descriptor callable must return PolicyDescriptors")
        prediction.validate(observation.edge_count, observation.satellite_count)
        return prediction.clone()


class TGNDescriptorProvider:
    """Inference adapter for a TGN with an :class:`IntensityFlowOutput` head.

    The adapter owns recurrent memory for one closed-loop episode. It supplies
    only current graph inputs to ``predict_step``; no ``y`` or simulator target
    is present in the model step.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.memory: torch.Tensor | None = None
        digest = hashlib.sha256()
        digest.update(type(model).__module__.encode())
        digest.update(type(model).__qualname__.encode())
        digest.update(repr(model).encode())
        model_cfg = getattr(model, "cfg", None)
        if isinstance(model_cfg, dict):
            digest.update(protocol_fingerprint(model_cfg).encode())
        for name, value in sorted(model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        self._fingerprint = digest.hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def reset(self) -> None:
        self.memory = None

    def predict_next(self, observation: ControlObservation) -> PolicyDescriptors:
        observation.validate()
        if not hasattr(self.model, "predict_step"):
            raise TypeError("model must implement predict_step(step, memory, device)")
        self.model.eval()
        with torch.no_grad():
            output, self.memory = self.model.predict_step(
                observation.as_model_step(),
                self.memory,
                self.device,
            )
        if not isinstance(output, IntensityFlowOutput):
            raise TypeError("model must return IntensityFlowOutput")
        output.validate(observation.edge_count, observation.satellite_count)

        oracle = observation.sim_descriptors.policy_fields
        predicted = output.policy_descriptors
        # A model can live on a different device from the simulator. The
        # policy comparison itself is deliberately carried out in simulator
        # precision and on the simulator device.
        aligned = PolicyDescriptors(
            gamma_edge=predicted.gamma_edge.detach().to(
                device=oracle.gamma_edge.device,
                dtype=oracle.gamma_edge.dtype,
            ),
            intensity_edge=predicted.intensity_edge.detach().to(
                device=oracle.intensity_edge.device,
                dtype=oracle.intensity_edge.dtype,
            ),
            flow_node=predicted.flow_node.detach().to(
                device=oracle.flow_node.device,
                dtype=oracle.flow_node.dtype,
            ),
        )
        aligned.validate(observation.edge_count, observation.satellite_count)
        return aligned


def _clone_action(action: ServingAction) -> ServingAction:
    return ServingAction(
        observation_id=action.observation_id,
        requested_serving=action.requested_serving.clone(),
    )


def _clone_execution(result: ExecutionResult) -> ExecutionResult:
    return ExecutionResult(
        observation_id=result.observation_id,
        requested_serving=result.requested_serving.clone(),
        executed_serving=result.executed_serving.clone(),
        admitted=result.admitted.clone(),
        failure_reason=result.failure_reason.clone(),
        handover_attempted=result.handover_attempted.clone(),
        handover_executed=result.handover_executed.clone(),
        flow_before=result.flow_before.clone(),
        flow_after=result.flow_after.clone(),
    )


@dataclass
class ClosedLoopStep:
    """Auditable record of one descriptor-to-action-to-state transition."""

    observation_id: tuple[int, int]
    mode: SubstitutionMode
    candidate_edge_ids: torch.Tensor
    sim_descriptors: SimulatorDescriptors
    model_input_descriptors: PolicyDescriptors
    model_descriptors: PolicyDescriptors | None
    next_model_prediction: PolicyDescriptors | None
    policy_descriptors: PolicyDescriptors
    initialized_edge: torch.Tensor
    feasible_edge: torch.Tensor
    action: ServingAction
    execution: ExecutionResult


@dataclass
class ClosedLoopResult:
    """One complete action-coupled evaluation condition."""

    mode: SubstitutionMode
    episode_seed: int
    protocol_version: int
    protocol_fingerprint: str
    provider_fingerprint: str
    initializer_kind: str
    initializer_fingerprint: str
    policy_config: FixedRankPolicyConfig
    hard_feasibility_mask: bool
    initial_policy_strategy: str
    new_edge_strategy: str
    records: tuple[ClosedLoopStep, ...]
    final_serving: torch.Tensor
    final_flow: torch.Tensor

    @property
    def action_count(self) -> int:
        return len(self.records)


def run_closed_loop(
    environment: PaperAlignedLEOEnv,
    *,
    mode: SubstitutionMode | str,
    descriptor_provider: DescriptorProvider | None = None,
    policy_initializer: PolicyStreamInitializer | None = None,
    policy: FixedRankPolicy | None = None,
    allow_oracle_warm_start: bool = False,
) -> ClosedLoopResult:
    """Run one paper-protocol condition through the real simulator transition.

    ``oracle`` bypasses the model provider. In every model condition, the action
    at epoch ``t`` consumes the descriptor prediction already staged for ``t``.
    The provider predicts ``t+1`` from the uncorrected staged model stream;
    partial oracle substitution is confined to the copy evaluated by the
    controller. The fresh output is aligned to persistent candidate ids only
    after the action is executed and is never used prematurely for ``a_t``.
    Simulator descriptors and the hard feasibility mask are never overwritten.
    """

    selected_mode = SubstitutionMode.parse(mode)
    if selected_mode is not SubstitutionMode.ORACLE and descriptor_provider is None:
        raise ValueError(f"mode {selected_mode.value!r} requires a descriptor provider")
    if selected_mode is SubstitutionMode.ORACLE and descriptor_provider is not None:
        raise ValueError("oracle mode does not use a descriptor provider")
    if selected_mode is not SubstitutionMode.ORACLE and policy_initializer is None:
        raise ValueError(f"mode {selected_mode.value!r} requires a policy_stream_initializer")
    if selected_mode is SubstitutionMode.ORACLE and policy_initializer is not None:
        raise ValueError("oracle mode does not use a policy_stream_initializer")
    initializer_kind = (
        None
        if policy_initializer is None
        else PolicyInitializerKind.parse(policy_initializer.kind)
    )
    if (
        initializer_kind is PolicyInitializerKind.ORACLE_WARM_START
        and not allow_oracle_warm_start
    ):
        raise ValueError(
            "oracle warm starts are excluded by the no-teacher-warm-start paper protocol; "
            "set allow_oracle_warm_start=True only for an explicitly labelled diagnostic"
        )
    if descriptor_provider is not None:
        descriptor_provider.reset()
    environment_policy_config = environment.fixed_policy_config()
    controller = policy or FixedRankPolicy(environment_policy_config)
    if controller.config != environment_policy_config:
        raise ValueError(
            "paper-protocol policy must match environment.fixed_policy_config(); "
            "policy-swap experiments require a separately labelled runner"
        )
    observation = environment.reset_control()
    if selected_mode is not SubstitutionMode.ORACLE:
        assert policy_initializer is not None
        initial_mask = torch.ones(
            observation.edge_count,
            dtype=torch.bool,
            device=observation.candidate_edge_ids.device,
        )
        observation = cold_start_policy_stream(
            observation,
            policy_initializer.initialize(observation, initial_mask),
        )
    records: list[ClosedLoopStep] = []

    while True:
        model_descriptors = (
            None
            if selected_mode is SubstitutionMode.ORACLE
            else observation.policy_descriptors.clone()
        )
        policy_observation = apply_substitution(
            observation,
            model_descriptors,
            selected_mode,
        )
        next_model_prediction = None
        has_next_epoch = observation.epoch + 1 < environment.horizon_steps
        if selected_mode is not SubstitutionMode.ORACLE and has_next_epoch:
            assert descriptor_provider is not None
            next_model_prediction = descriptor_provider.predict_next(observation)
            next_model_prediction.validate(
                observation.edge_count,
                observation.satellite_count,
            )
        action = controller.select_action(policy_observation)
        next_observation, execution, done = environment.step_action(action)
        records.append(
            ClosedLoopStep(
                observation_id=observation.observation_id,
                mode=selected_mode,
                candidate_edge_ids=observation.candidate_edge_ids.clone(),
                sim_descriptors=observation.sim_descriptors.clone(),
                model_input_descriptors=observation.policy_descriptors.clone(),
                model_descriptors=(
                    None if model_descriptors is None else model_descriptors.clone()
                ),
                next_model_prediction=(
                    None
                    if next_model_prediction is None
                    else next_model_prediction.clone()
                ),
                policy_descriptors=policy_observation.policy_descriptors.clone(),
                initialized_edge=observation.meta.get(
                    "policy_initialized_edge",
                    torch.zeros(
                        observation.edge_count,
                        dtype=torch.bool,
                        device=observation.candidate_edge_ids.device,
                    ),
                ).clone(),
                feasible_edge=observation.sim_descriptors.feasible_edge.clone(),
                action=_clone_action(action),
                execution=_clone_execution(execution),
            )
        )
        if done:
            break
        if next_observation is None:
            raise RuntimeError("environment returned no observation before episode end")
        if selected_mode is SubstitutionMode.ORACLE:
            observation = next_observation
        else:
            if next_model_prediction is None:
                raise RuntimeError("model condition did not produce a next-epoch prediction")
            assert policy_initializer is not None
            persistent = persistent_candidate_mask(observation, next_observation)
            initialized_descriptors = None
            if bool((~persistent).any().item()):
                initialized_descriptors = policy_initializer.initialize(
                    next_observation,
                    ~persistent,
                )
            observation = carry_policy_stream(
                observation,
                next_model_prediction,
                next_observation,
                initialized_descriptors,
            )

    return ClosedLoopResult(
        mode=selected_mode,
        episode_seed=environment.seed,
        protocol_version=PAPER_PROTOCOL_VERSION,
        protocol_fingerprint=environment.protocol_fingerprint,
        provider_fingerprint=(
            "oracle"
            if descriptor_provider is None
            else str(descriptor_provider.fingerprint)
        ),
        initializer_kind=(
            "simulator_oracle"
            if initializer_kind is None
            else initializer_kind.value
        ),
        initializer_fingerprint=(
            "oracle"
            if policy_initializer is None
            else str(policy_initializer.fingerprint)
        ),
        policy_config=controller.config,
        hard_feasibility_mask=environment.hard_feasibility_mask,
        initial_policy_strategy=(
            "simulator_oracle" if selected_mode is SubstitutionMode.ORACLE else "explicit"
        ),
        new_edge_strategy=(
            "simulator_oracle"
            if selected_mode is SubstitutionMode.ORACLE
            else "explicit_initializer_then_staged_prediction"
        ),
        records=tuple(records),
        final_serving=environment.current_serving.clone(),
        final_flow=environment.flow.clone(),
    )


def run_paired_conditions(
    environment_factory: Callable[[], PaperAlignedLEOEnv],
    *,
    modes: Iterable[SubstitutionMode | str],
    descriptor_provider_factory: Callable[[], DescriptorProvider] | None = None,
    policy_initializer_factory: Callable[[], PolicyStreamInitializer] | None = None,
    allow_oracle_warm_start: bool = False,
) -> Dict[SubstitutionMode, ClosedLoopResult]:
    """Run counterfactual conditions from fresh, identically seeded simulators.

    The paper environment keys exogenous noise by ``(seed, epoch, stream)``.
    Fresh environments with the same configuration therefore share geometry,
    user order, and PHY draws even after their executed actions diverge.
    """

    selected_modes = tuple(SubstitutionMode.parse(mode) for mode in modes)
    if not selected_modes:
        raise ValueError("at least one substitution mode is required")
    if len(set(selected_modes)) != len(selected_modes):
        raise ValueError("substitution modes must be unique")

    results: Dict[SubstitutionMode, ClosedLoopResult] = {}
    reference_seed: int | None = None
    reference_result: ClosedLoopResult | None = None
    reference_protocol_fingerprint: str | None = None
    reference_provider_fingerprint: str | None = None
    reference_initializer_fingerprint: str | None = None
    reference_initializer_kind: PolicyInitializerKind | None = None
    for selected_mode in selected_modes:
        environment = environment_factory()
        if not isinstance(environment, PaperAlignedLEOEnv):
            raise TypeError("environment_factory must return PaperAlignedLEOEnv")
        if reference_seed is None:
            reference_seed = environment.seed
        elif environment.seed != reference_seed:
            raise ValueError("paired environments must use the same episode seed")
        if reference_protocol_fingerprint is None:
            reference_protocol_fingerprint = environment.protocol_fingerprint
        elif environment.protocol_fingerprint != reference_protocol_fingerprint:
            raise ValueError("paired environments must use the same normalized configuration")
        provider = None
        initializer = None
        if selected_mode is not SubstitutionMode.ORACLE:
            if descriptor_provider_factory is None:
                raise ValueError(
                    f"mode {selected_mode.value!r} requires descriptor_provider_factory"
                )
            provider = descriptor_provider_factory()
            if reference_provider_fingerprint is None:
                reference_provider_fingerprint = provider.fingerprint
            elif provider.fingerprint != reference_provider_fingerprint:
                raise ValueError("paired model conditions must use one frozen predictor")
            if policy_initializer_factory is None:
                raise ValueError(
                    f"mode {selected_mode.value!r} requires policy_initializer_factory"
                )
            initializer = policy_initializer_factory()
            initializer_kind = PolicyInitializerKind.parse(initializer.kind)
            if reference_initializer_fingerprint is None:
                reference_initializer_fingerprint = initializer.fingerprint
                reference_initializer_kind = initializer_kind
            elif (
                initializer.fingerprint != reference_initializer_fingerprint
                or initializer_kind is not reference_initializer_kind
            ):
                raise ValueError(
                    "paired model conditions must use one frozen policy initializer"
                )
        result = run_closed_loop(
            environment,
            mode=selected_mode,
            descriptor_provider=provider,
            policy_initializer=initializer,
            allow_oracle_warm_start=allow_oracle_warm_start,
        )
        if reference_result is not None:
            if result.action_count != reference_result.action_count:
                raise ValueError("paired environments must use the same horizon")
            reference_initial = reference_result.records[0]
            current_initial = result.records[0]
            initial_pairs = (
                (reference_initial.candidate_edge_ids, current_initial.candidate_edge_ids),
                (reference_initial.feasible_edge, current_initial.feasible_edge),
                (
                    reference_initial.sim_descriptors.policy_fields.gamma_edge,
                    current_initial.sim_descriptors.policy_fields.gamma_edge,
                ),
                (
                    reference_initial.sim_descriptors.policy_fields.intensity_edge,
                    current_initial.sim_descriptors.policy_fields.intensity_edge,
                ),
                (
                    reference_initial.sim_descriptors.policy_fields.flow_node,
                    current_initial.sim_descriptors.policy_fields.flow_node,
                ),
            )
            if not all(torch.equal(left, right) for left, right in initial_pairs):
                raise ValueError(
                    "paired environments must have identical initial simulator state"
                )
        else:
            reference_result = result
        results[selected_mode] = result
    return results
