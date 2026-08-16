"""Closed-loop evaluator for the isolated Snapshot paper conditions.

The runner stages ``D_t`` before action ``a_t`` and applies a fresh model
prediction only to ``D_{t+1}``.  Simulator feasibility and transitions remain
authoritative.  Model input and the fixed-rank controller receive only
``SnapshotControlInput``; integrated intensity and raw/EMA flow are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

import torch
import torch.nn as nn

from ..sim.paper_environment import PAPER_PROTOCOL_VERSION, PaperAlignedLEOEnv
from ..sim.state import ExecutionResult, ServingAction, SimulatorDescriptors
from .snapshot import (
    SnapshotFixedRankController,
    SnapshotMarginConfig,
    SnapshotOutput,
    simulator_snapshot_output,
)
from .snapshot_data import (
    SnapshotControlInput,
    SnapshotFeatureConfig,
    SnapshotInitializationRequest,
    SnapshotStreamInitializer,
    carry_snapshot_stream,
    instantaneous_load_after_execution,
    resolve_initial_admitted_load,
    snapshot_control_from_observation,
)


class SnapshotEvaluationMode(str, Enum):
    MODEL = "model"
    ORACLE = "oracle"

    @classmethod
    def parse(cls, value: "SnapshotEvaluationMode | str") -> "SnapshotEvaluationMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError("Snapshot evaluation mode must be model or oracle") from exc


class SnapshotDescriptorProvider(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def reset(self) -> None: ...

    def predict_next(self, control: SnapshotControlInput) -> SnapshotOutput: ...


class SnapshotModelProvider:
    """Stateful adapter from ``predict_step`` to Snapshot closed-loop timing."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str,
        fingerprint: str,
    ) -> None:
        if not hasattr(model, "predict_step"):
            raise TypeError("Snapshot model must implement predict_step")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("Snapshot provider requires an explicit checkpoint fingerprint")
        self.model = model.to(device)
        self.device = torch.device(device)
        self._fingerprint = fingerprint.strip()
        self._state: Any = None

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def reset(self) -> None:
        self._state = None

    @torch.no_grad()
    def predict_next(self, control: SnapshotControlInput) -> SnapshotOutput:
        control.validate()
        self.model.eval()
        output, self._state = self.model.predict_step(
            control.as_model_step(), self._state, self.device
        )
        if not isinstance(output, SnapshotOutput):
            raise TypeError("Snapshot provider model must return SnapshotOutput")
        output.validate(control.edge_count, control.satellite_count)
        return output.detach()


def _clone_action(value: ServingAction) -> ServingAction:
    return ServingAction(
        observation_id=value.observation_id,
        requested_serving=value.requested_serving.clone(),
    )


def _clone_execution(value: ExecutionResult) -> ExecutionResult:
    return ExecutionResult(
        observation_id=value.observation_id,
        requested_serving=value.requested_serving.clone(),
        executed_serving=value.executed_serving.clone(),
        admitted=value.admitted.clone(),
        failure_reason=value.failure_reason.clone(),
        handover_attempted=value.handover_attempted.clone(),
        handover_executed=value.handover_executed.clone(),
        flow_before=value.flow_before.clone(),
        flow_after=value.flow_after.clone(),
    )


@dataclass(frozen=True)
class SnapshotClosedLoopStep:
    observation_id: tuple[int, int]
    candidate_edge_ids: torch.Tensor
    feasible_edge: torch.Tensor
    simulator_snapshot: SnapshotOutput
    controller_snapshot: SnapshotOutput
    initialized_edge: torch.Tensor
    next_model_prediction: SnapshotOutput | None
    action: ServingAction
    execution: ExecutionResult


@dataclass(frozen=True)
class SnapshotClosedLoopResult:
    mode: SnapshotEvaluationMode
    episode_seed: int
    protocol_version: int
    protocol_fingerprint: str
    provider_fingerprint: str
    initializer_fingerprint: str
    policy_config: Any
    records: tuple[SnapshotClosedLoopStep, ...]
    final_serving: torch.Tensor
    final_flow: torch.Tensor

    @property
    def action_count(self) -> int:
        return len(self.records)


def run_snapshot_closed_loop(
    environment: PaperAlignedLEOEnv,
    *,
    mode: SnapshotEvaluationMode | str,
    controller: SnapshotFixedRankController,
    margin_config: SnapshotMarginConfig,
    feature_config: SnapshotFeatureConfig,
    initial_oracle_admitted_load: float | Sequence[float] | torch.Tensor,
    descriptor_provider: SnapshotDescriptorProvider | None = None,
    stream_initializer: SnapshotStreamInitializer | None = None,
) -> SnapshotClosedLoopResult:
    """Run one Snapshot oracle/model condition with exact staged timing."""

    selected = SnapshotEvaluationMode.parse(mode)
    if selected is SnapshotEvaluationMode.MODEL:
        if descriptor_provider is None or stream_initializer is None:
            raise ValueError("Snapshot model mode requires provider and initializer")
        descriptor_provider.reset()
    elif descriptor_provider is not None or stream_initializer is not None:
        raise ValueError("Snapshot oracle mode does not use model provider/initializer")
    if controller.config.hard_feasibility_mask != environment.hard_feasibility_mask:
        raise ValueError("Snapshot controller hard mask must match environment protocol")
    if controller.config.min_dwell_steps != environment.min_dwell_steps:
        raise ValueError("Snapshot controller dwell must match environment protocol")
    if controller.config.hysteresis != environment.hysteresis:
        raise ValueError("Snapshot controller hysteresis must match environment protocol")

    observation = environment.reset_control()
    oracle_load = resolve_initial_admitted_load(
        initial_oracle_admitted_load,
        satellite_count=environment.S,
        dtype=environment.flow.dtype,
        device=environment.device,
    )
    oracle_snapshot = simulator_snapshot_output(
        observation, oracle_load, margin_config
    )
    oracle_control = snapshot_control_from_observation(
        observation, oracle_snapshot, feature_config
    )
    if selected is SnapshotEvaluationMode.ORACLE:
        control = oracle_control
    else:
        assert stream_initializer is not None
        request = SnapshotInitializationRequest.from_observation(observation)
        all_edges = torch.ones(
            request.edge_count, dtype=torch.bool, device=request.device
        )
        initial = stream_initializer.initialize(request, all_edges)
        control = snapshot_control_from_observation(
            observation,
            initial,
            feature_config,
            initialized_edge=all_edges,
        )

    records: list[SnapshotClosedLoopStep] = []
    while True:
        has_next = observation.epoch + 1 < environment.horizon_steps
        prediction = None
        if selected is SnapshotEvaluationMode.MODEL and has_next:
            assert descriptor_provider is not None
            prediction = descriptor_provider.predict_next(control)
        action = controller.select_action(control, control.descriptors)
        next_observation, execution, done = environment.step_action(action)
        records.append(
            SnapshotClosedLoopStep(
                observation_id=control.observation_id,
                candidate_edge_ids=control.candidate_edge_ids.clone(),
                feasible_edge=control.feasible_edge.clone(),
                simulator_snapshot=oracle_snapshot.clone(),
                controller_snapshot=control.descriptors.clone(),
                initialized_edge=control.initialized_edge.clone(),
                next_model_prediction=None if prediction is None else prediction.clone(),
                action=_clone_action(action),
                execution=_clone_execution(execution),
            )
        )
        if done:
            break
        if next_observation is None:
            raise RuntimeError("Snapshot environment omitted non-terminal observation")
        oracle_load = instantaneous_load_after_execution(environment, execution)
        oracle_snapshot = simulator_snapshot_output(
            next_observation, oracle_load, margin_config
        )
        next_oracle_control = snapshot_control_from_observation(
            next_observation, oracle_snapshot, feature_config
        )
        if selected is SnapshotEvaluationMode.ORACLE:
            control = next_oracle_control
        else:
            if prediction is None:
                raise RuntimeError("Snapshot model mode omitted staged t+1 prediction")
            control = carry_snapshot_stream(
                control,
                prediction,
                next_oracle_control,
                stream_initializer,
            )
        observation = next_observation

    return SnapshotClosedLoopResult(
        mode=selected,
        episode_seed=environment.seed,
        protocol_version=PAPER_PROTOCOL_VERSION,
        protocol_fingerprint=environment.protocol_fingerprint,
        provider_fingerprint=(
            "snapshot_oracle"
            if descriptor_provider is None
            else descriptor_provider.fingerprint
        ),
        initializer_fingerprint=(
            "snapshot_oracle"
            if stream_initializer is None
            else stream_initializer.fingerprint
        ),
        policy_config=controller.config,
        records=tuple(records),
        final_serving=environment.current_serving.clone(),
        final_flow=environment.flow.clone(),
    )


__all__ = [
    "SnapshotClosedLoopResult",
    "SnapshotClosedLoopStep",
    "SnapshotDescriptorProvider",
    "SnapshotEvaluationMode",
    "SnapshotModelProvider",
    "run_snapshot_closed_loop",
]
