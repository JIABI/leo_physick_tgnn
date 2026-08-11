"""Training adapter for the isolated Snapshot interface.

This module supplies the differentiable core expected by the paper training
script without reusing the Intensity--Flow trainer's descriptor carriers.  It
supports recorded Snapshot-oracle episodes, scheduled model roll-in with an
explicit new-edge initializer, and action-coupled simulator rollouts.  Optimizer
and checkpoint policy remain owned by the calling script.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from ..sim.paper_environment import PaperAlignedLEOEnv
from .snapshot import (
    SnapshotFixedRankController,
    SnapshotLoss,
    SnapshotLossWeights,
    SnapshotMarginConfig,
    SnapshotOutput,
    simulator_snapshot_output,
    snapshot_one_step_loss,
)
from .snapshot_data import (
    SnapshotControlInput,
    SnapshotFeatureConfig,
    SnapshotInitializationRequest,
    SnapshotStreamInitializer,
    build_snapshot_target,
    carry_snapshot_stream,
    instantaneous_load_after_execution,
    snapshot_control_from_observation,
    snapshot_control_from_record,
    snapshot_target_from_record,
)


@dataclass(frozen=True)
class SnapshotLossSummary:
    total: float
    gamma: float
    feasibility_margin: float
    admitted_load: float
    decision_aware: float
    transitions: int
    persistent_edges: int
    model_rollin_steps: int
    rollin_opportunities: int


@dataclass(frozen=True)
class SnapshotEpisodeObjective:
    objective: torch.Tensor
    summary: SnapshotLossSummary


class _Accumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.gamma = 0.0
        self.margin = 0.0
        self.load = 0.0
        self.decision = 0.0
        self.transitions = 0
        self.persistent = 0
        self.model_steps = 0
        self.opportunities = 0

    def add(self, loss: SnapshotLoss, decision: torch.Tensor) -> None:
        self.total += float((loss.total + decision).detach().item())
        self.gamma += float(loss.gamma.detach().item())
        self.margin += float(loss.feasibility_margin.detach().item())
        self.load += float(loss.admitted_load.detach().item())
        self.decision += float(decision.detach().item())
        self.transitions += 1
        self.persistent += loss.persistent_edge_count

    def rollin(self, model: bool) -> None:
        self.opportunities += 1
        self.model_steps += int(model)

    def summary(self) -> SnapshotLossSummary:
        denominator = max(1, self.transitions)
        return SnapshotLossSummary(
            total=self.total / denominator,
            gamma=self.gamma / denominator,
            feasibility_margin=self.margin / denominator,
            admitted_load=self.load / denominator,
            decision_aware=self.decision / denominator,
            transitions=self.transitions,
            persistent_edges=self.persistent,
            model_rollin_steps=self.model_steps,
            rollin_opportunities=self.opportunities,
        )

    def merge(self, other: "_Accumulator") -> None:
        for name in ("total", "gamma", "margin", "load", "decision"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in ("transitions", "persistent", "model_steps", "opportunities"):
            setattr(self, name, getattr(self, name) + getattr(other, name))


def _sample(probability: float, generator: torch.Generator | None) -> bool:
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("model_rollin_probability must lie in [0,1]")
    if value == 0.0:
        return False
    if value == 1.0:
        return True
    return bool(torch.rand((), generator=generator).item() < value)


def _episode_records(episode: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = episode.get("steps")
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("Snapshot training episode requires at least two steps")
    if not all(isinstance(record, Mapping) for record in records):
        raise TypeError("Snapshot step records must be mappings")
    return records


def _detach_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_state(item) for item in value)
    if isinstance(value, list):
        return [_detach_state(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _detach_state(item) for key, item in value.items()}
    if value is None:
        return None
    raise TypeError(f"unsupported Snapshot recurrent state {type(value).__name__}")


class SnapshotTrainingAdapter:
    """Typed loss/roll-in adapter for any Snapshot ``predict_step`` model."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_weights: SnapshotLossWeights,
        device: torch.device | str,
        decision_aware_weight: float = 0.0,
    ) -> None:
        if not hasattr(model, "predict_step"):
            raise TypeError("Snapshot model must implement predict_step")
        if not isinstance(loss_weights, SnapshotLossWeights):
            raise TypeError("loss_weights must be SnapshotLossWeights")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_weights = loss_weights
        self.decision_aware_weight = float(decision_aware_weight)
        if not math.isfinite(self.decision_aware_weight) or self.decision_aware_weight < 0.0:
            raise ValueError("decision_aware_weight must be finite and non-negative")

    def predict(
        self, control: SnapshotControlInput, state: Any
    ) -> tuple[SnapshotOutput, Any]:
        control.validate()
        output, next_state = self.model.predict_step(
            control.as_model_step(), state, self.device
        )
        if not isinstance(output, SnapshotOutput):
            raise TypeError("Snapshot model predict_step must return SnapshotOutput")
        output.validate(control.edge_count, control.satellite_count)
        return output, next_state

    def loss(
        self,
        prediction: SnapshotOutput,
        target: Any,
        source: SnapshotControlInput,
    ) -> tuple[SnapshotLoss, torch.Tensor, torch.Tensor]:
        typed = snapshot_one_step_loss(prediction, target, self.loss_weights)
        decision = typed.total.new_zeros(())
        if self.decision_aware_weight > 0.0:
            hook = getattr(self.model, "decision_aware_loss_hook", None)
            if not callable(hook):
                raise ValueError(
                    "positive decision_aware_weight requires decision_aware_loss_hook"
                )
            raw = hook(
                prediction,
                target.as_output(),
                source.candidate_edge_ids,
                target.persistent_edge,
            )
            if raw.ndim != 0 or not bool(torch.isfinite(raw)):
                raise FloatingPointError("Snapshot decision-aware loss is invalid")
            decision = self.decision_aware_weight * raw
        return typed, decision, typed.total + decision

    def episode_objective(
        self,
        episode: Mapping[str, Any],
        *,
        model_rollin_probability: float = 0.0,
        initializer: SnapshotStreamInitializer | None = None,
        generator: torch.Generator | None = None,
    ) -> SnapshotEpisodeObjective:
        records = _episode_records(episode)
        controls = [snapshot_control_from_record(record, self.device) for record in records]
        targets = [snapshot_target_from_record(record, self.device) for record in records[:-1]]
        current = controls[0]
        state: Any = None
        objectives: list[torch.Tensor] = []
        accumulator = _Accumulator()
        for index, target in enumerate(targets):
            prediction, state = self.predict(current, state)
            typed, decision, combined = self.loss(prediction, target, current)
            objectives.append(combined)
            accumulator.add(typed, decision)
            if index + 1 < len(targets):
                use_model = _sample(model_rollin_probability, generator)
                accumulator.rollin(use_model)
                current = (
                    carry_snapshot_stream(
                        current,
                        prediction.detach(),
                        controls[index + 1],
                        initializer,
                    )
                    if use_model
                    else controls[index + 1]
                )
        objective = torch.stack(objectives).mean()
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("Snapshot episode objective is NaN or Inf")
        return SnapshotEpisodeObjective(objective=objective, summary=accumulator.summary())

    def train_epoch(
        self,
        episodes: Sequence[Mapping[str, Any]],
        *,
        optimizer: torch.optim.Optimizer,
        clip_grad_norm: float,
        batch_size_control_graphs: int = 32,
        model_rollin_probability: float = 0.0,
        initializer: SnapshotStreamInitializer | None = None,
        generator: torch.Generator | None = None,
    ) -> SnapshotLossSummary:
        if not episodes:
            raise ValueError("train_epoch requires at least one Snapshot episode")
        if (
            isinstance(batch_size_control_graphs, bool)
            or int(batch_size_control_graphs) != batch_size_control_graphs
            or int(batch_size_control_graphs) <= 0
        ):
            raise ValueError("batch_size_control_graphs must be positive")
        batch_size = int(batch_size_control_graphs)
        clip = float(clip_grad_norm)
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("clip_grad_norm must be finite and positive")
        self.model.train()
        total = _Accumulator()
        pending: list[torch.Tensor] = []
        optimizer.zero_grad(set_to_none=True)

        def apply_batch() -> None:
            if not pending:
                return
            objective = torch.stack(pending).mean()
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("Snapshot batch objective is NaN or Inf")
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            if not bool(torch.isfinite(torch.as_tensor(norm))):
                raise FloatingPointError("Snapshot gradient norm is NaN or Inf")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending.clear()

        for episode in episodes:
            records = _episode_records(episode)
            controls = [
                snapshot_control_from_record(record, self.device) for record in records
            ]
            targets = [
                snapshot_target_from_record(record, self.device)
                for record in records[:-1]
            ]
            current = controls[0]
            state: Any = None
            for index, target in enumerate(targets):
                prediction, state = self.predict(current, state)
                typed, decision, combined = self.loss(prediction, target, current)
                pending.append(combined)
                total.add(typed, decision)
                # The manuscript objective is per-control-epoch one-step TF.
                # Preserve the numerical memory but cut the gradient graph at
                # every control step (TBPTT=1).
                state = _detach_state(state)
                if index + 1 < len(targets):
                    use_model = _sample(model_rollin_probability, generator)
                    total.rollin(use_model)
                    current = (
                        carry_snapshot_stream(
                            current,
                            prediction.detach(),
                            controls[index + 1],
                            initializer,
                        )
                        if use_model
                        else controls[index + 1]
                    )
                if len(pending) == batch_size:
                    apply_batch()
        apply_batch()
        return total.summary()

    @torch.no_grad()
    def evaluate_epoch(
        self,
        episodes: Sequence[Mapping[str, Any]],
        *,
        model_rollin_probability: float = 0.0,
        initializer: SnapshotStreamInitializer | None = None,
        generator: torch.Generator | None = None,
    ) -> SnapshotLossSummary:
        self.model.eval()
        total = _Accumulator()
        for episode in episodes:
            result = self.episode_objective(
                episode,
                model_rollin_probability=model_rollin_probability,
                initializer=initializer,
                generator=generator,
            ).summary
            partial = _Accumulator()
            partial.total = result.total * result.transitions
            partial.gamma = result.gamma * result.transitions
            partial.margin = result.feasibility_margin * result.transitions
            partial.load = result.admitted_load * result.transitions
            partial.decision = result.decision_aware * result.transitions
            partial.transitions = result.transitions
            partial.persistent = result.persistent_edges
            partial.model_steps = result.model_rollin_steps
            partial.opportunities = result.rollin_opportunities
            total.merge(partial)
        return total.summary()

    def action_coupled_objective(
        self,
        environment: PaperAlignedLEOEnv,
        *,
        controller: SnapshotFixedRankController,
        margin_config: SnapshotMarginConfig,
        feature_config: SnapshotFeatureConfig,
        d0_initializer: SnapshotStreamInitializer,
        new_edge_initializer: SnapshotStreamInitializer,
        rollout_horizon: int,
        model_rollin_probability: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> SnapshotEpisodeObjective:
        """Run an action-coupled differentiable Snapshot rollout.

        D0 and new edges are always explicit.  Fresh predictions are staged for
        ``t+1`` only after action ``a_t`` has been committed.
        """

        observation = environment.reset_control()
        request = SnapshotInitializationRequest.from_observation(observation)
        all_edges = torch.ones(
            request.edge_count, dtype=torch.bool, device=request.device
        )
        initial = d0_initializer.initialize(request, all_edges)
        control = snapshot_control_from_observation(
            observation, initial, feature_config, initialized_edge=all_edges
        )
        steps = min(int(rollout_horizon), environment.horizon_steps - 1)
        if steps <= 0:
            raise ValueError("action-coupled rollout requires at least one transition")
        state: Any = None
        objectives: list[torch.Tensor] = []
        accumulator = _Accumulator()
        for index in range(steps):
            action = controller.select_action(control, control.descriptors)
            prediction, state = self.predict(control, state)
            next_observation, execution, done = environment.step_action(action)
            if done or next_observation is None:
                raise RuntimeError("environment ended before Snapshot rollout horizon")
            admitted_load = instantaneous_load_after_execution(environment, execution)
            oracle_snapshot = simulator_snapshot_output(
                next_observation, admitted_load, margin_config
            )
            oracle_control = snapshot_control_from_observation(
                next_observation, oracle_snapshot, feature_config
            )
            target = build_snapshot_target(control, oracle_control)
            typed, decision, combined = self.loss(prediction, target, control)
            objectives.append(combined)
            accumulator.add(typed, decision)
            if index + 1 < steps:
                use_model = _sample(model_rollin_probability, generator)
                accumulator.rollin(use_model)
                control = (
                    carry_snapshot_stream(
                        control,
                        prediction,
                        oracle_control,
                        new_edge_initializer,
                    )
                    if use_model
                    else oracle_control
                )
        objective = torch.stack(objectives).mean()
        return SnapshotEpisodeObjective(objective=objective, summary=accumulator.summary())

    def action_coupled_windows(
        self,
        environment: PaperAlignedLEOEnv,
        *,
        controller: SnapshotFixedRankController,
        margin_config: SnapshotMarginConfig,
        feature_config: SnapshotFeatureConfig,
        d0_initializer: SnapshotStreamInitializer,
        new_edge_initializer: SnapshotStreamInitializer,
        rollout_horizon: int,
        model_rollin_probability: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Iterator[SnapshotEpisodeObjective]:
        """Yield successive H-step losses over one complete live episode.

        Each yielded objective retains gradients only within its H-step window.
        The recurrent state and staged descriptor stream are detached before
        the yield, so an optimizer may update the model between windows without
        retaining the preceding graph. Simulator state and executed actions
        continue without reset, preserving genuine action coupling across the
        full training episode.
        """

        horizon = int(rollout_horizon)
        if horizon <= 1:
            raise ValueError("action-coupled rollout_horizon must exceed one")
        observation = environment.reset_control()
        request = SnapshotInitializationRequest.from_observation(observation)
        all_edges = torch.ones(
            request.edge_count, dtype=torch.bool, device=request.device
        )
        initial = d0_initializer.initialize(request, all_edges)
        control = snapshot_control_from_observation(
            observation, initial, feature_config, initialized_edge=all_edges
        )
        transition_count = environment.horizon_steps - 1
        if transition_count <= 0:
            raise ValueError("action-coupled training requires horizon_steps >= 2")
        state: Any = None
        objectives: list[torch.Tensor] = []
        accumulator = _Accumulator()

        for transition_index in range(transition_count):
            action = controller.select_action(control, control.descriptors)
            prediction, state = self.predict(control, state)
            next_observation, execution, done = environment.step_action(action)
            if done or next_observation is None:
                raise RuntimeError(
                    "environment ended before the final supervised transition"
                )
            admitted_load = instantaneous_load_after_execution(environment, execution)
            oracle_snapshot = simulator_snapshot_output(
                next_observation, admitted_load, margin_config
            )
            oracle_control = snapshot_control_from_observation(
                next_observation, oracle_snapshot, feature_config
            )
            target = build_snapshot_target(control, oracle_control)
            typed, decision, combined = self.loss(prediction, target, control)
            objectives.append(combined)
            accumulator.add(typed, decision)

            if transition_index + 1 < transition_count:
                use_model = _sample(model_rollin_probability, generator)
                accumulator.rollin(use_model)
                control = (
                    carry_snapshot_stream(
                        control,
                        prediction,
                        oracle_control,
                        new_edge_initializer,
                    )
                    if use_model
                    else oracle_control
                )

            boundary = (
                len(objectives) == horizon
                or transition_index + 1 == transition_count
            )
            if boundary:
                objective = torch.stack(objectives).mean()
                if not bool(torch.isfinite(objective.detach())):
                    raise FloatingPointError(
                        "Snapshot rollout-window objective is NaN or Inf"
                    )
                summary = accumulator.summary()
                # Cut both recurrent and descriptor feedback graphs at H.
                state = _detach_state(state)
                if transition_index + 1 < transition_count:
                    control = control.with_descriptors(
                        control.descriptors.detach(),
                        initialized_edge=control.initialized_edge,
                    )
                objectives = []
                accumulator = _Accumulator()
                yield SnapshotEpisodeObjective(
                    objective=objective,
                    summary=summary,
                )


__all__ = [
    "SnapshotEpisodeObjective",
    "SnapshotLossSummary",
    "SnapshotTrainingAdapter",
]
