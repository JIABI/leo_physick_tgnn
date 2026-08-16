"""Outcome, decision-fidelity, and matched inference for UAV rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .evaluation import (
    UAVClosedLoopResult,
    UAVClosedLoopStep,
    UAVSubstitutionMode,
)
from .policy import (
    UAVFixedRankPolicyConfig,
    normalized_service_rank,
)
from .state import ServiceFailureReason
from .training import (
    UAVLossWeights,
    UAVRecordAdapter,
    uav_descriptor_one_step_loss,
)
from .protocol import UAVSharedProtocol


UAV_METRIC_CONTRACT_VERSION = 3
UAV_PINGPONG_CONTRACT = (
    "accepted station sequence A->B->A whose return epoch minus A epoch is "
    "at most ceil(window_seconds/control_dt_s)"
)
UAV_TAIL_SERVICE_CONTRACT = (
    "nearest-rank p10 of cumulative completed missions across UAVs at each "
    "epoch, followed by a uniform mean over epochs"
)


def _strict_positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class UAVOutcomeMetricConfig:
    control_dt_s: float = 1.0
    pingpong_window_seconds: float = 2.0
    tail_probability: float = 0.10
    delay_tail_probability: float = 0.90
    fifo_queue_capacity: int = 8
    active_slots_per_station: int = 2

    def __post_init__(self) -> None:
        _strict_positive("control_dt_s", self.control_dt_s)
        _strict_positive("pingpong_window_seconds", self.pingpong_window_seconds)
        if not 0.0 < float(self.tail_probability) <= 1.0:
            raise ValueError("tail_probability must lie in (0,1]")
        if not 0.0 < float(self.delay_tail_probability) <= 1.0:
            raise ValueError("delay_tail_probability must lie in (0,1]")
        if not math.isclose(
            float(self.tail_probability), 0.10, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("the named p10 UAV endpoints require tail_probability=0.10")
        if not math.isclose(
            float(self.delay_tail_probability), 0.90, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "the named p90 UAV endpoints require delay_tail_probability=0.90"
            )
        for name in ("fifo_queue_capacity", "active_slots_per_station"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class UAVEpisodeOutcomeMetrics:
    episode_seed: int
    epochs: int
    uav_count: int
    uav_time_exposure_s: float
    assignment_attempts: int
    assignment_failures: int
    assignment_failure_rate: float | None
    service_admission_attempts: int
    service_admission_failures: int
    service_failure_per_attempt: float | None
    service_failure_denominator_contract: str
    queue_admissions: int
    queue_rejections: int
    queue_rejection_rate: float | None
    service_starts: int
    service_completions: int
    service_completion_rate: float | None
    uavs_with_completion_fraction: float
    reassociations: int
    reassociation_rate_uav_second: float
    pingpong_events: int
    pingpong_rate_uav_second: float
    pingpong_window_seconds: float
    pingpong_window_steps: int
    pingpong_contract: str
    completed_wait_samples: int
    censored_wait_samples: int
    mean_waiting_delay_s: float
    p90_waiting_delay_s: float
    completed_service_delay_samples: int
    censored_service_delay_samples: int
    mean_service_delay_s: float
    p90_service_delay_s: float
    delay_tail_probability: float
    energy_depletion_events: int
    energy_depleted_uav_fraction: float
    mean_final_energy_fraction: float
    tail_service_p10_completed_missions: float
    tail_service_ratio_to_oracle: float | None
    tail_service_contract: str
    tail_residual_energy_p10: float
    tail_residual_energy_ratio_to_oracle: float | None
    station_load_cv_mean: float
    station_load_cv_p90: float
    station_flow_cv_mean: float
    station_flow_cv_p90: float
    station_flow_pressure_pearson: float | None
    failure_reason_counts: Mapping[str, int]
    metric_contract_version: int = UAV_METRIC_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UAVDecisionFidelityMetrics:
    comparisons: int
    top1_rank_agreement: float
    candidate_kendall_tau: float
    near_tie_comparisons: int
    near_tie_flip_rate: float | None
    near_tie_margin: float
    oracle_action_agreement: float
    oracle_rank_score_regret: float
    metric_contract_version: int = UAV_METRIC_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UAVPairedBootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    units: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UAVOneStepEpisodeMetrics:
    total: float
    eta: float
    log1p_intensity: float
    flow: float
    feasibility: float
    transitions: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UAVHierarchicalBootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    run_clusters: int
    episode_units: int
    seed: int
    contract: str = (
        "resample run seeds, then resample matched episodes within each sampled run"
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nearest_rank(values: torch.Tensor, probability: float) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("nearest-rank input must be a non-empty vector")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nearest-rank input contains NaN or Inf")
    if not 0.0 < float(probability) <= 1.0:
        raise ValueError("nearest-rank probability must lie in (0,1]")
    index = max(1, math.ceil(float(probability) * values.numel())) - 1
    return torch.sort(values).values[index]


def evaluate_uav_teacher_forced_one_step_episode(
    model: nn.Module,
    episode: Mapping[str, Any] | Sequence[Any],
    *,
    protocol: UAVSharedProtocol,
    loss_weights: UAVLossWeights,
    device: torch.device | str,
) -> UAVOneStepEpisodeMetrics:
    """Evaluate a serialized episode with recurrent teacher-forced inputs.

    Only ``graph`` is passed to ``predict_step``.  The typed target is consumed
    afterwards by :func:`uav_descriptor_one_step_loss`; no optimizer is created
    and no target can enter the model inference API.
    """

    if not hasattr(model, "predict_step"):
        raise TypeError("UAV one-step model must implement predict_step")
    if not isinstance(protocol, UAVSharedProtocol):
        raise TypeError("protocol must be UAVSharedProtocol")
    if not isinstance(loss_weights, UAVLossWeights):
        raise TypeError("loss_weights must be UAVLossWeights")
    target_device = torch.device(device)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = target_device
    if model_device != target_device:
        raise ValueError("one-step model and adapter must use the same device")
    if isinstance(episode, Mapping) and "seed" in episode:
        if int(episode["seed"]) != protocol.episode_seed:
            raise ValueError("serialized episode seed differs from its protocol")
    adapter = UAVRecordAdapter(protocol, target_device)
    transitions = adapter.transitions(episode)
    if not transitions:
        raise ValueError("serialized UAV test episode has no supervised transitions")
    sums = {
        "total": 0.0,
        "eta": 0.0,
        "log1p_intensity": 0.0,
        "flow": 0.0,
        "feasibility": 0.0,
    }
    state: Any = None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for graph, target in transitions:
                returned = model.predict_step(graph, state)
                if not isinstance(returned, tuple) or len(returned) != 2:
                    raise TypeError("UAV predict_step must return (prediction, state)")
                prediction, state = returned
                if state is not None and hasattr(state, "detach"):
                    state = state.detach()
                loss = uav_descriptor_one_step_loss(
                    prediction,
                    target,
                    loss_weights,
                )
                sums["total"] += float(loss.total.detach().item())
                sums["eta"] += float(loss.eta.detach().item())
                sums["log1p_intensity"] += float(
                    loss.log1p_intensity.detach().item()
                )
                sums["flow"] += float(loss.station_flow.detach().item())
                sums["feasibility"] += float(loss.feasibility.detach().item())
    finally:
        model.train(was_training)
    count = len(transitions)
    return UAVOneStepEpisodeMetrics(
        total=sums["total"] / count,
        eta=sums["eta"] / count,
        log1p_intensity=sums["log1p_intensity"] / count,
        flow=sums["flow"] / count,
        feasibility=sums["feasibility"] / count,
        transitions=count,
    )


def _sample_summary(values: Sequence[float], probability: float) -> tuple[int, float, float]:
    if not values:
        return 0, 0.0, 0.0
    tensor = torch.tensor(tuple(float(value) for value in values), dtype=torch.float64)
    return (
        int(tensor.numel()),
        float(tensor.mean().item()),
        float(_nearest_rank(tensor, probability).item()),
    )


def _pingpong_events(
    records: Sequence[UAVClosedLoopStep],
    *,
    user_count: int,
    window_steps: int,
) -> int:
    histories: list[list[tuple[int, int]]] = [[] for _ in range(user_count)]
    for epoch, record in enumerate(records):
        execution = record.execution
        for user in range(user_count):
            requested = int(execution.requested_station[user].item())
            before = int(execution.target_before[user].item())
            accepted = bool(execution.assignment_accepted[user].item())
            if requested < 0 or not accepted or requested == before:
                continue
            histories[user].append((epoch, requested))
    events = 0
    for history in histories:
        for index in range(2, len(history)):
            first_epoch, first = history[index - 2]
            _, middle = history[index - 1]
            return_epoch, returned = history[index]
            if (
                first == returned
                and first != middle
                and return_epoch - first_epoch <= window_steps
            ):
                events += 1
    return events


def _delay_samples(
    records: Sequence[UAVClosedLoopStep],
    *,
    user_count: int,
    dt: float,
) -> tuple[list[float], int, list[float], int]:
    queued_at: list[int | None] = [None] * user_count
    service_at: list[int | None] = [None] * user_count
    waiting: list[float] = []
    service: list[float] = []
    censored_wait = 0
    censored_service = 0
    for epoch, record in enumerate(records):
        execution = record.execution
        for user in range(user_count):
            # Reassociation detaches the UAV from an old queue/slot before the
            # same transition advances service. That interval is right-censored.
            if bool(execution.reassociation_executed[user].item()):
                if queued_at[user] is not None:
                    censored_wait += 1
                    queued_at[user] = None
                if service_at[user] is not None:
                    censored_service += 1
                    service_at[user] = None
            if bool(execution.energy_depleted[user].item()):
                if queued_at[user] is not None:
                    censored_wait += 1
                    queued_at[user] = None
                if service_at[user] is not None:
                    censored_service += 1
                    service_at[user] = None
            if bool(execution.queue_admitted[user].item()):
                if queued_at[user] is not None:
                    censored_wait += 1
                queued_at[user] = epoch
            if bool(execution.service_started[user].item()):
                if queued_at[user] is None:
                    waiting.append(0.0)
                else:
                    waiting.append((epoch - queued_at[user]) * dt)
                    queued_at[user] = None
                if service_at[user] is not None:
                    censored_service += 1
                service_at[user] = epoch
            if bool(execution.service_completed[user].item()):
                if service_at[user] is not None:
                    service.append((epoch - service_at[user]) * dt)
                    service_at[user] = None
    censored_wait += sum(value is not None for value in queued_at)
    censored_service += sum(value is not None for value in service_at)
    return waiting, censored_wait, service, censored_service


def _station_flow_cvs(records: Sequence[UAVClosedLoopStep]) -> list[float]:
    values: list[float] = []
    for record in records:
        flow = record.execution.station_flow_after.to(torch.float64)
        mean = float(flow.mean().item())
        values.append(
            0.0
            if mean <= 0.0
            else float((flow.std(unbiased=False) / flow.mean()).item())
        )
    return values


def _station_load_diagnostics(
    records: Sequence[UAVClosedLoopStep],
    config: UAVOutcomeMetricConfig,
) -> tuple[list[float], float | None]:
    cvs: list[float] = []
    flow_values: list[torch.Tensor] = []
    pressure_values: list[torch.Tensor] = []
    for record in records:
        execution = record.execution
        queue = execution.queue_length_after.to(torch.float64)
        active = execution.active_slots_after.to(torch.float64)
        load = queue + active
        active_load = load[load > 0.0]
        mean = (
            0.0 if active_load.numel() == 0 else float(active_load.mean().item())
        )
        cvs.append(
            0.0
            if active_load.numel() < 2 or mean <= 0.0
            else float(
                (active_load.std(unbiased=False) / active_load.mean()).item()
            )
        )
        pressure = 0.5 * (
            queue / float(config.fifo_queue_capacity)
        ) + 0.5 * (
            active / float(config.active_slots_per_station)
        )
        flow_values.append(execution.station_flow_after.to(torch.float64).cpu())
        pressure_values.append(pressure.cpu())
    flow = torch.cat(flow_values)
    pressure = torch.cat(pressure_values)
    flow_centered = flow - flow.mean()
    pressure_centered = pressure - pressure.mean()
    denominator = torch.linalg.vector_norm(flow_centered) * torch.linalg.vector_norm(
        pressure_centered
    )
    correlation = (
        None
        if float(denominator.item()) <= 0.0
        else float((flow_centered @ pressure_centered / denominator).item())
    )
    return cvs, correlation


def _tail_mission_service(
    records: Sequence[UAVClosedLoopStep],
    *,
    user_count: int,
    probability: float,
) -> float:
    """Apply the manuscript's user-first, time-second tail aggregation."""

    cumulative = torch.zeros(user_count, dtype=torch.float64)
    per_epoch: list[float] = []
    for record in records:
        cumulative += record.execution.service_completed.detach().cpu().to(
            torch.float64
        )
        per_epoch.append(float(_nearest_rank(cumulative, probability).item()))
    if not per_epoch:
        raise ValueError("tail mission service requires at least one epoch")
    return sum(per_epoch) / len(per_epoch)


def evaluate_uav_outcomes(
    result: UAVClosedLoopResult,
    *,
    config: UAVOutcomeMetricConfig | None = None,
    oracle_result: UAVClosedLoopResult | None = None,
) -> UAVEpisodeOutcomeMetrics:
    """Compute one episode's outcomes; epochs are not treated as IID units."""

    config = config or UAVOutcomeMetricConfig()
    if not result.records:
        raise ValueError("UAV closed-loop result contains no epochs")
    if oracle_result is not None:
        if (
            oracle_result.episode_seed != result.episode_seed
            or oracle_result.action_count != result.action_count
            or oracle_result.protocol_fingerprint != result.protocol_fingerprint
        ):
            raise ValueError("oracle result is not the matched episode/protocol")
    users = int(result.final_energy_fraction.numel())
    epochs = len(result.records)
    dt = float(config.control_dt_s)
    duration = epochs * dt

    assignment_attempts = sum(
        int(record.execution.assignment_attempted.sum().item())
        for record in result.records
    )
    assignment_failures = sum(
        int(
            (
                record.execution.assignment_attempted
                & ~record.execution.assignment_accepted
            ).sum().item()
        )
        for record in result.records
    )
    queue_admissions = sum(
        int(record.execution.queue_admitted.sum().item())
        for record in result.records
    )
    service_starts = sum(
        int(record.execution.service_started.sum().item())
        for record in result.records
    )
    service_completions = sum(
        int(record.execution.service_completed.sum().item())
        for record in result.records
    )
    energy_depletions = sum(
        int(record.execution.energy_depleted.sum().item())
        for record in result.records
    )
    reassociations = sum(
        int(record.execution.reassociation_executed.sum().item())
        for record in result.records
    )

    reason_counts = {
        reason.name.lower(): sum(
            int((record.execution.failure_reason == int(reason)).sum().item())
            for record in result.records
        )
        for reason in ServiceFailureReason
        if reason is not ServiceFailureReason.NONE
    }
    queue_rejections = reason_counts[ServiceFailureReason.QUEUE_FULL.name.lower()]
    reserve_failures = reason_counts[
        ServiceFailureReason.RESERVE_VIOLATION.name.lower()
    ]
    service_admission_failures = sum(reason_counts.values())
    service_admission_attempts = (
        assignment_attempts + reserve_failures + queue_rejections
    )

    waiting, censored_wait, service, censored_service = _delay_samples(
        result.records,
        user_count=users,
        dt=dt,
    )
    wait_count, wait_mean, wait_tail = _sample_summary(
        waiting, config.delay_tail_probability
    )
    service_count, service_mean, service_tail = _sample_summary(
        service, config.delay_tail_probability
    )
    window_steps = max(
        1, math.ceil(config.pingpong_window_seconds / config.control_dt_s)
    )
    pingpong = _pingpong_events(
        result.records,
        user_count=users,
        window_steps=window_steps,
    )
    flow_cvs = _station_flow_cvs(result.records)
    _, flow_cv_mean, flow_cv_tail = _sample_summary(
        flow_cvs, config.delay_tail_probability
    )
    station_load_cvs, flow_pressure_pearson = _station_load_diagnostics(
        result.records, config
    )
    _, station_load_cv_mean, station_load_cv_tail = _sample_summary(
        station_load_cvs, config.delay_tail_probability
    )
    missions = result.final_missions_completed.to(torch.float64).cpu()
    tail_completion = _tail_mission_service(
        result.records,
        user_count=users,
        probability=config.tail_probability,
    )
    tail_ratio: float | None = None
    residual_energy = result.final_energy_fraction.to(torch.float64).cpu()
    tail_residual_energy = float(
        _nearest_rank(residual_energy, config.tail_probability).item()
    )
    tail_residual_ratio: float | None = None
    if oracle_result is not None:
        oracle_tail = _tail_mission_service(
            oracle_result.records,
            user_count=users,
            probability=config.tail_probability,
        )
        tail_ratio = tail_completion / oracle_tail if oracle_tail > 0.0 else None
        oracle_residual_tail = float(
            _nearest_rank(
                oracle_result.final_energy_fraction.to(torch.float64).cpu(),
                config.tail_probability,
            ).item()
        )
        tail_residual_ratio = (
            tail_residual_energy / oracle_residual_tail
            if oracle_residual_tail > 0.0
            else None
        )

    depleted_ever = torch.zeros(users, dtype=torch.bool)
    for record in result.records:
        depleted_ever |= record.execution.energy_depleted.detach().cpu()
    return UAVEpisodeOutcomeMetrics(
        episode_seed=result.episode_seed,
        epochs=epochs,
        uav_count=users,
        uav_time_exposure_s=users * duration,
        assignment_attempts=assignment_attempts,
        assignment_failures=assignment_failures,
        assignment_failure_rate=(
            assignment_failures / assignment_attempts
            if assignment_attempts
            else None
        ),
        service_admission_attempts=service_admission_attempts,
        service_admission_failures=service_admission_failures,
        service_failure_per_attempt=(
            service_admission_failures / service_admission_attempts
            if service_admission_attempts
            else None
        ),
        service_failure_denominator_contract=(
            "assignment_attempts + reserve_violation_events + queue_full_events"
        ),
        queue_admissions=queue_admissions,
        queue_rejections=queue_rejections,
        queue_rejection_rate=(
            queue_rejections / (queue_admissions + queue_rejections)
            if queue_admissions + queue_rejections
            else None
        ),
        service_starts=service_starts,
        service_completions=service_completions,
        service_completion_rate=(
            service_completions / service_starts if service_starts else None
        ),
        uavs_with_completion_fraction=float((missions > 0).to(torch.float64).mean().item()),
        reassociations=reassociations,
        reassociation_rate_uav_second=reassociations / (users * duration),
        pingpong_events=pingpong,
        pingpong_rate_uav_second=pingpong / (users * duration),
        pingpong_window_seconds=float(config.pingpong_window_seconds),
        pingpong_window_steps=window_steps,
        pingpong_contract=UAV_PINGPONG_CONTRACT,
        completed_wait_samples=wait_count,
        censored_wait_samples=censored_wait,
        mean_waiting_delay_s=wait_mean,
        p90_waiting_delay_s=wait_tail,
        completed_service_delay_samples=service_count,
        censored_service_delay_samples=censored_service,
        mean_service_delay_s=service_mean,
        p90_service_delay_s=service_tail,
        delay_tail_probability=float(config.delay_tail_probability),
        energy_depletion_events=energy_depletions,
        energy_depleted_uav_fraction=float(
            depleted_ever.to(torch.float64).mean().item()
        ),
        mean_final_energy_fraction=float(
            result.final_energy_fraction.to(torch.float64).mean().item()
        ),
        tail_service_p10_completed_missions=tail_completion,
        tail_service_ratio_to_oracle=tail_ratio,
        tail_service_contract=UAV_TAIL_SERVICE_CONTRACT,
        tail_residual_energy_p10=tail_residual_energy,
        tail_residual_energy_ratio_to_oracle=tail_residual_ratio,
        station_load_cv_mean=station_load_cv_mean,
        station_load_cv_p90=station_load_cv_tail,
        station_flow_cv_mean=flow_cv_mean,
        station_flow_cv_p90=flow_cv_tail,
        station_flow_pressure_pearson=flow_pressure_pearson,
        failure_reason_counts=reason_counts,
    )


def _policy_config(result: UAVClosedLoopResult) -> UAVFixedRankPolicyConfig:
    value: Any = result.policy_config
    if isinstance(value, Mapping) and isinstance(value.get("config"), Mapping):
        value = value["config"]
    if not isinstance(value, Mapping):
        raise TypeError("result policy_config does not contain a mapping")
    fields = set(UAVFixedRankPolicyConfig.__dataclass_fields__)
    return UAVFixedRankPolicyConfig(
        **{key: item for key, item in value.items() if key in fields}
    )


def _candidate_scores(
    record: UAVClosedLoopStep,
    descriptors: Any,
    config: UAVFixedRankPolicyConfig,
) -> torch.Tensor:
    edge_ids = record.candidate_edge_ids
    result = torch.full_like(descriptors.eta_edge, -torch.inf)
    for user in torch.unique(edge_ids[:, 0]).tolist():
        rows = torch.nonzero(edge_ids[:, 0] == int(user), as_tuple=False).flatten()
        if config.hard_feasible_start_mask:
            rows = rows[record.feasible_start_edge.index_select(0, rows)]
        if rows.numel() == 0:
            continue
        stations = edge_ids.index_select(0, rows)[:, 1]
        eta = normalized_service_rank(
            descriptors.eta_edge.index_select(0, rows),
            stations,
            higher_is_better=True,
        )
        intensity = normalized_service_rank(
            descriptors.intensity_edge.index_select(0, rows),
            stations,
            higher_is_better=False,
        )
        flow = normalized_service_rank(
            descriptors.station_flow_node.index_select(0, stations),
            stations,
            higher_is_better=False,
        )
        result[rows] = (
            config.eta_weight * eta
            + config.intensity_weight * intensity
            + config.flow_weight * flow
        )
    return result


def _ordered_rows(
    rows: torch.Tensor,
    scores: torch.Tensor,
    station_ids: torch.Tensor,
) -> torch.Tensor:
    by_station = rows.index_select(
        0,
        torch.argsort(station_ids.index_select(0, rows), stable=True),
    )
    return by_station.index_select(
        0,
        torch.argsort(
            scores.index_select(0, by_station), descending=True, stable=True
        ),
    )


def _kendall_tau(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.numel() <= 1:
        return 1.0
    concordant = 0
    discordant = 0
    for left in range(first.numel()):
        for right in range(left + 1, first.numel()):
            product = float(
                ((first[left] - first[right]) * (second[left] - second[right])).item()
            )
            if product > 0.0:
                concordant += 1
            elif product < 0.0:
                discordant += 1
    denominator = concordant + discordant
    return 0.0 if denominator == 0 else (concordant - discordant) / denominator


def _reference_oracle_station(
    record: UAVClosedLoopStep,
    user: int,
    rows: torch.Tensor,
    scores: torch.Tensor,
    config: UAVFixedRankPolicyConfig,
) -> int:
    station_ids = record.candidate_edge_ids[:, 1]
    eligible = rows[torch.isfinite(scores.index_select(0, rows))]
    if eligible.numel() == 0:
        return -1
    ordered = _ordered_rows(eligible, scores, station_ids)
    best_edge = int(ordered[0].item())
    best_station = int(station_ids[best_edge].item())
    current = int(record.current_station_before[user].item())
    if current < 0:
        return best_station
    current_rows = rows[station_ids.index_select(0, rows) == current]
    if current_rows.numel() != 1 or not bool(
        torch.isfinite(scores[int(current_rows[0].item())]).item()
    ):
        return best_station
    if current == best_station:
        return current
    current_edge = int(current_rows[0].item())
    hysteresis = config.hysteresis
    if hysteresis is None:
        hysteresis = 1.0 / float(eligible.numel())
    if float(scores[best_edge].item()) + 1e-8 >= (
        float(scores[current_edge].item()) + float(hysteresis)
    ):
        return best_station
    return current


def evaluate_uav_decision_fidelity(
    result: UAVClosedLoopResult,
    *,
    near_tie_margin: float = 1.0 / 3.0,
) -> UAVDecisionFidelityMetrics:
    """Compare policy-facing ranks/actions with the same-state oracle policy."""

    near_tie_margin = float(near_tie_margin)
    if not math.isfinite(near_tie_margin) or near_tie_margin < 0.0:
        raise ValueError("near_tie_margin must be finite and non-negative")
    config = _policy_config(result)
    top1: list[float] = []
    tau: list[float] = []
    action_agreement: list[float] = []
    regret: list[float] = []
    near_tie_flips: list[float] = []
    for record in result.records:
        policy_scores = _candidate_scores(
            record, record.policy_descriptors, config
        )
        oracle_scores = _candidate_scores(
            record, record.simulator_descriptors, config
        )
        edge_users = record.candidate_edge_ids[:, 0]
        stations = record.candidate_edge_ids[:, 1]
        for user in torch.unique(edge_users).tolist():
            rows = torch.nonzero(edge_users == int(user), as_tuple=False).flatten()
            comparable = rows[
                torch.isfinite(policy_scores.index_select(0, rows))
                & torch.isfinite(oracle_scores.index_select(0, rows))
            ]
            if comparable.numel() == 0:
                continue
            policy_order = _ordered_rows(comparable, policy_scores, stations)
            oracle_order = _ordered_rows(comparable, oracle_scores, stations)
            policy_best = int(policy_order[0].item())
            oracle_best = int(oracle_order[0].item())
            top1.append(float(policy_best == oracle_best))
            if oracle_order.numel() > 1:
                oracle_gap = float(
                    (
                        oracle_scores[int(oracle_order[0].item())]
                        - oracle_scores[int(oracle_order[1].item())]
                    ).item()
                )
                if oracle_gap <= near_tie_margin:
                    near_tie_flips.append(float(policy_best != oracle_best))
            tau.append(
                _kendall_tau(
                    policy_scores.index_select(0, comparable),
                    oracle_scores.index_select(0, comparable),
                )
            )
            requested = int(record.action.requested_station[int(user)].item())
            reference = _reference_oracle_station(
                record,
                int(user),
                rows,
                oracle_scores,
                config,
            )
            action_agreement.append(float(requested == reference))
            requested_rows = comparable[
                stations.index_select(0, comparable) == requested
            ]
            requested_score = (
                float(oracle_scores[int(requested_rows[0].item())].item())
                if requested_rows.numel() == 1
                else 0.0
            )
            oracle_best_score = float(oracle_scores[oracle_best].item())
            regret.append(max(0.0, oracle_best_score - requested_score))
    if not top1:
        raise ValueError("no UAV policy/oracle candidate comparisons are available")
    return UAVDecisionFidelityMetrics(
        comparisons=len(top1),
        top1_rank_agreement=sum(top1) / len(top1),
        candidate_kendall_tau=sum(tau) / len(tau),
        near_tie_comparisons=len(near_tie_flips),
        near_tie_flip_rate=(
            sum(near_tie_flips) / len(near_tie_flips)
            if near_tie_flips
            else None
        ),
        near_tie_margin=near_tie_margin,
        oracle_action_agreement=sum(action_agreement) / len(action_agreement),
        oracle_rank_score_regret=sum(regret) / len(regret),
    )


def evaluate_uav_result(
    result: UAVClosedLoopResult,
    *,
    outcome_config: UAVOutcomeMetricConfig | None = None,
    oracle_result: UAVClosedLoopResult | None = None,
    near_tie_margin: float = 1.0 / 3.0,
) -> dict[str, Any]:
    return {
        "episode_seed": result.episode_seed,
        "mode": result.mode.value,
        "protocol_fingerprint": result.protocol_fingerprint,
        "paired_fingerprint": result.paired_fingerprint,
        "outcomes": evaluate_uav_outcomes(
            result,
            config=outcome_config,
            oracle_result=oracle_result,
        ).as_dict(),
        "decision_fidelity": evaluate_uav_decision_fidelity(
            result, near_tie_margin=near_tie_margin
        ).as_dict(),
    }


def evaluate_uav_paired_results(
    results: Mapping[UAVSubstitutionMode | str, UAVClosedLoopResult],
    *,
    outcome_config: UAVOutcomeMetricConfig | None = None,
    near_tie_margin: float = 1.0 / 3.0,
) -> dict[str, dict[str, Any]]:
    normalized = {
        UAVSubstitutionMode.parse(mode): result for mode, result in results.items()
    }
    oracle = normalized.get(UAVSubstitutionMode.ORACLE)
    if oracle is None:
        raise ValueError("paired UAV metrics require an oracle condition")
    fingerprints = {result.paired_fingerprint for result in normalized.values()}
    seeds = {result.episode_seed for result in normalized.values()}
    if len(fingerprints) != 1 or len(seeds) != 1:
        raise ValueError("UAV results are not one matched paired unit")
    return {
        mode.value: evaluate_uav_result(
            result,
            outcome_config=outcome_config,
            oracle_result=oracle,
            near_tie_margin=near_tie_margin,
        )
        for mode, result in normalized.items()
    }


def paired_percentile_bootstrap(
    paired_differences: Sequence[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17001,
) -> UAVPairedBootstrapInterval:
    """Bootstrap matched episode/seed units; never resample decision epochs."""

    values = torch.tensor(
        tuple(float(value) for value in paired_differences), dtype=torch.float64
    )
    if values.ndim != 1 or values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("paired_differences must be a non-empty finite vector")
    if isinstance(resamples, bool) or int(resamples) != resamples or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        0,
        values.numel(),
        (int(resamples), values.numel()),
        generator=generator,
    )
    draws = values[indices].mean(dim=1)
    tail = (1.0 - float(confidence)) / 2.0
    return UAVPairedBootstrapInterval(
        estimate=float(values.mean().item()),
        lower=float(_nearest_rank(draws, tail).item()),
        upper=float(_nearest_rank(draws, 1.0 - tail).item()),
        confidence=float(confidence),
        resamples=int(resamples),
        units=int(values.numel()),
        seed=int(seed),
    )


def hierarchical_paired_bootstrap(
    paired_differences: Sequence[float],
    run_seeds: Sequence[int],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17001,
) -> UAVHierarchicalBootstrapInterval:
    """Primary inference over independent training-run clusters.

    A bootstrap draw first samples run seeds with replacement.  For every
    sampled run occurrence, its matched test episodes are then sampled with
    replacement.  The draw is the uniform mean of sampled-run means, preventing
    5 x 80 episodes from being misrepresented as 400 independent training runs.
    """

    differences = torch.tensor(
        tuple(float(value) for value in paired_differences), dtype=torch.float64
    )
    clusters = tuple(int(value) for value in run_seeds)
    if differences.ndim != 1 or differences.numel() == 0:
        raise ValueError("paired_differences must be a non-empty vector")
    if len(clusters) != differences.numel():
        raise ValueError("run_seeds must align one-to-one with paired differences")
    if not bool(torch.isfinite(differences).all()):
        raise ValueError("paired differences contain NaN or Inf")
    if isinstance(resamples, bool) or int(resamples) != resamples or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    unique_runs = tuple(sorted(set(clusters)))
    if not unique_runs:
        raise ValueError("at least one run cluster is required")
    cluster_values = [
        differences[
            torch.tensor(
                [value == run_seed for value in clusters], dtype=torch.bool
            )
        ]
        for run_seed in unique_runs
    ]
    if any(values.numel() == 0 for values in cluster_values):
        raise RuntimeError("hierarchical bootstrap constructed an empty run")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    draws_per_cluster: list[torch.Tensor] = []
    cluster_count = len(cluster_values)
    for values in cluster_values:
        indices = torch.randint(
            0,
            values.numel(),
            (int(resamples), cluster_count, values.numel()),
            generator=generator,
        )
        draws_per_cluster.append(values[indices].mean(dim=2))
    # [bootstrap, sampled-run-position, available-run]
    available = torch.stack(draws_per_cluster, dim=2)
    sampled_runs = torch.randint(
        0,
        cluster_count,
        (int(resamples), cluster_count),
        generator=generator,
    )
    selected = available.gather(2, sampled_runs.unsqueeze(2)).squeeze(2)
    draws = selected.mean(dim=1)
    estimate = torch.stack([values.mean() for values in cluster_values]).mean()
    tail = (1.0 - float(confidence)) / 2.0
    return UAVHierarchicalBootstrapInterval(
        estimate=float(estimate.item()),
        lower=float(_nearest_rank(draws, tail).item()),
        upper=float(_nearest_rank(draws, 1.0 - tail).item()),
        confidence=float(confidence),
        resamples=int(resamples),
        run_clusters=cluster_count,
        episode_units=int(differences.numel()),
        seed=int(seed),
    )


_DIRECTIONS: Mapping[str, str] = {
    "assignment_failure_rate": "lower_is_better",
    "service_failure_per_attempt": "lower_is_better",
    "queue_rejection_rate": "lower_is_better",
    "service_completion_rate": "higher_is_better",
    "uavs_with_completion_fraction": "higher_is_better",
    "reassociation_rate_uav_second": "lower_is_better",
    "pingpong_rate_uav_second": "lower_is_better",
    "mean_waiting_delay_s": "lower_is_better",
    "p90_waiting_delay_s": "lower_is_better",
    "mean_service_delay_s": "lower_is_better",
    "p90_service_delay_s": "lower_is_better",
    "energy_depleted_uav_fraction": "lower_is_better",
    "mean_final_energy_fraction": "higher_is_better",
    "tail_service_p10_completed_missions": "higher_is_better",
    "tail_service_ratio_to_oracle": "higher_is_better",
    "tail_residual_energy_p10": "higher_is_better",
    "tail_residual_energy_ratio_to_oracle": "higher_is_better",
    "station_load_cv_mean": "lower_is_better",
    "station_load_cv_p90": "lower_is_better",
    "station_flow_cv_mean": "lower_is_better",
    "station_flow_cv_p90": "lower_is_better",
    "station_flow_pressure_pearson": "diagnostic",
    "top1_rank_agreement": "higher_is_better",
    "candidate_kendall_tau": "higher_is_better",
    "near_tie_flip_rate": "lower_is_better",
    "oracle_action_agreement": "higher_is_better",
    "oracle_rank_score_regret": "lower_is_better",
}


def _numeric_endpoints(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for section in ("outcomes", "decision_fidelity"):
        source = row.get(section)
        if not isinstance(source, Mapping):
            raise TypeError(f"metric row is missing mapping {section}")
        for name in _DIRECTIONS:
            value = source.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                values[name] = numeric
    return values


def aggregate_matched_uav_metrics(
    units: Sequence[Mapping[str, Any]],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 17001,
) -> dict[str, Any]:
    """Aggregate mode-minus-oracle effects over matched rollout units.

    Each unit must contain ``run_seed``, ``local_episode_index``,
    ``episode_seed``, ``paired_episode_id``, ``exogenous_sequence_id``,
    ``paired_fingerprint``, and a ``metrics`` mapping returned by
    :func:`evaluate_uav_paired_results`.
    """

    if not units:
        raise ValueError("matched UAV aggregation requires at least one unit")
    materialized = [dict(unit) for unit in units]
    identities: list[dict[str, Any]] = []
    mode_names: tuple[str, ...] | None = None
    rows_by_mode: dict[str, list[Mapping[str, Any]]] = {}
    one_step_rows: list[Mapping[str, Any]] = []
    one_step_presence: set[bool] = set()
    seen_units: set[tuple[int, int]] = set()
    seen_episode_seeds: set[int] = set()
    for unit in materialized:
        episode_seed = int(unit["episode_seed"])
        paired_episode_id = str(
            unit.get(
                "paired_episode_id",
                (
                    f"uav_run_{int(unit['run_seed'])}_episode_"
                    f"{int(unit['local_episode_index'])}"
                ),
            )
        )
        exogenous_sequence_id = int(
            unit.get("exogenous_sequence_id", episode_seed)
        )
        paired_fingerprint = str(unit["paired_fingerprint"])
        run_seed = int(unit["run_seed"])
        local_episode_index = int(unit["local_episode_index"])
        key = (run_seed, local_episode_index)
        if key in seen_units or episode_seed in seen_episode_seeds:
            raise ValueError("duplicate matched UAV evaluation unit")
        if exogenous_sequence_id != episode_seed:
            raise ValueError(
                "exogenous_sequence_id must equal the simulator episode seed"
            )
        seen_units.add(key)
        seen_episode_seeds.add(episode_seed)
        metric_container = unit.get("metrics")
        if not isinstance(metric_container, Mapping):
            raise ValueError("every unit requires a metrics mapping")
        if "closed_loop" in metric_container:
            metrics = metric_container.get("closed_loop")
            one_step = metric_container.get("one_step_model")
        else:
            metrics = metric_container
            one_step = unit.get("one_step_model")
        if not isinstance(metrics, Mapping) or "oracle" not in metrics:
            raise ValueError("every unit requires metrics with an oracle row")
        one_step_presence.add(one_step is not None)
        if one_step is not None:
            if not isinstance(one_step, Mapping):
                raise TypeError("one_step_model must be a mapping")
            one_step_rows.append(one_step)
        observed = tuple(sorted(str(name) for name in metrics))
        if mode_names is None:
            mode_names = observed
            rows_by_mode = {name: [] for name in observed}
        elif observed != mode_names:
            raise ValueError("substitution modes differ across matched units")
        identities.append(
            {
                "run_seed": run_seed,
                "local_episode_index": local_episode_index,
                "episode_seed": episode_seed,
                "paired_episode_id": paired_episode_id,
                "exogenous_sequence_id": exogenous_sequence_id,
                "paired_fingerprint": paired_fingerprint,
            }
        )
        for name in observed:
            row = metrics[name]
            if not isinstance(row, Mapping):
                raise TypeError("each mode metric row must be a mapping")
            if int(row.get("episode_seed", -1)) != episode_seed:
                raise ValueError("metric row episode seed differs from its unit")
            if str(row.get("paired_fingerprint", "")) != paired_fingerprint:
                raise ValueError("metric row paired fingerprint differs from its unit")
            rows_by_mode[name].append(row)
    if len(one_step_presence) > 1:
        raise ValueError("one-step loss must be present for either all or no units")
    assert mode_names is not None
    oracle_rows = [_numeric_endpoints(row) for row in rows_by_mode["oracle"]]
    run_identity = [identity["run_seed"] for identity in identities]

    def run_weighted_mean(values: Sequence[float], runs: Sequence[int]) -> float:
        unique_runs = sorted(set(runs))
        return sum(
            sum(value for value, run in zip(values, runs) if run == run_seed)
            / sum(1 for run in runs if run == run_seed)
            for run_seed in unique_runs
        ) / len(unique_runs)

    summaries: dict[str, Any] = {}
    effects: dict[str, Any] = {}
    for mode in mode_names:
        condition_rows = [_numeric_endpoints(row) for row in rows_by_mode[mode]]
        names = sorted(set().union(*(set(row) for row in condition_rows)))
        summaries[mode] = {}
        if mode != "oracle":
            effects[mode] = {}
        for name in names:
            valid = [index for index, row in enumerate(condition_rows) if name in row]
            values = [condition_rows[index][name] for index in valid]
            runs = [run_identity[index] for index in valid]
            summaries[mode][name] = {
                "mean": run_weighted_mean(values, runs),
                "units": len(values),
                "run_clusters": len(set(runs)),
                "omitted_undefined_units": len(condition_rows) - len(values),
                "aggregation": "uniform mean of per-run episode means",
                "direction": _DIRECTIONS[name],
            }
            if mode == "oracle":
                continue
            paired_indices = [
                index
                for index, (condition, oracle) in enumerate(
                    zip(condition_rows, oracle_rows)
                )
                if name in condition and name in oracle
            ]
            if not paired_indices:
                continue
            differences = [
                condition_rows[index][name] - oracle_rows[index][name]
                for index in paired_indices
            ]
            paired_runs = [run_identity[index] for index in paired_indices]
            primary_interval = hierarchical_paired_bootstrap(
                differences,
                paired_runs,
                resamples=resamples,
                confidence=confidence,
                seed=bootstrap_seed,
            )
            sensitivity_interval = paired_percentile_bootstrap(
                differences,
                resamples=resamples,
                confidence=confidence,
                seed=bootstrap_seed + 1,
            )
            effects[mode][name] = {
                "difference_definition": "condition - matched oracle",
                "direction": _DIRECTIONS[name],
                "unit_differences": [
                    {**identities[index], "value": value}
                    for index, value in zip(paired_indices, differences)
                ],
                "omitted_undefined_pairs": len(condition_rows) - len(differences),
                "hierarchical_interval_primary": primary_interval.as_dict(),
                "episode_level_sensitivity_interval": sensitivity_interval.as_dict(),
            }
    one_step_aggregate: dict[str, Any] | None = None
    if one_step_rows:
        required = (
            "total",
            "eta",
            "log1p_intensity",
            "flow",
            "feasibility",
            "transitions",
        )
        for row in one_step_rows:
            missing = [name for name in required if name not in row]
            if missing:
                raise ValueError(f"one-step metric row is missing {missing!r}")
        loss_summaries: dict[str, Any] = {}
        for name in required[:-1]:
            values = [float(row[name]) for row in one_step_rows]
            if not all(math.isfinite(value) and value >= 0.0 for value in values):
                raise ValueError(f"one-step {name} values must be finite/non-negative")
            primary = hierarchical_paired_bootstrap(
                values,
                run_identity,
                resamples=resamples,
                confidence=confidence,
                seed=bootstrap_seed,
            )
            sensitivity = paired_percentile_bootstrap(
                values,
                resamples=resamples,
                confidence=confidence,
                seed=bootstrap_seed + 1,
            )
            loss_summaries[name] = {
                "mean": run_weighted_mean(values, run_identity),
                "direction": "lower_is_better",
                "aggregation": "uniform mean of per-run episode means",
                "unit_values": [
                    {**identity, "value": value}
                    for identity, value in zip(identities, values)
                ],
                "hierarchical_interval_primary": primary.as_dict(),
                "episode_level_sensitivity_interval": sensitivity.as_dict(),
            }
        transition_counts = [int(row["transitions"]) for row in one_step_rows]
        if any(value <= 0 for value in transition_counts):
            raise ValueError("one-step transition counts must be positive")
        one_step_aggregate = {
            "condition": "model_teacher_forced_recurrent",
            "oracle_gap": None,
            "target_not_passed_to_predict_step": True,
            "run_clusters": len(set(run_identity)),
            "episode_units": len(one_step_rows),
            "transitions": {
                "minimum_per_episode": min(transition_counts),
                "maximum_per_episode": max(transition_counts),
                "total": sum(transition_counts),
            },
            "losses": loss_summaries,
        }
    return {
        "metric_contract_version": UAV_METRIC_CONTRACT_VERSION,
        "independent_unit": "training run seed cluster",
        "within_cluster_unit": "matched test episode",
        "units": identities,
        "summaries": summaries,
        "paired_effects": effects,
        "one_step_model": one_step_aggregate,
        "bootstrap": {
            "primary_kind": "matched hierarchical run/episode bootstrap",
            "sensitivity_kind": "flat matched episode percentile bootstrap",
            "resamples": int(resamples),
            "confidence": float(confidence),
            "primary_seed": int(bootstrap_seed),
            "sensitivity_seed": int(bootstrap_seed) + 1,
        },
    }


__all__ = [
    "UAVDecisionFidelityMetrics",
    "UAVEpisodeOutcomeMetrics",
    "UAVHierarchicalBootstrapInterval",
    "UAVOutcomeMetricConfig",
    "UAVOneStepEpisodeMetrics",
    "UAVPairedBootstrapInterval",
    "UAV_METRIC_CONTRACT_VERSION",
    "UAV_PINGPONG_CONTRACT",
    "UAV_TAIL_SERVICE_CONTRACT",
    "aggregate_matched_uav_metrics",
    "evaluate_uav_decision_fidelity",
    "evaluate_uav_outcomes",
    "evaluate_uav_paired_results",
    "evaluate_uav_result",
    "evaluate_uav_teacher_forced_one_step_episode",
    "hierarchical_paired_bootstrap",
    "paired_percentile_bootstrap",
]
