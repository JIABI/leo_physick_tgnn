"""Paper endpoint and paired-inference definitions for closed-loop traces."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

from leo_pg.control.policy import FixedRankPolicyConfig, normalized_ordinal_rank
from leo_pg.eval.closed_loop import ClosedLoopResult, ClosedLoopStep
from leo_pg.paper.calibration import nearest_rank_quantile
from leo_pg.sim.state import PolicyDescriptors


METRIC_CONTRACT_VERSION = 2
METRIC_AGGREGATION_SCHEMA_VERSION = 1
TAIL_AGGREGATION_CONTRACT = (
    "nearest-rank p10 across users at each epoch, then uniform mean across epochs"
)


@dataclass(frozen=True)
class OutcomeMetricConfig:
    dt_ctrl: float = 0.1
    outage_gamma_db: float = -3.0
    pingpong_window_seconds: float = 2.0
    bandwidth_hz: float = 1.0
    throughput_load_discount: bool = True
    user_tail_probability: float = 0.10

    def __post_init__(self) -> None:
        for name in ("dt_ctrl", "pingpong_window_seconds", "bandwidth_hz"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.outage_gamma_db)):
            raise ValueError("outage_gamma_db must be finite")
        if not 0.0 < float(self.user_tail_probability) <= 1.0:
            raise ValueError("user_tail_probability must lie in (0,1]")


@dataclass(frozen=True)
class EpisodeOutcomeMetrics:
    episode_seed: int
    epochs: int
    user_count: int
    outage_rate: float
    handover_attempts: int
    handover_failures: int
    handover_failure_rate: float
    handovers_executed: int
    pingpong_events: int
    pingpong_rate_user_second: float
    pingpong_fraction_executed_decisions: float
    user_p10_throughput_mean: float
    user_p10_throughput_ratio_to_oracle: float | None
    active_satellite_load_cv_mean: float
    mean_throughput: float
    throughput_contract: str
    tail_aggregation_contract: str = TAIL_AGGREGATION_CONTRACT
    metric_contract_version: int = METRIC_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _served_gamma(step: ClosedLoopStep) -> torch.Tensor:
    users = int(step.execution.executed_serving.numel())
    result = torch.full(
        (users,),
        -torch.inf,
        dtype=step.sim_descriptors.policy_fields.gamma_edge.dtype,
        device=step.sim_descriptors.policy_fields.gamma_edge.device,
    )
    lookup = {
        (int(user), int(satellite)): edge
        for edge, (user, satellite) in enumerate(step.candidate_edge_ids.tolist())
    }
    for user, satellite in enumerate(step.execution.executed_serving.tolist()):
        edge = lookup.get((user, int(satellite)))
        if edge is not None:
            result[user] = step.sim_descriptors.policy_fields.gamma_edge[edge]
    return result


def _throughput(step: ClosedLoopStep, config: OutcomeMetricConfig) -> torch.Tensor:
    gamma_db = _served_gamma(step)
    connected = step.execution.executed_serving >= 0
    gamma_linear = torch.pow(
        torch.tensor(10.0, dtype=gamma_db.dtype, device=gamma_db.device),
        gamma_db / 10.0,
    )
    rate = config.bandwidth_hz * torch.log2(1.0 + gamma_linear)
    rate = torch.where(connected, rate, torch.zeros_like(rate))
    if config.throughput_load_discount:
        serving = step.execution.executed_serving.clamp_min(0)
        discount = (1.0 - step.execution.flow_after[serving]).clamp(0.0, 1.0)
        rate = torch.where(connected, rate * discount, torch.zeros_like(rate))
    return rate


def _epoch_user_tail_mean(
    throughput: torch.Tensor,
    *,
    probability: float,
) -> float:
    """Average the per-epoch nearest-rank quantile across users.

    The manuscript contract is ``Q_p(users at t)`` followed by a uniform mean
    over decision epochs.  Keeping this helper shared by learned, oracle, and
    classical traces prevents an accidental reversal of the time/user axes.
    """

    if throughput.ndim != 2 or throughput.size(0) == 0 or throughput.size(1) == 0:
        raise ValueError("throughput must have non-empty shape [epoch,user]")
    if not bool(torch.isfinite(throughput).all()):
        raise ValueError("throughput contains NaN or Inf")
    epoch_tail = torch.stack(
        tuple(
            nearest_rank_quantile(throughput[epoch], probability)
            for epoch in range(throughput.size(0))
        )
    )
    return float(epoch_tail.mean().item())


def _pingpong_events(
    serving: torch.Tensor,
    *,
    window_steps: int,
) -> int:
    """Count user-level A→B→A returns within the declared time window."""

    if serving.ndim != 2:
        raise ValueError("serving history must have shape [time,user]")
    events = 0
    epochs, users = serving.shape
    for user in range(users):
        changes: list[tuple[int, int]] = []
        previous = int(serving[0, user].item())
        if previous >= 0:
            changes.append((0, previous))
        for epoch in range(1, epochs):
            current = int(serving[epoch, user].item())
            if current != previous:
                if current >= 0:
                    changes.append((epoch, current))
                previous = current
        for index in range(2, len(changes)):
            t0, a = changes[index - 2]
            _, b = changes[index - 1]
            t2, returned = changes[index]
            if a == returned and a != b and t2 - t0 <= window_steps:
                events += 1
    return events


def evaluate_closed_loop_outcomes(
    result: ClosedLoopResult,
    *,
    config: OutcomeMetricConfig | None = None,
    oracle_result: ClosedLoopResult | None = None,
) -> EpisodeOutcomeMetrics:
    config = config or OutcomeMetricConfig()
    if not result.records:
        raise ValueError("closed-loop result contains no epochs")
    if oracle_result is not None:
        if oracle_result.episode_seed != result.episode_seed:
            raise ValueError("oracle result must be the matched episode")
        if oracle_result.action_count != result.action_count:
            raise ValueError("oracle result must use the same horizon")

    serving = torch.stack(
        tuple(record.execution.executed_serving.cpu() for record in result.records)
    )
    throughput = torch.stack(
        tuple(_throughput(record, config).cpu() for record in result.records)
    )
    gamma = torch.stack(tuple(_served_gamma(record).cpu() for record in result.records))
    connected = serving >= 0
    outage = (~connected) | (gamma < config.outage_gamma_db)
    attempts = sum(
        int(record.execution.handover_attempted.sum().item()) for record in result.records
    )
    executed = sum(
        int(record.execution.handover_executed.sum().item()) for record in result.records
    )
    failures = attempts - executed
    window_steps = max(1, math.ceil(config.pingpong_window_seconds / config.dt_ctrl))
    pingpong = _pingpong_events(serving, window_steps=window_steps)
    duration = len(result.records) * config.dt_ctrl
    user_count = int(serving.size(1))

    tail_mean = _epoch_user_tail_mean(
        throughput,
        probability=config.user_tail_probability,
    )
    oracle_ratio: float | None = None
    if oracle_result is not None:
        oracle_throughput = torch.stack(
            tuple(_throughput(record, config).cpu() for record in oracle_result.records)
        )
        denominator = _epoch_user_tail_mean(
            oracle_throughput,
            probability=config.user_tail_probability,
        )
        oracle_ratio = tail_mean / denominator if denominator > 0 else None

    cvs: list[float] = []
    for record in result.records:
        active = torch.unique(record.execution.executed_serving)
        active = active[active >= 0]
        if active.numel() <= 1:
            cvs.append(0.0)
            continue
        load = record.execution.flow_after.index_select(0, active)
        mean = float(load.mean().item())
        cvs.append(
            0.0
            if mean <= 0
            else float((load.std(unbiased=False) / load.mean()).item())
        )

    return EpisodeOutcomeMetrics(
        episode_seed=result.episode_seed,
        epochs=len(result.records),
        user_count=user_count,
        outage_rate=float(outage.to(torch.float64).mean().item()),
        handover_attempts=attempts,
        handover_failures=failures,
        handover_failure_rate=(failures / attempts if attempts else 0.0),
        handovers_executed=executed,
        pingpong_events=pingpong,
        pingpong_rate_user_second=pingpong / (user_count * duration),
        pingpong_fraction_executed_decisions=(
            pingpong / int(connected.sum().item()) if bool(connected.any()) else 0.0
        ),
        user_p10_throughput_mean=tail_mean,
        user_p10_throughput_ratio_to_oracle=oracle_ratio,
        active_satellite_load_cv_mean=float(sum(cvs) / len(cvs)),
        mean_throughput=float(throughput.mean().item()),
        throughput_contract=(
            "bandwidth*log2(1+SINR)*(1-flow)"
            if config.throughput_load_discount
            else "bandwidth*log2(1+SINR)"
        ),
        tail_aggregation_contract=TAIL_AGGREGATION_CONTRACT,
    )


@dataclass(frozen=True)
class DecisionFidelityMetrics:
    comparisons: int
    top1_agreement: float
    candidate_kendall_tau: float
    near_tie_flip_rate: float
    oracle_score_regret: float
    near_tie_margin: float
    metric_contract_version: int = METRIC_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClassicalOracleRegretMetrics:
    """Decision regret for a classical action on its own observed state.

    The counterfactual reference is the fixed rank policy evaluated on the
    simulator-authoritative descriptors, candidate set, serving state, dwell
    state, and feasibility mask recorded for that same classical decision.
    This deliberately does not compare actions from two diverged trajectories.
    """

    comparisons: int
    oracle_score_regret: float
    reference_policy: str
    evaluated_action: str
    unavailable_action_score: float
    policy_config: Mapping[str, Any]
    metric_contract_version: int = METRIC_CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rank_scores(
    edge_ids: torch.Tensor,
    descriptors: PolicyDescriptors,
    config: FixedRankPolicyConfig,
    feasible_edge: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = torch.full(
        (edge_ids.size(0),),
        -torch.inf,
        dtype=descriptors.gamma_edge.dtype,
        device=descriptors.gamma_edge.device,
    )
    if feasible_edge is not None:
        feasible_edge = torch.as_tensor(feasible_edge, device=edge_ids.device)
        if feasible_edge.dtype != torch.bool or feasible_edge.shape != (edge_ids.size(0),):
            raise ValueError("feasible_edge must be bool with shape [edge_count]")
    for user in torch.unique(edge_ids[:, 0]).tolist():
        rows = torch.nonzero(edge_ids[:, 0] == int(user), as_tuple=False).flatten()
        if config.hard_feasibility_mask:
            if feasible_edge is None:
                raise ValueError(
                    "hard-feasibility decision metrics require feasible_edge"
                )
            rows = rows[feasible_edge.index_select(0, rows)]
        if rows.numel() == 0:
            continue
        satellites = edge_ids.index_select(0, rows)[:, 1]
        gamma = normalized_ordinal_rank(
            descriptors.gamma_edge.index_select(0, rows),
            satellites,
            higher_is_better=True,
        )
        intensity = normalized_ordinal_rank(
            descriptors.intensity_edge.index_select(0, rows),
            satellites,
            higher_is_better=False,
        )
        flow = normalized_ordinal_rank(
            descriptors.flow_node.index_select(0, satellites),
            satellites,
            higher_is_better=False,
        )
        scores[rows] = (
            config.gamma_weight * gamma
            + config.load_weight * flow
            + config.intensity_weight * intensity
        )
    return scores


def _kendall_tau(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() <= 1:
        return 1.0
    concordant = 0
    discordant = 0
    for left in range(a.numel()):
        for right in range(left + 1, a.numel()):
            first = float((a[left] - a[right]).item())
            second = float((b[left] - b[right]).item())
            product = first * second
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denominator = concordant + discordant
    return 0.0 if denominator == 0 else (concordant - discordant) / denominator


def evaluate_decision_fidelity(
    records: Sequence[ClosedLoopStep],
    *,
    policy_config: FixedRankPolicyConfig,
    near_tie_margin: float = 1.0 / 6.0,
    descriptor_source: str = "policy",
) -> DecisionFidelityMetrics:
    """Compare the descriptors that actually reached the controller with oracle ranks.

    ``descriptor_source='policy'`` is the paper protocol: partial-oracle modes are
    scored after the requested field replacement. ``'model'`` remains available
    for diagnosing the uncorrected model stream.
    """

    if descriptor_source not in {"policy", "model"}:
        raise ValueError("descriptor_source must be 'policy' or 'model'")
    top1: list[float] = []
    taus: list[float] = []
    flips: list[float] = []
    regrets: list[float] = []
    for record in records:
        model = (
            record.policy_descriptors
            if descriptor_source == "policy"
            else record.model_descriptors
        )
        if model is None:
            continue
        oracle = record.sim_descriptors.policy_fields
        feasible_edge = record.sim_descriptors.feasible_edge
        model_scores = _rank_scores(
            record.candidate_edge_ids,
            model,
            policy_config,
            feasible_edge,
        )
        oracle_scores = _rank_scores(
            record.candidate_edge_ids,
            oracle,
            policy_config,
            feasible_edge,
        )
        for user in torch.unique(record.candidate_edge_ids[:, 0]).tolist():
            rows = torch.nonzero(
                record.candidate_edge_ids[:, 0] == int(user), as_tuple=False
            ).flatten()
            if policy_config.hard_feasibility_mask:
                rows = rows[feasible_edge.index_select(0, rows)]
            if rows.numel() == 0:
                continue
            satellites = record.candidate_edge_ids.index_select(0, rows)[:, 1]
            # Deterministic satellite-id tie ordering.
            satellite_order = torch.argsort(satellites, stable=True)
            ordered = rows.index_select(0, satellite_order)
            model_order = ordered[
                torch.argsort(model_scores[ordered], descending=True, stable=True)
            ]
            oracle_order = ordered[
                torch.argsort(oracle_scores[ordered], descending=True, stable=True)
            ]
            model_best = int(model_order[0].item())
            oracle_best = int(oracle_order[0].item())
            top1.append(float(model_best == oracle_best))
            taus.append(_kendall_tau(model_scores[rows], oracle_scores[rows]))
            regret = float((oracle_scores[oracle_best] - oracle_scores[model_best]).item())
            regrets.append(max(0.0, regret))
            if rows.numel() > 1:
                oracle_gap = float(
                    (oracle_scores[oracle_order[0]] - oracle_scores[oracle_order[1]]).item()
                )
                if oracle_gap <= near_tie_margin:
                    flips.append(float(model_best != oracle_best))
    if not top1:
        raise ValueError("no model-vs-oracle decision comparisons are available")
    return DecisionFidelityMetrics(
        comparisons=len(top1),
        top1_agreement=sum(top1) / len(top1),
        candidate_kendall_tau=sum(taus) / len(taus),
        near_tie_flip_rate=(sum(flips) / len(flips) if flips else 0.0),
        oracle_score_regret=sum(regrets) / len(regrets),
        near_tie_margin=float(near_tie_margin),
    )


def _serialized_classical_observation(
    observation: Mapping[str, Any],
) -> tuple[
    torch.Tensor,
    PolicyDescriptors,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    edge_ids = torch.as_tensor(observation["candidate_edge_ids"]).long().cpu()
    if edge_ids.ndim != 2 or edge_ids.size(1) != 2:
        raise ValueError("candidate_edge_ids must have shape [edge_count,2]")
    sim = observation.get("sim_descriptors")
    if not isinstance(sim, Mapping):
        raise TypeError("classical observation requires simulator descriptors")
    descriptors = PolicyDescriptors(
        gamma_edge=torch.as_tensor(sim["gamma_edge"]).float().cpu(),
        intensity_edge=torch.as_tensor(sim["intensity_edge"]).float().cpu(),
        flow_node=torch.as_tensor(sim["flow_node"]).float().cpu(),
    )
    satellite_count = int(observation["satellite_count"])
    descriptors.validate(edge_ids.size(0), satellite_count)
    feasible = torch.as_tensor(sim["feasible_edge"]).bool().cpu()
    if feasible.shape != (edge_ids.size(0),):
        raise ValueError("classical feasible_edge shape disagrees with candidates")
    current = torch.as_tensor(observation["current_serving"]).long().cpu()
    hold = torch.as_tensor(observation["hold_steps"]).long().cpu()
    user_order = torch.as_tensor(observation["user_order"]).long().cpu()
    user_count = int(observation["user_count"])
    if current.shape != (user_count,) or hold.shape != (user_count,):
        raise ValueError("classical serving/hold state shape disagrees with user_count")
    if user_order.shape != (user_count,) or torch.unique(user_order).numel() != user_count:
        raise ValueError("classical user_order must be a user permutation")
    return edge_ids, descriptors, feasible, current, hold, user_order


def _best_scored_edge(
    rows: torch.Tensor,
    scores: torch.Tensor,
    satellite_ids: torch.Tensor,
) -> int:
    satellite_order = torch.argsort(
        satellite_ids.index_select(0, rows),
        stable=True,
    )
    ordered = rows.index_select(0, satellite_order)
    score_order = torch.argsort(
        scores.index_select(0, ordered),
        descending=True,
        stable=True,
    )
    return int(ordered[score_order[0]].item())


def _action_score(
    target: int,
    rows: torch.Tensor,
    *,
    edge_ids: torch.Tensor,
    scores: torch.Tensor,
) -> float:
    if target < 0 or rows.numel() == 0:
        return 0.0
    matches = rows[edge_ids.index_select(0, rows)[:, 1] == int(target)]
    if matches.numel() != 1:
        return 0.0
    value = float(scores[int(matches[0].item())].item())
    return value if math.isfinite(value) else 0.0


def evaluate_classical_oracle_regret(
    result: Mapping[str, Any],
    *,
    policy_config: FixedRankPolicyConfig,
) -> ClassicalOracleRegretMetrics:
    """Compare committed classical requests with fixed-rank oracle requests.

    Both scores are evaluated before simulator admission on the same classical
    observation. A disconnect, non-candidate request, or hard-mask-ineligible
    request receives the explicit finite score floor zero. The fixed-rank
    counterfactual follows the released dwell and hysteresis rules exactly.
    """

    records = result.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("classical result contains no records")
    regrets: list[float] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("classical record must be a mapping")
        observation = record.get("observation")
        committed = record.get("action")
        if not isinstance(observation, Mapping) or not isinstance(committed, Mapping):
            raise TypeError("classical regret requires observation and committed action")
        edge_ids, descriptors, feasible, current, hold, user_order = (
            _serialized_classical_observation(observation)
        )
        requested = torch.as_tensor(committed["requested_serving"]).long().cpu()
        if requested.shape != current.shape:
            raise ValueError("classical committed action shape disagrees with user_count")
        scores = _rank_scores(edge_ids, descriptors, policy_config, feasible)
        edge_users = edge_ids[:, 0]
        edge_satellites = edge_ids[:, 1]
        for user in user_order.tolist():
            rows = torch.nonzero(edge_users == int(user), as_tuple=False).flatten()
            eligible = rows[torch.isfinite(scores.index_select(0, rows))]
            if eligible.numel() == 0:
                continue
            best_edge = _best_scored_edge(eligible, scores, edge_satellites)
            best_satellite = int(edge_satellites[best_edge].item())
            current_satellite = int(current[user].item())
            reference = current_satellite
            if current_satellite < 0:
                reference = best_satellite
            else:
                current_rows = rows[
                    edge_satellites.index_select(0, rows) == current_satellite
                ]
                current_is_eligible = (
                    current_rows.numel() == 1
                    and bool(torch.isfinite(scores[int(current_rows[0].item())]).item())
                )
                if (
                    current_is_eligible
                    and best_satellite != current_satellite
                    and int(hold[user].item()) >= policy_config.min_dwell_steps
                ):
                    current_edge = int(current_rows[0].item())
                    hysteresis = policy_config.hysteresis
                    if hysteresis is None:
                        hysteresis = 1.0 / float(eligible.numel())
                    best_score = float(scores[best_edge].item())
                    required = float(scores[current_edge].item()) + float(hysteresis)
                    if best_score >= required or math.isclose(
                        best_score,
                        required,
                        rel_tol=1e-7,
                        abs_tol=1e-8,
                    ):
                        reference = best_satellite
            reference_score = _action_score(
                reference,
                rows,
                edge_ids=edge_ids,
                scores=scores,
            )
            committed_score = _action_score(
                int(requested[user].item()),
                rows,
                edge_ids=edge_ids,
                scores=scores,
            )
            regrets.append(max(0.0, reference_score - committed_score))
    if not regrets:
        raise ValueError("no classical fixed-rank regret comparisons are available")
    return ClassicalOracleRegretMetrics(
        comparisons=len(regrets),
        oracle_score_regret=sum(regrets) / len(regrets),
        reference_policy=(
            "FixedRankPolicy on simulator descriptors and the same classical "
            "candidate/serving/dwell/feasibility state"
        ),
        evaluated_action="committed request after optional shield, before admission",
        unavailable_action_score=0.0,
        policy_config=asdict(policy_config),
    )


def evaluate_closed_loop_result(
    result: ClosedLoopResult | Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluation-runner adapter for outcome and decision metrics.

    The evaluator places the matched oracle result in the transient ``context``
    mapping. It supplies the same-contract tail denominator and, for classical
    traces, the fixed-policy configuration. The live object is never serialized.
    """

    context = dict(context or {})
    options = dict(options or {})
    outcome_options = dict(options.get("outcome", options))
    outcome_fields = set(OutcomeMetricConfig.__dataclass_fields__)
    outcome_config = OutcomeMetricConfig(
        **{
            key: value
            for key, value in outcome_options.items()
            if key in outcome_fields
        }
    )
    matched_oracle = context.get("matched_oracle_result")
    if matched_oracle is not None and not isinstance(matched_oracle, ClosedLoopResult):
        raise TypeError("matched_oracle_result must be a ClosedLoopResult")
    if isinstance(result, Mapping):
        if matched_oracle is None:
            raise ValueError(
                "classical metrics require a matched oracle result for normalized "
                "tail throughput and fixed-score regret"
            )
        return {
            "outcomes": _evaluate_serialized_classical_outcomes(
                result,
                config=outcome_config,
                oracle_result=matched_oracle,
            ),
            "classical_oracle_regret": evaluate_classical_oracle_regret(
                result,
                policy_config=matched_oracle.policy_config,
            ).as_dict(),
        }
    if not isinstance(result, ClosedLoopResult):
        raise TypeError("result must be a ClosedLoopResult or serialized classical trace")
    outcome_oracle = matched_oracle
    if result.mode.value == "oracle" and outcome_oracle is None:
        outcome_oracle = result
    outcome = evaluate_closed_loop_outcomes(
        result,
        config=outcome_config,
        oracle_result=outcome_oracle,
    )
    row: dict[str, Any] = {"outcomes": outcome.as_dict()}
    if result.mode.value != "oracle":
        decision_options = dict(options.get("decision", {}))
        fidelity = evaluate_decision_fidelity(
            result.records,
            policy_config=result.policy_config,
            near_tie_margin=float(
                decision_options.get("near_tie_margin", 1.0 / 6.0)
            ),
            descriptor_source=str(
                decision_options.get("descriptor_source", "policy")
            ),
        )
        row["decision_fidelity"] = fidelity.as_dict()
    return row


def _evaluate_serialized_classical_outcomes(
    result: Mapping[str, Any],
    *,
    config: OutcomeMetricConfig,
    oracle_result: ClosedLoopResult,
) -> dict[str, Any]:
    """Apply the same endpoint contract to a classical-controller trace."""

    records = result.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("classical result contains no records")
    serving_rows: list[torch.Tensor] = []
    gamma_rows: list[torch.Tensor] = []
    throughput_rows: list[torch.Tensor] = []
    flow_rows: list[torch.Tensor] = []
    attempts = 0
    executed = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("classical record must be a mapping")
        observation = record.get("observation")
        execution = record.get("execution")
        if not isinstance(observation, Mapping) or not isinstance(execution, Mapping):
            raise TypeError("classical record requires observation and execution mappings")
        edge_ids = torch.as_tensor(observation["candidate_edge_ids"]).long().cpu()
        sim = observation.get("sim_descriptors")
        if not isinstance(sim, Mapping):
            raise TypeError("classical observation requires simulator descriptors")
        edge_gamma = torch.as_tensor(sim["gamma_edge"]).float().cpu()
        serving = torch.as_tensor(execution["executed_serving"]).long().cpu()
        flow_after = torch.as_tensor(execution["flow_after"]).float().cpu()
        if edge_ids.shape != (edge_gamma.numel(), 2):
            raise ValueError("classical candidate ids and gamma length disagree")
        served_gamma = torch.full((serving.numel(),), -torch.inf)
        lookup = {
            (int(user), int(satellite)): edge
            for edge, (user, satellite) in enumerate(edge_ids.tolist())
        }
        for user, satellite in enumerate(serving.tolist()):
            edge = lookup.get((user, int(satellite)))
            if edge is not None:
                served_gamma[user] = edge_gamma[edge]
        connected = serving >= 0
        gamma_linear = torch.pow(torch.tensor(10.0), served_gamma / 10.0)
        throughput = config.bandwidth_hz * torch.log2(1.0 + gamma_linear)
        throughput = torch.where(connected, throughput, torch.zeros_like(throughput))
        if config.throughput_load_discount:
            destination = serving.clamp_min(0)
            discount = (1.0 - flow_after[destination]).clamp(0.0, 1.0)
            throughput = torch.where(
                connected,
                throughput * discount,
                torch.zeros_like(throughput),
            )
        serving_rows.append(serving)
        gamma_rows.append(served_gamma)
        throughput_rows.append(throughput)
        flow_rows.append(flow_after)
        attempts += int(
            torch.as_tensor(execution["handover_attempted"]).bool().sum().item()
        )
        executed += int(
            torch.as_tensor(execution["handover_executed"]).bool().sum().item()
        )

    serving_history = torch.stack(serving_rows)
    gamma_history = torch.stack(gamma_rows)
    throughput_history = torch.stack(throughput_rows)
    connected = serving_history >= 0
    outage = (~connected) | (gamma_history < config.outage_gamma_db)
    user_count = int(serving_history.size(1))
    window_steps = max(1, math.ceil(config.pingpong_window_seconds / config.dt_ctrl))
    pingpong = _pingpong_events(serving_history, window_steps=window_steps)
    duration = len(records) * config.dt_ctrl
    tail_mean = _epoch_user_tail_mean(
        throughput_history,
        probability=config.user_tail_probability,
    )
    if oracle_result.episode_seed != int(result["episode_seed"]):
        raise ValueError("classical oracle result must be the matched episode")
    if oracle_result.action_count != len(records):
        raise ValueError("classical oracle result must use the same horizon")
    oracle_throughput = torch.stack(
        tuple(_throughput(record, config).cpu() for record in oracle_result.records)
    )
    oracle_tail_mean = _epoch_user_tail_mean(
        oracle_throughput,
        probability=config.user_tail_probability,
    )
    oracle_ratio = tail_mean / oracle_tail_mean if oracle_tail_mean > 0 else None
    cvs: list[float] = []
    for serving, flow in zip(serving_rows, flow_rows):
        active = torch.unique(serving)
        active = active[active >= 0]
        if active.numel() <= 1:
            cvs.append(0.0)
            continue
        active_flow = flow.index_select(0, active)
        mean = float(active_flow.mean().item())
        cvs.append(
            0.0
            if mean <= 0.0
            else float(
                (active_flow.std(unbiased=False) / active_flow.mean()).item()
            )
        )
    failures = attempts - executed
    return EpisodeOutcomeMetrics(
        episode_seed=int(result["episode_seed"]),
        epochs=len(records),
        user_count=user_count,
        outage_rate=float(outage.to(torch.float64).mean().item()),
        handover_attempts=attempts,
        handover_failures=failures,
        handover_failure_rate=(failures / attempts if attempts else 0.0),
        handovers_executed=executed,
        pingpong_events=pingpong,
        pingpong_rate_user_second=pingpong / (user_count * duration),
        pingpong_fraction_executed_decisions=(
            pingpong / int(connected.sum().item()) if bool(connected.any()) else 0.0
        ),
        user_p10_throughput_mean=tail_mean,
        user_p10_throughput_ratio_to_oracle=oracle_ratio,
        active_satellite_load_cv_mean=float(sum(cvs) / len(cvs)),
        mean_throughput=float(throughput_history.mean().item()),
        throughput_contract=(
            "bandwidth*log2(1+SINR)*(1-flow)"
            if config.throughput_load_discount
            else "bandwidth*log2(1+SINR)"
        ),
        tail_aggregation_contract=TAIL_AGGREGATION_CONTRACT,
    ).as_dict()


@dataclass(frozen=True)
class PairedBootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    units: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_percentile_bootstrap(
    paired_differences: Sequence[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> PairedBootstrapInterval:
    """Bootstrap matched seed/episode units without treating epochs as IID."""

    differences = torch.tensor(tuple(float(v) for v in paired_differences), dtype=torch.float64)
    if differences.ndim != 1 or differences.numel() == 0:
        raise ValueError("paired_differences must be non-empty")
    if not torch.isfinite(differences).all():
        raise ValueError("paired differences contain NaN or Inf")
    if isinstance(resamples, bool) or int(resamples) != resamples or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        0,
        differences.numel(),
        (int(resamples), differences.numel()),
        generator=generator,
    )
    draws = differences[indices].mean(dim=1)
    tail = (1.0 - float(confidence)) / 2.0
    lower = float(nearest_rank_quantile(draws, tail).item())
    upper = float(nearest_rank_quantile(draws, 1.0 - tail).item())
    return PairedBootstrapInterval(
        estimate=float(differences.mean().item()),
        lower=lower,
        upper=upper,
        confidence=float(confidence),
        resamples=int(resamples),
        units=int(differences.numel()),
        seed=int(seed),
    )


def _tail_risk_p10(values: Sequence[float], *, probability: float) -> float:
    ratios = torch.tensor(tuple(float(value) for value in values), dtype=torch.float64)
    if ratios.ndim != 1 or ratios.numel() == 0 or not bool(torch.isfinite(ratios).all()):
        raise ValueError("tail-risk ratios must be a non-empty finite vector")
    return 1.0 - float(nearest_rank_quantile(ratios, probability).item())


def _paired_tail_risk_bootstrap(
    condition_ratios: Sequence[float],
    oracle_ratios: Sequence[float],
    *,
    probability: float,
    resamples: int,
    confidence: float,
    seed: int,
) -> PairedBootstrapInterval:
    """Matched-unit bootstrap for the nonlinear ``1 - Q_p(ratio)`` endpoint."""

    condition = torch.tensor(
        tuple(float(value) for value in condition_ratios),
        dtype=torch.float64,
    )
    oracle = torch.tensor(
        tuple(float(value) for value in oracle_ratios),
        dtype=torch.float64,
    )
    if condition.ndim != 1 or condition.numel() == 0 or condition.shape != oracle.shape:
        raise ValueError("tail-risk condition/oracle ratios must be matched vectors")
    if not bool(torch.isfinite(condition).all()) or not bool(torch.isfinite(oracle).all()):
        raise ValueError("tail-risk condition/oracle ratios must be finite")
    if not 0.0 < float(probability) <= 1.0:
        raise ValueError("tail-risk probability must lie in (0,1]")
    if isinstance(resamples, bool) or int(resamples) != resamples or int(resamples) <= 0:
        raise ValueError("resamples must be a positive integer")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        0,
        condition.numel(),
        (int(resamples), condition.numel()),
        generator=generator,
    )
    rank = max(1, math.ceil(float(probability) * condition.numel())) - 1
    condition_q = torch.sort(condition[indices], dim=1).values[:, rank]
    oracle_q = torch.sort(oracle[indices], dim=1).values[:, rank]
    effects = (1.0 - condition_q) - (1.0 - oracle_q)
    tail = (1.0 - float(confidence)) / 2.0
    return PairedBootstrapInterval(
        estimate=(
            _tail_risk_p10(condition.tolist(), probability=probability)
            - _tail_risk_p10(oracle.tolist(), probability=probability)
        ),
        lower=float(nearest_rank_quantile(effects, tail).item()),
        upper=float(nearest_rank_quantile(effects, 1.0 - tail).item()),
        confidence=float(confidence),
        resamples=int(resamples),
        units=int(condition.numel()),
        seed=int(seed),
    )


_OUTCOME_METRICS: dict[str, str] = {
    "outage_rate": "lower_is_better",
    "handover_failure_rate": "lower_is_better",
    "pingpong_rate_user_second": "lower_is_better",
    "user_p10_throughput_mean": "higher_is_better",
    "user_p10_throughput_ratio_to_oracle": "higher_is_better",
    "active_satellite_load_cv_mean": "lower_is_better",
    "mean_throughput": "higher_is_better",
}
_DECISION_METRICS: dict[str, str] = {
    "top1_agreement": "higher_is_better",
    "candidate_kendall_tau": "higher_is_better",
    "near_tie_flip_rate": "lower_is_better",
    "oracle_score_regret": "lower_is_better",
}
_ORACLE_DECISION_REFERENCE: dict[str, float] = {
    "top1_agreement": 1.0,
    "candidate_kendall_tau": 1.0,
    "near_tie_flip_rate": 0.0,
    "oracle_score_regret": 0.0,
}


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _canonical_metric_values(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    outcomes = row.get("outcomes")
    if isinstance(outcomes, Mapping):
        for name in _OUTCOME_METRICS:
            value = _finite_scalar(outcomes.get(name))
            if value is not None:
                values[name] = value
    decision = row.get("decision_fidelity")
    if isinstance(decision, Mapping):
        for name in _DECISION_METRICS:
            value = _finite_scalar(decision.get(name))
            if value is not None:
                values[name] = value
    classical = row.get("classical_oracle_regret")
    if isinstance(classical, Mapping):
        value = _finite_scalar(classical.get("oracle_score_regret"))
        if value is not None:
            values["oracle_score_regret"] = value
    return values


def _unit_identity(unit: Mapping[str, Any]) -> dict[str, int]:
    return {
        "base_seed": int(unit["base_seed"]),
        "episode_index": int(unit["episode_index"]),
        "episode_seed": int(unit["episode_seed"]),
    }


def aggregate_evaluation_units(
    units: Sequence[Mapping[str, Any]],
    *,
    options: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate standard metric rows over matched seed-by-episode units.

    Epochs remain inside their episode metric and are never resampled. Every
    reported effect retains its full matched unit-difference vector, followed by
    a percentile interval over exactly those independent rollout units.
    """

    context = dict(context or {})
    options = dict(options or {})
    if not units:
        raise ValueError("metric aggregation requires at least one evaluation unit")
    raw_resamples = options.get("bootstrap_resamples", 10_000)
    if isinstance(raw_resamples, bool) or int(raw_resamples) != raw_resamples:
        raise ValueError("bootstrap_resamples must be a positive integer")
    resamples = int(raw_resamples)
    if resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    bootstrap_seed = int(options.get("bootstrap_seed", 17))
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")
    confidence = float(options.get("bootstrap_confidence", 0.95))
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must lie in (0,1)")
    tail_probability = float(options.get("tail_risk_probability", 0.10))
    if not 0.0 < tail_probability <= 1.0:
        raise ValueError("tail_risk_probability must lie in (0,1]")

    materialized = [dict(unit) for unit in units]
    identities = [_unit_identity(unit) for unit in materialized]
    identity_keys = [
        (item["base_seed"], item["episode_index"], item["episode_seed"])
        for item in identities
    ]
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError("metric aggregation received duplicate matched units")
    expected_seeds = context.get("base_seeds")
    if expected_seeds is not None:
        expected_seed_set = {int(value) for value in expected_seeds}
        observed_seed_set = {item["base_seed"] for item in identities}
        if observed_seed_set != expected_seed_set:
            raise ValueError("aggregated base seeds disagree with evaluation context")
    expected_episodes = context.get("episodes_per_seed")
    if expected_episodes is not None:
        expected_episodes = int(expected_episodes)
        for base_seed in {item["base_seed"] for item in identities}:
            observed = sorted(
                item["episode_index"]
                for item in identities
                if item["base_seed"] == base_seed
            )
            if observed != list(range(expected_episodes)):
                raise ValueError(
                    "aggregated episode indices do not form the declared per-seed grid"
                )

    first_metrics = materialized[0].get("metrics")
    if not isinstance(first_metrics, Mapping):
        raise ValueError("evaluation units do not contain standard metric rows")
    first_paired = first_metrics.get("paired")
    first_classical = first_metrics.get("classical", {})
    if not isinstance(first_paired, Mapping) or "oracle" not in first_paired:
        raise ValueError("metric aggregation requires a paired oracle row in every unit")
    if not isinstance(first_classical, Mapping):
        raise TypeError("classical metric rows must be a mapping")
    paired_names = tuple(sorted(str(name) for name in first_paired))
    classical_names = tuple(sorted(str(name) for name in first_classical))
    for unit in materialized:
        metric_rows = unit.get("metrics")
        if not isinstance(metric_rows, Mapping):
            raise ValueError("every evaluation unit must contain metric rows")
        paired_rows = metric_rows.get("paired")
        classical_rows = metric_rows.get("classical", {})
        if not isinstance(paired_rows, Mapping) or not isinstance(classical_rows, Mapping):
            raise TypeError("paired/classical metric rows must be mappings")
        if tuple(sorted(str(name) for name in paired_rows)) != paired_names:
            raise ValueError("paired metric conditions differ across matched units")
        if tuple(sorted(str(name) for name in classical_rows)) != classical_names:
            raise ValueError("classical metric conditions differ across matched units")

    condition_specs: list[tuple[str, str, str]] = []
    for name in paired_names:
        condition_specs.append((f"paired/{name}", "paired", name))
    for name in classical_names:
        condition_specs.append((f"classical/{name}", "classical", name))

    oracle_rows = [
        _canonical_metric_values(unit["metrics"]["paired"]["oracle"])
        for unit in materialized
    ]
    condition_summaries: dict[str, Any] = {}
    paired_effects: dict[str, Any] = {}
    tail_risk: dict[str, Any] = {}
    controller_candidates: dict[str, list[dict[str, Any]]] = {}

    for condition_key, condition_type, condition_name in condition_specs:
        condition_rows = [
            _canonical_metric_values(
                unit["metrics"][condition_type][condition_name]
            )
            for unit in materialized
        ]
        metric_names = sorted(set.intersection(*(set(row) for row in condition_rows)))
        summary_metrics: dict[str, Any] = {}
        effect_metrics: dict[str, Any] = {}
        for metric_name in metric_names:
            condition_values = [row[metric_name] for row in condition_rows]
            if metric_name in _ORACLE_DECISION_REFERENCE:
                oracle_values = [
                    row.get(metric_name, _ORACLE_DECISION_REFERENCE[metric_name])
                    for row in oracle_rows
                ]
            else:
                if not all(metric_name in row for row in oracle_rows):
                    continue
                oracle_values = [row[metric_name] for row in oracle_rows]
            differences = [
                condition - oracle
                for condition, oracle in zip(condition_values, oracle_values)
            ]
            direction = (
                _OUTCOME_METRICS.get(metric_name)
                or _DECISION_METRICS.get(metric_name)
                or "unspecified"
            )
            summary_metrics[metric_name] = {
                "mean": sum(condition_values) / len(condition_values),
                "units": len(condition_values),
                "direction": direction,
            }
            interval = paired_percentile_bootstrap(
                differences,
                resamples=resamples,
                confidence=confidence,
                seed=bootstrap_seed,
            )
            effect_metrics[metric_name] = {
                "difference_definition": "condition - matched oracle",
                "direction": direction,
                "unit_differences": [
                    {**identity, "value": float(value)}
                    for identity, value in zip(identities, differences)
                ],
                "percentile_interval": interval.as_dict(),
            }

        metadata: dict[str, Any] = {
            "condition_type": condition_type,
            "condition": condition_name,
        }
        if condition_type == "classical":
            trace = materialized[0]["classical"][condition_name]
            if not isinstance(trace, Mapping):
                raise TypeError("classical trace metadata must be a mapping")
            metadata.update(
                {
                    "controller_name": str(trace.get("controller_name", condition_name)),
                    "controller_parameters": dict(trace.get("controller_parameters", {})),
                    "sweep_provenance": dict(trace.get("sweep_provenance", {})),
                }
            )
        condition_summaries[condition_key] = {
            **metadata,
            "metrics": summary_metrics,
        }
        if condition_name != "oracle" or condition_type != "paired":
            paired_effects[condition_key] = {
                **metadata,
                "reference": "paired/oracle",
                "metrics": effect_metrics,
            }

        ratio_name = "user_p10_throughput_ratio_to_oracle"
        if ratio_name in metric_names and all(ratio_name in row for row in oracle_rows):
            condition_ratios = [row[ratio_name] for row in condition_rows]
            oracle_ratios = [row[ratio_name] for row in oracle_rows]
            risk_interval = _paired_tail_risk_bootstrap(
                condition_ratios,
                oracle_ratios,
                probability=tail_probability,
                resamples=resamples,
                confidence=confidence,
                seed=bootstrap_seed,
            )
            tail_risk[condition_key] = {
                **metadata,
                "definition": (
                    "1 - nearest-rank P10 of per-episode tail-throughput ratio "
                    "to the matched oracle"
                ),
                "probability": tail_probability,
                "condition_estimate": _tail_risk_p10(
                    condition_ratios,
                    probability=tail_probability,
                ),
                "oracle_estimate": _tail_risk_p10(
                    oracle_ratios,
                    probability=tail_probability,
                ),
                "unit_ratios": [
                    {
                        **identity,
                        "condition": float(condition),
                        "oracle": float(oracle),
                        "paired_degradation_difference": float(oracle - condition),
                    }
                    for identity, condition, oracle in zip(
                        identities,
                        condition_ratios,
                        oracle_ratios,
                    )
                ],
                "paired_percentile_interval": risk_interval.as_dict(),
            }

        if condition_type == "classical" and "oracle_score_regret" in summary_metrics:
            controller_name = str(metadata["controller_name"])
            controller_candidates.setdefault(controller_name, []).append(
                {
                    "condition": condition_key,
                    "parameters": metadata["controller_parameters"],
                    "sweep_provenance": metadata["sweep_provenance"],
                    "mean_oracle_score_regret": summary_metrics[
                        "oracle_score_regret"
                    ]["mean"],
                    "units": summary_metrics["oracle_score_regret"]["units"],
                }
            )

    controller_selection: dict[str, Any] = {}
    for controller_name, candidates in sorted(controller_candidates.items()):
        ordered = sorted(
            candidates,
            key=lambda item: (
                float(item["mean_oracle_score_regret"]),
                str(item["condition"]),
            ),
        )
        controller_selection[controller_name] = {
            "criterion": (
                "minimum mean per-episode classical oracle fixed-score regret; "
                "lexicographic condition label breaks exact ties"
            ),
            "selected": ordered[0],
            "candidate_count": len(ordered),
            "ranking": ordered,
        }

    return {
        "schema_version": METRIC_AGGREGATION_SCHEMA_VERSION,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "unit_contract": {
            "unit": "matched (base_seed, episode_index) rollout",
            "unit_count": len(materialized),
            "base_seed_count": len({item["base_seed"] for item in identities}),
            "units_per_base_seed": {
                str(base_seed): sum(
                    int(item["base_seed"] == base_seed) for item in identities
                )
                for base_seed in sorted({item["base_seed"] for item in identities})
            },
            "episode_indices_by_base_seed": {
                str(base_seed): sorted(
                    item["episode_index"]
                    for item in identities
                    if item["base_seed"] == base_seed
                )
                for base_seed in sorted({item["base_seed"] for item in identities})
            },
            "epochs_resampled": False,
        },
        "bootstrap": {
            "method": "matched-unit percentile bootstrap",
            "resamples": resamples,
            "confidence": confidence,
            "seed": bootstrap_seed,
            "quantile_convention": "nearest-rank without interpolation",
        },
        "oracle_condition": "paired/oracle",
        "condition_summaries": condition_summaries,
        "paired_effects_vs_oracle": paired_effects,
        "tail_risk_p10": tail_risk,
        "controller_selection": controller_selection,
    }


def aggregate_metric_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Average numeric fields across already independent episode/seed rows."""

    materialized = tuple(dict(row) for row in rows)
    if not materialized:
        raise ValueError("rows must be non-empty")
    result: dict[str, float] = {}
    shared = set.intersection(*(set(row) for row in materialized))
    for name in sorted(shared):
        values = [row[name] for row in materialized]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            numeric = [float(value) for value in values]
            if all(math.isfinite(value) for value in numeric):
                result[name] = sum(numeric) / len(numeric)
    return result


__all__ = [
    "ClassicalOracleRegretMetrics",
    "DecisionFidelityMetrics",
    "EpisodeOutcomeMetrics",
    "METRIC_AGGREGATION_SCHEMA_VERSION",
    "METRIC_CONTRACT_VERSION",
    "OutcomeMetricConfig",
    "PairedBootstrapInterval",
    "TAIL_AGGREGATION_CONTRACT",
    "aggregate_metric_rows",
    "aggregate_evaluation_units",
    "evaluate_classical_oracle_regret",
    "evaluate_closed_loop_outcomes",
    "evaluate_closed_loop_result",
    "evaluate_decision_fidelity",
    "paired_percentile_bootstrap",
]
