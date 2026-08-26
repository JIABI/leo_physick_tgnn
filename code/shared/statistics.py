"""Run-first inference and registered paired analyses.

The satellite manuscript uses equal-weight run means followed by Student-t
intervals.  The UAV study retains its run-first hierarchical bootstrap.  Both
contracts live here so callers cannot silently exchange their independent
units.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

import numpy as np
from scipy import stats

from .schemas import ResultRecord


@dataclass(frozen=True)
class StudentTSummary:
    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    run_n: int
    degrees_of_freedom: int
    standard_error: float


@dataclass(frozen=True)
class EquivalenceSummary:
    ratio: float
    ci_low: float
    ci_high: float
    lower_bound: float
    upper_bound: float
    alpha: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool
    paired_n: int


def _finite_vector(name: str, values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size < 2:
        raise ValueError(f"{name} requires at least two independent runs")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def run_first_student_t(
    run_means: Iterable[float], *, confidence: float = 0.95
) -> StudentTSummary:
    """Equal-weight mean and two-sided Student-t interval over run means."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    values = _finite_vector("run_means", run_means)
    n = int(values.size)
    estimate = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(n))
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    half_width = critical * standard_error
    return StudentTSummary(
        estimate=estimate,
        ci_low=estimate - half_width,
        ci_high=estimate + half_width,
        confidence=float(confidence),
        run_n=n,
        degrees_of_freedom=n - 1,
        standard_error=standard_error,
    )


def paired_run_first_student_t(
    comparison_run_means: Iterable[float],
    reference_run_means: Iterable[float],
    *,
    confidence: float = 0.95,
) -> StudentTSummary:
    """Student-t interval for matched run-level differences (comparison-reference)."""

    comparison = _finite_vector("comparison_run_means", comparison_run_means)
    reference = _finite_vector("reference_run_means", reference_run_means)
    if comparison.shape != reference.shape:
        raise ValueError("paired run vectors must have the same length")
    return run_first_student_t(comparison - reference, confidence=confidence)


def paired_geometric_ratio_tost(
    comparison: Iterable[float],
    reference: Iterable[float],
    *,
    bounds: tuple[float, float] = (0.95, 1.05),
    alpha: float = 0.05,
) -> EquivalenceSummary:
    """Paired log-scale geometric ratio and two one-sided equivalence tests.

    Checkpoint pairs are the independent units.  The returned interval is the
    ``1-2*alpha`` interval used by TOST (90% when ``alpha=0.05``).
    """

    x = _finite_vector("comparison", comparison)
    y = _finite_vector("reference", reference)
    if x.shape != y.shape:
        raise ValueError("paired vectors must have the same length")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("geometric ratios require strictly positive values")
    lower, upper = map(float, bounds)
    if not 0 < lower < 1 < upper:
        raise ValueError("equivalence bounds must satisfy 0 < lower < 1 < upper")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    log_ratio = np.log(x) - np.log(y)
    n = int(log_ratio.size)
    mean = float(log_ratio.mean())
    se = float(log_ratio.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        p_lower = 0.0 if mean > np.log(lower) else 1.0
        p_upper = 0.0 if mean < np.log(upper) else 1.0
        ci_low = ci_high = float(np.exp(mean))
    else:
        df = n - 1
        t_lower = (mean - np.log(lower)) / se
        t_upper = (mean - np.log(upper)) / se
        p_lower = float(stats.t.sf(t_lower, df=df))
        p_upper = float(stats.t.cdf(t_upper, df=df))
        critical = float(stats.t.ppf(1.0 - alpha, df=df))
        ci_low = float(np.exp(mean - critical * se))
        ci_high = float(np.exp(mean + critical * se))
    p_tost = max(p_lower, p_upper)
    return EquivalenceSummary(
        ratio=float(np.exp(mean)),
        ci_low=ci_low,
        ci_high=ci_high,
        lower_bound=lower,
        upper_bound=upper,
        alpha=float(alpha),
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        equivalent=bool(p_tost < alpha and ci_low >= lower and ci_high <= upper),
        paired_n=n,
    )


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm step-down adjusted P values in original order."""

    p = np.asarray(list(p_values), dtype=float).reshape(-1)
    if p.size == 0:
        return []
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be finite values in [0, 1]")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(
        np.minimum(1.0, (p.size - np.arange(p.size)) * ranked)
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return [float(value) for value in adjusted]


def continuity_corrected_ratio_of_rate_ratios(
    *,
    a1_count: float,
    a1_exposure: float,
    a2_count: float,
    a2_exposure: float,
    b1_count: float,
    b1_exposure: float,
    b2_count: float,
    b2_exposure: float,
    correction: float = 0.5,
) -> float:
    """EXP3 C7 ratio-of-rate-ratios with +0.5 in each event-count cell."""

    counts = np.asarray([a1_count, a2_count, b1_count, b2_count], dtype=float)
    exposure = np.asarray(
        [a1_exposure, a2_exposure, b1_exposure, b2_exposure], dtype=float
    )
    if np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("event counts must be finite and non-negative")
    if np.any(exposure <= 0) or not np.isfinite(exposure).all():
        raise ValueError("exposures must be finite and positive")
    if not np.isfinite(correction) or correction < 0:
        raise ValueError("correction must be finite and non-negative")
    rates = (counts + float(correction)) / exposure
    return float((rates[3] / rates[2]) / (rates[1] / rates[0]))


class BootstrapMode(str, Enum):
    """Episode sampling contracts.

    ``nested_within_run`` resamples matched episodes separately inside each
    selected run. ``fixed_panel`` conditions on the observed episode panel and
    resamples runs only. ``crossed_run_episode`` resamples one common episode
    panel and applies it to every selected run, appropriate only when episode IDs
    are identical across runs.
    """

    NESTED_WITHIN_RUN = "nested_within_run"
    FIXED_PANEL = "fixed_panel"
    CROSSED_RUN_EPISODE = "crossed_run_episode"


@dataclass(frozen=True)
class ContrastSummary:
    platform: str
    condition_id: str
    metric: str
    cell_a: str
    cell_b: str
    contrast: str
    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    bootstrap_draws: int
    bootstrap_mode: str
    stress_id: str
    split: str
    run_n: int
    paired_episode_n: int
    undefined_a_n: int
    undefined_b_n: int
    unpaired_n: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _condition_values(
    records: Iterable[ResultRecord],
    *,
    platform: str,
    condition_id: str,
    cell_id: str,
    metric: str,
    stress_id: str,
    split: str,
) -> tuple[dict[tuple[str, str, str], float], int]:
    values: dict[tuple[str, str, str], float] = {}
    undefined = 0
    for row in records:
        if (
            row.platform != platform
            or row.condition_id != condition_id
            or row.cell_id != cell_id
            or row.metric != metric
            or row.stress_id != stress_id
            or row.split != split
        ):
            continue
        if not row.defined or row.value is None:
            undefined += 1
            continue
        key = (row.run_id, row.episode_id, row.exogenous_sequence_id)
        if key in values:
            raise ValueError(f"duplicate result record for {cell_id}, {metric}, {key}")
        values[key] = float(row.value)
    return values, undefined


def _condition_identities(
    records: Iterable[ResultRecord],
    *,
    platform: str,
    condition_id: str,
    cell_id: str,
    metric: str,
    stress_id: str,
    split: str,
) -> set[tuple[str, str, str]]:
    """Return the complete raw panel, including undefined observations."""

    return {
        (row.run_id, row.episode_id, row.exogenous_sequence_id)
        for row in records
        if row.platform == platform
        and row.condition_id == condition_id
        and row.cell_id == cell_id
        and row.metric == metric
        and row.stress_id == stress_id
        and row.split == split
    }


def paired_hierarchical_bootstrap(
    records: Iterable[ResultRecord],
    *,
    platform: str,
    condition_id: str,
    cell_a: str,
    cell_b: str,
    metric: str,
    stress_id: str = "nominal",
    split: str = "test",
    draws: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260815,
    mode: BootstrapMode | str = BootstrapMode.NESTED_WITHIN_RUN,
) -> ContrastSummary:
    """Estimate A minus B using equal run weights and paired episode identities."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    mode = BootstrapMode(mode)
    records = list(records)
    raw_keys_a = _condition_identities(
        records,
        platform=platform,
        condition_id=condition_id,
        cell_id=cell_a,
        metric=metric,
        stress_id=stress_id,
        split=split,
    )
    raw_keys_b = _condition_identities(
        records,
        platform=platform,
        condition_id=condition_id,
        cell_id=cell_b,
        metric=metric,
        stress_id=stress_id,
        split=split,
    )
    if raw_keys_a != raw_keys_b:
        raise ValueError(
            "paired inference requires identical run/episode/exogenous panels "
            "before undefined-denominator filtering"
        )
    values_a, undefined_a = _condition_values(
        records,
        platform=platform,
        condition_id=condition_id,
        cell_id=cell_a,
        metric=metric,
        stress_id=stress_id,
        split=split,
    )
    values_b, undefined_b = _condition_values(
        records,
        platform=platform,
        condition_id=condition_id,
        cell_id=cell_b,
        metric=metric,
        stress_id=stress_id,
        split=split,
    )
    keys_a, keys_b = set(values_a), set(values_b)
    paired_keys = sorted(keys_a & keys_b)
    if not paired_keys:
        raise ValueError("no defined paired run-episode observations")
    unpaired_n = len(keys_a ^ keys_b)
    differences: dict[str, dict[str, float]] = defaultdict(dict)
    for run_id, episode_id, exogenous_sequence_id in paired_keys:
        paired_unit = f"{episode_id}\x1f{exogenous_sequence_id}"
        differences[run_id][paired_unit] = (
            values_a[(run_id, episode_id, exogenous_sequence_id)]
            - values_b[(run_id, episode_id, exogenous_sequence_id)]
        )
    run_ids = sorted(differences)
    if len(run_ids) < 2:
        raise ValueError("run-first inference requires at least two independent runs")
    point_run_means = [float(np.mean(list(differences[run_id].values()))) for run_id in run_ids]
    estimate = float(np.mean(point_run_means))

    common_episodes: list[str] = []
    if mode is BootstrapMode.CROSSED_RUN_EPISODE:
        panels = [set(differences[run_id]) for run_id in run_ids]
        if any(panel != panels[0] for panel in panels[1:]):
            raise ValueError("crossed_run_episode requires the same paired episode IDs in every run")
        common_episodes = sorted(panels[0])

    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled_runs = rng.choice(run_ids, size=len(run_ids), replace=True)
        sampled_common = None
        if mode is BootstrapMode.CROSSED_RUN_EPISODE:
            sampled_common = rng.choice(common_episodes, size=len(common_episodes), replace=True)
        run_means: list[float] = []
        for run_id in sampled_runs:
            episode_values = differences[str(run_id)]
            if mode is BootstrapMode.FIXED_PANEL:
                selected = list(episode_values.values())
            elif mode is BootstrapMode.CROSSED_RUN_EPISODE:
                assert sampled_common is not None
                selected = [episode_values[str(episode_id)] for episode_id in sampled_common]
            else:
                episode_ids = list(episode_values)
                sampled_episodes = rng.choice(episode_ids, size=len(episode_ids), replace=True)
                selected = [episode_values[str(episode_id)] for episode_id in sampled_episodes]
            run_means.append(float(np.mean(selected)))
        boot[draw] = float(np.mean(run_means))

    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(boot, [alpha, 1.0 - alpha])
    return ContrastSummary(
        platform=platform,
        condition_id=condition_id,
        metric=metric,
        cell_a=cell_a,
        cell_b=cell_b,
        contrast=f"{cell_a} - {cell_b}",
        estimate=estimate,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        bootstrap_draws=draws,
        bootstrap_mode=mode.value,
        stress_id=stress_id,
        split=split,
        run_n=len(run_ids),
        paired_episode_n=len(paired_keys),
        undefined_a_n=undefined_a,
        undefined_b_n=undefined_b,
        unpaired_n=unpaired_n,
        seed=int(seed),
    )


def summarize_condition(
    records: Iterable[ResultRecord],
    *,
    platform: str,
    condition_id: str,
    cell_id: str,
    metric: str,
    stress_id: str = "nominal",
    split: str = "test",
) -> dict[str, float | int | str | None]:
    """Equal-run-weight point summary, preserving missingness counts."""

    grouped: dict[str, list[float]] = defaultdict(list)
    undefined_n = 0
    for row in records:
        if (
            row.platform == platform
            and row.condition_id == condition_id
            and row.cell_id == cell_id
            and row.metric == metric
            and row.stress_id == stress_id
            and row.split == split
        ):
            if row.defined and row.value is not None:
                grouped[row.run_id].append(float(row.value))
            else:
                undefined_n += 1
    run_means = [float(np.mean(values)) for values in grouped.values() if values]
    return {
        "platform": platform,
        "condition_id": condition_id,
        "cell_id": cell_id,
        "metric": metric,
        "stress_id": stress_id,
        "split": split,
        "estimate": float(np.mean(run_means)) if run_means else None,
        "between_run_sd": float(np.std(run_means, ddof=1)) if len(run_means) > 1 else None,
        "run_n": len(run_means),
        "defined_episode_n": sum(len(values) for values in grouped.values()),
        "undefined_episode_n": undefined_n,
    }
