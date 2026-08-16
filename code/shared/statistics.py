"""Run-first paired hierarchical uncertainty for matched experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from .schemas import ResultRecord


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
