"""Metric primitives with explicit aggregation and denominator rules."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from .schemas import MetricObservation


def nearest_rank(values: Sequence[float] | np.ndarray, probability: float) -> float:
    """Nearest-rank empirical quantile with no interpolation."""

    if not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    ordered = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if ordered.size == 0:
        raise ValueError("nearest_rank is undefined for an empty sample")
    rank = int(math.ceil(probability * ordered.size))
    return float(ordered[rank - 1])


def safe_rate(metric: str, numerator: float, denominator: float, *, unit: str = "dimensionless") -> MetricObservation:
    if denominator <= 0:
        return MetricObservation(
            metric=metric,
            value=None,
            defined=False,
            numerator=float(numerator),
            denominator=float(denominator),
            unit=unit,
            missing_reason="zero_or_negative_denominator",
        )
    return MetricObservation(
        metric=metric,
        value=float(numerator) / float(denominator),
        defined=True,
        numerator=float(numerator),
        denominator=float(denominator),
        unit=unit,
    )


def mean_observation(metric: str, values: Iterable[float], *, unit: str = "dimensionless") -> MetricObservation:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return MetricObservation(metric=metric, value=None, defined=False, unit=unit, missing_reason="empty_support")
    return MetricObservation(metric=metric, value=float(np.mean(array)), defined=True, support_n=int(array.size), unit=unit)


def tail_service_user_first(rates_by_time_user: np.ndarray, probability: float = 0.1) -> MetricObservation:
    """Quantile over users at each time, followed by a time mean."""

    values = np.asarray(rates_by_time_user, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        return MetricObservation(
            metric="tail_service",
            value=None,
            defined=False,
            unit="rate",
            missing_reason="empty_time_or_user_axis",
        )
    step_values = [nearest_rank(row[np.isfinite(row)], probability) for row in values if np.isfinite(row).any()]
    return mean_observation("tail_service", step_values, unit="rate")


def population_cv(values: Sequence[float] | np.ndarray, *, active_mask: Sequence[bool] | np.ndarray | None = None) -> MetricObservation:
    array = np.asarray(values, dtype=float).reshape(-1)
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if mask.shape != array.shape:
            raise ValueError("active_mask and values must have the same shape")
        array = array[mask]
    array = array[np.isfinite(array)]
    if array.size == 0:
        return MetricObservation(metric="active_load_cv", value=None, defined=False, missing_reason="empty_active_set")
    mean = float(np.mean(array))
    if mean == 0:
        return MetricObservation(metric="active_load_cv", value=None, defined=False, missing_reason="zero_active_mean")
    return MetricObservation(
        metric="active_load_cv",
        value=float(np.std(array, ddof=0) / mean),
        defined=True,
        support_n=int(array.size),
    )


def aba_event_count(associations: Sequence[str | int | None], times: Sequence[float], window_s: float) -> int:
    """Count A-B-A patterns after compressing consecutive executed targets."""

    if len(associations) != len(times):
        raise ValueError("associations and times must have equal length")
    if window_s < 0:
        raise ValueError("window_s must be non-negative")
    changes: list[tuple[float, str | int]] = []
    last: str | int | None = None
    for association, time in zip(associations, times):
        if association is None:
            continue
        if association != last:
            changes.append((float(time), association))
            last = association
    events = 0
    for index in range(2, len(changes)):
        t0, a0 = changes[index - 2]
        _, a1 = changes[index - 1]
        t2, a2 = changes[index]
        if a2 == a0 and a2 != a1 and t2 - t0 <= window_s:
            events += 1
    return events


def aggregate_defined(observations: Iterable[MetricObservation]) -> dict[str, float | int | None]:
    observations = list(observations)
    values = [float(item.value) for item in observations if item.defined and item.value is not None]
    return {
        "estimate": float(np.mean(values)) if values else None,
        "defined_n": len(values),
        "undefined_n": len(observations) - len(values),
        "total_n": len(observations),
    }
