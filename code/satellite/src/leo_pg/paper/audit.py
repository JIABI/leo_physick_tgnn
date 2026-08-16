"""Shrink--jump and dwell-sensitivity audit utilities.

This module computes diagnostics from stored descriptor errors and switch
events.  It deliberately does not label the resulting index a stability
guarantee; it is an empirical regime summary whose inputs and quantile rule are
recorded explicitly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import torch

from leo_pg.paper.calibration import nearest_rank_quantile
from leo_pg.sim.state import PolicyDescriptors


@dataclass(frozen=True)
class DescriptorScales:
    gamma: float = 10.0
    intensity: float = 1.0
    flow: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"descriptor scale {name} must be finite and positive")


def descriptor_error_vector(
    prediction: PolicyDescriptors,
    oracle: PolicyDescriptors,
    *,
    scales: DescriptorScales | None = None,
) -> torch.Tensor:
    """Return the normalized controller-facing error vector for one epoch."""

    scales = scales or DescriptorScales()
    if prediction.gamma_edge.shape != oracle.gamma_edge.shape:
        raise ValueError("gamma descriptor shapes differ")
    if prediction.intensity_edge.shape != oracle.intensity_edge.shape:
        raise ValueError("intensity descriptor shapes differ")
    if prediction.flow_node.shape != oracle.flow_node.shape:
        raise ValueError("flow descriptor shapes differ")
    components = (
        (prediction.gamma_edge - oracle.gamma_edge) / scales.gamma,
        (prediction.intensity_edge - oracle.intensity_edge) / scales.intensity,
        (prediction.flow_node - oracle.flow_node) / scales.flow,
    )
    vector = torch.cat(tuple(value.reshape(-1) for value in components))
    if not torch.isfinite(vector).all():
        raise ValueError("descriptor error contains NaN or Inf")
    return vector


def descriptor_error_norm(
    prediction: PolicyDescriptors,
    oracle: PolicyDescriptors,
    *,
    scales: DescriptorScales | None = None,
    norm: str = "l2",
) -> float:
    vector = descriptor_error_vector(prediction, oracle, scales=scales)
    normalized = str(norm).strip().lower()
    if normalized == "l1":
        value = torch.linalg.vector_norm(vector, ord=1)
    elif normalized == "l2":
        value = torch.linalg.vector_norm(vector, ord=2)
    elif normalized in {"linf", "inf"}:
        value = torch.linalg.vector_norm(vector, ord=float("inf"))
    else:
        raise ValueError("norm must be one of l1, l2, linf")
    return float(value.item())


@dataclass(frozen=True)
class ShrinkJumpSample:
    previous_error: float
    next_error: float
    switched: bool
    seed: int | None = None
    user: int | None = None
    epoch: int | None = None

    @property
    def ratio(self) -> float:
        return self.next_error / self.previous_error


@dataclass(frozen=True)
class ShrinkJumpSummary:
    alpha_quantile: float
    beta_quantile: float
    alpha_probability: float
    beta_probability: float
    dwell_steps: int
    stability_index: float
    no_switch_count: int
    switch_count: int
    expansive_fraction: float
    epsilon: float
    interpretation: str = "empirical_diagnostic_not_a_guarantee"

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def build_shrink_jump_samples(
    error_norms: Sequence[float],
    switch_between: Sequence[bool],
    *,
    epsilon: float = 1e-8,
    seed: int | None = None,
    user: int | None = None,
) -> tuple[ShrinkJumpSample, ...]:
    """Pair ``e_t`` and ``e_{t+1}`` with the action event between them."""

    if len(error_norms) < 2:
        raise ValueError("at least two error norms are required")
    if len(switch_between) != len(error_norms) - 1:
        raise ValueError("switch_between must have len(error_norms)-1 entries")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    samples: list[ShrinkJumpSample] = []
    for epoch, (previous, next_value, switched) in enumerate(
        zip(error_norms[:-1], error_norms[1:], switch_between)
    ):
        previous = float(previous)
        next_value = float(next_value)
        if not math.isfinite(previous) or not math.isfinite(next_value):
            raise ValueError("error norms must be finite")
        if previous < 0 or next_value < 0:
            raise ValueError("error norms must be non-negative")
        samples.append(
            ShrinkJumpSample(
                previous_error=max(previous, epsilon),
                next_error=next_value,
                switched=bool(switched),
                seed=seed,
                user=user,
                epoch=epoch,
            )
        )
    return tuple(samples)


def summarize_shrink_jump(
    samples: Iterable[ShrinkJumpSample],
    *,
    dwell_steps: int,
    alpha_probability: float = 0.95,
    beta_probability: float = 0.95,
    epsilon: float = 1e-8,
) -> ShrinkJumpSummary:
    """Estimate ``alpha``, ``beta`` and ``I_sj=beta*alpha**D``."""

    if isinstance(dwell_steps, bool) or int(dwell_steps) != dwell_steps or dwell_steps < 0:
        raise ValueError("dwell_steps must be a non-negative integer")
    values = tuple(samples)
    if not values:
        raise ValueError("at least one shrink-jump sample is required")
    ratios = torch.tensor([sample.ratio for sample in values], dtype=torch.float64)
    if not torch.isfinite(ratios).all() or torch.any(ratios < 0):
        raise ValueError("shrink-jump ratios must be finite and non-negative")
    switches = torch.tensor([sample.switched for sample in values], dtype=torch.bool)
    no_switch_ratios = ratios[~switches]
    switch_ratios = ratios[switches]
    if no_switch_ratios.numel() == 0 or switch_ratios.numel() == 0:
        raise ValueError("both switch and no-switch samples are required")
    alpha = float(nearest_rank_quantile(no_switch_ratios, alpha_probability).item())
    beta = float(nearest_rank_quantile(switch_ratios, beta_probability).item())
    index = beta * (alpha ** int(dwell_steps))
    return ShrinkJumpSummary(
        alpha_quantile=alpha,
        beta_quantile=beta,
        alpha_probability=float(alpha_probability),
        beta_probability=float(beta_probability),
        dwell_steps=int(dwell_steps),
        stability_index=float(index),
        no_switch_count=int(no_switch_ratios.numel()),
        switch_count=int(switch_ratios.numel()),
        expansive_fraction=float(
            (no_switch_ratios > 1.0).to(torch.float64).mean().item()
        ),
        epsilon=float(epsilon),
    )


def dwell_sweep(
    samples: Iterable[ShrinkJumpSample],
    dwell_values: Iterable[int],
    *,
    alpha_probability: float = 0.95,
    beta_probability: float = 0.95,
) -> tuple[ShrinkJumpSummary, ...]:
    frozen = tuple(samples)
    values = tuple(int(value) for value in dwell_values)
    if not values:
        raise ValueError("dwell_values must be non-empty")
    return tuple(
        summarize_shrink_jump(
            frozen,
            dwell_steps=value,
            alpha_probability=alpha_probability,
            beta_probability=beta_probability,
        )
        for value in values
    )


def default_dwell_sweep() -> Mapping[str, object]:
    """Configuration-only default spanning 0.2--2.0 s at 100 ms control."""

    return {
        "dwell_steps": [5, 10, 20, 40],
        "alpha_probability": 0.95,
        "beta_probability": 0.95,
        "provenance": "estimated_sensitivity_grid",
        "rationale": (
            "The main protocol fixes Dmin=10; neighbouring values are an "
            "explicit sensitivity grid rather than recovered publication runs."
        ),
    }


__all__ = [
    "DescriptorScales",
    "ShrinkJumpSample",
    "ShrinkJumpSummary",
    "build_shrink_jump_samples",
    "default_dwell_sweep",
    "descriptor_error_norm",
    "descriptor_error_vector",
    "dwell_sweep",
    "summarize_shrink_jump",
]
