"""Core simulator-side Intensity--Flow descriptor operations.

The functions in this module implement the manuscript equations without owning
simulator state.  In particular, all inputs are simulator-side values; policy
predictions must not be passed back here as authoritative state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Literal, TypeAlias

import torch


PhiMode: TypeAlias = Literal["zero", "identity", "global_mean", "mean_reversion"]
Phi: TypeAlias = PhiMode | Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FirstViolationCoxResult:
    r"""Outputs of the feasible-start first-violation Cox calculation.

    ``integrated_intensity`` is the policy-facing :math:`\Lambda^{viol}`.
    ``log1p_target`` is its numerically conditioned learning target.  A false
    entry in ``feasible`` has exactly zero gated hazard and integrated intensity;
    that zero is an already-infeasible sentinel, not a low-risk prediction.
    """

    feasible: torch.Tensor
    gated_hazard: torch.Tensor
    integrated_intensity: torch.Tensor
    log1p_target: torch.Tensor


def _require_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _finite_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _coefficient(
    name: str,
    value: Real | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device != reference.device:
            raise ValueError(f"{name} must be on the same device as the covariates")
        result = value.to(dtype=reference.dtype)
    else:
        result = torch.as_tensor(
            _finite_real(name, value), dtype=reference.dtype, device=reference.device
        )
    _require_finite_tensor(name, result)
    return result


def nearest_rank_quantile(
    samples: torch.Tensor,
    q: float,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Return a nearest-rank empirical quantile without interpolation.

    The selected (one-based) order statistic is ``ceil(q * N)``.  This is the
    convention used for the manuscript's control-window p10 aggregation.
    """

    samples = _require_tensor("samples", samples)
    _require_finite_tensor("samples", samples)
    q_value = _finite_real("q", q)
    if not 0.0 < q_value <= 1.0:
        raise ValueError("q must lie in (0, 1]")
    if samples.ndim == 0:
        raise ValueError("samples must have at least one dimension")
    if not -samples.ndim <= dim < samples.ndim:
        raise ValueError(f"dim={dim} is out of range for a {samples.ndim}-D tensor")

    normalized_dim = dim % samples.ndim
    sample_count = samples.shape[normalized_dim]
    if sample_count == 0:
        raise ValueError("the aggregation dimension must be non-empty")
    rank = math.ceil(q_value * sample_count)
    return torch.kthvalue(samples, rank, dim=normalized_dim).values


def nearest_rank_p10(samples: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Return the nearest-rank p10 used for ``gamma_sim`` aggregation."""

    return nearest_rank_quantile(samples, 0.1, dim=dim)


def feasibility_gate(
    gamma_sim: torch.Tensor,
    flow_sim: torch.Tensor,
    *,
    gamma_min: float = -5.0,
    flow_max: float = 0.95,
) -> torch.Tensor:
    """Evaluate the inclusive simulator-side SINR/load feasibility predicate."""

    gamma_sim = _require_tensor("gamma_sim", gamma_sim)
    flow_sim = _require_tensor("flow_sim", flow_sim)
    _require_finite_tensor("gamma_sim", gamma_sim)
    _require_finite_tensor("flow_sim", flow_sim)
    gamma_threshold = _finite_real("gamma_min", gamma_min)
    flow_threshold = _finite_real("flow_max", flow_max)
    if flow_threshold < 0.0:
        raise ValueError("flow_max must be non-negative")
    if flow_sim.numel() and bool((flow_sim < 0).any()):
        raise ValueError("flow_sim must be non-negative")
    if gamma_sim.device != flow_sim.device:
        raise ValueError("gamma_sim and flow_sim must be on the same device")
    try:
        gamma_b, flow_b = torch.broadcast_tensors(gamma_sim, flow_sim)
    except RuntimeError as exc:
        raise ValueError("gamma_sim and flow_sim must be broadcast-compatible") from exc
    return (gamma_b >= gamma_threshold) & (flow_b <= flow_threshold)


def _evaluate_phi(load: torch.Tensor, phi: Phi) -> torch.Tensor:
    if callable(phi):
        raw_feedback = phi(load)
        if not isinstance(raw_feedback, torch.Tensor):
            raise TypeError("a callable phi must return a torch.Tensor")
    elif phi == "zero":
        raw_feedback = torch.zeros_like(load)
    elif phi == "identity":
        raw_feedback = load
    elif phi == "global_mean":
        raw_feedback = load.mean().expand_as(load)
    elif phi == "mean_reversion":
        raw_feedback = load.mean() - load
    else:
        raise ValueError(
            "phi must be callable or one of: zero, identity, global_mean, "
            "mean_reversion"
        )

    if raw_feedback.device != load.device:
        raise ValueError("phi output must be on the same device as load")
    try:
        feedback = torch.broadcast_to(raw_feedback, load.shape).to(dtype=load.dtype)
    except RuntimeError as exc:
        raise ValueError("phi output must be broadcast-compatible with load") from exc
    _require_finite_tensor("phi output", feedback)
    return feedback


def instantaneous_mean_flow(
    load: torch.Tensor,
    admitted_users: torch.Tensor,
    *,
    dt_ctrl: float,
    arrival_rate: float,
    service_rate: float,
    feedback_strength: float,
    phi: Phi,
    clip_bounds: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Compute the clipped instantaneous mean-flow component.

    Implements

    ``clip(L + dt * (arrival_rate * n - service_rate) + kappa * Phi(L))``.

    ``phi`` is deliberately explicit because the manuscript equation leaves the
    coupling map configurable.  Built-in modes are provided for common choices,
    and a callable can reproduce a released experiment-specific map exactly.
    """

    load = _require_tensor("load", load)
    admitted_users = _require_tensor("admitted_users", admitted_users)
    if load.ndim != 1 or admitted_users.ndim != 1:
        raise ValueError("load and admitted_users must both be one-dimensional")
    if load.shape != admitted_users.shape:
        raise ValueError("load and admitted_users must have identical shapes")
    if load.numel() == 0:
        raise ValueError("load and admitted_users must be non-empty")
    if load.device != admitted_users.device:
        raise ValueError("load and admitted_users must be on the same device")
    if not load.is_floating_point():
        raise TypeError("load must have a floating-point dtype")
    _require_finite_tensor("load", load)
    _require_finite_tensor("admitted_users", admitted_users)
    if bool((admitted_users < 0).any()):
        raise ValueError("admitted_users must be non-negative")

    dt = _finite_real("dt_ctrl", dt_ctrl)
    arrival = _finite_real("arrival_rate", arrival_rate)
    service = _finite_real("service_rate", service_rate)
    strength = _finite_real("feedback_strength", feedback_strength)
    if dt <= 0.0:
        raise ValueError("dt_ctrl must be positive")
    if arrival < 0.0 or service < 0.0 or strength < 0.0:
        raise ValueError("rates and feedback_strength must be non-negative")
    if not isinstance(clip_bounds, tuple) or len(clip_bounds) != 2:
        raise TypeError("clip_bounds must be a (lower, upper) tuple")
    lower = _finite_real("clip_bounds[0]", clip_bounds[0])
    upper = _finite_real("clip_bounds[1]", clip_bounds[1])
    if not 0.0 <= lower < upper:
        raise ValueError("clip bounds must satisfy 0 <= lower < upper")
    if bool(((load < lower) | (load > upper)).any()):
        raise ValueError("load must lie within clip_bounds")

    feedback = _evaluate_phi(load, phi)
    instantaneous = load + dt * (arrival * admitted_users.to(load.dtype) - service)
    instantaneous = instantaneous + strength * feedback
    instantaneous = instantaneous.clamp(min=lower, max=upper)
    _require_finite_tensor("instantaneous mean-flow output", instantaneous)
    return instantaneous


def mean_flow_update(
    load: torch.Tensor,
    admitted_users: torch.Tensor,
    *,
    dt_ctrl: float,
    arrival_rate: float,
    service_rate: float,
    feedback_strength: float,
    ema_factor: float,
    phi: Phi,
    clip_bounds: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Apply the manuscript EMA to the instantaneous mean-flow component."""

    rho = _finite_real("ema_factor", ema_factor)
    if not 0.0 < rho < 1.0:
        raise ValueError("ema_factor must lie strictly between 0 and 1")
    instantaneous = instantaneous_mean_flow(
        load,
        admitted_users,
        dt_ctrl=dt_ctrl,
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        feedback_strength=feedback_strength,
        phi=phi,
        clip_bounds=clip_bounds,
    )
    updated = rho * load + (1.0 - rho) * instantaneous
    _require_finite_tensor("EMA mean-flow output", updated)
    return updated


def integrated_violation_intensity(
    gamma_sim: torch.Tensor,
    flow_sim: torch.Tensor,
    lookahead_covariates: torch.Tensor,
    *,
    baseline_hazard: float,
    beta_gamma: float | torch.Tensor,
    beta_flow: float | torch.Tensor,
    beta_lookahead: float | torch.Tensor,
    horizon: float,
    gamma_min: float = -5.0,
    flow_max: float = 0.95,
    num_intervals: int = 128,
) -> FirstViolationCoxResult:
    """Compute a frozen-state feasible-start first-violation descriptor.

    ``gamma_sim`` and ``flow_sim`` are one-dimensional, authoritative current
    simulator values.  The SINR covariate uses the manuscript's
    violation-aligned sign ``-gamma_sim``; load uses ``flow_sim``.  Geometry-only
    Cox covariates may evolve over the look-ahead window and are supplied either
    as ``[num_intervals + 1, E]`` or ``[num_intervals + 1, E, G]``.  For the
    latter, ``beta_lookahead`` must be scalar or have shape ``[G]``.

    The default is exactly 128 trapezoidal sub-intervals (129 grid endpoints).
    The current feasibility gate is computed once and held fixed over the whole
    window.  Thus currently infeasible edges receive an exact zero sentinel.
    """

    gamma_sim = _require_tensor("gamma_sim", gamma_sim)
    flow_sim = _require_tensor("flow_sim", flow_sim)
    lookahead_covariates = _require_tensor(
        "lookahead_covariates", lookahead_covariates
    )
    if gamma_sim.ndim != 1 or flow_sim.ndim != 1:
        raise ValueError("gamma_sim and flow_sim must both be one-dimensional")
    if gamma_sim.shape != flow_sim.shape:
        raise ValueError("gamma_sim and flow_sim must have identical shapes")
    if gamma_sim.device != flow_sim.device or gamma_sim.device != lookahead_covariates.device:
        raise ValueError("all Cox inputs must be on the same device")
    if not gamma_sim.is_floating_point() or not lookahead_covariates.is_floating_point():
        raise TypeError("gamma_sim and lookahead_covariates must be floating-point")
    _require_finite_tensor("gamma_sim", gamma_sim)
    _require_finite_tensor("flow_sim", flow_sim)
    _require_finite_tensor("lookahead_covariates", lookahead_covariates)

    if isinstance(num_intervals, bool) or not isinstance(num_intervals, int):
        raise TypeError("num_intervals must be an integer")
    if num_intervals < 1:
        raise ValueError("num_intervals must be at least 1")
    expected_points = num_intervals + 1
    edge_count = gamma_sim.numel()
    if lookahead_covariates.ndim not in (2, 3):
        raise ValueError("lookahead_covariates must have shape [grid, E] or [grid, E, G]")
    if lookahead_covariates.shape[:2] != (expected_points, edge_count):
        raise ValueError(
            "lookahead_covariates must have shape "
            f"[{expected_points}, {edge_count}] (plus an optional feature axis)"
        )
    if lookahead_covariates.ndim == 3 and lookahead_covariates.shape[2] == 0:
        raise ValueError("the look-ahead feature axis must be non-empty")

    duration = _finite_real("horizon", horizon)
    baseline = _finite_real("baseline_hazard", baseline_hazard)
    if duration <= 0.0:
        raise ValueError("horizon must be positive")
    if baseline <= 0.0:
        raise ValueError("baseline_hazard must be positive")

    reference = lookahead_covariates
    beta_g = _coefficient("beta_gamma", beta_gamma, reference)
    beta_l = _coefficient("beta_flow", beta_flow, reference)
    if beta_g.numel() != 1 or beta_l.numel() != 1:
        raise ValueError("beta_gamma and beta_flow must be scalar")
    beta_geo = _coefficient("beta_lookahead", beta_lookahead, reference)

    if lookahead_covariates.ndim == 2:
        if beta_geo.numel() != 1:
            raise ValueError("beta_lookahead must be scalar for [grid, E] covariates")
        geometry_linear = lookahead_covariates * beta_geo.reshape(())
    else:
        feature_count = lookahead_covariates.shape[2]
        if beta_geo.numel() == 1:
            beta_geo = beta_geo.expand(feature_count)
        elif beta_geo.shape != (feature_count,):
            raise ValueError(
                f"beta_lookahead must be scalar or have shape [{feature_count}]"
            )
        geometry_linear = torch.einsum(
            "teg,g->te", lookahead_covariates, beta_geo
        )

    dtype = lookahead_covariates.dtype
    frozen_linear = (
        beta_g.reshape(()) * -gamma_sim.to(dtype)
        + beta_l.reshape(()) * flow_sim.to(dtype)
    )
    log_hazard = frozen_linear.unsqueeze(0) + geometry_linear
    hazard = torch.as_tensor(baseline, dtype=dtype, device=reference.device) * torch.exp(
        log_hazard
    )
    _require_finite_tensor("Cox hazard", hazard)

    feasible = feasibility_gate(
        gamma_sim,
        flow_sim,
        gamma_min=gamma_min,
        flow_max=flow_max,
    )
    gated_hazard = hazard * feasible.to(dtype=dtype).unsqueeze(0)
    step = duration / num_intervals
    integrated = torch.trapezoid(gated_hazard, dx=step, dim=0)
    # Make the sentinel exact even on unusual low-precision backends.
    integrated = torch.where(feasible, integrated, torch.zeros_like(integrated))
    target = torch.log1p(integrated)
    _require_finite_tensor("integrated Cox intensity", integrated)
    _require_finite_tensor("log1p Cox target", target)
    if bool((integrated < 0).any()):
        raise ValueError("integrated Cox intensity must be non-negative")

    return FirstViolationCoxResult(
        feasible=feasible,
        gated_hazard=gated_hazard,
        integrated_intensity=integrated,
        log1p_target=target,
    )


# Descriptive compatibility names retained for equation-level call sites.
simulator_feasibility_gate = feasibility_gate
ema_mean_flow_update = mean_flow_update
first_violation_cox_intensity = integrated_violation_intensity
