import math

import pytest
import torch

from leo_pg.sim.intensity_flow import (
    ema_mean_flow_update,
    first_violation_cox_intensity,
    instantaneous_mean_flow,
    nearest_rank_p10,
    nearest_rank_quantile,
    simulator_feasibility_gate,
)


def test_nearest_rank_p10_uses_tenth_order_statistic_without_interpolation():
    descending = torch.arange(100.0, 0.0, -1.0).reshape(1, 100)
    result = nearest_rank_p10(descending)
    assert torch.equal(result, torch.tensor([10.0]))


def test_nearest_rank_quantile_supports_an_explicit_axis():
    samples = torch.tensor([[4.0, 1.0], [2.0, 3.0], [8.0, 0.0]])
    result = nearest_rank_quantile(samples, 0.5, dim=0)
    assert torch.equal(result, torch.tensor([4.0, 1.0]))


@pytest.mark.parametrize(
    ("samples", "q", "message"),
    [
        (torch.empty(2, 0), 0.1, "non-empty"),
        (torch.tensor([1.0, float("nan")]), 0.1, "finite"),
        (torch.tensor([1.0]), 0.0, "lie in"),
        (torch.tensor([1.0]), 1.1, "lie in"),
    ],
)
def test_nearest_rank_validates_inputs(samples, q, message):
    with pytest.raises(ValueError, match=message):
        nearest_rank_quantile(samples, q)


def test_simulator_feasibility_gate_is_inclusive_at_both_thresholds():
    gamma = torch.tensor([-5.0, -5.01, 2.0, 2.0])
    flow = torch.tensor([0.95, 0.5, 0.951, 0.2])
    feasible = simulator_feasibility_gate(gamma, flow)
    assert torch.equal(feasible, torch.tensor([True, False, False, True]))


def test_simulator_feasibility_gate_broadcasts_node_flow_to_edges():
    gamma = torch.tensor([[-4.0, -6.0], [0.0, 1.0]])
    flow = torch.tensor([0.2, 0.96])
    feasible = simulator_feasibility_gate(gamma, flow)
    assert torch.equal(
        feasible, torch.tensor([[True, False], [True, False]])
    )


def test_instantaneous_and_ema_mean_flow_match_the_manuscript_equations():
    load = torch.tensor([0.2, 0.8])
    admitted = torch.tensor([2, 0])
    instantaneous = instantaneous_mean_flow(
        load,
        admitted,
        dt_ctrl=0.1,
        arrival_rate=0.8,
        service_rate=1.0,
        feedback_strength=0.25,
        phi="global_mean",
    )
    # L + dt * (lambda_a * n - mu) + kappa * mean(L)
    expected_instantaneous = torch.tensor([0.385, 0.825])
    assert torch.allclose(instantaneous, expected_instantaneous)

    updated = ema_mean_flow_update(
        load,
        admitted,
        dt_ctrl=0.1,
        arrival_rate=0.8,
        service_rate=1.0,
        feedback_strength=0.25,
        ema_factor=0.9,
        phi="global_mean",
    )
    expected_ema = 0.9 * load + 0.1 * expected_instantaneous
    assert torch.allclose(updated, expected_ema)


def test_mean_flow_clips_before_ema_and_accepts_callable_phi():
    load = torch.tensor([0.9, 0.1])
    admitted = torch.tensor([10.0, 0.0])
    def phi(current):
        return 2.0 * current
    instantaneous = instantaneous_mean_flow(
        load,
        admitted,
        dt_ctrl=1.0,
        arrival_rate=1.0,
        service_rate=1.0,
        feedback_strength=1.0,
        phi=phi,
    )
    assert torch.equal(instantaneous, torch.tensor([1.0, 0.0]))

    updated = ema_mean_flow_update(
        load,
        admitted,
        dt_ctrl=1.0,
        arrival_rate=1.0,
        service_rate=1.0,
        feedback_strength=1.0,
        ema_factor=0.5,
        phi=phi,
    )
    assert torch.allclose(updated, torch.tensor([0.95, 0.05]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt_ctrl": 0.0},
        {"arrival_rate": -0.1},
        {"service_rate": float("inf")},
        {"feedback_strength": -1.0},
        {"clip_bounds": (1.0, 0.0)},
    ],
)
def test_mean_flow_validates_parameter_ranges(kwargs):
    valid = dict(
        dt_ctrl=0.1,
        arrival_rate=0.8,
        service_rate=1.0,
        feedback_strength=0.25,
        phi="zero",
        clip_bounds=(0.0, 1.0),
    )
    valid.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        instantaneous_mean_flow(torch.tensor([0.2]), torch.tensor([1]), **valid)


def test_mean_flow_rejects_nonfinite_or_out_of_range_state_and_phi():
    common = dict(
        dt_ctrl=0.1,
        arrival_rate=0.8,
        service_rate=1.0,
        feedback_strength=0.25,
    )
    with pytest.raises(ValueError, match="clip_bounds"):
        instantaneous_mean_flow(
            torch.tensor([1.2]), torch.tensor([1]), phi="zero", **common
        )
    with pytest.raises(ValueError, match="finite"):
        instantaneous_mean_flow(
            torch.tensor([0.2]),
            torch.tensor([1]),
            phi=lambda value: torch.full_like(value, float("nan")),
            **common,
        )
    with pytest.raises(ValueError, match="strictly"):
        ema_mean_flow_update(
            torch.tensor([0.2]),
            torch.tensor([1]),
            ema_factor=1.0,
            phi="zero",
            **common,
        )


def test_constant_cox_hazard_integrates_exactly_and_uses_zero_sentinel():
    intervals = 128
    gamma = torch.tensor([0.0, -6.0, 2.0])
    flow = torch.tensor([0.5, 0.2, 0.96])
    geometry = torch.zeros(intervals + 1, 3)
    result = first_violation_cox_intensity(
        gamma,
        flow,
        geometry,
        baseline_hazard=0.2,
        beta_gamma=0.0,
        beta_flow=0.0,
        beta_lookahead=0.0,
        horizon=10.0,
        num_intervals=intervals,
    )
    assert torch.equal(result.feasible, torch.tensor([True, False, False]))
    assert torch.allclose(result.integrated_intensity, torch.tensor([2.0, 0.0, 0.0]))
    assert torch.equal(result.gated_hazard[:, 1:], torch.zeros(intervals + 1, 2))
    assert torch.allclose(result.log1p_target, torch.log1p(torch.tensor([2.0, 0.0, 0.0])))


def test_cox_freezes_state_and_gate_while_geometry_evolves():
    # Hazard is exp(tau); two trapezoids over [0, 2] give
    # 0.5 * [1 + 2e + e^2]. The second edge is infeasible at t and stays gated.
    gamma = torch.tensor([1.0, -10.0])
    flow = torch.tensor([0.2, 0.2])
    geometry = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    result = first_violation_cox_intensity(
        gamma,
        flow,
        geometry,
        baseline_hazard=1.0,
        beta_gamma=0.0,
        beta_flow=0.0,
        beta_lookahead=1.0,
        horizon=2.0,
        num_intervals=2,
    )
    expected = 0.5 * (1.0 + 2.0 * math.e + math.e**2)
    assert result.integrated_intensity[0].item() == pytest.approx(expected)
    assert result.integrated_intensity[1].item() == 0.0


def test_cox_supports_multiple_geometry_covariates():
    geometry = torch.zeros(3, 1, 2)
    geometry[:, 0, 0] = torch.tensor([0.0, 1.0, 2.0])
    geometry[:, 0, 1] = 2.0
    result = first_violation_cox_intensity(
        torch.tensor([0.0]),
        torch.tensor([0.1]),
        geometry,
        baseline_hazard=1.0,
        beta_gamma=0.0,
        beta_flow=0.0,
        beta_lookahead=torch.tensor([0.0, 0.5]),
        horizon=2.0,
        num_intervals=2,
    )
    assert result.integrated_intensity.item() == pytest.approx(2.0 * math.e)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"horizon": 0.0}, "horizon"),
        ({"baseline_hazard": 0.0}, "baseline_hazard"),
        ({"num_intervals": 0}, "num_intervals"),
        ({"beta_gamma": float("nan")}, "beta_gamma"),
    ],
)
def test_cox_validates_scalar_parameters(changes, error):
    kwargs = dict(
        baseline_hazard=1.0,
        beta_gamma=0.0,
        beta_flow=0.0,
        beta_lookahead=0.0,
        horizon=1.0,
        num_intervals=2,
    )
    kwargs.update(changes)
    with pytest.raises((TypeError, ValueError), match=error):
        first_violation_cox_intensity(
            torch.tensor([0.0]),
            torch.tensor([0.1]),
            torch.zeros(3, 1),
            **kwargs,
        )


def test_cox_validates_grid_shape_finiteness_and_numerical_overflow():
    common = dict(
        baseline_hazard=1.0,
        beta_gamma=0.0,
        beta_flow=0.0,
        beta_lookahead=1.0,
        horizon=1.0,
        num_intervals=2,
    )
    with pytest.raises(ValueError, match="shape"):
        first_violation_cox_intensity(
            torch.tensor([0.0]),
            torch.tensor([0.1]),
            torch.zeros(2, 1),
            **common,
        )
    with pytest.raises(ValueError, match="finite"):
        first_violation_cox_intensity(
            torch.tensor([0.0]),
            torch.tensor([0.1]),
            torch.tensor([[0.0], [float("nan")], [0.0]]),
            **common,
        )
    with pytest.raises(ValueError, match="Cox hazard"):
        first_violation_cox_intensity(
            torch.tensor([0.0]),
            torch.tensor([0.1]),
            torch.full((3, 1), 1_000.0),
            **common,
        )
