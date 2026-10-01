from types import SimpleNamespace
import numpy as np
import pytest
import torch
from cox_physick.models import (KANLinear, TGNModel, GraphMemory, physical_kernels,
    parameter_counts, transformed_descriptors, DQNNetwork, dqn_features, action_mask,
    FeatureNormalizer, hysteretic_huber)
from cox_physick.controllers import RuleController, rank_slots
from cox_physick.config import Config


def observation():
    sat = np.array([[0, 1, -1, -1, -1, -1, -1, -1], [1, 0, -1, -1, -1, -1, -1, -1]])
    beam = np.where(sat >= 0, 0, -1)
    z = np.zeros((2, 8, 6), np.float32)
    z[:, :2] = [.4, .6, .8, 3., .2, .1]
    return SimpleNamespace(sat_id=sat, beam_id=beam, prev_sat=np.array([0, -1]), prev_beam=np.array([0, -1]),
        feasible=sat >= 0, z=z, survival=np.full((2, 8), .5, np.float32),
        analytic=np.tile([1., 1., 0., 0., 0., 0., 0., 0.], (2, 1)),
        rst=np.ones((2, 8)), sinr=np.tile([1., 3., 0., 0., 0., 0., 0., 0.], (2, 1)))


def test_exact_paper_parameter_counts():
    assert parameter_counts() == dict(tgn_mlp=83411, tgn_kan=112480, kan_coefficient=78378,
                                      physick_full=83498, mlp_coefficient=78361)


def test_cubic_basis_partition_of_unity_saturation_and_gradient():
    module = KANLinear(2, 3).double()
    x = torch.linspace(-2., 2., 101, dtype=torch.float64)[:, None].repeat(1, 2).requires_grad_()
    basis = module.basis(x)
    torch.testing.assert_close(basis.sum(-1), torch.ones_like(x))
    assert (basis >= -1e-12).all()
    torch.testing.assert_close(basis[0], basis[24])
    module(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert module.spline_weight.grad.abs().sum() > 0


def test_physical_bank_nonnegative_partition_and_units():
    z = torch.tensor([[.2, np.log(11.) / 2, np.log(21.) * .3, 20., .7, .1]], dtype=torch.float32)
    k = physical_kernels(z, torch.tensor([.8]))
    torch.testing.assert_close(k[:, :8].sum(-1), torch.ones(1))
    torch.testing.assert_close(k[:, 8:], torch.tensor([[.8, .7]]))


@pytest.mark.parametrize("method", ["full", "tgn_mlp", "tgn_kan", "no_arrival_rst", "rst_retained", "mlp_coeff"])
def test_zero_residual_and_isolated_memory(method):
    obs = observation()
    model = TGNModel(method)
    state = model.initial_state(2, 3)
    state.beam[-1] = 2.
    residual, load, next_state = model.forward_observation(obs, state)
    assert torch.equal(residual, torch.zeros_like(residual))
    torch.testing.assert_close(next_state.beam[-1], state.beam[-1])
    assert torch.isfinite(load).all()
    loss = (residual[0, 0] - 2).square() + load.square().mean()
    loss.backward()
    assert model.readout[-1].weight.grad.abs().sum() > 0


def test_synchronous_aggregation_is_edge_order_invariant():
    model = TGNModel("full")
    obs = observation()
    state = model.initial_state(2, 3)
    _, _, first = model.forward_observation(obs, state)
    order = [1, 0, 2, 3, 4, 5, 6, 7]
    for key in ("z", "survival", "sat_id", "beam_id"):
        setattr(obs, key, getattr(obs, key)[:, order])
    _, _, second = model.forward_observation(obs, state)
    torch.testing.assert_close(first.user, second.user)
    torch.testing.assert_close(first.beam, second.beam)


def test_interventions_affect_inputs_and_survival():
    obs = observation()
    z, s = transformed_descriptors(obs, "no_cox_rst")
    assert np.all(z[..., :2] == 0)
    assert np.array_equal(z[..., 2], obs.z[..., 2])
    assert np.all(s == 1)
    z, _ = transformed_descriptors(obs, "no_triplet")
    assert np.all(z[..., :3] == 0)
    with pytest.raises(ValueError):
        transformed_descriptors(obs, "eph_physick")


def test_frozen_training_normalizer_and_masked_dqn():
    obs = observation()
    norm = FeatureNormalizer()
    norm.update(obs.z, obs.sat_id >= 0)
    norm.freeze()
    with pytest.raises(RuntimeError):
        norm.update(obs.z)
    x = dqn_features(obs)
    assert x.shape == (2, 73)
    mask = action_mask(obs)
    assert not mask[:, -1].any()
    obs.feasible[1] = False
    assert action_mask(obs)[1, -1]
    assert action_mask(obs)[1].sum() == 1
    network = DQNNetwork()
    assert network(torch.tensor(x)).shape == (2, 9)


def test_hysteretic_weight_is_point_two_not_point_zero_four():
    pred = torch.tensor([0., 0.], requires_grad=True)
    target = torch.tensor([.5, -.5])
    loss = hysteretic_huber(pred, target)
    assert loss.item() == pytest.approx((.125 + .2 * .125) / 2)
    loss.backward()
    assert pred.grad[1] / -pred.grad[0] == pytest.approx(.2)


def test_ties_hysteresis_empty_actions_and_ttt():
    obs = observation()
    cfg = Config()
    assert rank_slots(obs, obs.analytic).tolist() == [0, 1]
    obs.analytic[0, 1] += .04
    assert RuleController("greedy", cfg).select(obs)[0] == 0
    obs.analytic[0, 1] += .02
    assert RuleController("greedy", cfg).select(obs)[0] == 1
    ttt = RuleController("maxsinr_ttt", cfg)
    assert ttt.select(obs)[0] == 0
    assert ttt.select(obs)[0] == 1
    obs.feasible[:] = False
    assert np.all(RuleController("greedy", cfg).select(obs) == -1)
