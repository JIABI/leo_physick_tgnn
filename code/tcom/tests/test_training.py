"""Functional training tests use a deterministic interface fixture, not paper data."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from cox_physick.config import Config
from cox_physick import training
from cox_physick.models import DQNNetwork, FeatureNormalizer
from cox_physick.controllers import RuleController


class InterfaceEnvironment:
    def __init__(self, cfg, seed, episode_id):
        self.cfg, self.seed, self.episode_id = cfg, seed, episode_id

    def reset(self):
        self.t = 0
        self.prev = np.full(2, -1, np.int64)
        return self.observe()

    def observe(self):
        sat = np.tile([0, 1, -1, -1, -1, -1, -1, -1], (2, 1))
        z = np.zeros((2, 8, 6), np.float32)
        z[:, :2] = np.array([.3, .2, .8, 2. + self.t / 10, .1, .1], np.float32)
        analytic = np.zeros((2, 8), np.float32)
        analytic[:, :2] = [.5, .6]
        return SimpleNamespace(sat_id=sat, beam_id=np.where(sat >= 0, 0, -1),
            z=z, survival=np.full((2, 8), .4, np.float32), prev_sat=self.prev.copy(),
            prev_beam=np.where(self.prev >= 0, 0, -1), feasible=sat >= 0,
            analytic=analytic, rst=np.ones((2, 8)), sinr=z[..., 3])

    def step(self, request):
        obs = self.observe()
        sat = np.array([obs.sat_id[k, r] if r >= 0 else -1 for k, r in enumerate(request)])
        occupancy = np.bincount(sat[sat >= 0], minlength=self.cfg.environment['n_satellites'])
        served = sat >= 0
        rate = np.where(served, 80. + 10. * sat, 0.)
        c = np.where((self.prev >= 0) & served & (self.prev != sat), .3, 0.)
        cost = np.where(served, .5 * c + .5 * occupancy[np.maximum(sat, 0)] / 10 - rate / 240, 5.)
        attempt = (self.prev >= 0) & served & (self.prev != sat)
        self.prev = sat
        self.t += 1
        return SimpleNamespace(next_observation=None if self.t >= self.cfg.environment['horizon'] else self.observe(),
            executed_sat=sat, executed_beam=np.where(served, 0, -1), executed_subband=np.where(served, np.arange(2), -1),
            rate_mbps=rate, c_exec=c, cost=cost, reward=-cost, occupancy=occupancy,
            attempt=attempt, failure=np.zeros(2, bool), info=dict(check_sinr=np.ones(2)))


def cfg():
    return Config(environment=dict(n_satellites=20, n_users=2, horizon=4, planning_points=2),
        training=dict(rounds=2, episodes_per_round=1, updates_per_round=2,
            validation_episodes=1, sequence_length=4, burn_in=2, batch_sequences=1,
            selection_interval=2, warmup_updates=1))


def test_replay_masked_target_and_real_optimizer_update():
    torch.manual_seed(2)
    online, target = DQNNetwork(), DQNNetwork()
    target.load_state_dict(online.state_dict())
    replay = training.ReplayBuffer(8)
    masks = np.zeros((4, 9), bool)
    masks[:, 8] = True
    replay.add_batch(np.zeros((4, 73)), np.zeros(4, int), np.ones(4), np.zeros((4, 73)), masks, np.zeros(4, bool))
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    before = online.layers[-1].weight.detach().clone()
    loss = training.dqn_update(online, target, optimizer, replay, np.random.default_rng(1), 'cpu', batch_size=4)
    assert np.isfinite(loss)
    assert not torch.equal(before, online.layers[-1].weight)


def test_residual_train_select_load_and_episode_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(training, '_environment', InterfaceEnvironment)
    config = cfg()
    selected = training.train_residual(config, 'mlp213', 1, tmp_path)
    saved = training._load(selected, 'cpu')
    assert saved['update'] in (2, 4)
    assert bool(saved['model']['normalizer.frozen'])
    controller = training.load_controller(selected, config)
    result = training.evaluate_episode(controller, config, 20001, 0, record_diagnostics=True)
    assert result['association'].shape == (4, 2)
    assert result['diagnostic_check_sinr'].shape == (4, 2)
    assert result['mean_cost'] == pytest.approx(result['cost'].mean())
    assert np.array_equal(result['proposal'], result['requests'])
    assert len(list((tmp_path / 'rollouts').rglob('*.npz'))) == 2
    with pytest.raises(ValueError):
        training.load_controller(selected, Config(environment={'n_beams': 8}))


def test_residual_resume_reproduces_uninterrupted_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(training, '_environment', InterfaceEnvironment)
    config = cfg()
    reference = tmp_path / 'reference'
    training.train_residual(config, 'mlp213', 2, reference)
    original_save = training._save
    interrupted = tmp_path / 'interrupted'
    stopped = False
    def interrupt(path, payload):
        nonlocal stopped
        original_save(path, payload)
        if Path(path).name == 'latest.pt' and payload['update'] == 2 and not stopped:
            stopped = True
            raise RuntimeError('test interruption')
    monkeypatch.setattr(training, '_save', interrupt)
    with pytest.raises(RuntimeError, match='test interruption'):
        training.train_residual(config, 'mlp213', 2, interrupted)
    monkeypatch.setattr(training, '_save', original_save)
    training.train_residual(config, 'mlp213', 2, interrupted, resume=interrupted / 'latest.pt')
    a = training._load(reference / 'latest.pt', 'cpu')
    b = training._load(interrupted / 'latest.pt', 'cpu')
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['history'] == b['history']


def test_dqn_train_load_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(training, '_environment', InterfaceEnvironment)
    config = cfg()
    options = dict(dqn_epochs=8, dqn_anneal_epochs=4, dqn_replay_capacity=32,
                   dqn_replay_warmup=4, dqn_batch_size=2, dqn_target_interval=2)
    reference = tmp_path / 'reference'
    selected = training.train_dqn(config, 3, reference, overrides=options)
    saved = training._load(selected, 'cpu')
    assert saved['update'] > 0
    assert saved['update'] % 2 == 0
    ctrl = training.load_controller(selected, config)
    assert ctrl.method == 'madrl'
    assert np.isfinite(training.evaluate_episode(ctrl, config, 20001, 1)['mean_cost'])
    original_save = training._save
    interrupted = tmp_path / 'interrupted'
    stopped = False
    def interrupt(path, payload):
        nonlocal stopped
        original_save(path, payload)
        if Path(path).name == 'latest.pt' and not stopped:
            stopped = True
            raise RuntimeError('test interruption')
    monkeypatch.setattr(training, '_save', interrupt)
    with pytest.raises(RuntimeError, match='test interruption'):
        training.train_dqn(config, 3, interrupted, overrides=options)
    monkeypatch.setattr(training, '_save', original_save)
    training.train_dqn(config, 3, interrupted, overrides=options, resume=interrupted / 'latest.pt')
    a = training._load(reference / 'latest.pt', 'cpu')
    b = training._load(interrupted / 'latest.pt', 'cpu')
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['history'] == b['history']
    assert a['replay']['position'] == b['replay']['position']
