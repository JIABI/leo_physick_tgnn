"""All decisions use one frozen observation; execution belongs to Environment."""
from __future__ import annotations
import numpy as np
import torch
from .models import TGNModel, DQNNetwork, RESIDUAL_METHODS, canonical_method, dqn_features, action_mask

RULE_ALIASES = {"greedy": "greedy_analytic", "load_aware_greedy": "greedy_analytic",
                "max-sinr+ttt": "max_sinr_ttt", "max_sinr": "max_sinr_ttt",
                "max-rst": "max_rst", "maxrst": "max_rst", "maxsinr_ttt": "max_sinr_ttt", "cox-only": "cox_only"}


def current_mask(obs):
    return ((obs.sat_id >= 0) & (obs.sat_id == np.asarray(obs.prev_sat)[:, None])
            & (obs.beam_id == np.asarray(obs.prev_beam)[:, None]))


def rank_slots(obs, scores, hysteresis=None):
    """Largest score, exact ties: retain source, then satellite and beam ID."""
    scores = np.asarray(scores)
    feasible = np.asarray(obs.feasible, dtype=bool) & (obs.sat_id >= 0)
    if not np.isfinite(scores[feasible]).all():
        raise FloatingPointError("Nonfinite candidate score")
    current = current_mask(obs)
    requests = np.full(len(scores), -1, dtype=np.int64)
    for k in range(len(scores)):
        candidates = np.flatnonzero(feasible[k])
        if not len(candidates):
            continue
        best = min(candidates, key=lambda j: (-float(scores[k, j]), not bool(current[k, j]),
                                              int(obs.sat_id[k, j]), int(obs.beam_id[k, j])))
        source = np.flatnonzero(current[k] & feasible[k])
        if hysteresis is not None and len(source) and scores[k, best] - scores[k, source[0]] <= hysteresis:
            best = source[0]
        requests[k] = best
    return requests


class RuleController:
    def __init__(self, method, cfg):
        self.method = RULE_ALIASES.get(canonical_method(method), canonical_method(method))
        if self.method not in {"greedy_analytic", "cox_only", "max_sinr_ttt", "max_rst"}:
            raise ValueError(method)
        self.cfg = cfg
        self.last_scores = None
        self.reset()

    def reset(self):
        self.pending = None
        self.streak = None

    def select(self, obs):
        if self.method == "max_rst":
            self.last_scores = np.asarray(obs.rst)
            return rank_slots(obs, self.last_scores)
        if self.method in {"greedy_analytic", "cox_only"}:
            self.last_scores = np.asarray(obs.analytic).copy()
            if self.method == "cox_only":
                self.last_scores -= np.asarray(obs.survival)
            return rank_slots(obs, self.last_scores, self.cfg.environment["hysteresis"])
        self.last_scores = np.asarray(obs.sinr)
        best = rank_slots(obs, self.last_scores)
        feasible = np.asarray(obs.feasible, bool)
        source = current_mask(obs) & feasible
        if self.pending is None or len(self.pending) != len(best):
            self.pending = np.full((len(best), 2), -1, np.int64)
            self.streak = np.zeros(len(best), np.int64)
        chosen = best.copy()
        for k, target in enumerate(best):
            src = np.flatnonzero(source[k])
            if target < 0 or not len(src):
                self.pending[k] = -1
                self.streak[k] = 0
                continue
            s = int(src[0])
            if target == s or self.last_scores[k, target] < self.last_scores[k, s] * 10 ** (3 / 10):
                chosen[k] = s
                self.pending[k] = -1
                self.streak[k] = 0
                continue
            pair = np.array([obs.sat_id[k, target], obs.beam_id[k, target]])
            self.streak[k] = self.streak[k] + 1 if np.array_equal(self.pending[k], pair) else 1
            self.pending[k] = pair
            if self.streak[k] < 2:
                chosen[k] = s
            else:
                self.streak[k] = 0
                self.pending[k] = -1
        return chosen


class NeuralController:
    def __init__(self, model: TGNModel, cfg, device="cpu"):
        self.model = model.to(device)
        self.method = model.method
        self.cfg = cfg
        self.device = torch.device(device)
        self.last_scores = None
        self.reset()

    def reset(self):
        self.state = None

    @torch.no_grad()
    def select(self, obs):
        self.model.eval()
        if self.state is None:
            self.state = self.model.initial_state(len(obs.prev_sat), self.cfg.environment["n_satellites"])
        residual, _, self.state = self.model.forward_observation(obs, self.state)
        self.last_scores = np.asarray(obs.analytic) + self.cfg.model["residual_weight"] * residual.cpu().numpy()
        return rank_slots(obs, self.last_scores, self.cfg.environment["hysteresis"])


class DQNController:
    def __init__(self, model: DQNNetwork, cfg, device="cpu", epsilon=0., seed=0):
        self.model = model.to(device)
        self.method = "madrl"
        self.cfg = cfg
        self.device = torch.device(device)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.last_scores = None

    def reset(self):
        pass

    @torch.no_grad()
    def select(self, obs):
        self.model.eval()
        inputs = torch.as_tensor(dqn_features(obs), device=self.device)
        q = self.model(inputs).cpu().numpy()
        mask = action_mask(obs)
        masked = np.where(mask, q, -np.inf)
        self.last_scores = masked[:, :8]
        requests = rank_slots(obs, q[:, :8])
        for k in range(len(q)):
            if self.rng.random() < self.epsilon:
                action = int(self.rng.choice(np.flatnonzero(mask[k])))
                requests[k] = action if action < 8 else -1
        return requests


def make_controller(method, cfg, model=None, device="cpu"):
    method = canonical_method(method)
    if method in RESIDUAL_METHODS:
        if model is None:
            raise ValueError("A trained model is required; use TGNModel explicitly for an initialization diagnostic")
        return NeuralController(model, cfg, device)
    if method == "madrl":
        if model is None:
            raise ValueError("A trained DQN model is required")
        return DQNController(model, cfg, device)
    return RuleController(method, cfg)


def explore_requests(obs, requests, probability, rng):
    """Independent feasible-candidate exploration, after all scores are frozen."""
    out = np.asarray(requests).copy()
    for k in range(len(out)):
        feasible = np.flatnonzero(np.asarray(obs.feasible[k]) & (obs.sat_id[k] >= 0))
        if len(feasible) and rng.random() < probability:
            out[k] = rng.choice(feasible)
    return out
