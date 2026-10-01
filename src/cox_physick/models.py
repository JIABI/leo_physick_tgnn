"""TCOM message operators, recurrent graph memories, and DQN networks.

Physical coordinates remain in the units of (34)--(38); only learned-network
inputs are standardized. There are no fixed-size user IDs in learned inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import itertools
import math
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

METHOD_ALIASES = {
    "cox_physick": "full", "cox-physick": "full", "physick": "full",
    "tgn-mlp": "tgn_mlp", "mlp213": "tgn_mlp", "kan_generic": "tgn_kan", "eph_physick": "ephemeris",
    "no_triplet": "no_arrival_rst", "no_cox_rst": "rst_retained", "no_kernel": "tgn_mlp", "no_kernel_bank": "tgn_mlp",
    "no-kernel": "tgn_mlp", "tgn-kan": "tgn_kan",
    "no_arrival_rst": "no_arrival_rst", "no_cox": "no_arrival_rst",
    "no_cox_rst_retained": "rst_retained", "no_cox_keep_rst": "rst_retained",
    "ephemeris_physick": "ephemeris", "ephemeris-physick": "ephemeris",
    "mlp_coefficient": "mlp_coeff", "mlp_coefficient_head": "mlp_coeff",
    "leo-madrl": "madrl", "leo_madrl": "madrl", "dqn": "madrl",
}
RESIDUAL_METHODS = {"full", "tgn_mlp", "tgn_kan", "no_arrival_rst", "rst_retained", "ephemeris", "mlp_coeff"}


def canonical_method(method: str) -> str:
    key = method.strip().lower().replace(" ", "_")
    return METHOD_ALIASES.get(key, key)


class FeatureNormalizer(nn.Module):
    """Population training statistics accumulated in FP64, frozen at inference."""
    def __init__(self, n_features: int = 6):
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(n_features, dtype=torch.float64))
        self.register_buffer("m2", torch.zeros(n_features, dtype=torch.float64))
        self.register_buffer("frozen", torch.tensor(False))

    @torch.no_grad()
    def update(self, values: Any, valid: Any = None) -> None:
        if bool(self.frozen):
            raise RuntimeError("Training normalizer is frozen")
        x = torch.as_tensor(values, dtype=torch.float64, device=self.mean.device)
        if valid is not None:
            x = x[torch.as_tensor(valid, dtype=torch.bool, device=x.device)]
        x = x.reshape(-1, self.mean.numel())
        x = x[torch.isfinite(x).all(-1)]
        if not len(x):
            return
        n = float(len(x))
        mu = x.mean(0)
        delta = mu - self.mean
        total = self.count + n
        self.m2.add_(((x - mu) ** 2).sum(0) + delta.square() * self.count * n / total)
        self.mean.add_(delta * n / total)
        self.count.copy_(total)

    @torch.no_grad()
    def freeze(self) -> None:
        if self.count.item() == 0:
            raise ValueError("Cannot freeze normalization without training observations")
        self.frozen.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        if self.count.item() == 0:
            return x
        scale = (self.m2 / self.count.clamp_min(1)).sqrt()
        scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
        return (x - self.mean.to(x)) / scale.to(x)


class KANLinear(nn.Module):
    """Five fixed grid intervals, degree-three open-clamped B-splines.

    Eight spline coefficients and one SiLU coefficient per scalar connection,
    plus one bias per output. Values outside [-1,1] saturate the spline branch.
    The SiLU branch uses the unsaturated learned input.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, 8))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("knots", torch.tensor([-1.] * 4 + [-.6, -.2, .2, .6] + [1.] * 4))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.02 / math.sqrt(in_features))

    def basis(self, x: Tensor) -> Tensor:
        y = x.clamp(-1., 1.)
        knots = self.knots.to(y)
        # Evaluate the right endpoint by its left limit before overriding it.
        eval_y = y.clamp(max=1. - torch.finfo(y.dtype).eps).unsqueeze(-1)
        basis = ((eval_y >= knots[:-1]) & (eval_y < knots[1:])).to(x.dtype)
        for degree in range(1, 4):
            count = len(knots) - degree - 1
            dl = knots[degree:degree + count] - knots[:count]
            dr = knots[degree + 1:degree + 1 + count] - knots[1:1 + count]
            left = torch.where(dl > 0, (eval_y - knots[:count]) / dl.clamp_min(1e-12), 0.)
            right = torch.where(dr > 0, (knots[degree + 1:degree + 1 + count] - eval_y) / dr.clamp_min(1e-12), 0.)
            basis = left * basis[..., :count] + right * basis[..., 1:count + 1]
        endpoint = torch.zeros_like(basis)
        endpoint[..., -1] = 1.
        return torch.where((y >= 1.).unsqueeze(-1), endpoint, basis)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.silu(x), self.base_weight, self.bias) + F.linear(
            self.basis(x).flatten(-2), self.spline_weight.flatten(1))


class KAN(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(KANLinear(input_dim, hidden_dim), KANLinear(hidden_dim, output_dim))


def physical_kernels(z: Tensor, survival: Tensor) -> Tensor:
    """Eight trilinear products, candidate-window Cox void, and broadcast load."""
    scale = z.new_tensor([1., math.log(11.), math.log(21.)])
    u = (z[..., :3] / scale).clamp(0., 1.)
    factors = torch.stack((1. - u, u), -1)
    products = [factors[..., 0, r] * factors[..., 1, q] * factors[..., 2, v]
                for r, q, v in itertools.product(range(2), repeat=3)]
    return torch.stack(products + [survival, z[..., 4]], -1)


class PhysiCKMessage(nn.Module):
    def __init__(self, coefficient: str = "kan"):
        super().__init__()
        if coefficient == "kan":
            self.coefficient = KAN(262, 32, 10)
        elif coefficient == "mlp":
            self.coefficient = nn.Sequential(nn.Linear(262, 287), nn.SiLU(), nn.Linear(287, 10))
        else:
            raise ValueError(coefficient)
        self.lifts = nn.Parameter(torch.empty(10, 128, 4))
        nn.init.xavier_uniform_(self.lifts)

    def forward(self, learned_input: Tensor, z: Tensor, survival: Tensor) -> Tensor:
        alpha = self.coefficient(learned_input).softmax(-1)
        kernels = physical_kernels(z, survival)
        inputs = torch.stack((torch.ones_like(z[..., 0]), z[..., 3], z[..., 4], z[..., 5]), -1)
        bases = torch.einsum("...f,mhf->...mh", inputs, self.lifts)
        return (bases * (alpha * kernels).unsqueeze(-1)).sum(-2)


@dataclass
class GraphMemory:
    user: Tensor
    beam: Tensor

    def detach(self) -> "GraphMemory":
        return GraphMemory(self.user.detach(), self.beam.detach())


def transformed_descriptors(obs: Any, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Explicit interventions, applied to both bases and learned inputs."""
    method = canonical_method(method)
    if method == "ephemeris":
        if getattr(obs, "eph_z", None) is None or getattr(obs, "eph_survival", None) is None:
            raise ValueError("Ephemeris variant requires geometric entry-count descriptors from the environment")
        z = np.asarray(obs.eph_z).copy()
        survival = np.asarray(obs.eph_survival).copy()
    else:
        z = np.asarray(obs.z).copy()
        survival = np.asarray(obs.survival).copy()
    if method in {"no_arrival_rst", "rst_retained"}:
        z[..., :3 if method == "no_arrival_rst" else 2] = 0.
        survival[...] = 1.
    return z, survival


class TGNModel(nn.Module):
    """Synchronous bipartite TGN with a common 128-dimensional memory scaffold."""
    def __init__(self, method: str = "full", n_beams: int = 7):
        super().__init__()
        self.method = canonical_method(method)
        if self.method not in RESIDUAL_METHODS:
            raise ValueError(f"Unknown residual model {method}")
        self.n_beams = n_beams
        self.normalizer = FeatureNormalizer(6)
        if self.method == "tgn_mlp":
            self.message = nn.Sequential(nn.Linear(262, 213), nn.SiLU(), nn.Linear(213, 128))
        elif self.method == "tgn_kan":
            self.message = KAN(262, 32, 128)
        else:
            self.message = PhysiCKMessage("mlp" if self.method == "mlp_coeff" else "kan")
        self.user_gru = nn.GRUCell(128, 128)
        self.beam_gru = nn.GRUCell(128, 128)
        self.readout = nn.Sequential(nn.Linear(262, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.load_head = nn.Sequential(nn.Linear(262, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def initial_state(self, n_users: int, n_satellites: int) -> GraphMemory:
        prototype = next(self.parameters())
        return GraphMemory(prototype.new_zeros(n_users, 128), prototype.new_zeros(n_satellites * self.n_beams, 128))

    def forward(self, z: Tensor, survival: Tensor, sat_id: Tensor, beam_id: Tensor,
                state: GraphMemory) -> tuple[Tensor, Tensor, GraphMemory]:
        valid = (sat_id >= 0) & (beam_id >= 0)
        users, slots = valid.nonzero(as_tuple=True)
        residual = z.new_zeros(valid.shape)
        load = z.new_zeros(valid.shape)
        if not len(users):
            return residual, load, state
        beams = sat_id[users, slots].long() * self.n_beams + beam_id[users, slots].long()
        if int(beams.max()) >= len(state.beam):
            raise ValueError("Beam IDs exceed the model's declared constellation state")
        physical = z[users, slots]
        normalized = self.normalizer(physical)
        inp = torch.cat((state.user[users], state.beam[beams], normalized), -1)
        messages = (self.message(inp, physical, survival[users, slots])
                    if isinstance(self.message, PhysiCKMessage) else self.message(inp))
        user_sum = torch.zeros_like(state.user).index_add(0, users, messages)
        beam_sum = torch.zeros_like(state.beam).index_add(0, beams, messages)
        uc = torch.bincount(users, minlength=len(state.user)).to(z)
        bc = torch.bincount(beams, minlength=len(state.beam)).to(z)
        active_users, active_beams = (uc > 0).nonzero().flatten(), (bc > 0).nonzero().flatten()
        updated_u = self.user_gru(user_sum[active_users] / uc[active_users, None], state.user[active_users])
        updated_b = self.beam_gru(beam_sum[active_beams] / bc[active_beams, None], state.beam[active_beams])
        next_u = state.user.index_copy(0, active_users, updated_u)
        next_b = state.beam.index_copy(0, active_beams, updated_b)
        read = torch.cat((next_u[users], next_b[beams], normalized), -1)
        residual = residual.index_put((users, slots), self.readout(read).squeeze(-1))
        load = load.index_put((users, slots), self.load_head(read).squeeze(-1))
        return residual, load, GraphMemory(next_u, next_b)

    def forward_observation(self, obs: Any, state: GraphMemory) -> tuple[Tensor, Tensor, GraphMemory]:
        device = next(self.parameters()).device
        z, survival = transformed_descriptors(obs, self.method)
        return self(torch.as_tensor(z, dtype=torch.float32, device=device),
                    torch.as_tensor(survival, dtype=torch.float32, device=device),
                    torch.as_tensor(obs.sat_id, dtype=torch.long, device=device),
                    torch.as_tensor(obs.beam_id, dtype=torch.long, device=device), state)


def dqn_features(obs: Any) -> np.ndarray:
    z = np.asarray(obs.z, dtype=np.float32)
    if z.shape[1] != 8:
        raise ValueError("The paper DQN requires exactly eight candidate slots")
    exists = np.asarray(obs.sat_id) >= 0
    current = exists & (obs.sat_id == np.asarray(obs.prev_sat)[:, None]) & (obs.beam_id == np.asarray(obs.prev_beam)[:, None])
    features = np.concatenate((z, current[..., None], exists[..., None], np.asarray(obs.feasible)[..., None]), -1).astype(np.float32)
    features[~exists] = 0.
    return np.concatenate((features.reshape(len(z), 72), (np.asarray(obs.prev_sat) < 0)[:, None]), -1).astype(np.float32)


def action_mask(obs: Any) -> np.ndarray:
    feasible = np.asarray(obs.feasible, dtype=bool) & (np.asarray(obs.sat_id) >= 0)
    return np.concatenate((feasible, ~feasible.any(-1, keepdims=True)), -1)


class DQNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = FeatureNormalizer(6)
        self.layers = nn.Sequential(nn.Linear(73, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 9))

    def forward(self, x: Tensor) -> Tensor:
        slots = x[..., :72].reshape(*x.shape[:-1], 8, 9)
        continuous = self.normalizer(slots[..., :6])
        # Missing slots stay zero after normalization; 0/1 flags are never standardized.
        continuous = continuous * slots[..., 7:8]
        normalized = torch.cat((continuous, slots[..., 6:]), -1).flatten(-2)
        return self.layers(torch.cat((normalized, x[..., 72:]), -1))


def hysteretic_huber(prediction: Tensor, target: Tensor) -> Tensor:
    delta = target - prediction
    weights = torch.where(delta.detach() >= 0, 1., .2)
    return (weights * F.smooth_l1_loss(prediction, target, reduction="none", beta=1.)).mean()


def parameter_counts() -> dict[str, int]:
    modules = {"tgn_mlp": TGNModel("tgn_mlp").message, "tgn_kan": KAN(262, 32, 128),
               "kan_coefficient": KAN(262, 32, 10), "physick_full": PhysiCKMessage(),
               "mlp_coefficient": PhysiCKMessage("mlp").coefficient}
    return {name: sum(p.numel() for p in module.parameters()) for name, module in modules.items()}
