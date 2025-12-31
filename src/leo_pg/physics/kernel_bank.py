from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_atanh(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # atanh is unstable near +/-1
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _inv_tanh_squash(z: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    # z = tanh(x/scale) -> x = scale * atanh(z)
    return scale * _safe_atanh(z)


def _inv_log1p_tanh(z: torch.Tensor) -> torch.Tensor:
    # z = tanh(log1p(x)) -> log1p(x) = atanh(z) -> x = exp(atanh(z)) - 1
    return torch.expm1(_safe_atanh(z))


def _inv_logit_tanh(z: torch.Tensor) -> torch.Tensor:
    # z = tanh(logit(p)) -> logit(p) = atanh(z) -> p = sigmoid(atanh(z))
    return torch.sigmoid(_safe_atanh(z))


class KernelBank(nn.Module):
    r"""Paper-shaped kernel primitives κ_m(z) for PhysiCK.

    Input z is the *normalized* physics-guided descriptor in approx [-1,1] with ordering:
        z = [P_risk, tanh(log1p(ET0)), tanh(log1p(Lambda)), tanh(rate/5), tanh(H/2), tanh(load/2)]

    This bank explicitly encodes the closed-form structure used in the Cox-process paper:

    - Eq.(29) (moment / Laplace form):      κ_exp = exp(-Λ * T0)
    - Eq.(30) (uniform-window closed form): κ_uni = (1 - exp(-Λ D)) / (Λ D), with D ≈ 2 T0
    - Eq.(37) (service capability proxy):   C ≈ rate * E[T0]
    - Eq.(39)/(40)/(41) (penalties/utility): cost/load terms entering a utility-like proxy

    The goal is to *not* force the network to relearn these nonlinearities; instead we provide
    these as basis functions and only learn the mixture weights α(z, mem).
    """

    def __init__(self, edge_dim: int = 6, num_kernels: int = 12):
        super().__init__()
        if edge_dim < 6:
            raise ValueError(f"KernelBank expects edge_dim>=6, got {edge_dim}")
        self.edge_dim = int(edge_dim)
        self.num_kernels = int(num_kernels)

        # Learnable but lightweight scales to keep magnitudes stable without destroying interpretability.
        self.service_scale = nn.Parameter(torch.tensor(10.0))  # tanh(C / scale)
        self.utility_scale = nn.Parameter(torch.tensor(10.0))  # tanh(U / scale)

        # Utility proxy weights (Eq.41-like); initialized conservative.
        self.w_cost = nn.Parameter(torch.tensor(1.0))
        self.w_load = nn.Parameter(torch.tensor(1.0))

        # Optional temperature on exp(-ΛT) to compensate approximation/units mismatch.
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [E, edge_dim] normalized descriptors (see class docstring)

        Returns:
            K: [E, num_kernels] kernel features
        """
        if z.dim() != 2:
            raise ValueError(f"KernelBank expects z as [E,D], got shape {tuple(z.shape)}")

        # Decode approximate raw physical quantities (monotonic inverses of normalize.py transforms)
        p_risk = _inv_logit_tanh(z[:, 0])               # [0,1]
        T0 = _inv_log1p_tanh(z[:, 1])                   # >=0
        lam = _inv_log1p_tanh(z[:, 2])                  # >=0
        rate = _inv_tanh_squash(z[:, 3], scale=5.0)     # >=0-ish
        H = _inv_tanh_squash(z[:, 4], scale=2.0)        # >=0-ish
        load = _inv_tanh_squash(z[:, 5], scale=2.0)     # >=0-ish

        # Safety clamps (prevents NaNs when z saturates)
        eps = 1e-8
        T0 = torch.clamp(T0, 0.0, 1e6)
        lam = torch.clamp(lam, 0.0, 1e6)
        rate = torch.clamp(rate, 0.0, 1e6)
        H = torch.clamp(H, 0.0, 1e6)
        load = torch.clamp(load, 0.0, 1e6)

        lamT = lam * T0
        D = 2.0 * T0
        lamD = lam * D

        # Eq.(29)-shaped kernel (Laplace / moment approximation)
        k_exp = torch.exp(-F.softplus(self.beta) * lamT)

        # Eq.(30)-shaped kernel (uniform-window closed form); safe for small lamD
        k_uni = (1.0 - torch.exp(-lamD)) / (lamD + eps)

        # Eq.(37) service capability proxy
        C = rate * T0

        # Eq.(41)-like utility proxy (very lightweight)
        U = C - F.softplus(self.w_cost) * H - F.softplus(self.w_load) * load

        # Build deterministic kernel list (M=12 default)
        k_list = [
            k_exp,                                 # κ0  exp(-Λ T0)           (Eq.29-like)
            k_uni,                                 # κ1  (1-exp(-ΛD))/(ΛD)    (Eq.30-like)
            p_risk,                                # κ2  P_risk               (Eq.32 / risk)
            1.0 - p_risk,                          # κ3  reliability
            torch.tanh(C / (F.softplus(self.service_scale) + eps)),  # κ4  normalized capability
            torch.tanh(rate / 10.0),               # κ5  instantaneous rate (squashed)
            torch.tanh(T0 / 10.0),                 # κ6  remaining time (squashed)
            torch.tanh(lam / 10.0),                # κ7  feasible intensity (squashed)
            -torch.tanh(H / 10.0),                 # κ8  negative HO cost penalty
            -torch.tanh(load / 10.0),              # κ9  negative load penalty
            torch.tanh(U / (F.softplus(self.utility_scale) + eps)),  # κ10 utility proxy
            torch.tanh(lamT / 10.0),               # κ11 interaction scale ΛT0
        ]

        K = torch.stack(k_list, dim=-1)  # [E, 12]

        # Pad/trim to requested num_kernels
        if K.size(-1) > self.num_kernels:
            K = K[:, : self.num_kernels]
        elif K.size(-1) < self.num_kernels:
            pad = torch.zeros((K.size(0), self.num_kernels - K.size(-1)), device=K.device, dtype=K.dtype)
            K = torch.cat([K, pad], dim=-1)
        return K
