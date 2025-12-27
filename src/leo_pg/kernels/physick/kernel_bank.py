from __future__ import annotations

import torch
import torch.nn as nn


def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # numerically stable atanh for |x|<1
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class KernelBank(nn.Module):
    """
    Paper-aligned kernel bank for Cox-driven handover management.

    This module is intentionally *formula-shaped*: each κ_m corresponds to a closed-form term
    appearing in the Cox-process derivation / utility design:

      - Feasibility-thinned arrival intensity Λ^{feas} (Eq. 26)
      - Handover success probability P^{HO} = 1 - exp(-Λ^{feas} T0) (Eq. 28)
      - Handover risk probability P^{HO}_{risk} = E[exp(-Λ^{feas} T0)] (Eq. 29)
      - Conditional expectation closed form (uniform residual window):
            E[exp(-Λ T0) | s, φ] = (1 - exp(-Λ D_s(φ))) / (Λ D_s(φ))   (Eq. 30),
        with E[T0 | s,φ] = D_s(φ)/2  (Eq. 21)  ->  D ≈ 2 E[T0]
      - Remaining service capability C ≈ \bar R · E[T] (Eq. 37)
      - Signaling overhead H (Eq. 39)
      - Load term ℓ (Eq. 40)
      - Risk-adaptive trigger Γ(P_risk) (Eq. 34) as a soft gate

    Important: In this codebase, z is *normalized* by leo_pg.physics.normalize.normalize_descriptor()
    and has default ordering:
        z = [P_risk, ET0, Lambda_feas, rate, H_norm, load]  (all ~ in [-1,1])

    We invert those normalizations (approximately exactly) to recover raw positive quantities
    before applying the analytical shapes, so the kernels remain faithful to the paper.

    Output:
        κ(z) ∈ R^{E×M} with M=num_kernels (trim/pad deterministically).
    """

    def __init__(
        self,
        edge_dim: int,
        num_kernels: int = 12,
        # risk-adaptive trigger parameters (Eq. 34):
        rho1: float = 0.3,
        rho2: float = 0.7,
        gamma_L: float = 2.0,
        gamma_M: float = 5.0,
        gamma_H: float = 10.0,
        # numeric constants:
        eps: float = 1e-8,
    ):
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.num_kernels = int(num_kernels)
        self.eps = float(eps)

        # Risk breakpoints (ρ1 < ρ2). Keep as parameters so you can tune/calibrate.
        # They are not required to be trained; but leaving them as Parameters lets you learn them if desired.
        self.rho1 = nn.Parameter(torch.tensor(float(rho1)))
        self.rho2 = nn.Parameter(torch.tensor(float(rho2)))

        # Γ(P_risk) levels (Γ_L < Γ_M < Γ_H). Keep learnable but initialized sensibly.
        self.gamma_L = nn.Parameter(torch.tensor(float(gamma_L)))
        self.gamma_M = nn.Parameter(torch.tensor(float(gamma_M)))
        self.gamma_H = nn.Parameter(torch.tensor(float(gamma_H)))

        # Optional scales for soft gates; small and safe.
        self.gate_scale = nn.Parameter(torch.tensor(10.0))

    # --------- Inverse transforms for normalized descriptors ---------
    def _inv_prisk(self, z_pr: torch.Tensor) -> torch.Tensor:
        """
        normalize_descriptor does: z_pr = tanh(logit(p))
        Invert: p = sigmoid(atanh(z_pr))
        """
        return torch.sigmoid(_atanh(z_pr))

    def _inv_log1p_tanh(self, z_x: torch.Tensor) -> torch.Tensor:
        """
        normalize_descriptor does: z_x = tanh(log1p(x)), for x>=0
        Invert: x = expm1(atanh(z_x))
        """
        return torch.expm1(_atanh(z_x)).clamp_min(0.0)

    def _inv_rate(self, z_rate: torch.Tensor) -> torch.Tensor:
        """
        normalize_descriptor does: z_rate = tanh(rate/5)
        Invert: rate = 5 * atanh(z_rate)
        """
        return (5.0 * _atanh(z_rate)).clamp_min(0.0)

    def _inv_H(self, z_H: torch.Tensor) -> torch.Tensor:
        """
        normalize_descriptor does: z_H = tanh(H/2) where H is already normalized cost surrogate
        Invert: H_raw = 2 * atanh(z_H)
        """
        return 2.0 * _atanh(z_H)

    def _inv_load(self, z_load: torch.Tensor) -> torch.Tensor:
        """
        normalize_descriptor does: z_load = tanh(load/2), load usually in [0,1] (or small positive)
        Invert: load_raw = 2 * atanh(z_load)
        """
        return 2.0 * _atanh(z_load)

    # --------- Paper-shaped primitives ---------
    def _soft_piecewise_gamma(self, p_risk: torch.Tensor) -> torch.Tensor:
        """
        Smooth version of Eq. (34):
            Γ(p) = Γ_L if p<=ρ1; Γ_M if ρ1<p<=ρ2; Γ_H if p>ρ2.
        We implement with sigmoids for differentiability:
            w1 = σ(s*(p-ρ1)), w2 = σ(s*(p-ρ2))
            Γ = Γ_L*(1-w1) + Γ_M*(w1-w2) + Γ_H*w2
        """
        s = torch.nn.functional.softplus(self.gate_scale) + 1.0  # positive
        rho1 = torch.clamp(self.rho1, 1e-3, 1.0 - 1e-3)
        rho2 = torch.clamp(self.rho2, rho1 + 1e-3, 1.0 - 1e-3)

        w1 = torch.sigmoid(s * (p_risk - rho1))
        w2 = torch.sigmoid(s * (p_risk - rho2))

        gL = torch.nn.functional.softplus(self.gamma_L)
        gM = torch.nn.functional.softplus(self.gamma_M)
        gH = torch.nn.functional.softplus(self.gamma_H)

        return gL * (1.0 - w1) + gM * (w1 - w2) + gH * w2

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # unpack
        pr_n = z[..., 0] if z.size(-1) > 0 else torch.zeros_like(z[..., 0])
        et0_n = z[..., 1] if z.size(-1) > 1 else torch.zeros_like(pr_n)
        lam_n = z[..., 2] if z.size(-1) > 2 else torch.zeros_like(pr_n)
        rate_n = z[..., 3] if z.size(-1) > 3 else torch.zeros_like(pr_n)
        H_n = z[..., 4] if z.size(-1) > 4 else torch.zeros_like(pr_n)
        load_n = z[..., 5] if z.size(-1) > 5 else torch.zeros_like(pr_n)

        # invert normalization to recover paper-space quantities
        P_risk = self._inv_prisk(pr_n).clamp(self.eps, 1.0)         # (0,1]
        ET0 = self._inv_log1p_tanh(et0_n)                            # >=0 (seconds)
        Lam = self._inv_log1p_tanh(lam_n)                            # >=0 (1/s)
        Rbar = self._inv_rate(rate_n)                                # >=0 (normalized rate)
        H = self._inv_H(H_n)                                         # can be negative if H_n<0; that's okay
        load = self._inv_load(load_n)                                # load proxy (typically >=0)

        # Derived terms from paper
        # Eq. (28): P_HO = 1 - exp(-Lam*T0).  In moment matching, T0~ET0.
        Nbar = Lam * ET0                                              # expected feasible arrivals during ET0
        P_HO = 1.0 - torch.exp(-Nbar.clamp_min(0.0))
        # Eq. (29): risk is E[exp(-Lam*T0)] ~ exp(-Lam*ET0) (moment matching)
        exp_term = torch.exp(-Nbar.clamp_min(0.0))

        # Eq. (30) using uniform residual window: D ≈ 2*ET0
        D = (2.0 * ET0).clamp_min(self.eps)
        LamD = (Lam * D).clamp_min(self.eps)
        eq30 = (1.0 - torch.exp(-LamD)) / LamD

        # Eq. (37): remaining service capability C ≈ Rbar * E[T]
        C = Rbar * ET0

        # Eq. (34): risk-adaptive trigger level Γ(P_risk) (soft version)
        Gamma = self._soft_piecewise_gamma(P_risk)

        # Utility components (paper's beam selection uses a utility over {C, H, load}; we keep them separable)
        # Provide monotone transforms to help learning without destroying interpretability.
        load_gate = torch.sigmoid(-load)          # high load -> small
        cost_gate = torch.sigmoid(-H)             # high cost -> small
        cap_sat = C / (1.0 + C)                   # in (0,1)

        # --- Kernel list (each κ is a paper-aligned primitive) ---
        k = []
        k.append(torch.ones_like(P_risk))         # κ1: bias / constant

        # Risk / success probabilities
        k.append(P_risk)                          # κ2: P_risk^{HO} (Eq. 29)
        k.append(P_HO)                            # κ3: P^{HO}=1-exp(-Lam ET0) (Eq. 28)
        k.append(exp_term)                        # κ4: exp(-Lam ET0) (moment matching core)
        k.append(eq30)                            # κ5: Eq. (30) conditional closed form with D≈2ET0

        # Arrival / intensity terms
        k.append(Lam)                             # κ6: Λ^{feas} (Eq. 26)
        k.append(Nbar)                            # κ7: expected arrivals \bar N = Λ^{feas} E[T0] (Eq. 33)

        # Service capability
        k.append(ET0)                             # κ8: E[T0] (residual time proxy)
        k.append(Rbar)                            # κ9: \bar R (rate proxy)
        k.append(C)                               # κ10: C = \bar R * E[T] (Eq. 37)
        k.append(cap_sat)                         # κ11: bounded capability

        # Decision shaping: trigger & penalties
        k.append(Gamma)                           # κ12: Γ(P_risk) (Eq. 34) soft gate

        # If you requested more kernels, keep adding paper-consistent penalties/couplings.
        # These are still interpretable and correspond to the utility terms (Eq. 39-40).
        if self.num_kernels > 12:
            k.append(H)                           # κ13: signaling overhead (Eq. 39)
        if self.num_kernels > 13:
            k.append(cost_gate)                   # κ14: exp-like/soft gate on cost
        if self.num_kernels > 14:
            k.append(load)                        # κ15: load proxy (Eq. 40)
        if self.num_kernels > 15:
            k.append(load_gate)                   # κ16: soft gate on load
        if self.num_kernels > 16:
            k.append(C * cost_gate)               # κ17: capability penalized by cost
        if self.num_kernels > 17:
            k.append(C * load_gate)               # κ18: capability penalized by load
        if self.num_kernels > 18:
            k.append(P_HO * cost_gate)            # κ19: success penalized by cost
        if self.num_kernels > 19:
            k.append(P_HO * load_gate)            # κ20: success penalized by load

        K = torch.stack(k, dim=-1)  # [E, >=12]

        # Deterministic trim/pad to num_kernels
        if K.size(-1) > self.num_kernels:
            K = K[..., : self.num_kernels]
        elif K.size(-1) < self.num_kernels:
            pad = torch.zeros(
                K.shape[:-1] + (self.num_kernels - K.size(-1),),
                device=K.device,
                dtype=K.dtype,
            )
            K = torch.cat([K, pad], dim=-1)

        # Final safety: replace NaN/Inf (can happen if z contains NaNs)
        K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        return K
