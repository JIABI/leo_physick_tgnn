from __future__ import annotations
import torch


def p_risk_ho(lam_feas: torch.Tensor, ET0: torch.Tensor, mode: str = "moment", eps: float = 1e-8) -> torch.Tensor:
    """Handover risk probability (paper-aligned).

    Paper definition (Eq. 29):
        P_risk^{HO}(t) = E[ exp(-Lambda^{feas}(t) * T0) ].

    Implemented modes:
      - "moment": moment-matching approximation:
            P_risk ≈ exp(-Lambda^{feas} * E[T0])
      - "eq30": uniform residual window closed form (Eq. 30) with D ≈ 2*E[T0]:
            E[exp(-Lambda*T0) | s,phi] = (1 - exp(-Lambda * D)) / (Lambda * D)
        We use D = 2*ET0 as a practical approximation in this codebase.
      - "mc": small Monte Carlo perturbation around ET0 (debug/sanity-check).

    Args:
      lam_feas: [E] >=0 feasibility-thinned intensity (1/s)
      ET0:      [E] >=0 residual service time expectation (s)
    Returns:
      [E] tensor in (0,1].
    """
    lam = torch.clamp(lam_feas, min=0.0)
    t0 = torch.clamp(ET0, min=0.0)

    if mode == "moment":
        return torch.exp(-lam * t0).clamp_min(eps)

    if mode == "eq30":
        D = torch.clamp(2.0 * t0, min=eps)
        lamD = torch.clamp(lam * D, min=eps)
        out = (1.0 - torch.exp(-lamD)) / lamD
        return out.clamp_min(eps)

    if mode == "mc":
        B = 32
        noise = torch.rand((B,) + t0.shape, device=t0.device, dtype=t0.dtype)
        T = torch.clamp(t0 * (0.5 + noise), min=0.0)
        return torch.exp(-lam * T).mean(dim=0).clamp_min(eps)

    raise ValueError(f"Unknown mode={mode}")
