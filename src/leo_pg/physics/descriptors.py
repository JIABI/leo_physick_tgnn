from __future__ import annotations
from typing import Dict, Any
import torch

from .cox_intensity import lambda_feas
from .residual_time import expected_residual_time
from .risk_probability import p_risk_ho
from .handover_cost import normalize_cost
from .normalize import normalize_descriptor


def compute_edge_descriptors(
    edge_ctx: Dict[str, Any],
    rate: torch.Tensor,             # [E]
    ho_cost: torch.Tensor,          # [E]
    sat_load: torch.Tensor,         # [S]
    cox_cfg: Dict[str, Any],
    device: torch.device,
    visibility_radius: float = 0.7,
) -> torch.Tensor:
    """Compute physics-guided descriptor z_ij(t) for each user->sat edge.

    Output order (edge_in_dim=6):
      [P_risk, ET0, Lambda_feas, rate, H_norm, load]

    Notes:
      - P_risk^{HO} uses paper Eq.(29); its computation mode can be selected by
        cox_cfg["risk_mode"] in {"eq30","moment","mc"}.
      - All returned values are normalized to approx [-1,1] for KAN friendliness.
    """
    E = int(edge_ctx["dist"].numel())
    if E == 0:
        return torch.zeros((0, 6), device=device)

    # Residual time expectation E[T0] (currently a proxy; can be upgraded to paper Eq.(20-21))
    ET0 = expected_residual_time(edge_ctx, visibility_radius=visibility_radius).to(device)

    # Feasibility-thinned intensity Lambda^{feas} (paper Eq.(26); current implementation may be simplified)
    lam = lambda_feas(edge_ctx, cox_cfg=cox_cfg, device=device)                  # [E]

    # Paper-aligned risk probability (Eq.(29)), with Eq.(30) option
    risk_mode = str(cox_cfg.get("risk_mode", "eq30")).lower()
    p_risk = p_risk_ho(lam, ET0, mode=risk_mode)                                 # [E]

    # load for each edge's destination satellite
    dst_s = edge_ctx["dst_s"].to(device)  # local sat idx [E]
    load = sat_load[dst_s]                # [E]

    Hn = normalize_cost(ho_cost.to(device))

    z = torch.stack([p_risk, ET0, lam, rate.to(device), Hn, load], dim=-1)       # [E,6]
    return normalize_descriptor(z)
