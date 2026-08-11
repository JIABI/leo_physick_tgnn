from __future__ import annotations

from typing import Any, Dict

import torch

from .cox_intensity import lambda_feas
from .handover_cost import normalize_cost
from .normalize import normalize_descriptor
from .residual_time import expected_residual_time
from .risk_probability import p_risk_ho


def _infer_dst_s(edge_ctx: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Infer local satellite indices (0..S-1) for each edge.

    Expected inputs (one of):
      - edge_ctx['dst_s'] : already local indices
      - edge_ctx['dst'] and edge_ctx['K_users'] : global node indices (user+sat)

    Returns:
      dst_s: LongTensor [E]
    """
    if "dst_s" in edge_ctx:
        return edge_ctx["dst_s"].to(device=device, dtype=torch.long)

    if "dst" in edge_ctx and "K_users" in edge_ctx:
        K = int(edge_ctx["K_users"])
        return (edge_ctx["dst"].to(device=device, dtype=torch.long) - K)

    raise KeyError(
        "compute_edge_descriptors requires edge_ctx['dst_s'] (local sat idx) "
        "or edge_ctx['dst'] plus edge_ctx['K_users'] to infer it."
    )


def compute_edge_descriptors(
    edge_ctx: Dict[str, Any],
    rate: torch.Tensor,  # [E]
    ho_cost: torch.Tensor,  # [E]
    sat_load: torch.Tensor,  # [S]
    cox_cfg: Dict[str, Any],
    device: torch.device,
    visibility_radius: float = 0.7,
) -> torch.Tensor:
    """Compute physics-guided descriptor z_ij(t) for each user->sat edge.

    Output order (edge_in_dim=6):
      [P_risk, log1p(ET0), log1p(Lambda_feas), rate, H_norm, load]

    Notes:
      - This function is robust to empty edge sets (E=0).
      - Risk probability mode can be selected via cox_cfg['risk_mode'].
      - All returned values are normalized (approximately) to [-1, 1].
    """

    # Robust empty-edge handling
    if "dist" not in edge_ctx:
        # if the builder didn't populate dist, treat as empty
        return torch.zeros((0, 6), device=device)

    E = int(edge_ctx["dist"].numel())
    if E == 0:
        return torch.zeros((0, 6), device=device)

    # Core Cox / geometry-derived descriptors
    ET0 = expected_residual_time(edge_ctx, visibility_radius=float(visibility_radius)).to(device)
    lam = lambda_feas(edge_ctx, cox_cfg=cox_cfg, device=device)  # [E]

    risk_mode = str(cox_cfg.get("risk_mode", "moment"))
    p_risk = p_risk_ho(lam, ET0, mode=risk_mode)  # [E]

    # Destination satellite load lookup
    dst_s = _infer_dst_s(edge_ctx, device=device)  # [E]
    load = sat_load.to(device)[dst_s]  # [E]

    # Normalized handover cost
    Hn = normalize_cost(ho_cost.to(device))  # [E]

    # Stack into z and normalize
    z = torch.stack(
        [
            p_risk,
            ET0,
            lam,
            rate.to(device),
            Hn,
            load,
        ],
        dim=-1,
    )

    return normalize_descriptor(z)
