from __future__ import annotations
from typing import Dict, Any, List
import math
import torch

def lambda_feas_from_shells(shells: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    """Compute Lambda^{feas}(t) using a simplified closed-form inspired by:
        Lambda^{feas}(t) = sum_s (P_s Q_s theta_s / (2π)) * sin(alpha_s) * p_feas^(s)(t)

    In real usage, p_feas may be time-varying; here it is a scalar per shell.
    Returns scalar Tensor on device.
    """
    if shells is None or len(shells) == 0:
        return torch.tensor(0.0, device=device)
    total = 0.0
    for sh in shells:
        P = float(sh.get("P", 0))
        Q = float(sh.get("Q", 0))
        theta = float(sh.get("theta", 0))
        alpha_deg = float(sh.get("alpha_deg", 0))
        p_feas = float(sh.get("p_feas", 1.0))
        total += (P * Q * theta / (2.0 * math.pi)) * math.sin(math.radians(alpha_deg)) * p_feas
    return torch.tensor(total, device=device)

def lambda_feas(edge_ctx: Dict[str, Any], cox_cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Edge-wise Lambda^{feas}(t). For now, treat as global background intensity.
    Extend later by conditioning on latitude/elevation and time.
    Returns [E] tensor.
    """
    shells = cox_cfg.get("shells", [])
    lam = lambda_feas_from_shells(shells, device=device)
    E = int(edge_ctx["dist"].numel())
    return lam.expand(E)
