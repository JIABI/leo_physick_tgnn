from __future__ import annotations
import torch

def safe_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))

def safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)

def squash_tanh(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.tanh(x / scale)

def normalize_descriptor(z: torch.Tensor) -> torch.Tensor:
    """Default numerically stable transforms; output in approx [-1,1]."""
    zt = z.clone()
    # Assumed ordering: [P_risk, ET0, Lambda, rate, H, load]
    if zt.size(-1) >= 1:
        zt[..., 0] = torch.tanh(safe_logit(zt[..., 0]))
    if zt.size(-1) >= 2:
        zt[..., 1] = torch.tanh(safe_log1p(zt[..., 1]))
    if zt.size(-1) >= 3:
        zt[..., 2] = torch.tanh(safe_log1p(zt[..., 2]))
    if zt.size(-1) >= 4:
        zt[..., 3] = torch.tanh(zt[..., 3] / 5.0)
    if zt.size(-1) >= 5:
        zt[..., 4] = torch.tanh(zt[..., 4] / 2.0)
    if zt.size(-1) >= 6:
        zt[..., 5] = torch.tanh(zt[..., 5] / 2.0)
    return zt
