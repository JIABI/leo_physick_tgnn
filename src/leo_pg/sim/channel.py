from __future__ import annotations
import torch

def compute_sinr(user_pos: torch.Tensor, sat_pos: torch.Tensor, sat_load: torch.Tensor,
                 base_sinr: float = 10.0, dist_scale: float = 5.0) -> torch.Tensor:
    """A lightweight SINR surrogate:
      sinr = base / (1 + dist*dist_scale) - 0.5*load
    returns: [E]
    """
    dist = torch.norm(user_pos - sat_pos, dim=-1)
    sinr = base_sinr / (1.0 + dist * dist_scale) - 0.5 * sat_load
    return torch.clamp(sinr, min=-10.0, max=40.0)

def sinr_to_rate(sinr: torch.Tensor) -> torch.Tensor:
    """Shannon-like mapping (surrogate)."""
    # if sinr is in linear-ish scale, keep positive; if negative, rate near 0
    return torch.log1p(torch.relu(sinr))
