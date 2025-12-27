from __future__ import annotations
import torch

def update_satellite_load(sat_load: torch.Tensor, incoming_users: torch.Tensor, momentum: float = 0.8) -> torch.Tensor:
    """sat_load: [S], incoming_users: [S] counts or fractions"""
    incoming = incoming_users.float()
    return momentum * sat_load + (1.0 - momentum) * incoming
