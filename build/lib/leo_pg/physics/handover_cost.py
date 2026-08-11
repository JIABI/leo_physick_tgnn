from __future__ import annotations
import torch

def normalize_cost(cost: torch.Tensor, max_cost: float = 2.0) -> torch.Tensor:
    return torch.clamp(cost / max_cost, min=0.0, max=1.0)
