from __future__ import annotations
from typing import Dict, Any
import torch

def expected_residual_time(edge_ctx: Dict[str, Any], visibility_radius: float = 0.7) -> torch.Tensor:
    """A debug residual service time expectation E[T0].
    In the Cox-based derivation, T0 comes from residual time distribution (often uniform-like in window).
    Here we approximate remaining time as proportional to how deep inside visibility region a link is.

    Returns [E] tensor (seconds).
    """
    dist = edge_ctx["dist"]
    # closer => longer, clipped
    return torch.clamp((visibility_radius - dist) * 20.0, min=0.1)
