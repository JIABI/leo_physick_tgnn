from __future__ import annotations
from typing import Dict, Any
import torch

def expected_residual_time(edge_ctx: Dict[str, Any], visibility_radius: float = 0.7) -> torch.Tensor:
    """Return the fixed radial residual-time proxy of the compatibility API.

    This deterministic proxy is used only by the historical descriptor stack.
    The manuscript path computes integrated first-violation Intensity from the
    explicit elevation trajectory in ``leo_pg.sim.intensity_flow``.

    Returns [E] tensor (seconds).
    """
    dist = edge_ctx["dist"]
    # closer => longer, clipped
    return torch.clamp((visibility_radius - dist) * 20.0, min=0.1)
