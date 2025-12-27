from __future__ import annotations
import torch

def build_satellite_interference_edges(sat_pos: torch.Tensor, radius: float = 0.5):
    """Create sat<->sat edges for nearby satellites (undirected as two directed edges).
    Returns edge_index [2, E_ss].
    """
    S = sat_pos.size(0)
    dist = torch.cdist(sat_pos, sat_pos)  # [S,S]
    mask = (dist < radius) & (dist > 0)
    src, dst = torch.where(mask)
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index
