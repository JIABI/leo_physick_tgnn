from __future__ import annotations
import torch

from typing import Tuple, Dict, Any


def build_user_sat_edges(

        user_pos: torch.Tensor,

        sat_pos: torch.Tensor,

        visibility_radius: float,

        user_offset: int,

        sat_offset: int,

        topk: int = 8,

) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """

    Returns:

      edge_index: [2, E] (src=user, dst=sat global index)

      edge_ctx: dict with per-edge context (dist, rel_vec, etc.)

    """

    # dist: [K, S]

    dist = torch.cdist(user_pos, sat_pos)

    # mask by visibility

    mask = dist <= visibility_radius

    dist_masked = dist.masked_fill(~mask, float("inf"))

    # top-k nearest among visible (per user)

    k = min(topk, sat_pos.shape[0])

    vals, idx = torch.topk(dist_masked, k=k, largest=False, dim=1)  # [K,k]

    valid = torch.isfinite(vals)  # [K,k]

    src_users = torch.arange(user_pos.shape[0], device=user_pos.device).unsqueeze(1).expand_as(idx)

    src = src_users[valid] + user_offset

    dst = idx[valid] + sat_offset

    edge_index = torch.stack([src, dst], dim=0)

    # build edge context

    # compute rel vector only for chosen edges (avoid huge tensor)

    u_local = src - user_offset

    s_local = dst - sat_offset

    rel = sat_pos[s_local] - user_pos[u_local]

    edge_ctx = {

        "dist": vals[valid],

        "rel": rel,

        "u_local": u_local,

        "s_local": s_local,

    }

    return edge_index, edge_ctx

