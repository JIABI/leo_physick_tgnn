from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import torch


def build_user_sat_edges(
    user_pos: torch.Tensor,
    sat_pos: torch.Tensor,
    visibility_radius: float,
    user_offset: int = 0,
    sat_offset: int = 0,
    topk: Optional[int] = None,
    fallback_to_topk: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build directed edges from users to candidate satellites.

    This function is intentionally robust: it will not silently return an empty graph
    unless there are literally no satellites.

    Args:
      user_pos: [K,3]
      sat_pos:  [S,3]
      visibility_radius: threshold in the *same units* as user_pos/sat_pos.
      user_offset/sat_offset: global index offsets in the concatenated node list.
      topk: if provided, for each user select up to topk closest satellites.
      fallback_to_topk: if True, when no satellites are within visibility_radius for a user,
        still keep its topk nearest satellites so the graph is never empty.

    Returns:
      edge_index: [2,E] global indices
      edge_ctx: dict with local indices and distances
    """
    device = user_pos.device
    K = int(user_pos.size(0))
    S = int(sat_pos.size(0))

    if S == 0 or K == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        return edge_index, {
            "src_u": torch.zeros((0,), dtype=torch.long, device=device),
            "dst_s": torch.zeros((0,), dtype=torch.long, device=device),
            "dist": torch.zeros((0,), dtype=torch.float32, device=device),
            "K": K,
            "S": S,
        }

    # Pairwise distances [K,S]
    dist = torch.cdist(user_pos, sat_pos)  # float32

    if topk is None or int(topk) <= 0:
        # Pure radius-based visibility
        mask = dist < float(visibility_radius)
        src_u, dst_s = torch.where(mask)
        if src_u.numel() == 0:
            # No visible satellites at all
            if not fallback_to_topk:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                edge_ctx = {"src_u": src_u, "dst_s": dst_s, "dist": torch.zeros((0,), device=device), "K": K, "S": S}
                return edge_index, edge_ctx
            # Fallback: connect each user to its nearest satellite
            nn = torch.argmin(dist, dim=1)  # [K]
            src_u = torch.arange(K, device=device)
            dst_s = nn
        edge_index = torch.stack([src_u + user_offset, dst_s + sat_offset], dim=0)
        edge_ctx = {
            "src_u": src_u,
            "dst_s": dst_s,
            "dist": dist[src_u, dst_s],
            "K": K,
            "S": S,
        }
        return edge_index, edge_ctx

    # Top-k selection per user
    k = min(int(topk), S)
    # indices of k nearest sats for each user
    topk_dist, topk_idx = torch.topk(dist, k=k, largest=False, dim=1)  # [K,k]

    # Apply visibility radius mask within top-k
    vis_mask = topk_dist < float(visibility_radius)

    if vis_mask.any():
        # Keep only visible among top-k
        src_list = []
        dst_list = []
        for u in range(K):
            m = vis_mask[u]
            if m.any():
                dst_u = topk_idx[u][m]
            elif fallback_to_topk:
                dst_u = topk_idx[u]
            else:
                continue
            src_u = torch.full((dst_u.numel(),), u, dtype=torch.long, device=device)
            src_list.append(src_u)
            dst_list.append(dst_u)
        if len(src_list) == 0:
            # Fallback for pathological cases
            nn = torch.argmin(dist, dim=1)
            src_u = torch.arange(K, device=device)
            dst_s = nn
        else:
            src_u = torch.cat(src_list, dim=0)
            dst_s = torch.cat(dst_list, dim=0)
    else:
        # Nobody has anything within radius; fallback to top-k for everyone
        if not fallback_to_topk:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_ctx = {"src_u": torch.zeros((0,), dtype=torch.long, device=device), "dst_s": torch.zeros((0,), dtype=torch.long, device=device), "dist": torch.zeros((0,), device=device), "K": K, "S": S}
            return edge_index, edge_ctx
        src_u = torch.arange(K, device=device).repeat_interleave(k)
        dst_s = topk_idx.reshape(-1)

    edge_index = torch.stack([src_u + user_offset, dst_s + sat_offset], dim=0)
    edge_ctx = {
        "src_u": src_u,
        "dst_s": dst_s,
        "dist": dist[src_u, dst_s],
        "K": K,
        "S": S,
    }
    return edge_index, edge_ctx
