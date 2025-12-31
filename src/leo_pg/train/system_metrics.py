from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import torch

from leo_pg.sim.channel import compute_sinr, sinr_to_rate


def pingpong_rate(serving_seq: torch.Tensor, invalid: int = -1) -> float:
    """Compute ping-pong rate: A->B->A over consecutive steps.

    serving_seq: [T, K] int, local sat ids (or invalid)
    """
    if serving_seq.ndim != 2:
        raise ValueError("serving_seq must be [T,K]")
    T, K = serving_seq.shape
    if T < 3:
        return 0.0
    a = serving_seq[:-2]
    b = serving_seq[1:-1]
    c = serving_seq[2:]
    valid = (a != invalid) & (b != invalid) & (c != invalid)
    ping = valid & (a == c) & (a != b)
    denom = float(valid.numel())
    return float(ping.float().sum().item()) / max(1.0, denom)


def ho_failure_rate(ho_fail_seq: torch.Tensor) -> float:
    """ho_fail_seq: [T,K] bool."""
    if ho_fail_seq.numel() == 0:
        return 0.0
    return float(ho_fail_seq.float().mean().item())


def load_stats(load_seq: torch.Tensor) -> Tuple[float, float]:
    """Compute (mean variance, mean peak) across time.

    load_seq: [T,S] float (utilization)
    """
    if load_seq.numel() == 0:
        return 0.0, 0.0
    var_t = load_seq.var(dim=1, unbiased=False)
    peak_t = load_seq.max(dim=1).values
    return float(var_t.mean().item()), float(peak_t.mean().item())


def greedy_assign_from_load(
    *,
    node_x: torch.Tensor,          # [N,6]
    edge_index: torch.Tensor,      # [2,E] (may contain other types)
    edge_type: torch.Tensor,       # [E]
    sat_load: torch.Tensor,        # [S] utilization
    K_users: int,
    sat_capacity_users: int,
    channel_cfg: Dict[str, Any],
    sinr_min: float = -1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute serving decisions from a predicted sat_load.

    Returns:
      serving_sat [K] (local sat id, -1 for fail)
      handover    [K] bool (dummy here; caller can compute from prev)
      ho_fail     [K] bool
    """
    device = node_x.device
    # Consider only user->sat edges
    m = (edge_type == 0)  # EdgeType.USER_SAT == 0
    if not torch.any(m):
        serving = torch.full((K_users,), -1, device=device, dtype=torch.long)
        ho_fail = torch.ones((K_users,), device=device, dtype=torch.bool)
        handover = torch.zeros((K_users,), device=device, dtype=torch.bool)
        return serving, handover, ho_fail

    ei = edge_index[:, m]
    src_u = ei[0]                  # global user idx
    dst_g = ei[1]                  # global sat node idx
    dst_s = dst_g - int(K_users)   # local sat idx

    user_pos = node_x[:K_users, :3]
    sat_pos = node_x[K_users:, :3]

    sinr = compute_sinr(
        user_pos=user_pos[src_u],
        sat_pos=sat_pos[dst_s],
        sat_load=sat_load[dst_s],
        base_sinr=float(channel_cfg.get("base_sinr", 10.0)),
        dist_scale=float(channel_cfg.get("dist_scale", 5.0)),
    )
    rate = sinr_to_rate(sinr)

    serving = torch.full((K_users,), -1, device=device, dtype=torch.long)
    ho_fail = torch.zeros((K_users,), device=device, dtype=torch.bool)
    incoming = torch.zeros((sat_load.numel(),), device=device, dtype=torch.float32)

    cap = max(1, int(sat_capacity_users))
    delta_util = 1.0 / float(cap)

    user_order = torch.randperm(K_users, device=device)

    for u in user_order.tolist():
        mask_u = (src_u == int(u))
        if not torch.any(mask_u):
            ho_fail[u] = True
            continue
        idx = torch.nonzero(mask_u, as_tuple=False).view(-1)
        r = rate[idx]
        s = dst_s[idx]
        q = sinr[idx]
        order = torch.argsort(r, descending=True)
        picked = -1
        for j in order.tolist():
            sj = int(s[j].item())
            if float(q[j].item()) < float(sinr_min):
                continue
            util = float(sat_load[sj].item()) + float(incoming[sj].item()) * delta_util
            if util + delta_util <= 1.0 + 1e-6:
                picked = sj
                incoming[sj] += 1.0
                break
        if picked < 0:
            ho_fail[u] = True
        else:
            serving[u] = picked

    handover = torch.zeros((K_users,), device=device, dtype=torch.bool)
    return serving, handover, ho_fail
