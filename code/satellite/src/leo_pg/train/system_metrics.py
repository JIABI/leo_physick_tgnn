from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import torch

from leo_pg.sim.channel import compute_sinr, sinr_to_rate
from leo_pg.sim.protocol import validate_allocator_parameters


def adjacent_aba_fraction(serving_seq: torch.Tensor, invalid: int = -1) -> float:
    """Fraction of valid consecutive triplets that follow A->B->A.

    This dimensionless diagnostic is not a time-windowed events/user/s metric.
    ``serving_seq`` has shape [T,K] and contains local satellite ids.
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
    denom = float(valid.sum().item())
    return float(ping.float().sum().item()) / max(1.0, denom)


def pingpong_rate(serving_seq: torch.Tensor, invalid: int = -1) -> float:
    """Backward-compatible alias for :func:`adjacent_aba_fraction`."""
    return adjacent_aba_fraction(serving_seq, invalid=invalid)


def assignment_failure_fraction(failure_seq: torch.Tensor) -> float:
    """Fraction of user-time assignments marked failed.

    This is not handover failures divided by handover attempts.
    """
    if failure_seq.numel() == 0:
        return 0.0
    return float(failure_seq.float().mean().item())


def ho_failure_rate(ho_fail_seq: torch.Tensor) -> float:
    """Legacy alias for the assignment-failure fraction diagnostic."""
    return assignment_failure_fraction(ho_fail_seq)


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
    user_order: Optional[torch.Tensor] = None,
    previous_serving: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute serving decisions from a predicted sat_load.

    Returns:
      serving_sat [K] (local sat id, -1 for fail)
      handover    [K] bool (valid previous and new serving ids differ)
      ho_fail     [K] bool
    """
    device = node_x.device
    if node_x.ndim != 2 or node_x.size(1) < 3:
        raise ValueError("node_x must have shape [N,feature_dim>=3]")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_type.ndim != 1 or edge_type.numel() != edge_index.size(1):
        raise ValueError("edge_type must have one entry per edge")
    if sat_load.ndim != 1 or K_users <= 0 or node_x.size(0) != K_users + sat_load.numel():
        raise ValueError("sat_load/K_users dimensions do not match node_x")
    if not torch.isfinite(node_x).all() or not torch.isfinite(sat_load).all():
        raise ValueError("allocator inputs must be finite")
    if previous_serving is not None:
        previous_serving = previous_serving.to(device=device, dtype=torch.long)
        if previous_serving.shape != (K_users,):
            raise ValueError("previous_serving must have shape [K_users]")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= node_x.size(0)
    ):
        raise ValueError("edge_index contains a node id outside node_x")
    supported = (edge_type == 0) | (edge_type == 1)
    if edge_type.numel() and not bool(torch.all(supported)):
        raise ValueError("edge_type contains unsupported values")
    user_sat_mask = edge_type == 0
    if torch.any(user_sat_mask):
        user_sat_edges = edge_index[:, user_sat_mask]
        if torch.any(user_sat_edges[0] >= K_users) or torch.any(user_sat_edges[1] < K_users):
            raise ValueError("USER_SAT edges must run from a user node to a satellite node")
    sat_sat_mask = edge_type == 1
    if torch.any(sat_sat_mask) and torch.any(edge_index[:, sat_sat_mask] < K_users):
        raise ValueError("SAT_SAT edges must connect two satellite nodes")
    base_sinr, dist_scale, sinr_min, _ = validate_allocator_parameters(
        base_sinr=channel_cfg.get("base_sinr", 10.0),
        dist_scale=channel_cfg.get("dist_scale", 5.0),
        sinr_min=sinr_min,
        load_momentum=0.0,
    )
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
        base_sinr=base_sinr,
        dist_scale=dist_scale,
    )
    rate = sinr_to_rate(sinr)

    serving = torch.full((K_users,), -1, device=device, dtype=torch.long)
    ho_fail = torch.zeros((K_users,), device=device, dtype=torch.bool)
    incoming = torch.zeros((sat_load.numel(),), device=device, dtype=torch.float32)

    cap = max(1, int(sat_capacity_users))
    delta_util = 1.0 / float(cap)

    # Evaluation must not depend on how many random numbers a previous model or
    # horizon consumed. Callers can supply an explicit permutation; otherwise
    # use a deterministic order.
    if user_order is None:
        user_order = torch.arange(K_users, device=device)
    else:
        user_order = user_order.to(device=device, dtype=torch.long)
        if user_order.numel() != K_users:
            raise ValueError("user_order must contain exactly K_users entries")
        if (
            torch.unique(user_order).numel() != K_users
            or int(user_order.min()) < 0
            or int(user_order.max()) >= K_users
        ):
            raise ValueError("user_order must be a permutation of [0,K_users)")

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

    if previous_serving is None:
        handover = torch.zeros((K_users,), device=device, dtype=torch.bool)
    else:
        handover = (
            (previous_serving >= 0)
            & (serving >= 0)
            & (previous_serving != serving)
        )
    return serving, handover, ho_fail
