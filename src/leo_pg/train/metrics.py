from __future__ import annotations
import torch

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())

def load_variance(pred_load: torch.Tensor, K_users: int, S_sats: int) -> float:
    # pred_load expected at satellite nodes only; here assume nodes include users first
    sat = pred_load[K_users:]
    return float(torch.var(sat).item())

def ping_pong_rate(actions: torch.Tensor) -> float:
    """Legacy alias: return the dimensionless valid-triplet A->B->A fraction.

    This is not the manuscript's time-windowed ping-pong event rate.
    """
    if actions.ndim != 2:
        raise ValueError("actions must have shape [T,K]")
    if actions.size(0) < 3:
        return 0.0
    a = actions[:-2]
    b = actions[1:-1]
    c = actions[2:]
    valid = (a >= 0) & (b >= 0) & (c >= 0)
    denominator = int(valid.sum().item())
    if denominator == 0:
        return 0.0
    ping = valid & (a == c) & (a != b)
    return float(ping.sum().item()) / float(denominator)
