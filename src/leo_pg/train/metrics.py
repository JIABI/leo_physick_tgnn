from __future__ import annotations
import torch

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())

def load_variance(pred_load: torch.Tensor, K_users: int, S_sats: int) -> float:
    # pred_load expected at satellite nodes only; here assume nodes include users first
    sat = pred_load[K_users:]
    return float(torch.var(sat).item())

def ping_pong_rate(actions: torch.Tensor) -> float:
    # Placeholder: compute rate of alternation in chosen sat id per user
    # actions: [T, K] int
    if actions.numel() == 0:
        return 0.0
    changes = (actions[1:] != actions[:-1]).float().mean()
    return float(changes.item())
