from __future__ import annotations
import torch
import torch.nn as nn

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def satellite_target_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    step: dict,
) -> torch.Tensor:
    """MSE over satellite target rows, excluding zero-padded user rows."""
    if pred.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    meta = step.get("meta", {})
    if "K_users" not in meta:
        raise ValueError("step.meta.K_users is required for satellite-target MSE")
    user_count = int(meta["K_users"])
    if not 0 <= user_count < pred.size(0):
        raise ValueError(
            f"K_users={user_count} leaves no valid satellite target rows for shape {tuple(pred.shape)}"
        )
    return mse_loss(pred[user_count:], target[user_count:])

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff * mask
    return diff.sum() / (mask.sum() + 1e-12)
