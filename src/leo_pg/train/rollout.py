from __future__ import annotations

from typing import Dict, Any, Optional, Sequence, List

import torch


def rollout_teacher_forcing(
    model,
    episode: Dict[str, Any],
    device: torch.device,
    H: int = 30,
    weights: Optional[Sequence[float]] = None,
    return_preds_cpu: bool = False,
) -> Dict[str, Any]:
    """
    Teacher-forcing rollout evaluation (streaming, memory-safe).

    Key change vs old version:
      - DO NOT call model.forward_episode() (which stores all-step preds on GPU).
      - Instead, iterate steps and call model.forward_step(step, mem, device).
      - Optionally collect preds on CPU only (return_preds_cpu=True) for system metrics.

    Returns dict:
      - mse_mean, mse_last, mse_series, effective_H
      - preds_cpu (optional): list[Tensor] length effective_H, each on CPU
    """
    steps = episode.get("steps", [])
    if len(steps) == 0:
        return {
            "mse_mean": float("nan"),
            "mse_last": float("nan"),
            "mse_series": [],
            "effective_H": 0,
            **({"preds_cpu": []} if return_preds_cpu else {}),
        }

    # Safety: clip H
    effective_H = int(min(max(0, H), len(steps)))
    if effective_H <= 0:
        return {
            "mse_mean": float("nan"),
            "mse_last": float("nan"),
            "mse_series": [],
            "effective_H": 0,
            **({"preds_cpu": []} if return_preds_cpu else {}),
        }

    # Prepare weights
    if weights is not None:
        w = torch.tensor(list(weights[:effective_H]), device="cpu", dtype=torch.float32)
        w = w / (w.sum().clamp_min(1e-12))
    else:
        w = None

    preds_cpu: List[torch.Tensor] = []

    # Streaming rollout using forward_step
    # We assume your model implements forward_step(step, mem, device) -> (pred, y, mem).
    # And memory init is accessible via model.memory.init(node_x).
    with torch.no_grad():
        # init memory from step0 node features
        s0 = steps[0]
        node_x0 = s0["node_x"].to(device)
        mem = model.memory.init(node_x0)

        mses: List[float] = []
        mse_weighted_sum = 0.0
        wsum = 0.0

        for t in range(effective_H):
            step = steps[t]
            pred, y, mem = model.forward_step(step, mem, device=device)

            # compute MSE in fp32 on CPU to avoid AMP dtype issues
            mse_t = torch.mean((pred.float() - y.float()) ** 2).detach().cpu().item()
            mses.append(float(mse_t))

            if return_preds_cpu:
                preds_cpu.append(pred.detach().cpu())

            if w is not None:
                wt = float(w[t].item())
                mse_weighted_sum += wt * float(mse_t)
                wsum += wt

        if w is not None:
            mse_mean = float(mse_weighted_sum / max(1e-12, wsum))
        else:
            mse_mean = float(sum(mses) / max(1, len(mses)))

        out = {
            "mse_mean": mse_mean,
            "mse_last": float(mses[-1]),
            "mse_series": mses,
            "effective_H": effective_H,
        }
        if return_preds_cpu:
            out["preds_cpu"] = preds_cpu
        return out


def rollout_episode(
    model,
    episode: Dict[str, Any],
    device: torch.device,
    H: int = 30,
    weights: Optional[Sequence[float]] = None,
    return_preds_cpu: bool = False,
) -> Dict[str, Any]:
    """
    Wrapper for rollout evaluation (teacher forcing in this repo).

    return_preds_cpu=True makes it export-friendly:
      - preds are stored only on CPU
      - export script can compute pingpong/HO fail/load stats without GPU OOM
    """
    return rollout_teacher_forcing(
        model=model,
        episode=episode,
        device=device,
        H=H,
        weights=weights,
        return_preds_cpu=return_preds_cpu,
    )
