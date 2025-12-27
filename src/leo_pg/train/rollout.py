from __future__ import annotations
from typing import Dict, Any
import torch

def rollout_teacher_forcing(model, episode: Dict[str, Any], device: torch.device, H: int = 30) -> Dict[str, Any]:
    """Teacher-forcing rollout evaluation: compare predicted y against ground truth y per step.
    This baseline does not feed predicted y back into descriptors; in your final system you may close the loop
    by updating load/interference predictions that influence z_ij.
    """
    out = model.forward_episode(episode, device=device)
    preds = out["preds"]
    ys = out["ys"]
    T = min(len(preds), H)
    mses = []
    for t in range(T):
        mses.append(torch.mean((preds[t] - ys[t]) ** 2).item())
    return {"mse_mean": float(sum(mses) / max(1,len(mses))), "mse_series": mses}
