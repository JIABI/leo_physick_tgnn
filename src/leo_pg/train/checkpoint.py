from __future__ import annotations
import torch
from typing import Optional, Any, Dict

def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    """Return a model state_dict from a checkpoint payload.

    Supports multiple historical key conventions:
      - {'model': state_dict}
      - {'model_state': state_dict}
      - {'state_dict': state_dict}
      - {'model_state_dict': state_dict}
    Also supports checkpoints that *are* the state_dict themselves.
    """
    if isinstance(payload, dict):
        for k in ("model", "model_state", "state_dict", "model_state_dict"):
            if k in payload and isinstance(payload[k], dict):
                return payload[k]
    if isinstance(payload, dict):
        # Heuristic: if most values are tensors, assume it's already a state_dict
        tensor_vals = sum(1 for v in payload.values() if torch.is_tensor(v))
        if tensor_vals > 0 and tensor_vals / max(1, len(payload)) > 0.5:
            return payload  # type: ignore[return-value]
    raise KeyError(
        "Could not find a model state_dict in checkpoint. "
        "Expected keys one of: 'model', 'model_state', 'state_dict', 'model_state_dict', "
        "or the checkpoint itself to be a state_dict."
    )

def save_ckpt(
    path: str,
    model: torch.nn.Module,
    opt: Optional[torch.optim.Optimizer] = None,
    **meta: Any,
) -> None:
    """Save a checkpoint with a stable, backward-compatible schema."""
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),  # preferred
        "model": model.state_dict(),        # legacy compat
        **meta,
    }
    if opt is not None:
        payload["optimizer_state"] = opt.state_dict()
        payload["optimizer"] = opt.state_dict()  # legacy compat
    torch.save(payload, path)

def load_ckpt(
    path: str,
    model: torch.nn.Module,
    opt: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load checkpoint. Returns checkpoint metadata dict (payload)."""
    payload = torch.load(path, map_location=map_location)
    state = _extract_state_dict(payload)
    model.load_state_dict(state, strict=strict)

    if opt is not None and isinstance(payload, dict):
        opt_state = payload.get("optimizer_state") or payload.get("optimizer") or payload.get("opt")
        if isinstance(opt_state, dict):
            opt.load_state_dict(opt_state)
    return payload if isinstance(payload, dict) else {"_payload_type": type(payload).__name__}
