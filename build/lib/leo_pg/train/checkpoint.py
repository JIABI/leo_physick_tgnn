from __future__ import annotations
import torch
from typing import Optional, Any, Dict

CHECKPOINT_SCHEMA_VERSION = 2


def model_signature_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize every configuration field that changes model forward semantics."""
    model_cfg = cfg.get("model", {})
    head_cfg = cfg.get("head", {})
    message_type = str(model_cfg.get("message_type", "mlp")).lower()
    signature: Dict[str, Any] = {
        "model": {
            "node_in_dim": int(model_cfg.get("node_in_dim", 0)),
            "edge_in_dim": int(model_cfg.get("edge_in_dim", 0)),
            "mem_dim": int(model_cfg.get("mem_dim", 0)),
            "msg_dim": int(model_cfg.get("msg_dim", 0)),
            "emb_dim": int(model_cfg.get("emb_dim", 0)),
            "message_type": message_type,
            "aggregator": str(model_cfg.get("aggregator", "sum")).lower(),
            "dropout": float(model_cfg.get("dropout", 0.0)),
            "use_edge_type": bool(model_cfg.get("use_edge_type", True)),
            "edge_type_vocab": int(model_cfg.get("edge_type_vocab", 8)),
            "edge_chunk": int(
                cfg.get("train", {}).get("edge_chunk", model_cfg.get("edge_chunk", 50000))
            ),
        },
        "head": {
            "type": str(head_cfg.get("type", "forecast")).lower(),
            "out_dim": int(head_cfg.get("out_dim", 1)),
            "output_activation": str(head_cfg.get("output_activation", "none")).lower(),
        },
    }
    if message_type == "physick":
        physick_cfg: Dict[str, Any] = {}
        if isinstance(cfg.get("physick"), dict):
            physick_cfg.update(cfg["physick"])
        if isinstance(model_cfg.get("physick"), dict):
            physick_cfg.update(model_cfg["physick"])
        projection_radius = physick_cfg.get("projection_radius", 1.0)
        signature["physick"] = {
            "num_kernels": int(physick_cfg.get("num_kernels", 16)),
            "num_knots": int(physick_cfg.get("num_knots", 16)),
            "coeff_impl": str(physick_cfg.get("coeff_impl", "mlp")).lower(),
            "mix_impl": str(physick_cfg.get("mix_impl", "dot")).lower(),
            "coeff_hidden": int(physick_cfg.get("coeff_hidden", 64)),
            "coeff_chunk": int(physick_cfg.get("coeff_chunk", 256)),
            "mix_hidden": int(physick_cfg.get("mix_hidden", 128)),
            "dropout": float(physick_cfg.get("dropout", model_cfg.get("dropout", 0.0))),
            "projection_radius": None if projection_radius is None else float(projection_radius),
        }
    return signature

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
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),  # preferred
        "model": model.state_dict(),        # legacy compat
        **meta,
    }
    config = meta.get("config", getattr(model, "cfg", None))
    if isinstance(config, dict):
        payload["model_signature"] = model_signature_from_config(config)
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
    allow_config_mismatch: bool = False,
) -> Dict[str, Any]:
    """Load checkpoint. Returns checkpoint metadata dict (payload)."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    state = _extract_state_dict(payload)
    if strict:
        expected = model.state_dict()
        missing = sorted(set(expected).difference(state))
        unexpected = sorted(set(state).difference(expected))
        shape_mismatches = [
            f"{name}: checkpoint={tuple(state[name].shape)}, model={tuple(expected[name].shape)}"
            for name in sorted(set(expected).intersection(state))
            if tuple(state[name].shape) != tuple(expected[name].shape)
        ]
        if missing or unexpected or shape_mismatches:
            details = []
            if shape_mismatches:
                details.append("shape mismatches: " + "; ".join(shape_mismatches[:8]))
            if missing:
                details.append("missing keys: " + ", ".join(missing[:8]))
            if unexpected:
                details.append("unexpected keys: " + ", ".join(unexpected[:8]))
            raise RuntimeError(
                "Checkpoint architecture is incompatible with the requested model ("
                + " | ".join(details)
                + "). Check the saved config and kernel count; legacy PhysiCK checkpoints used 12 kernels, "
                "whereas the repaired default uses 16."
            )
        saved_signature = payload.get("model_signature") if isinstance(payload, dict) else None
        current_cfg = getattr(model, "cfg", None)
        if saved_signature is not None and isinstance(current_cfg, dict):
            requested_signature = model_signature_from_config(current_cfg)
            if saved_signature != requested_signature and not allow_config_mismatch:
                raise RuntimeError(
                    "Checkpoint model signature does not match the requested configuration. "
                    "Use the checkpoint's saved config, or explicitly set allow_config_mismatch=True "
                    "only for a deliberate migration."
                )
    model.load_state_dict(state, strict=strict)

    if opt is not None and isinstance(payload, dict):
        opt_state = payload.get("optimizer_state") or payload.get("optimizer") or payload.get("opt")
        if isinstance(opt_state, dict):
            opt.load_state_dict(opt_state)
    return payload if isinstance(payload, dict) else {"_payload_type": type(payload).__name__}
