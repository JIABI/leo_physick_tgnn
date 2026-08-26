from __future__ import annotations
import torch
from typing import Optional, Any, Dict

from leo_pg.models.heads.intensity_flow import INTENSITY_FLOW_HEAD_CONTRACT_VERSION
from leo_pg.sim.state import (
    PAPER_EDGE_FEATURE_NAMES,
    PAPER_FEATURE_CONTRACT_VERSION,
    PAPER_NODE_FEATURE_NAMES,
)

CHECKPOINT_SCHEMA_VERSION = 2


def _canonical_signature_value(value: Any, *, path: str) -> Any:
    """Restrict nested semantic config to a deterministic primitive tree."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [
            _canonical_signature_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            str(key): _canonical_signature_value(value[key], path=f"{path}.{key}")
            for key in sorted(value)
        }
    raise TypeError(
        f"model signature field {path} must contain only primitive config values, "
        f"got {type(value).__name__}"
    )


def model_signature_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize every configuration field that changes model forward semantics."""
    model_cfg = cfg.get("model", {})
    head_cfg = cfg.get("head", {})
    head_type = str(head_cfg.get("type", "forecast")).lower()
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
            "message_passing_layers": int(
                model_cfg.get("message_passing_layers", 1)
            ),
            "time_encoding_dim": (
                None
                if model_cfg.get("time_encoding_dim") is None
                else int(model_cfg["time_encoding_dim"])
            ),
            "edge_chunk": int(
                cfg.get("train", {}).get("edge_chunk", model_cfg.get("edge_chunk", 50000))
            ),
        },
        "head": {
            "type": head_type,
            "out_dim": int(head_cfg.get("out_dim", 1)),
            "output_activation": str(head_cfg.get("output_activation", "none")).lower(),
        },
    }
    paper_method = str(cfg.get("paper_method", "")).strip().lower()
    paper_interface = str(cfg.get("paper_interface", "")).strip().lower()
    if paper_method:
        signature["paper_method"] = paper_method
    if paper_interface:
        signature["paper_interface"] = paper_interface
    paper_baseline = cfg.get("paper_baseline")
    if paper_baseline is not None:
        if not isinstance(paper_baseline, dict):
            raise TypeError("paper_baseline must be a mapping")
        signature["paper_baseline"] = _canonical_signature_value(
            paper_baseline,
            path="paper_baseline",
        )
    if paper_interface == "snapshot_v1":
        from leo_pg.paper.snapshot_data import (
            SNAPSHOT_EDGE_FEATURE_NAMES,
            SNAPSHOT_FEATURE_CONTRACT_VERSION,
            SNAPSHOT_NODE_FEATURE_NAMES,
        )

        snapshot_cfg = cfg.get("paper_snapshot")
        if not isinstance(snapshot_cfg, dict):
            raise TypeError("paper_snapshot must be a mapping")
        snapshot_head = snapshot_cfg.get("head")
        if not isinstance(snapshot_head, dict):
            raise TypeError("paper_snapshot.head must be a mapping")
        signature["head"] = {
            "type": "snapshot_v1",
            "feature_contract_version": SNAPSHOT_FEATURE_CONTRACT_VERSION,
            "node_feature_names": SNAPSHOT_NODE_FEATURE_NAMES,
            "edge_feature_names": SNAPSHOT_EDGE_FEATURE_NAMES,
            "hidden_dim": int(
                snapshot_head.get("hidden_dim", model_cfg.get("emb_dim", 64))
            ),
            "dropout": float(
                snapshot_head.get("dropout", model_cfg.get("dropout", 0.0))
            ),
            "load_bounds": _canonical_signature_value(
                snapshot_head.get("load_bounds"),
                path="paper_snapshot.head.load_bounds",
            ),
            "descriptor_domains": [
                "current_gamma",
                "current_feasibility_margin",
                "instantaneous_pre_ema_admitted_load",
            ],
            "forbidden_domains": ["integrated_intensity", "raw_flow", "ema_flow"],
        }
        snapshot_model_semantics: Dict[str, Any] = {
            "head": _canonical_signature_value(
                snapshot_head,
                path="paper_snapshot.head",
            ),
            "feature": _canonical_signature_value(
                snapshot_cfg.get("feature"),
                path="paper_snapshot.feature",
            ),
        }
        if paper_method == "snapshot_da_gwm":
            policies = snapshot_cfg.get("policies")
            selected_policy = (
                policies.get(paper_method) if isinstance(policies, dict) else None
            )
            snapshot_model_semantics["decision_loss"] = _canonical_signature_value(
                snapshot_cfg.get("decision_loss"),
                path="paper_snapshot.decision_loss",
            )
            snapshot_model_semantics["score_weights"] = _canonical_signature_value(
                selected_policy.get("score_weights")
                if isinstance(selected_policy, dict)
                else None,
                path=f"paper_snapshot.policies.{paper_method}.score_weights",
            )
        signature["paper_snapshot_model"] = snapshot_model_semantics
    elif head_type == "intensity_flow":
        signature["head"].update(
            {
                "contract_version": INTENSITY_FLOW_HEAD_CONTRACT_VERSION,
                "feature_contract_version": PAPER_FEATURE_CONTRACT_VERSION,
                "node_feature_names": PAPER_NODE_FEATURE_NAMES,
                "edge_feature_names": PAPER_EDGE_FEATURE_NAMES,
                "hidden_dim": int(head_cfg.get("hidden_dim", model_cfg.get("emb_dim", 64))),
                "dropout": float(head_cfg.get("dropout", model_cfg.get("dropout", 0.0))),
                "edge_input": "source_destination_node_embeddings_and_edge_descriptors",
                "gamma_activation": "identity",
                "intensity_activation": "softplus",
                "flow_activation": "sigmoid",
                "feasibility_role": "auxiliary_logit",
            }
        )
    if message_type == "physick":
        physick_cfg: Dict[str, Any] = {}
        if isinstance(cfg.get("physick"), dict):
            physick_cfg.update(cfg["physick"])
        if isinstance(model_cfg.get("physick"), dict):
            physick_cfg.update(model_cfg["physick"])
        projection_radius = physick_cfg.get("projection_radius", 1.0)
        signature["physick"] = {
            "implementation": str(
                physick_cfg.get("implementation", "legacy_cox_prototype")
            ).lower(),
            "num_kernels": int(physick_cfg.get("num_kernels", 16)),
            "num_knots": int(physick_cfg.get("num_knots", 16)),
            "coeff_impl": str(physick_cfg.get("coeff_impl", "mlp")).lower(),
            "mix_impl": str(physick_cfg.get("mix_impl", "dot")).lower(),
            "coeff_hidden": int(physick_cfg.get("coeff_hidden", 64)),
            "coeff_chunk": int(physick_cfg.get("coeff_chunk", 256)),
            "mix_hidden": int(physick_cfg.get("mix_hidden", 128)),
            "dropout": float(physick_cfg.get("dropout", model_cfg.get("dropout", 0.0))),
            "projection_radius": None if projection_radius is None else float(projection_radius),
            "latent_dim": int(physick_cfg.get("latent_dim", model_cfg.get("msg_dim", 0))),
            "descriptor_dim": int(physick_cfg.get("descriptor_dim", 16)),
            "kernel_hidden_dim": int(
                physick_cfg.get("kernel_hidden_dim", model_cfg.get("msg_dim", 0))
            ),
            "operating_clip": float(physick_cfg.get("operating_clip", 5.0)),
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
        priority = ("model_state", "model", "model_state_dict", "state_dict")
        candidates = [(key, payload[key]) for key in priority if isinstance(payload.get(key), dict)]
        if candidates:
            preferred_key, preferred = candidates[0]
            for other_key, other in candidates[1:]:
                conflict = set(preferred) != set(other)
                if not conflict:
                    for name in preferred:
                        left, right = preferred[name], other[name]
                        if torch.is_tensor(left) and torch.is_tensor(right):
                            conflict = not torch.equal(left, right)
                        else:
                            conflict = type(left) is not type(right) or left != right
                        if conflict:
                            break
                if conflict:
                    raise ValueError(
                        "checkpoint contains conflicting model states under "
                        f"{preferred_key!r} and {other_key!r}"
                    )
            return preferred
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
    nonfinite = [
        name
        for name, value in model.state_dict().items()
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise FloatingPointError(f"refusing to save non-finite model state: {', '.join(nonfinite[:8])}")
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
    allow_legacy_checkpoint: bool = False,
) -> Dict[str, Any]:
    """Load checkpoint. Returns checkpoint metadata dict (payload)."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    state = _extract_state_dict(payload)
    nonfinite = [
        name
        for name, value in state.items()
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise FloatingPointError(f"checkpoint contains non-finite tensors: {', '.join(nonfinite[:8])}")
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
        if saved_signature is None and isinstance(payload, dict) and isinstance(payload.get("config"), dict):
            saved_signature = model_signature_from_config(payload["config"])
        current_cfg = getattr(model, "cfg", None)
        if saved_signature is None and not allow_legacy_checkpoint:
            raise RuntimeError(
                "Checkpoint has no model semantic signature. Its activation, aggregation, and "
                "other same-shape semantics cannot be verified. Pass allow_legacy_checkpoint=True "
                "only for an explicitly reviewed legacy migration."
            )
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
