"""Model factory for the manuscript's Snapshot factorial cells.

The factory is deliberately separate from the Intensity--Flow registry.  Every
returned model consumes the seven-field contract in ``snapshot_data`` and
returns :class:`~leo_pg.paper.snapshot.SnapshotOutput` through
``predict_step``.

``snapshot_mlp`` and ``snapshot_physick`` are the two primary 2 x 2 cells.
``snapshot_ltt_r`` exposes the manuscript's LTT-R-on-Snapshot condition, and
``snapshot_da_gwm`` exposes the Snapshot head/loss variant of DA-GWM.  The
underlying controlled baseline implementations and their optional dependency
errors remain in :mod:`leo_pg.paper.baselines`.

Configuration assumptions
-------------------------
``model.node_in_dim`` and ``model.edge_in_dim`` (or the corresponding baseline
values) must both be 7.  ``paper_snapshot.head.load_bounds`` is mandatory; the
manuscript does not provide a universal numeric range that this factory could
safely invent.  DA-GWM additionally requires every field consumed by
``SnapshotDecisionAwareRankingLoss`` under ``paper_snapshot.decision_loss`` and
explicit ``paper_snapshot.score_weights``.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch.nn as nn

from ..models.tgn.tgn import TGN
from .baselines import build_paper_baseline
from .snapshot import (
    SnapshotDecisionAwareRankingLoss,
    SnapshotHead,
    SnapshotScoreWeights,
)
from .snapshot_data import SNAPSHOT_EDGE_FEATURE_NAMES, SNAPSHOT_NODE_FEATURE_NAMES


SNAPSHOT_METHODS = (
    "snapshot_mlp",
    "snapshot_physick",
    "snapshot_ltt_r",
    "snapshot_da_gwm",
)


def normalize_snapshot_method(value: str) -> str:
    method = str(value).strip().lower().replace("-", "_")
    aliases = {
        "snapshot_tgn_mlp": "snapshot_mlp",
        "snapshot_tgn_physick": "snapshot_physick",
        "ltt_r_snapshot": "snapshot_ltt_r",
        "da_gwm_snapshot": "snapshot_da_gwm",
    }
    method = aliases.get(method, method)
    if method not in SNAPSHOT_METHODS:
        raise ValueError(
            f"unknown Snapshot method {value!r}; expected {', '.join(SNAPSHOT_METHODS)}"
        )
    return method


def _mapping(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = root.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} mapping is required")
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _snapshot_head(
    root: Mapping[str, Any], *, embedding_dim: int, edge_in_dim: int
) -> SnapshotHead:
    snapshot = _mapping(root, "paper_snapshot")
    head_cfg = _mapping(snapshot, "head")
    bounds = head_cfg.get("load_bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("paper_snapshot.head.load_bounds must be an explicit pair")
    return SnapshotHead(
        in_dim=embedding_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=_positive_int(
            "paper_snapshot.head.hidden_dim",
            head_cfg.get("hidden_dim", embedding_dim),
        ),
        dropout=float(head_cfg.get("dropout", 0.0)),
        load_bounds=(float(bounds[0]), float(bounds[1])),
    )


def _require_snapshot_dims(node_in_dim: int, edge_in_dim: int) -> None:
    if node_in_dim != len(SNAPSHOT_NODE_FEATURE_NAMES):
        raise ValueError(
            f"Snapshot models require node_in_dim={len(SNAPSHOT_NODE_FEATURE_NAMES)}"
        )
    if edge_in_dim != len(SNAPSHOT_EDGE_FEATURE_NAMES):
        raise ValueError(
            f"Snapshot models require edge_in_dim={len(SNAPSHOT_EDGE_FEATURE_NAMES)}"
        )


def resolved_snapshot_model_config(
    cfg: Mapping[str, Any], method: str
) -> dict[str, Any]:
    """Return the exact method/interface config to persist in a checkpoint."""

    normalized = normalize_snapshot_method(method)
    resolved = copy.deepcopy(dict(cfg))
    resolved["paper_method"] = normalized
    resolved["paper_interface"] = "snapshot_v1"
    resolved["snapshot_feature_contract"] = {
        "node_feature_names": list(SNAPSHOT_NODE_FEATURE_NAMES),
        "edge_feature_names": list(SNAPSHOT_EDGE_FEATURE_NAMES),
        "forbidden_inputs": ["integrated_intensity", "raw_flow", "ema_flow"],
    }
    if normalized in {"snapshot_mlp", "snapshot_physick"}:
        model = _mapping(resolved, "model")
        if not isinstance(model, dict):
            # deepcopy(dict(cfg)) normally preserves a dict; make the error useful
            # for custom Mapping implementations.
            model = dict(model)
            resolved["model"] = model
        model["message_type"] = normalized.removeprefix("snapshot_")
        if normalized == "snapshot_physick":
            physick = model.setdefault("physick", {})
            if not isinstance(physick, dict):
                raise TypeError("model.physick must be a mapping")
            # ``paper`` selects the bounded signed kernel-bank implementation;
            # Snapshot semantics come exclusively from the isolated input/head.
            physick["implementation"] = "paper"
        return resolved

    base_kind = normalized.removeprefix("snapshot_")
    registry = _mapping(resolved, "paper_baselines")
    baseline = registry.get(base_kind)
    if not isinstance(baseline, Mapping):
        raise ValueError(f"paper_baselines.{base_kind} configuration is required")
    selected = copy.deepcopy(dict(baseline))
    selected["kind"] = base_kind
    resolved["paper_baseline"] = selected
    return resolved


def build_snapshot_model(cfg: Mapping[str, Any], method: str) -> nn.Module:
    """Build one isolated Snapshot predictor with a stable method name."""

    normalized = normalize_snapshot_method(method)
    resolved = resolved_snapshot_model_config(cfg, normalized)
    if normalized in {"snapshot_mlp", "snapshot_physick"}:
        model_cfg = _mapping(resolved, "model")
        node_in_dim = _positive_int("model.node_in_dim", model_cfg.get("node_in_dim"))
        edge_in_dim = _positive_int("model.edge_in_dim", model_cfg.get("edge_in_dim"))
        _require_snapshot_dims(node_in_dim, edge_in_dim)
        embedding_dim = _positive_int("model.emb_dim", model_cfg.get("emb_dim"))
        head = _snapshot_head(
            resolved, embedding_dim=embedding_dim, edge_in_dim=edge_in_dim
        )
        model = TGN(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            mem_dim=_positive_int("model.mem_dim", model_cfg.get("mem_dim")),
            msg_dim=_positive_int("model.msg_dim", model_cfg.get("msg_dim")),
            emb_dim=embedding_dim,
            message_type=str(model_cfg["message_type"]),
            aggregator=str(model_cfg.get("aggregator", "sum")),
            dropout=float(model_cfg.get("dropout", 0.0)),
            use_edge_type=bool(model_cfg.get("use_edge_type", True)),
            edge_type_vocab=_positive_int(
                "model.edge_type_vocab", model_cfg.get("edge_type_vocab", 8)
            ),
            head=head,
            edge_chunk=_positive_int(
                "train.edge_chunk",
                resolved.get("train", {}).get(
                    "edge_chunk", model_cfg.get("edge_chunk", 50000)
                ),
            ),
            cfg=resolved,
        )
    else:
        baseline_cfg = _mapping(resolved, "paper_baseline")
        node_in_dim = _positive_int(
            "paper_baseline.node_in_dim", baseline_cfg.get("node_in_dim")
        )
        edge_in_dim = _positive_int(
            "paper_baseline.edge_in_dim", baseline_cfg.get("edge_in_dim")
        )
        _require_snapshot_dims(node_in_dim, edge_in_dim)
        model = build_paper_baseline(resolved)
        embedding_dim = int(
            getattr(model, "model_dim", getattr(model, "hidden_dim", 0))
        )
        if embedding_dim <= 0:
            raise TypeError("Snapshot baseline does not expose its embedding dimension")
        model.head = _snapshot_head(
            resolved, embedding_dim=embedding_dim, edge_in_dim=edge_in_dim
        )
        if normalized == "snapshot_da_gwm":
            snapshot_cfg = _mapping(resolved, "paper_snapshot")
            decision = _mapping(snapshot_cfg, "decision_loss")
            policies = _mapping(snapshot_cfg, "policies")
            policy_cfg = policies.get(normalized)
            if not isinstance(policy_cfg, Mapping):
                raise ValueError(
                    f"paper_snapshot.policies.{normalized} configuration is required"
                )
            weights = SnapshotScoreWeights.from_mapping(
                _mapping(policy_cfg, "score_weights")
            )
            required = (
                "gamma_scale",
                "feasibility_scale",
                "load_scale",
                "soft_rank_temperature",
            )
            missing = [name for name in required if name not in decision]
            if missing:
                raise ValueError(
                    "paper_snapshot.decision_loss is missing: " + ", ".join(missing)
                )
            model.decision_loss = SnapshotDecisionAwareRankingLoss(
                weights=weights,
                gamma_scale=decision["gamma_scale"],
                feasibility_scale=decision["feasibility_scale"],
                load_scale=decision["load_scale"],
                soft_rank_temperature=decision["soft_rank_temperature"],
                pair_margin=decision.get("pair_margin", 0.0),
            )

    setattr(model, "cfg", resolved)
    setattr(model, "paper_method", normalized)
    setattr(model, "paper_interface", "snapshot_v1")
    return model


__all__ = [
    "SNAPSHOT_METHODS",
    "build_snapshot_model",
    "normalize_snapshot_method",
    "resolved_snapshot_model_config",
]
