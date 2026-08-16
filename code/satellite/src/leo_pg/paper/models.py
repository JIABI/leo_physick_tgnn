"""Unified model registry for the paper protocol and controlled baselines."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch.nn as nn

from leo_pg.models.registry import build_model
from leo_pg.paper.baselines import build_paper_baseline
from leo_pg.paper.snapshot_models import (
    SNAPSHOT_METHODS,
    build_snapshot_model,
    resolved_snapshot_model_config,
)


PAPER_METHODS = (
    "tgn_mlp",
    "tgn_kan",
    "tgn_physick",
    "ltt_r",
    "da_gwm",
    "s4",
    "mamba2",
    *SNAPSHOT_METHODS,
)

# Methods named in manuscript v8.  S4 and Mamba2 are registered only as
# exploratory temporal-backbone ablations.  Earlier workspace prototypes also
# contained Conformer, BigMLP, EdgeAttn and TempTrans implementations; those
# unreported systems are deliberately absent from this publication registry.
EXPLORATORY_METHODS = ("s4", "mamba2")


def normalize_paper_method(value: str) -> str:
    method = str(value).strip().lower().replace("-", "_")
    aliases = {
        "mlp": "tgn_mlp",
        "kan": "tgn_kan",
        "physick": "tgn_physick",
        "lttr": "ltt_r",
        "ltt_r": "ltt_r",
        "dagwm": "da_gwm",
        "da_gwm": "da_gwm",
        "snapshot_tgn_mlp": "snapshot_mlp",
        "snapshot_tgn_physick": "snapshot_physick",
        "ltt_r_snapshot": "snapshot_ltt_r",
        "da_gwm_snapshot": "snapshot_da_gwm",
    }
    method = aliases.get(method, method)
    if method not in PAPER_METHODS:
        raise ValueError(
            f"unknown paper method {value!r}; expected {', '.join(PAPER_METHODS)}"
        )
    return method


def resolved_model_config(
    cfg: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    """Create the method-specific frozen config saved with a checkpoint."""

    method = normalize_paper_method(method)
    if method in SNAPSHOT_METHODS:
        return resolved_snapshot_model_config(cfg, method)
    resolved = copy.deepcopy(dict(cfg))
    resolved["paper_method"] = method
    if method.startswith("tgn_"):
        model = resolved.get("model")
        if not isinstance(model, dict):
            raise ValueError("TGN paper methods require a model mapping")
        model["message_type"] = method.removeprefix("tgn_")
        if method == "tgn_physick":
            physick = model.setdefault("physick", {})
            if not isinstance(physick, dict):
                raise TypeError("model.physick must be a mapping")
            physick.setdefault("implementation", "paper_intensity_flow")
        return resolved

    registry = resolved.get("paper_baselines")
    if not isinstance(registry, Mapping) or method not in registry:
        raise ValueError(f"paper_baselines.{method} configuration is required")
    baseline = copy.deepcopy(dict(registry[method]))
    baseline["kind"] = method
    resolved["paper_baseline"] = baseline
    return resolved


def build_paper_model(cfg: Mapping[str, Any], method: str) -> nn.Module:
    method = normalize_paper_method(method)
    if method in SNAPSHOT_METHODS:
        model = build_snapshot_model(cfg, method)
        setattr(model, "paper_method", method)
        return model
    resolved = resolved_model_config(cfg, method)
    if method.startswith("tgn_"):
        model = build_model(resolved)
    else:
        model = build_paper_baseline(resolved)
    setattr(model, "cfg", resolved)
    setattr(model, "paper_method", method)
    return model


def parameter_count(model: nn.Module, *, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


__all__ = [
    "PAPER_METHODS",
    "EXPLORATORY_METHODS",
    "build_paper_model",
    "normalize_paper_method",
    "parameter_count",
    "resolved_model_config",
]
