"""Small runtime helpers kept local to the release package."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_cfg(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve an optional explicit ``extends`` chain."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"configuration must be a YAML mapping: {source}")
    current = dict(value)
    parent_value = current.pop("extends", None)
    if parent_value is None:
        return current
    if not isinstance(parent_value, str) or not parent_value.strip():
        raise TypeError("extends must be a non-empty relative YAML path")
    parent = load_cfg((source.parent / parent_value).resolve())
    return _deep_merge(parent, current)


def get_device(name: str, *, strict: bool = False) -> torch.device:
    """Resolve ``cpu`` or ``cuda`` and optionally reject unavailable CUDA."""

    requested = str(name).strip().lower()
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        if strict:
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed model initialization, optimizer-side randomness and data order."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


__all__ = ["get_device", "load_cfg", "set_seed"]
