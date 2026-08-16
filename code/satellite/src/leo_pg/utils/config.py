from __future__ import annotations

from pathlib import Path

import yaml


def deep_update(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_cfg(path: str, _seen: tuple[Path, ...] = ()) -> dict:
    """Load YAML, optionally inheriting files through a top-level ``base`` key."""
    config_path = Path(path).expanduser().resolve()
    if config_path in _seen:
        chain = " -> ".join(str(value) for value in (*_seen, config_path))
        raise ValueError(f"Cyclic config inheritance: {chain}")
    seen = (*_seen, config_path)

    with config_path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    if not isinstance(current, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    bases = current.pop("base", None)
    if bases is None:
        return current
    if isinstance(bases, (str, Path)):
        bases = [bases]
    if not isinstance(bases, list):
        raise ValueError("Config 'base' must be a path or list of paths")

    merged: dict = {}
    for base in bases:
        base_path = (config_path.parent / str(base)).resolve()
        deep_update(merged, load_cfg(str(base_path), _seen=seen))
    return deep_update(merged, current)
