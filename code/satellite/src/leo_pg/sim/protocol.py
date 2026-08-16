from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, Tuple


def validate_allocator_parameters(
    *,
    base_sinr: float,
    dist_scale: float,
    sinr_min: float,
    load_momentum: float,
) -> Tuple[float, float, float, float]:
    """Validate and normalize the scalar rate-greedy protocol parameters."""
    values = {
        "base_sinr": float(base_sinr),
        "dist_scale": float(dist_scale),
        "sinr_min": float(sinr_min),
        "load_momentum": float(load_momentum),
    }
    nonfinite = [name for name, value in values.items() if not math.isfinite(value)]
    if nonfinite:
        raise ValueError(
            "allocator protocol parameters must be finite: " + ", ".join(nonfinite)
        )
    if values["base_sinr"] < 0.0:
        raise ValueError("allocator base_sinr must be non-negative")
    if values["dist_scale"] < 0.0:
        raise ValueError("allocator dist_scale must be non-negative")
    if not 0.0 <= values["load_momentum"] <= 1.0:
        raise ValueError("allocator load_momentum must be in [0,1]")
    return (
        values["base_sinr"],
        values["dist_scale"],
        values["sinr_min"],
        values["load_momentum"],
    )


def validate_allocator_protocol(protocol: Any) -> Dict[str, float | str]:
    """Validate a serialized allocator protocol and return normalized values."""
    required = {"name", "base_sinr", "dist_scale", "sinr_min", "load_momentum"}
    if not isinstance(protocol, Mapping) or not required.issubset(protocol):
        raise ValueError("allocator_protocol is missing required fields")
    if protocol["name"] != "rate_greedy_v1":
        raise ValueError(f"unsupported allocator protocol: {protocol['name']!r}")
    base_sinr, dist_scale, sinr_min, load_momentum = validate_allocator_parameters(
        base_sinr=protocol["base_sinr"],
        dist_scale=protocol["dist_scale"],
        sinr_min=protocol["sinr_min"],
        load_momentum=protocol["load_momentum"],
    )
    return {
        "name": "rate_greedy_v1",
        "base_sinr": base_sinr,
        "dist_scale": dist_scale,
        "sinr_min": sinr_min,
        "load_momentum": load_momentum,
    }
