"""Strict YAML resolver and factory for the isolated UAV/shared-service platform."""

from __future__ import annotations

from dataclasses import dataclass, fields
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from .environment import UAVSharedServiceEnv
from .protocol import (
    EstimatedDynamicsParameters,
    ManuscriptExactParameters,
    UAV_SHARED_PROTOCOL_VERSION,
    UAVSharedProtocol,
)
from .runtime import load_cfg


_SECTION_KEYS = {
    "protocol_version",
    "episode_seed",
    "density_multiplier",
    "capacity_compression",
    "exact",
    "dynamics",
    "provenance",
}
_TUPLE_EXACT_FIELDS = {
    "density_multipliers",
    "capacity_compressions",
    "eta_clip",
    "station_flow_clip",
    "intensity_clip",
}


@dataclass(frozen=True)
class ResolvedUAVSharedConfig:
    """Validated platform configuration with its YAML-level provenance."""

    protocol: UAVSharedProtocol
    provenance: Dict[str, Dict[str, str]]
    source_path: str | None = None

    def manifest(self) -> Dict[str, Any]:
        manifest = self.protocol.manifest()
        manifest["configuration"] = {
            "source_path": self.source_path,
            "provenance": {
                group: dict(values) for group, values in self.provenance.items()
            },
        }
        return manifest


ConfigSource = (
    ResolvedUAVSharedConfig
    | Mapping[str, Any]
    | str
    | PathLike[str]
)


def _as_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    non_string_keys = [key for key in value if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(f"{name} keys must be strings: {non_string_keys!r}")
    return value


def _require_exact_keys(
    name: str,
    value: Mapping[str, Any],
    expected: set[str],
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{name} schema mismatch; missing={missing!r}, unknown={unknown!r}"
        )


def _strict_int(name: str, value: Any) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (not bool, float, or string)")
    return value


def _read_source(source: ConfigSource) -> tuple[Mapping[str, Any], str | None]:
    if isinstance(source, Mapping):
        return _as_mapping("configuration", source), None
    if isinstance(source, (str, PathLike)):
        path = Path(source).expanduser().resolve()
        loaded = load_cfg(path)
        return _as_mapping("YAML document", loaded), str(path)
    raise TypeError("config source must be a mapping or YAML path")


def _resolve_provenance(value: Any) -> Dict[str, Dict[str, str]]:
    provenance = _as_mapping("uav_shared.provenance", value)
    _require_exact_keys(
        "uav_shared.provenance", provenance, {"exact", "dynamics"}
    )
    expected_status = {"exact": "manuscript_exact"}
    resolved: Dict[str, Dict[str, str]] = {}
    for group, status in expected_status.items():
        entry = _as_mapping(f"uav_shared.provenance.{group}", provenance[group])
        _require_exact_keys(
            f"uav_shared.provenance.{group}", entry, {"status", "source"}
        )
        if entry["status"] != status:
            raise ValueError(
                f"uav_shared.provenance.{group}.status must be {status!r}"
            )
        source = entry["source"]
        if not isinstance(source, str) or not source.strip():
            raise TypeError(
                f"uav_shared.provenance.{group}.source must be a non-empty string"
            )
        resolved[group] = {"status": status, "source": source.strip()}
    dynamics = _as_mapping("uav_shared.provenance.dynamics", provenance["dynamics"])
    _require_exact_keys(
        "uav_shared.provenance.dynamics", dynamics, {"status", "source"}
    )
    dynamics_status = dynamics["status"]
    if dynamics_status not in {"configured", "author_supplied"}:
        raise ValueError(
            "uav_shared.provenance.dynamics.status must be configured or "
            "author_supplied"
        )
    dynamics_source = dynamics["source"]
    if not isinstance(dynamics_source, str) or not dynamics_source.strip():
        raise TypeError(
            "uav_shared.provenance.dynamics.source must be a non-empty string"
        )
    resolved["dynamics"] = {
        "status": str(dynamics_status),
        "source": dynamics_source.strip(),
    }
    return resolved


def resolve_uav_shared_config(source: ConfigSource) -> ResolvedUAVSharedConfig:
    """Resolve either the repository YAML or an extracted ``uav_shared`` block.

    Unlike permissive application configuration loaders, this resolver requires
    every protocol field and rejects unknown keys.  This prevents a manuscript
    run from silently falling back to a changed default.
    """

    if isinstance(source, ResolvedUAVSharedConfig):
        return source
    root, source_path = _read_source(source)
    raw_section = root["uav_shared"] if "uav_shared" in root else root
    section = _as_mapping("uav_shared", raw_section)
    _require_exact_keys("uav_shared", section, _SECTION_KEYS)

    protocol_version = _strict_int(
        "uav_shared.protocol_version", section["protocol_version"]
    )
    if protocol_version != UAV_SHARED_PROTOCOL_VERSION:
        raise ValueError(
            "uav_shared.protocol_version mismatch: "
            f"{protocol_version} != {UAV_SHARED_PROTOCOL_VERSION}"
        )

    exact_mapping = _as_mapping("uav_shared.exact", section["exact"])
    exact_field_names = {parameter.name for parameter in fields(ManuscriptExactParameters)}
    _require_exact_keys("uav_shared.exact", exact_mapping, exact_field_names)
    exact_kwargs = dict(exact_mapping)
    for name in _TUPLE_EXACT_FIELDS:
        value = exact_kwargs[name]
        if type(value) not in (list, tuple):
            raise TypeError(f"uav_shared.exact.{name} must be a YAML sequence")
        exact_kwargs[name] = tuple(value)
    exact = ManuscriptExactParameters(**exact_kwargs)

    estimated_mapping = _as_mapping("uav_shared.dynamics", section["dynamics"])
    estimated_field_names = {
        parameter.name for parameter in fields(EstimatedDynamicsParameters)
    }
    _require_exact_keys(
        "uav_shared.dynamics", estimated_mapping, estimated_field_names
    )
    estimated = EstimatedDynamicsParameters(**dict(estimated_mapping))

    protocol = UAVSharedProtocol(
        episode_seed=section["episode_seed"],
        density_multiplier=section["density_multiplier"],
        capacity_compression=section["capacity_compression"],
        exact=exact,
        estimated=estimated,
    )
    return ResolvedUAVSharedConfig(
        protocol=protocol,
        provenance=_resolve_provenance(section["provenance"]),
        source_path=source_path,
    )


def resolve_uav_shared_protocol(source: ConfigSource) -> UAVSharedProtocol:
    """Return only the validated runtime protocol."""

    return resolve_uav_shared_config(source).protocol


def build_uav_environment(
    source: ConfigSource,
    *,
    device: torch.device | str = "cpu",
) -> UAVSharedServiceEnv:
    """Build the isolated environment from a validated mapping or YAML file."""

    return UAVSharedServiceEnv(
        resolve_uav_shared_protocol(source),
        device=device,
    )


__all__ = [
    "ConfigSource",
    "ResolvedUAVSharedConfig",
    "build_uav_environment",
    "resolve_uav_shared_config",
    "resolve_uav_shared_protocol",
]
