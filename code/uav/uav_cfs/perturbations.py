"""Explicit structural perturbations for zero-shot UAV evaluation.

The current manuscript names four mismatch families but does not publish their
numeric magnitudes.  This module implements each transition-law change while
requiring the missing magnitude or rule to be supplied by the author config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping


PERTURBATION_KINDS = (
    "nominal",
    "service_time_inflation",
    "queue_law_swap",
    "station_memory_shift",
    "threshold_tightening",
)
QUEUE_LAWS = ("fifo", "lowest_energy_first", "highest_energy_first")


@dataclass(frozen=True)
class UAVStructuralPerturbation:
    """One fully specified simulator-side evaluation perturbation."""

    kind: str = "nominal"
    service_time_multiplier: float = 1.0
    queue_law: str = "fifo"
    station_flow_memory: float | None = None
    feasible_start_reserve_fraction: float | None = None
    author_provenance: str = "nominal protocol"

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in PERTURBATION_KINDS:
            raise ValueError(f"kind must be one of {PERTURBATION_KINDS}")
        object.__setattr__(self, "kind", kind)
        multiplier = float(self.service_time_multiplier)
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("service_time_multiplier must be finite and positive")
        object.__setattr__(self, "service_time_multiplier", multiplier)
        queue_law = str(self.queue_law).strip().lower()
        if queue_law not in QUEUE_LAWS:
            raise ValueError(f"queue_law must be one of {QUEUE_LAWS}")
        object.__setattr__(self, "queue_law", queue_law)
        if self.station_flow_memory is not None:
            memory = float(self.station_flow_memory)
            if not math.isfinite(memory) or not 0.0 <= memory <= 1.0:
                raise ValueError("station_flow_memory must lie in [0,1]")
            object.__setattr__(self, "station_flow_memory", memory)
        if self.feasible_start_reserve_fraction is not None:
            reserve = float(self.feasible_start_reserve_fraction)
            if not math.isfinite(reserve) or not 0.0 <= reserve < 1.0:
                raise ValueError(
                    "feasible_start_reserve_fraction must lie in [0,1)"
                )
            object.__setattr__(self, "feasible_start_reserve_fraction", reserve)
        provenance = str(self.author_provenance).strip()
        if not provenance:
            raise ValueError("author_provenance must be a non-empty string")
        object.__setattr__(self, "author_provenance", provenance)
        active = {
            "service_time_inflation": multiplier > 1.0,
            "queue_law_swap": queue_law != "fifo",
            "station_memory_shift": self.station_flow_memory is not None,
            "threshold_tightening": (
                self.feasible_start_reserve_fraction is not None
            ),
        }
        if kind == "nominal" and any(active.values()):
            raise ValueError("nominal perturbation cannot change simulator laws")
        if kind != "nominal" and not active[kind]:
            raise ValueError(f"{kind} is missing its author-supplied setting")
        if kind == "service_time_inflation" and multiplier <= 1.0:
            raise ValueError("service-time inflation requires a multiplier > 1")
        for name, enabled in active.items():
            if name != kind and enabled:
                raise ValueError(
                    f"{kind} must not also activate the {name} perturbation"
                )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self), "fingerprint": self.fingerprint}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "UAVStructuralPerturbation":
        """Resolve a strict mapping; no manuscript-missing magnitude is defaulted."""

        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("structural_perturbation must be a mapping")
        raw = dict(value)
        allowed = {
            "kind",
            "service_time_multiplier",
            "queue_law",
            "station_flow_memory",
            "feasible_start_reserve_fraction",
            "author_provenance",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown structural perturbation keys: {unknown}")
        if "kind" not in raw or "author_provenance" not in raw:
            raise ValueError(
                "structural_perturbation requires kind and author_provenance"
            )
        kind = str(raw["kind"]).strip().lower()
        required = {
            "service_time_inflation": "service_time_multiplier",
            "queue_law_swap": "queue_law",
            "station_memory_shift": "station_flow_memory",
            "threshold_tightening": "feasible_start_reserve_fraction",
        }
        if kind in required and required[kind] not in raw:
            raise ValueError(
                f"{kind} requires author-supplied {required[kind]}"
            )
        return cls(**raw)


__all__ = [
    "PERTURBATION_KINDS",
    "QUEUE_LAWS",
    "UAVStructuralPerturbation",
]
