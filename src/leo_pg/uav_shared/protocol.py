from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from numbers import Real
from typing import Any, Dict, Mapping


UAV_SHARED_PROTOCOL_VERSION = 1
UAV_SHARED_SNAPSHOT_SCHEMA_VERSION = 1


_EXACT_REFERENCE = (
    "main_final.tex:824-861; "
    "si_final.tex:2923-2985,3482-3523"
)


def _strict_int(
    name: str,
    value: int,
    *,
    non_negative: bool = False,
    positive: bool = False,
) -> int:
    """Validate an integer without accepting floats or numeric strings."""

    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (not bool, float, or string)")
    if non_negative and value < 0:
        raise ValueError(f"{name} must be non-negative")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    """Validate and normalize a real-valued field to a Python float."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number (not bool or string)")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class ManuscriptExactParameters:
    """Second-platform values explicitly reported in the manuscript/SI.

    The class deliberately rejects changed values.  Stress selections are made
    in :class:`UAVSharedProtocol`; the paper-reported axes themselves remain
    immutable and auditable.
    """

    control_dt_s: float = 1.0
    evaluation_horizon_steps: int = 120
    nominal_uav_count: int = 24
    station_count: int = 6
    density_multipliers: tuple[int, ...] = (1, 2, 3, 4, 5)
    capacity_compressions: tuple[float, ...] = (1.0, 0.75, 0.5)
    candidate_topk: int = 3
    active_slots_per_station: int = 2
    fifo_queue_capacity: int = 8
    feasible_start_reserve_fraction: float = 0.15
    intensity_lookahead_s: float = 12.0
    eta_clip: tuple[float, float] = (0.0, 1.0)
    station_flow_clip: tuple[float, float] = (0.0, 1.0)
    intensity_clip: tuple[float, float] = (0.0, 3.0)

    def __post_init__(self) -> None:
        for name in (
            "evaluation_horizon_steps",
            "nominal_uav_count",
            "station_count",
            "candidate_topk",
            "active_slots_per_station",
            "fifo_queue_capacity",
        ):
            _strict_int(name, getattr(self, name), positive=True)

        for name in (
            "control_dt_s",
            "feasible_start_reserve_fraction",
            "intensity_lookahead_s",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

        if type(self.density_multipliers) is not tuple:
            raise TypeError("density_multipliers must be a tuple of integers")
        density_multipliers = tuple(
            _strict_int(f"density_multipliers[{index}]", value, positive=True)
            for index, value in enumerate(self.density_multipliers)
        )
        object.__setattr__(self, "density_multipliers", density_multipliers)

        for name in (
            "capacity_compressions",
            "eta_clip",
            "station_flow_clip",
            "intensity_clip",
        ):
            values = getattr(self, name)
            if type(values) is not tuple:
                raise TypeError(f"{name} must be a tuple of real numbers")
            normalized = tuple(
                _finite(f"{name}[{index}]", value)
                for index, value in enumerate(values)
            )
            object.__setattr__(self, name, normalized)

        expected = {
            "control_dt_s": 1.0,
            "evaluation_horizon_steps": 120,
            "nominal_uav_count": 24,
            "station_count": 6,
            "density_multipliers": (1, 2, 3, 4, 5),
            "capacity_compressions": (1.0, 0.75, 0.5),
            "candidate_topk": 3,
            "active_slots_per_station": 2,
            "fifo_queue_capacity": 8,
            "feasible_start_reserve_fraction": 0.15,
            "intensity_lookahead_s": 12.0,
            "eta_clip": (0.0, 1.0),
            "station_flow_clip": (0.0, 1.0),
            "intensity_clip": (0.0, 3.0),
        }
        for name, expected_value in expected.items():
            if getattr(self, name) != expected_value:
                raise ValueError(
                    f"{name} is manuscript-exact and must remain {expected_value!r}"
                )


@dataclass(frozen=True)
class EstimatedDynamicsParameters:
    """Explicit placeholders for quantities not numerically fixed by the paper.

    These values make the environment executable; they are *not* represented
    as recovered experimental settings.  Every item appears under
    ``explicit_estimates`` in :meth:`UAVSharedProtocol.manifest`.
    """

    region_width_m: float = 1_000.0
    region_height_m: float = 1_000.0
    station_boundary_margin_fraction: float = 0.10
    max_speed_m_s: float = 25.0
    max_turn_rate_rad_s: float = math.pi / 4.0
    heading_jitter_rad: float = math.pi / 90.0
    docking_radius_m: float = 12.0
    maximum_reachable_distance_m: float = 800.0
    nominal_battery_units: float = 100.0
    initial_energy_fraction_min: float = 0.65
    initial_energy_fraction_max: float = 1.0
    travel_energy_units_per_m: float = 0.02
    idle_energy_units_per_s: float = 0.03
    service_energy_units_per_s: float = 1.5
    nominal_service_duration_s: float = 8.0
    service_duration_jitter_s: float = 2.0
    flow_memory: float = 0.85
    flow_queue_weight: float = 0.45
    flow_occupancy_weight: float = 0.35
    flow_inbound_weight: float = 0.20
    eta_travel_weight: float = 0.50
    eta_arrival_energy_weight: float = 0.50
    intensity_start_eta_weight: float = 0.45
    intensity_energy_margin_weight: float = 0.30
    intensity_station_flow_weight: float = 0.25

    def __post_init__(self) -> None:
        for parameter in fields(self):
            object.__setattr__(
                self,
                parameter.name,
                _finite(parameter.name, getattr(self, parameter.name)),
            )

        for name in (
            "region_width_m",
            "region_height_m",
            "max_speed_m_s",
            "max_turn_rate_rad_s",
            "docking_radius_m",
            "maximum_reachable_distance_m",
            "nominal_battery_units",
            "nominal_service_duration_s",
        ):
            _finite(name, getattr(self, name), positive=True)
        for name in (
            "heading_jitter_rad",
            "travel_energy_units_per_m",
            "idle_energy_units_per_s",
            "service_energy_units_per_s",
            "service_duration_jitter_s",
        ):
            value = _finite(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.service_duration_jitter_s >= self.nominal_service_duration_s:
            raise ValueError(
                "service_duration_jitter_s must be smaller than nominal_service_duration_s"
            )
        for name in (
            "station_boundary_margin_fraction",
            "initial_energy_fraction_min",
            "initial_energy_fraction_max",
            "flow_memory",
        ):
            value = _finite(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.initial_energy_fraction_min > self.initial_energy_fraction_max:
            raise ValueError(
                "initial_energy_fraction_min cannot exceed initial_energy_fraction_max"
            )
        for prefix, names in (
            (
                "flow",
                (
                    "flow_queue_weight",
                    "flow_occupancy_weight",
                    "flow_inbound_weight",
                ),
            ),
            (
                "eta",
                ("eta_travel_weight", "eta_arrival_energy_weight"),
            ),
            (
                "intensity",
                (
                    "intensity_start_eta_weight",
                    "intensity_energy_margin_weight",
                    "intensity_station_flow_weight",
                ),
            ),
        ):
            weights = [_finite(name, getattr(self, name)) for name in names]
            if any(weight < 0.0 for weight in weights):
                raise ValueError(f"{prefix} weights must be non-negative")
            if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{prefix} weights must sum to one")


_EXACT_DESCRIPTIONS: Mapping[str, str] = {
    "control_dt_s": "One-second control timeline.",
    "evaluation_horizon_steps": "Descriptor-autoregressive evaluation horizon.",
    "nominal_uav_count": "Nominal number of UAV agents.",
    "station_count": "Nominal number of fixed service stations.",
    "density_multipliers": "Reported UAV-density stress axis.",
    "capacity_compressions": "Reported service-capacity stress axis.",
    "candidate_topk": "Reachable candidates truncated to Top-k=3.",
    "active_slots_per_station": "Active service slots at each station.",
    "fifo_queue_capacity": "FIFO waiting capacity, excluding active slots.",
    "feasible_start_reserve_fraction": "Arrival reserve threshold.",
    "intensity_lookahead_s": "Frozen-decision feasible-start look-ahead.",
    "eta_clip": "Reported min-max normalization/clipping range.",
    "station_flow_clip": "Reported min-max normalization/clipping range.",
    "intensity_clip": "Reported Lambda_max=3 clipping range.",
}


_ESTIMATE_DESCRIPTIONS: Mapping[str, str] = {
    "region_width_m": "Operational-region width; not reported.",
    "region_height_m": "Operational-region height; not reported.",
    "station_boundary_margin_fraction": "Seeded station-layout margin; not reported.",
    "max_speed_m_s": "Bounded UAV speed value; only boundedness is reported.",
    "max_turn_rate_rad_s": "Bounded turn-rate value; only boundedness is reported.",
    "heading_jitter_rad": "Magnitude of keyed smooth-motion perturbation; not reported.",
    "docking_radius_m": "Station arrival/docking radius; not reported.",
    "maximum_reachable_distance_m": "Travel-distance reachability limit; not reported.",
    "nominal_battery_units": "Normalized finite battery budget; scale not reported.",
    "initial_energy_fraction_min": "Seeded initial-energy lower bound; not reported.",
    "initial_energy_fraction_max": "Seeded initial-energy upper bound; not reported.",
    "travel_energy_units_per_m": "Travel energy law coefficient; not reported.",
    "idle_energy_units_per_s": "Queue/idle energy law coefficient; not reported.",
    "service_energy_units_per_s": "Charging/service energy gain; not reported.",
    "nominal_service_duration_s": "Nominal slot occupation duration; not reported.",
    "service_duration_jitter_s": "Keyed service-duration spread; not reported.",
    "flow_memory": "Slow station-flow EMA memory; not reported.",
    "flow_queue_weight": "Queue contribution to station flow; not reported.",
    "flow_occupancy_weight": "Slot-occupancy contribution to station flow; not reported.",
    "flow_inbound_weight": "Inbound-target contribution to station flow; not reported.",
    "eta_travel_weight": "Travel component of local utility; not reported.",
    "eta_arrival_energy_weight": "Arrival-energy component of local utility; not reported.",
    "intensity_start_eta_weight": "Service-start-delay proxy weight; not reported.",
    "intensity_energy_margin_weight": "Energy-margin proxy weight; not reported.",
    "intensity_station_flow_weight": "Congestion proxy weight; not reported.",
}


@dataclass(frozen=True)
class UAVSharedProtocol:
    """Complete, fingerprinted environment protocol for the second platform."""

    episode_seed: int = 7
    density_multiplier: int = 1
    capacity_compression: float = 1.0
    exact: ManuscriptExactParameters = field(
        default_factory=ManuscriptExactParameters
    )
    estimated: EstimatedDynamicsParameters = field(
        default_factory=EstimatedDynamicsParameters
    )

    def __post_init__(self) -> None:
        if not isinstance(self.exact, ManuscriptExactParameters):
            raise TypeError("exact must be a ManuscriptExactParameters instance")
        if not isinstance(self.estimated, EstimatedDynamicsParameters):
            raise TypeError("estimated must be an EstimatedDynamicsParameters instance")
        object.__setattr__(
            self,
            "episode_seed",
            _strict_int("episode_seed", self.episode_seed, non_negative=True),
        )
        object.__setattr__(
            self,
            "density_multiplier",
            _strict_int("density_multiplier", self.density_multiplier, positive=True),
        )
        object.__setattr__(
            self,
            "capacity_compression",
            _finite("capacity_compression", self.capacity_compression, positive=True),
        )
        if self.density_multiplier not in self.exact.density_multipliers:
            raise ValueError(
                "density_multiplier must be one of the manuscript stress values "
                f"{self.exact.density_multipliers}"
            )
        if self.capacity_compression not in self.exact.capacity_compressions:
            raise ValueError(
                "capacity_compression must be one of the manuscript stress values "
                f"{self.exact.capacity_compressions}"
            )

    @property
    def uav_count(self) -> int:
        return self.exact.nominal_uav_count * self.density_multiplier

    @property
    def station_count(self) -> int:
        return self.exact.station_count

    @property
    def horizon_steps(self) -> int:
        return self.exact.evaluation_horizon_steps

    def _fingerprint_payload(self) -> Dict[str, Any]:
        return {
            "protocol_version": UAV_SHARED_PROTOCOL_VERSION,
            "episode_seed": self.episode_seed,
            "density_multiplier": self.density_multiplier,
            "capacity_compression": self.capacity_compression,
            "exact": asdict(self.exact),
            "estimated": asdict(self.estimated),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._fingerprint_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def manifest(self) -> Dict[str, Any]:
        exact_values = asdict(self.exact)
        estimate_values = asdict(self.estimated)
        return {
            "schema_version": 1,
            "protocol_version": UAV_SHARED_PROTOCOL_VERSION,
            "protocol_fingerprint": self.fingerprint,
            "selected_condition": {
                "episode_seed": self.episode_seed,
                "density_multiplier": self.density_multiplier,
                "capacity_compression": self.capacity_compression,
                "uav_count": self.uav_count,
            },
            "manuscript_exact": {
                name: {
                    "value": value,
                    "status": "manuscript_exact",
                    "source": _EXACT_REFERENCE,
                    "description": _EXACT_DESCRIPTIONS[name],
                }
                for name, value in exact_values.items()
            },
            "explicit_estimates": {
                name: {
                    "value": value,
                    "status": "explicit_estimate",
                    "source": "Not numerically specified in main_final.tex or si_final.tex",
                    "description": _ESTIMATE_DESCRIPTIONS[name],
                }
                for name, value in estimate_values.items()
            },
        }
