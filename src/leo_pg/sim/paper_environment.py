from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from ..graph.candidates import (
    ElevationCandidates,
    build_elevation_candidates,
    elevation_for_edges,
)
from .ephemeris import (
    EphemerisState,
    HybridUserSatEphemeris,
    SimpleKinematicEphemeris,
    load_tle_file,
)
from .intensity_flow import (
    feasibility_gate,
    integrated_violation_intensity,
    mean_flow_update,
    nearest_rank_p10,
)
from .randomness import keyed_standard_normal, keyed_user_order
from .state import (
    ControlObservation,
    ExecutionResult,
    FailureReason,
    PAPER_EDGE_FEATURE_NAMES,
    PAPER_FEATURE_CONTRACT_VERSION,
    PAPER_NODE_FEATURE_NAMES,
    PolicyDescriptors,
    ServingAction,
    SimulatorDescriptors,
)


PAPER_PROTOCOL_VERSION = 1
PROTOCOL_FINGERPRINT_SCHEMA_VERSION = 2
CALLABLE_PROTOCOL_FINGERPRINT_ATTRIBUTE = "__protocol_fingerprint__"


def _callable_protocol_identity(value: Any, *, path: str) -> Dict[str, str]:
    """Return a stable, author-supplied identity for a configured callable.

    A callable's module and qualified name are not enough to distinguish two
    closures with different captured state.  Source/bytecode hashing is also
    insufficient for that case and can vary across Python builds.  Requiring a
    stable semantic version keeps the run identity explicit and portable.
    """

    fingerprint = getattr(value, CALLABLE_PROTOCOL_FINGERPRINT_ATTRIBUTE, None)
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError(
            f"{path} is callable and must define a non-empty "
            f"{CALLABLE_PROTOCOL_FINGERPRINT_ATTRIBUTE!r} string"
        )
    return {
        "callable": (
            f"{getattr(value, '__module__', '')}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        ),
        "protocol_fingerprint": fingerprint.strip(),
    }


def _canonical_config(value: Any, *, path: str = "config") -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_config(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_config(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return _callable_protocol_identity(value, path=path)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _file_content_identity(path: Path, *, config_path: str) -> Dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{config_path} does not exist and cannot be protocol-fingerprinted: {path}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(
            f"{config_path} must refer to a regular file for protocol fingerprinting: "
            f"{resolved}"
        )

    digest = hashlib.sha256()
    size_bytes = 0
    with resolved.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size_bytes += len(block)
    return {
        "resolved_path": str(resolved),
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _protocol_file_inputs(cfg: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Collect content identities for files that define simulator semantics."""

    ephemeris = cfg.get("ephemeris", {})
    if not isinstance(ephemeris, Mapping):
        return {}
    if str(ephemeris.get("mode", "debug")).lower() != "skyfield_tle":
        return {}

    raw_tle_path = ephemeris.get("tle_path")
    if raw_tle_path is None or not str(raw_tle_path).strip():
        return {}
    return {
        "ephemeris.tle_path": _file_content_identity(
            Path(str(raw_tle_path)), config_path="ephemeris.tle_path"
        )
    }


def protocol_fingerprint(cfg: Mapping[str, Any]) -> str:
    """Hash normalized config plus stable callable and file-content identities.

    Configured callables must expose a non-empty ``__protocol_fingerprint__``
    string.  File-backed TLE ephemerides are bound to the input bytes as well as
    their configured path, so in-place file edits produce a new run identity.
    """

    payload = {
        "paper_protocol_version": PAPER_PROTOCOL_VERSION,
        "fingerprint_schema_version": PROTOCOL_FINGERPRINT_SCHEMA_VERSION,
        "config": _canonical_config(cfg),
        "file_inputs": _protocol_file_inputs(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mapping(parent: Mapping[str, Any], name: str) -> Dict[str, Any]:
    value = parent.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _parse_utc(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    timestamp = datetime.fromisoformat(text)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _sphere_points(
    count: int,
    radius: float,
    seed: int,
    stream: str,
    device: torch.device,
) -> torch.Tensor:
    values = keyed_standard_normal((count, 3), seed, 0, stream, device=device)
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(1e-12) * radius


def _tangent_velocities(
    positions: torch.Tensor,
    speed: float,
    seed: int,
    stream: str,
) -> torch.Tensor:
    raw = keyed_standard_normal(
        tuple(positions.shape),
        seed,
        0,
        stream,
        dtype=positions.dtype,
        device=positions.device,
    )
    radial = positions / torch.linalg.vector_norm(positions, dim=1, keepdim=True).clamp_min(1e-12)
    tangent = raw - (raw * radial).sum(dim=1, keepdim=True) * radial
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=1, keepdim=True).clamp_min(1e-12)
    return tangent * speed


class PaperAlignedLEOEnv:
    """Action-coupled Intensity--Flow simulator implementing the paper protocol.

    The environment never chooses an association internally. ``observe``
    creates simulator-authoritative descriptors; a controller returns a
    :class:`ServingAction`; and ``step_action`` alone mutates load, serving
    history, and time. Policy descriptor replacement therefore affects future
    state only through the requested and executed action.

    Configuration lives under ``paper_protocol``. Quantities that would change
    the executed protocol without being uniquely fixed by the manuscript are
    mandatory: ``flow.phi``, ``flow.clip_upper``,
    ``intensity.baseline_hazard``, ``intensity.gamma_covariate_scale_db``,
    ``intensity.elevation_covariate_scale_deg``, and
    ``policy.hard_feasibility_mask``.
    """

    def __init__(self, cfg: Mapping[str, Any], device: torch.device | str = "cpu"):
        self.cfg = dict(cfg)
        self.protocol_fingerprint = protocol_fingerprint(cfg)
        self.device = torch.device(device)
        protocol = _mapping(cfg, "paper_protocol")
        if not protocol:
            raise ValueError("paper_protocol configuration is required")
        self.protocol_cfg = protocol
        self.seed = int(cfg.get("seed", protocol.get("seed", 7)))
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        self.horizon_steps = _positive_int(
            "paper_protocol.horizon_steps",
            protocol.get("horizon_steps", cfg.get("T", 600)),
        )
        self.dt_ctrl = _finite(
            "paper_protocol.dt_ctrl",
            protocol.get("dt_ctrl", 0.1),
            positive=True,
        )
        self.dt_phy = _finite(
            "paper_protocol.dt_phy",
            protocol.get("dt_phy", 0.001),
            positive=True,
        )
        expected_samples = self.dt_ctrl / self.dt_phy
        if not math.isclose(expected_samples, round(expected_samples), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("dt_ctrl / dt_phy must be an integer")
        self.phy_samples = int(round(expected_samples))

        candidate_cfg = _mapping(protocol, "candidates")
        self.minimum_elevation_deg = _finite(
            "candidates.minimum_elevation_deg",
            candidate_cfg.get("minimum_elevation_deg", 10.0),
        )
        self.topk = _positive_int("candidates.topk", candidate_cfg.get("topk", 6))

        feasibility_cfg = _mapping(protocol, "feasibility")
        self.gamma_min_db = _finite(
            "feasibility.gamma_min_db", feasibility_cfg.get("gamma_min_db", -5.0)
        )
        self.flow_max = _finite("feasibility.flow_max", feasibility_cfg.get("flow_max", 0.95))
        if self.flow_max <= 0:
            raise ValueError("feasibility.flow_max must be positive")
        self.capacity_users = _positive_int(
            "feasibility.capacity_users", feasibility_cfg.get("capacity_users", 40)
        )
        self.admission_order = str(
            feasibility_cfg.get("admission_order", "keyed_random")
        ).lower()
        if self.admission_order != "keyed_random":
            raise ValueError("feasibility.admission_order must be 'keyed_random'")

        self.phy_cfg = _mapping(protocol, "phy")
        self.noise_power = _finite(
            "phy.noise_power", self.phy_cfg.get("noise_power", 1.0), positive=True
        )
        self.interference_eta = _finite(
            "phy.interference_eta", self.phy_cfg.get("interference_eta", 0.6)
        )
        if self.interference_eta < 0:
            raise ValueError("phy.interference_eta must be non-negative")
        self.reference_power = _finite(
            "phy.reference_power", self.phy_cfg.get("reference_power", 10.0), positive=True
        )
        self.distance_reference = _finite(
            "phy.distance_reference", self.phy_cfg.get("distance_reference", 1.0), positive=True
        )
        self.pathloss_exponent = _finite(
            "phy.pathloss_exponent", self.phy_cfg.get("pathloss_exponent", 2.0), positive=True
        )
        self.shadow_sigma = _finite(
            "phy.shadow_sigma", self.phy_cfg.get("shadow_sigma", 0.1)
        )
        if self.shadow_sigma < 0:
            raise ValueError("phy.shadow_sigma must be non-negative")
        self.interference_power = _finite(
            "phy.interference_power", self.phy_cfg.get("interference_power", 1.0), positive=True
        )

        self.flow_cfg = _mapping(protocol, "flow")
        if "phi" not in self.flow_cfg:
            raise ValueError(
                "paper_protocol.flow.phi is required because the manuscript does not "
                "numerically fix Phi(L)"
            )
        self.flow_phi = self.flow_cfg["phi"]
        self.flow_arrival_rate = _finite(
            "flow.arrival_rate", self.flow_cfg.get("arrival_rate", 0.8)
        )
        self.flow_service_rate = _finite(
            "flow.service_rate", self.flow_cfg.get("service_rate", 1.0)
        )
        self.flow_feedback_strength = _finite(
            "flow.feedback_strength", self.flow_cfg.get("feedback_strength", 0.25)
        )
        self.flow_ema_factor = _finite(
            "flow.ema_factor", self.flow_cfg.get("ema_factor", 0.9)
        )
        if "clip_upper" not in self.flow_cfg:
            raise ValueError(
                "paper_protocol.flow.clip_upper must be explicit"
            )
        self.flow_clip = (
            _finite("flow.clip_lower", self.flow_cfg.get("clip_lower", 0.0)),
            _finite("flow.clip_upper", self.flow_cfg["clip_upper"]),
        )
        self.load_gate_can_activate = self.flow_clip[1] > self.flow_max

        self.intensity_cfg = _mapping(protocol, "intensity")
        if "baseline_hazard" not in self.intensity_cfg:
            raise ValueError(
                "paper_protocol.intensity.baseline_hazard is required because the "
                "manuscript does not report lambda_0"
            )
        self.baseline_hazard = _finite(
            "intensity.baseline_hazard",
            self.intensity_cfg["baseline_hazard"],
            positive=True,
        )
        self.beta_gamma = _finite(
            "intensity.beta_gamma", self.intensity_cfg.get("beta_gamma", 0.8)
        )
        self.beta_flow = _finite(
            "intensity.beta_flow", self.intensity_cfg.get("beta_flow", 0.9)
        )
        self.beta_elevation = _finite(
            "intensity.beta_elevation", self.intensity_cfg.get("beta_elevation", 0.2)
        )
        for required_scale in (
            "gamma_covariate_scale_db",
            "elevation_covariate_scale_deg",
        ):
            if required_scale not in self.intensity_cfg:
                raise ValueError(
                    f"paper_protocol.intensity.{required_scale} must be explicit"
                )
        self.gamma_covariate_scale_db = _finite(
            "intensity.gamma_covariate_scale_db",
            self.intensity_cfg["gamma_covariate_scale_db"],
            positive=True,
        )
        self.elevation_covariate_scale_deg = _finite(
            "intensity.elevation_covariate_scale_deg",
            self.intensity_cfg["elevation_covariate_scale_deg"],
            positive=True,
        )
        self.intensity_horizon = _finite(
            "intensity.horizon", self.intensity_cfg.get("horizon", 10.0), positive=True
        )
        self.intensity_intervals = _positive_int(
            "intensity.num_intervals", self.intensity_cfg.get("num_intervals", 128)
        )

        policy_cfg = _mapping(protocol, "policy")
        if "hard_feasibility_mask" not in policy_cfg:
            raise ValueError(
                "paper_protocol.policy.hard_feasibility_mask must be explicit "
                "(False=full geometric ranking, True=pre-ranking hard mask)"
            )
        if not isinstance(policy_cfg["hard_feasibility_mask"], bool):
            raise TypeError("policy.hard_feasibility_mask must be a boolean")
        self.hard_feasibility_mask = policy_cfg["hard_feasibility_mask"]
        self.gamma_weight = _finite(
            "policy.gamma_weight", policy_cfg.get("gamma_weight", 1.0)
        )
        self.load_weight = _finite(
            "policy.load_weight", policy_cfg.get("load_weight", 0.4)
        )
        self.intensity_weight = _finite(
            "policy.intensity_weight", policy_cfg.get("intensity_weight", 0.6)
        )
        if min(self.gamma_weight, self.load_weight, self.intensity_weight) < 0:
            raise ValueError("policy score weights must be non-negative")
        self.min_dwell_steps = int(policy_cfg.get("min_dwell_steps", 10))
        if self.min_dwell_steps < 0:
            raise ValueError("policy.min_dwell_steps must be non-negative")
        self.hysteresis = _finite(
            "policy.hysteresis", policy_cfg.get("hysteresis", 1.0 / self.topk)
        )

        self.pos_scale = _finite(
            "ephemeris.pos_scale",
            _mapping(cfg, "ephemeris").get("pos_scale", 1.0),
            positive=True,
        )
        self.vel_scale = _finite(
            "ephemeris.vel_scale",
            _mapping(cfg, "ephemeris").get("vel_scale", 1.0),
            positive=True,
        )
        self.ephem, self.K, self.S = self._build_ephemeris(cfg)
        self.flow = torch.zeros(self.S, dtype=torch.float32, device=self.device)
        self.current_serving = torch.full(
            (self.K,), -1, dtype=torch.long, device=self.device
        )
        self.hold_steps = torch.zeros(self.K, dtype=torch.long, device=self.device)
        self.epoch = 0
        self._ephemeris_state: EphemerisState | None = None
        self._pending_observation: ControlObservation | None = None
        self._last_committed_epoch = -1

    def _build_ephemeris(
        self, cfg: Mapping[str, Any]
    ) -> tuple[SimpleKinematicEphemeris | HybridUserSatEphemeris, int, int]:
        ephemeris_cfg = _mapping(cfg, "ephemeris")
        mode = str(ephemeris_cfg.get("mode", "debug")).lower()
        K = _positive_int("K_users", cfg.get("K_users", 128))
        if mode == "skyfield_tle":
            raw_tle_path = ephemeris_cfg.get("tle_path")
            if raw_tle_path is None or not str(raw_tle_path).strip():
                raise ValueError("ephemeris.tle_path is required for skyfield_tle")
            tle_path = Path(str(raw_tle_path))
            records = load_tle_file(str(tle_path))
            S = len(records)
            configured_satellites = cfg.get("S_sats")
            if configured_satellites is not None and int(configured_satellites) != S:
                raise ValueError("S_sats does not match the number of TLE records")
            user_radius = _finite(
                "ephemeris.user_radius", ephemeris_cfg.get("user_radius", 6371.0), positive=True
            )
            users = _sphere_points(K, user_radius, self.seed, "paper-users", self.device)
            user_speed = _finite(
                "ephemeris.user_speed", ephemeris_cfg.get("user_speed", 0.008)
            )
            user_velocity = _tangent_velocities(
                users, user_speed, self.seed, "paper-user-velocity"
            )
            start = _parse_utc(
                str(ephemeris_cfg.get("start_time_utc", "2025-12-25T00:00:00Z"))
            )
            ephem = HybridUserSatEphemeris(
                user_pos0=users,
                user_vel0=user_velocity,
                tle_records=records,
                start_time_utc=start,
                device=self.device,
            )
            return ephem, K, S
        if mode != "debug":
            raise ValueError("ephemeris.mode must be debug or skyfield_tle")
        S = _positive_int("S_sats", cfg.get("S_sats", 1000))
        user_radius = _finite(
            "ephemeris.user_radius", ephemeris_cfg.get("user_radius", 1.0), positive=True
        )
        satellite_radius = _finite(
            "ephemeris.satellite_radius",
            ephemeris_cfg.get("satellite_radius", 1.2),
            positive=True,
        )
        if satellite_radius <= user_radius:
            raise ValueError("ephemeris.satellite_radius must exceed user_radius")
        if "initial_user_pos" in ephemeris_cfg or "initial_satellite_pos" in ephemeris_cfg:
            if "initial_user_pos" not in ephemeris_cfg or "initial_satellite_pos" not in ephemeris_cfg:
                raise ValueError(
                    "ephemeris initial_user_pos and initial_satellite_pos must be provided together"
                )
            users = torch.as_tensor(
                ephemeris_cfg["initial_user_pos"], dtype=torch.float32, device=self.device
            )
            satellites = torch.as_tensor(
                ephemeris_cfg["initial_satellite_pos"],
                dtype=torch.float32,
                device=self.device,
            )
            if users.shape != (K, 3) or satellites.shape != (S, 3):
                raise ValueError("explicit ephemeris positions have incompatible shapes")
            if not torch.isfinite(users).all() or not torch.isfinite(satellites).all():
                raise ValueError("explicit ephemeris positions must be finite")
        else:
            users = _sphere_points(K, user_radius, self.seed, "paper-users", self.device)
            satellites = _sphere_points(
                S, satellite_radius, self.seed, "paper-satellites", self.device
            )
        if "initial_user_vel" in ephemeris_cfg or "initial_satellite_vel" in ephemeris_cfg:
            if "initial_user_vel" not in ephemeris_cfg or "initial_satellite_vel" not in ephemeris_cfg:
                raise ValueError(
                    "ephemeris initial_user_vel and initial_satellite_vel must be provided together"
                )
            user_velocity = torch.as_tensor(
                ephemeris_cfg["initial_user_vel"], dtype=torch.float32, device=self.device
            )
            satellite_velocity = torch.as_tensor(
                ephemeris_cfg["initial_satellite_vel"],
                dtype=torch.float32,
                device=self.device,
            )
            if user_velocity.shape != (K, 3) or satellite_velocity.shape != (S, 3):
                raise ValueError("explicit ephemeris velocities have incompatible shapes")
        else:
            user_velocity = _tangent_velocities(
                users,
                _finite(
                    "ephemeris.user_speed", ephemeris_cfg.get("user_speed", 0.001)
                ),
                self.seed,
                "paper-user-velocity",
            )
            satellite_velocity = _tangent_velocities(
                satellites,
                _finite(
                    "ephemeris.satellite_speed",
                    ephemeris_cfg.get("satellite_speed", 0.01),
                ),
                self.seed,
                "paper-satellite-velocity",
            )
        ephem = SimpleKinematicEphemeris(
            torch.cat((users, satellites), dim=0),
            torch.cat((user_velocity, satellite_velocity), dim=0),
        )
        return ephem, K, S

    def reset_control(self) -> ControlObservation:
        self.ephem.reset()
        self._ephemeris_state = self.ephem.step(0.0)
        self.flow.zero_()
        self.current_serving.fill_(-1)
        self.hold_steps.zero_()
        self.epoch = 0
        self._last_committed_epoch = -1
        self._pending_observation = None
        return self.observe()

    def _require_state(self) -> EphemerisState:
        if self._ephemeris_state is None:
            raise RuntimeError("call reset_control() before observe()")
        return self._ephemeris_state

    def _candidates(self, state: EphemerisState) -> ElevationCandidates:
        return build_elevation_candidates(
            state.pos[: self.K],
            state.pos[self.K :],
            minimum_elevation_deg=self.minimum_elevation_deg,
            topk=self.topk,
            user_offset=0,
            satellite_offset=self.K,
        )

    def _interference_summary(self) -> torch.Tensor:
        active = torch.unique(self.current_serving[self.current_serving >= 0])
        active_flow = self.flow[active] if active.numel() else self.flow
        return active_flow.pow(self.interference_power).mean() if active_flow.numel() else self.flow.new_zeros(())

    def _gamma_sim(self, candidates: ElevationCandidates) -> torch.Tensor:
        edge_count = int(candidates.edge_ids.size(0))
        if edge_count == 0:
            return torch.empty(0, dtype=self.flow.dtype, device=self.device)
        normalized_distance = candidates.distance / self.distance_reference
        mean_power = self.reference_power / (
            1.0 + normalized_distance.pow(self.pathloss_exponent)
        )
        normal = keyed_standard_normal(
            (edge_count, self.phy_samples),
            self.seed,
            self.epoch,
            "phy-shadowing",
            dtype=mean_power.dtype,
            device=self.device,
        )
        multiplier = torch.exp(
            self.shadow_sigma * normal - 0.5 * self.shadow_sigma**2
        )
        received_power = mean_power[:, None] * multiplier
        denominator = self.noise_power + self.interference_eta * self._interference_summary()
        sinr_linear = received_power / denominator
        gamma_samples_db = 10.0 * torch.log10(sinr_linear.clamp_min(1e-12))
        return nearest_rank_p10(gamma_samples_db, dim=1)

    def _lookahead_elevation_covariate(
        self, candidates: ElevationCandidates
    ) -> torch.Tensor:
        edge_count = int(candidates.edge_ids.size(0))
        if edge_count == 0:
            return torch.empty(
                (self.intensity_intervals + 1, 0),
                dtype=self.flow.dtype,
                device=self.device,
            )
        offsets = torch.linspace(
            0.0,
            self.intensity_horizon,
            self.intensity_intervals + 1,
            dtype=self.flow.dtype,
            device=self.device,
        )
        elevations = []
        for offset in offsets.tolist():
            future = self.ephem.state_at(float(offset))
            elevations.append(
                elevation_for_edges(
                    future.pos[: self.K],
                    future.pos[self.K :],
                    candidates.edge_ids,
                )
            )
        # Worsening visibility must increase a positive Cox coefficient.
        return -torch.stack(elevations, dim=0) / self.elevation_covariate_scale_deg

    def _edge_features(
        self,
        candidates: ElevationCandidates,
        gamma: torch.Tensor,
        intensity: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.edge_ids.numel() == 0:
            return torch.empty(
                (0, len(PAPER_EDGE_FEATURE_NAMES)),
                dtype=self.flow.dtype,
                device=self.device,
            )
        users, satellites = candidates.edge_ids.unbind(dim=1)
        destination_flow = self.flow[satellites]
        current = (self.current_serving[users] == satellites).to(self.flow.dtype)
        dwell_scale = float(max(1, self.min_dwell_steps))
        dwell = (self.hold_steps[users].to(self.flow.dtype) / dwell_scale).clamp(0.0, 1.0)
        return torch.stack(
            (
                candidates.elevation_deg / 90.0,
                torch.log1p(candidates.distance / self.distance_reference),
                gamma / self.gamma_covariate_scale_db,
                destination_flow,
                torch.log1p(intensity),
                current,
                dwell,
            ),
            dim=1,
        )

    def observe(self) -> ControlObservation:
        if self.epoch >= self.horizon_steps:
            raise RuntimeError("episode is finished")
        state = self._require_state()
        candidates = self._candidates(state)
        gamma = self._gamma_sim(candidates)
        destination_flow = (
            self.flow[candidates.edge_ids[:, 1]]
            if candidates.edge_ids.numel()
            else torch.empty(0, dtype=self.flow.dtype, device=self.device)
        )
        feasible = feasibility_gate(
            gamma,
            destination_flow,
            gamma_min=self.gamma_min_db,
            flow_max=self.flow_max,
        )
        cox = integrated_violation_intensity(
            gamma,
            destination_flow,
            self._lookahead_elevation_covariate(candidates),
            baseline_hazard=self.baseline_hazard,
            beta_gamma=self.beta_gamma / self.gamma_covariate_scale_db,
            beta_flow=self.beta_flow,
            beta_lookahead=self.beta_elevation,
            horizon=self.intensity_horizon,
            gamma_min=self.gamma_min_db,
            flow_max=self.flow_max,
            num_intervals=self.intensity_intervals,
        )
        if not torch.equal(cox.feasible, feasible):
            raise RuntimeError("feasibility gate and intensity gate disagree")
        scaled_pos = state.pos * self.pos_scale
        scaled_vel = state.vel * self.vel_scale
        node_flow = torch.zeros((self.K + self.S, 1), dtype=self.flow.dtype, device=self.device)
        node_flow[self.K :, 0] = self.flow
        node_x = torch.cat((scaled_pos, scaled_vel, node_flow), dim=1)
        fields = PolicyDescriptors(
            gamma_edge=gamma,
            intensity_edge=cox.integrated_intensity,
            flow_node=self.flow.clone(),
        )
        edge_features = self._edge_features(
            candidates,
            gamma,
            cox.integrated_intensity,
        )
        observation = ControlObservation(
            observation_id=(self.seed, self.epoch),
            node_x=node_x,
            candidate_edge_index=candidates.edge_index,
            candidate_edge_ids=candidates.edge_ids,
            edge_features=edge_features,
            elevation_deg=candidates.elevation_deg,
            sim_descriptors=SimulatorDescriptors(
                policy_fields=fields.clone(),
                feasible_edge=feasible.bool(),
            ),
            policy_descriptors=fields.clone(),
            current_serving=self.current_serving.clone(),
            hold_steps=self.hold_steps.clone(),
            user_order=keyed_user_order(self.K, self.seed, self.epoch, device=self.device),
            meta={
                "paper_protocol_version": PAPER_PROTOCOL_VERSION,
                "protocol_fingerprint": self.protocol_fingerprint,
                "capacity_users": self.capacity_users,
                "hard_feasibility_mask": self.hard_feasibility_mask,
                "topk": self.topk,
                "minimum_elevation_deg": self.minimum_elevation_deg,
                "gamma_min_db": self.gamma_min_db,
                "flow_max": self.flow_max,
                "load_gate_can_activate": self.load_gate_can_activate,
                "feature_contract_version": PAPER_FEATURE_CONTRACT_VERSION,
                "node_feature_names": PAPER_NODE_FEATURE_NAMES,
                "edge_feature_names": PAPER_EDGE_FEATURE_NAMES,
                "node_flow_column": 6,
                "edge_gamma_column": 2,
                "edge_flow_column": 3,
                "edge_intensity_column": 4,
                "gamma_feature_scale": self.gamma_covariate_scale_db,
            },
        )
        observation.validate()
        self._pending_observation = observation
        return observation.with_policy_descriptors(observation.policy_descriptors)

    def _execute_requests(
        self, observation: ControlObservation, action: ServingAction
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        executed = torch.full_like(action.requested_serving, -1)
        reasons = torch.full_like(
            action.requested_serving, int(FailureReason.NONE), dtype=torch.long
        )
        admitted_counts = torch.zeros(self.S, dtype=self.flow.dtype, device=self.device)
        edge_lookup = {
            (int(user), int(satellite)): edge
            for edge, (user, satellite) in enumerate(observation.candidate_edge_ids.tolist())
        }
        for user in observation.user_order.tolist():
            requested = int(action.requested_serving[user].item())
            if requested < 0:
                reasons[user] = int(FailureReason.ABSTAIN)
                continue
            edge = edge_lookup.get((user, requested))
            if edge is None:
                reasons[user] = int(FailureReason.NOT_CANDIDATE)
                continue
            if not bool(observation.sim_descriptors.feasible_edge[edge].item()):
                reasons[user] = int(FailureReason.INFEASIBLE)
                continue
            if int(admitted_counts[requested].item()) >= self.capacity_users:
                reasons[user] = int(FailureReason.CAPACITY)
                continue
            executed[user] = requested
            admitted_counts[requested] += 1.0
        return executed, reasons, admitted_counts

    def step_action(
        self, action: ServingAction
    ) -> tuple[ControlObservation | None, ExecutionResult, bool]:
        if self._ephemeris_state is None:
            raise RuntimeError("call reset_control() before step_action()")
        if self.epoch >= self.horizon_steps:
            raise RuntimeError("episode is finished")
        if self._last_committed_epoch == self.epoch:
            raise RuntimeError("an action has already been committed for this epoch")
        observation = self._pending_observation or self.observe()
        if action.observation_id != observation.observation_id:
            raise ValueError(
                f"stale action for {action.observation_id}; expected {observation.observation_id}"
            )
        action.validate(self.K, self.S)
        executed, reasons, admitted_counts = self._execute_requests(observation, action)
        admitted = executed >= 0
        previous = self.current_serving.clone()
        attempted = (previous >= 0) & (action.requested_serving >= 0) & (
            action.requested_serving != previous
        )
        handover_executed = (previous >= 0) & admitted & (executed != previous)
        flow_before = self.flow.clone()
        flow_after = mean_flow_update(
            self.flow,
            admitted_counts,
            dt_ctrl=self.dt_ctrl,
            arrival_rate=self.flow_arrival_rate,
            service_rate=self.flow_service_rate,
            feedback_strength=self.flow_feedback_strength,
            ema_factor=self.flow_ema_factor,
            phi=self.flow_phi,
            clip_bounds=self.flow_clip,
        )
        self.flow = flow_after
        same = admitted & (executed == previous)
        newly_admitted = admitted & ~same
        self.hold_steps = torch.where(
            same,
            self.hold_steps + 1,
            torch.where(
                newly_admitted,
                torch.ones_like(self.hold_steps),
                torch.zeros_like(self.hold_steps),
            ),
        )
        self.current_serving = executed
        result = ExecutionResult(
            observation_id=observation.observation_id,
            requested_serving=action.requested_serving.clone(),
            executed_serving=executed.clone(),
            admitted=admitted.clone(),
            failure_reason=reasons.clone(),
            handover_attempted=attempted.clone(),
            handover_executed=handover_executed.clone(),
            flow_before=flow_before,
            flow_after=flow_after.clone(),
        )
        result.validate(self.K, self.S)
        self._last_committed_epoch = self.epoch
        self.epoch += 1
        self._ephemeris_state = self.ephem.step(self.dt_ctrl)
        self._pending_observation = None
        done = self.epoch >= self.horizon_steps
        next_observation = None if done else self.observe()
        return next_observation, result, done

    @property
    def done(self) -> bool:
        return self.epoch >= self.horizon_steps

    def fixed_policy_config(self):
        """Build the exact fixed-controller configuration for this environment."""
        from ..control.policy import FixedRankPolicyConfig

        return FixedRankPolicyConfig(
            gamma_weight=self.gamma_weight,
            load_weight=self.load_weight,
            intensity_weight=self.intensity_weight,
            hard_feasibility_mask=self.hard_feasibility_mask,
            min_dwell_steps=self.min_dwell_steps,
            hysteresis=self.hysteresis,
        )
