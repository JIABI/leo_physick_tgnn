from __future__ import annotations
from typing import Dict, Any
from datetime import datetime, timezone

import torch
from torch.onnx.symbolic_opset10 import topk

from .ephemeris import (
    SimpleKinematicEphemeris,
    HybridUserSatEphemeris,
    load_tle_file,
)
from .channel import compute_sinr, sinr_to_rate
from .load import update_satellite_load
from ..graph.builder import build_user_sat_edges
from ..physics.descriptors import compute_edge_descriptors
from ..graph.edge_types import EdgeType


def _parse_iso_utc(s: str) -> datetime:
    """Parse ISO8601 string into timezone-aware UTC datetime.
    Accepts 'Z' suffix.
    """
    s = str(s).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _random_points_on_sphere(n: int, r: float, device: torch.device) -> torch.Tensor:
    """Uniform random points on sphere of radius r."""
    u = torch.rand(n, device=device)
    v = torch.rand(n, device=device)
    theta = 2.0 * torch.pi * u
    phi = torch.acos(2.0 * v - 1.0)
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)


class MultiUserLEOEnv:
    """A unified environment producing time-varying graphs.

    Nodes:
      - users: 0..K-1
      - satellites: K..K+S-1

    Edges:
      - user->sat for visible/candidate sats at time t.

    Ephemeris switching (cfg['ephemeris']):
      - mode: 'debug' (default) or 'skyfield_tle'
      - tle_path: path to a TLE file (required for skyfield_tle)
      - start_time_utc: ISO string, default '2025-12-25T00:00:00Z'
      - pos_scale / vel_scale: optional scalar rescaling applied to ephemeris outputs
        (useful to keep legacy normalized thresholds when using km outputs from SGP4).

    Notes:
      - Skyfield returns satellite states in km and km/s.
      - If you keep your legacy visibility_radius ~ O(1), set pos_scale ~ 1e-4
        so Earth radius ~ 0.637 and LEO radius ~ 0.7 in normalized units.
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device

        self.T = int(cfg.get("T", 200))
        self.dt = float(cfg.get("dt", 1.0))
        self.K = int(cfg.get("K_users", 50))
        self.S = int(cfg.get("S_sats", 200))  # may be overridden by TLE count

        ephem_cfg = cfg.get("ephemeris", {}) or {}
        self.ephem_mode = str(ephem_cfg.get("mode", "debug")).lower()

        # Scaling to reconcile km with normalized coordinates (optional)
        self.pos_scale = float(ephem_cfg.get("pos_scale", 1.0))
        self.vel_scale = float(ephem_cfg.get("vel_scale", 1.0))

        # Init user positions/velocities
        torch.manual_seed(int(cfg.get("seed", 7)))
        self.user_pos0 = torch.randn(self.K, 3, device=device) * 0.2
        self.user_vel0 = torch.randn(self.K, 3, device=device) * 0.01

        # Optional: place users on an Earth-like sphere (recommended for skyfield_tle)
        # If enabled, we also scale the radius by pos_scale so it matches the chosen units.
        if bool(ephem_cfg.get("place_users_on_earth", False)):
            re_km = float(ephem_cfg.get("user_earth_radius_km", 6371.0))
            self.user_pos0 = _random_points_on_sphere(self.K, re_km * self.pos_scale, device=device)
            self.user_vel0 = torch.zeros_like(self.user_pos0)

        # Satellites and ephemeris initialization
        self._tle_records = None
        self._tle_start_time_utc = None

        if self.ephem_mode in ("skyfield_tle", "tle", "sgp4", "skyfield"):
            tle_path = ephem_cfg.get("tle_path", None)
            if not tle_path:
                raise ValueError("ephemeris.mode=skyfield_tle requires ephemeris.tle_path")
            self._tle_records = load_tle_file(str(tle_path))
            if len(self._tle_records) == 0:
                raise ValueError(f"No TLE records loaded from {tle_path}")
            # Override S to match TLE records
            self.S = int(len(self._tle_records))
            start_time = ephem_cfg.get("start_time_utc", "2025-12-25T00:00:00Z")
            self._tle_start_time_utc = _parse_iso_utc(start_time)

            # Hybrid: users kinematic, sats from TLE
            try:
                self.ephem = HybridUserSatEphemeris(
                    user_pos0=self.user_pos0,
                    user_vel0=self.user_vel0,
                    tle_records=self._tle_records,
                    start_time_utc=self._tle_start_time_utc,
                    device=self.device,
                )
            except ImportError as e:
                raise ImportError(
                    "Skyfield/SGP4 not available. Install with: pip install skyfield sgp4"
                ) from e
        else:
            # Debug kinematics for both users and sats
            self.sat_pos0 = torch.randn(self.S, 3, device=device) * 0.2
            self.sat_vel0 = torch.randn(self.S, 3, device=device) * 0.02
            pos0 = torch.cat([self.user_pos0, self.sat_pos0], dim=0)
            vel0 = torch.cat([self.user_vel0, self.sat_vel0], dim=0)
            self.ephem = SimpleKinematicEphemeris(pos0, vel0)

        self.N = self.K + self.S

        # graph/channel params
        self.visibility_radius = float(cfg.get("graph", {}).get("visibility_radius", 0.7))
        self.base_sinr = float(cfg.get("channel", {}).get("base_sinr", 10.0))
        self.dist_scale = float(cfg.get("channel", {}).get("dist_scale", 5.0))

        # Cox parameters
        self.cox_cfg = cfg.get("cox", {})

        # HO costs
        ho = cfg.get("handover", {})
        self.cost_intra_beam = float(ho.get("cost_intra_beam", 0.1))
        self.cost_intra_sat = float(ho.get("cost_intra_sat", 0.3))
        self.cost_inter_sat = float(ho.get("cost_inter_sat", 1.0))

        # satellite load state
        self.sat_load = torch.zeros(self.S, device=device)

        self.t = 0

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self.sat_load.zero_()
        return self._make_step()

    def step(self) -> Dict[str, Any]:
        assert self.t < self.T
        self.t += 1
        return self._make_step()

    def _make_step(self) -> Dict[str, Any]:
        state = self.ephem.step(self.dt)
        pos, vel = state.pos, state.vel

        # Apply optional rescaling (e.g., km -> normalized)
        if self.pos_scale != 1.0:
            pos = pos * self.pos_scale
        if self.vel_scale != 1.0:
            vel = vel * self.vel_scale

        user_pos, user_vel = pos[:self.K], vel[:self.K]
        sat_pos, sat_vel = pos[self.K:], vel[self.K:]

        # user->sat visible edges
        topk = int(self.cfg.get("graph", {}).get("topk", 8))
        edge_index_us, edge_ctx = build_user_sat_edges(
            user_pos=user_pos,
            sat_pos=sat_pos,
            visibility_radius=self.visibility_radius,
            user_offset=0,
            sat_offset=self.K,
            topk = topk
        )

        if edge_index_us.numel() == 0:
            edge_z = torch.zeros((0, 6), device=self.device)
            edge_type = torch.zeros((0,), dtype=torch.long, device=self.device)
        else:
            # edge-level SINR and rate
            src = edge_index_us[0]              # users
            dst = edge_index_us[1] - self.K     # satellite local idx (0..S-1)

            sinr = compute_sinr(
                user_pos=user_pos[src],
                sat_pos=sat_pos[dst],
                sat_load=self.sat_load[dst],
                base_sinr=self.base_sinr,
                dist_scale=self.dist_scale,
            )
            rate = sinr_to_rate(sinr)

            # HO signaling cost surrogate by edge type (placeholder: treat as inter-sat)
            # Replace this with true (v0,n0)->(v',n') typed costs if/when you track serving links.
            H = torch.full_like(rate, self.cost_inter_sat)

            edge_z = compute_edge_descriptors(
                edge_ctx=edge_ctx,
                rate=rate,
                ho_cost=H,
                sat_load=self.sat_load,
                cox_cfg=self.cox_cfg,
                device=self.device,
            )

            edge_type = torch.full(
                (edge_z.size(0),),
                int(EdgeType.USER_SAT),
                device=self.device,
                dtype=torch.long,
            )

        # node features: [pos, vel]
        node_x = torch.cat([pos, vel], dim=-1)

        # Debug target y: satellite load on satellite nodes
        if edge_index_us.numel() > 0:
            incoming = torch.bincount((edge_index_us[1] - self.K), minlength=self.S).float()
        else:
            incoming = torch.zeros(self.S, device=self.device)

        self.sat_load = update_satellite_load(
            self.sat_load,
            incoming / max(1.0, float(self.K)),
            momentum=0.8,
        )

        y = torch.zeros((self.N, 1), device=self.device)
        y[self.K:, 0] = self.sat_load

        return {
            "t": int(self.t),
            "node_x": node_x,
            "edge_index": edge_index_us,
            "edge_z": edge_z,
            "edge_type": edge_type,
            "y": y,
            "meta": {
                "K_users": self.K,
                "S_sats": self.S,
                "ephemeris_mode": self.ephem_mode,
                "pos_scale": self.pos_scale,
                "vel_scale": self.vel_scale,
            },
        }
