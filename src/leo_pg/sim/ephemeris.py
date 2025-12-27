from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union
from datetime import datetime, timedelta, timezone

import torch

@dataclass
class EphemerisState:
    pos: torch.Tensor  # [N,3] or [S,3]
    vel: torch.Tensor  # [N,3] or [S,3]

class SimpleKinematicEphemeris:
    """Debug ephemeris: constant-velocity kinematics.

    For real LEO satellites, use `SkyfieldTLEEphemeris` (SGP4) below.
    """
    def __init__(self, pos0: torch.Tensor, vel0: torch.Tensor):
        self.pos = pos0.clone()
        self.vel = vel0.clone()

    def step(self, dt: float) -> EphemerisState:
        self.pos = self.pos + self.vel * dt
        return EphemerisState(self.pos.clone(), self.vel.clone())

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

class SkyfieldTLEEphemeris:
    """SGP4 ephemeris via Skyfield.

    Inputs:
      - tle_records: (name,line1,line2) or (line1,line2)
      - start_time_utc: datetime (naive treated as UTC)

    Output:
      - satellite position/velocity in km and km/s (SGP4/TLE TEME-like inertial frame).
    """
    def __init__(
        self,
        tle_records: Sequence[Union[Tuple[str, str, str], Tuple[str, str]]],
        start_time_utc: datetime,
        device: torch.device = torch.device("cpu"),
    ):
        try:
            from skyfield.api import EarthSatellite, load
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Skyfield is required. Install with: pip install skyfield sgp4"
            ) from e

        self._ts = load.timescale()
        self._dt = _ensure_utc(start_time_utc)
        self._device = device

        sats = []
        for rec in tle_records:
            if len(rec) == 3:
                name, l1, l2 = rec  # type: ignore
            elif len(rec) == 2:
                name = f"SAT_{len(sats)}"
                l1, l2 = rec  # type: ignore
            else:
                raise ValueError("Each TLE record must be (name,line1,line2) or (line1,line2)")
            sats.append(EarthSatellite(l1, l2, name, self._ts))
        if len(sats) == 0:
            raise ValueError("tle_records is empty.")
        self._sats = sats

    @property
    def t_utc(self) -> datetime:
        return self._dt

    def step(self, dt_seconds: float) -> EphemerisState:
        self._dt = self._dt + timedelta(seconds=float(dt_seconds))
        t = self._ts.from_datetime(self._dt)

        pos_list = []
        vel_list = []
        for sat in self._sats:
            g = sat.at(t)
            p = torch.tensor(g.position.km, dtype=torch.float32, device=self._device).view(3)
            v = torch.tensor(g.velocity.km_per_s, dtype=torch.float32, device=self._device).view(3)
            pos_list.append(p)
            vel_list.append(v)
        pos = torch.stack(pos_list, dim=0)  # [S,3]
        vel = torch.stack(vel_list, dim=0)  # [S,3]
        return EphemerisState(pos=pos, vel=vel)

def load_tle_file(path: str) -> List[Tuple[str, str, str]]:
    """Load 3-line (name+2) or 2-line TLE file -> (name,line1,line2) list."""
    lines = [ln.strip() for ln in open(path, "r", encoding="utf-8") if ln.strip()]
    out: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name = f"SAT_{len(out)}"
            out.append((name, lines[i], lines[i + 1]))
            i += 2
        else:
            if i + 2 >= len(lines):
                break
            name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
            if not (l1.startswith("1 ") and l2.startswith("2 ")):
                raise ValueError(f"Malformed TLE at lines {i}-{i+2}")
            out.append((name, l1, l2))
            i += 3
    return out

class HybridUserSatEphemeris:
    """Users: kinematic. Satellites: Skyfield SGP4. Returns concatenated [users; sats]."""
    def __init__(
        self,
        user_pos0: torch.Tensor,
        user_vel0: torch.Tensor,
        tle_records: Sequence[Union[Tuple[str, str, str], Tuple[str, str]]],
        start_time_utc: datetime,
        device: torch.device = torch.device("cpu"),
    ):
        self.user = SimpleKinematicEphemeris(user_pos0.to(device), user_vel0.to(device))
        self.sat = SkyfieldTLEEphemeris(tle_records=tle_records, start_time_utc=start_time_utc, device=device)

    def step(self, dt_seconds: float) -> EphemerisState:
        u = self.user.step(dt_seconds)
        s = self.sat.step(dt_seconds)
        return EphemerisState(pos=torch.cat([u.pos, s.pos], dim=0),
                              vel=torch.cat([u.vel, s.vel], dim=0))
