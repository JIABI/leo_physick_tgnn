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
        self._pos0 = pos0.clone()
        self._vel0 = vel0.clone()
        self.reset()

    def reset(self) -> None:
        self.pos = self._pos0.clone()
        self.vel = self._vel0.clone()

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
        self._start_time = _ensure_utc(start_time_utc)
        self._dt = self._start_time
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

    def reset(self) -> None:
        self._dt = self._start_time

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

def _validate_tle_pair(line1: str, line2: str, line1_number: int, line2_number: int) -> None:
    for line, prefix, line_number in ((line1, "1 ", line1_number), (line2, "2 ", line2_number)):
        if not line.startswith(prefix):
            raise ValueError(f"Malformed TLE line {line_number}: expected prefix {prefix!r}")
        if len(line) != 69:
            raise ValueError(f"Malformed TLE line {line_number}: expected 69 characters, got {len(line)}")
        checksum = sum(int(char) for char in line[:68] if char.isdigit()) + line[:68].count("-")
        if not line[-1].isdigit() or checksum % 10 != int(line[-1]):
            raise ValueError(f"Invalid TLE checksum at line {line_number}")
    if line1[2:7] != line2[2:7]:
        raise ValueError(
            f"TLE satellite numbers differ at lines {line1_number} and {line2_number}"
        )


def load_tle_file(path: str) -> List[Tuple[str, str, str]]:
    """Strictly load complete 3-line (name+2) or 2-line TLE records."""
    with open(path, "r", encoding="utf-8") as handle:
        lines = [(number, text.strip()) for number, text in enumerate(handle, start=1) if text.strip()]
    out: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        line_number, text = lines[i]
        if text.startswith("1 "):
            if i + 1 >= len(lines):
                raise ValueError(f"Truncated two-line TLE record at line {line_number}")
            line2_number, line2 = lines[i + 1]
            name = f"SAT_{len(out)}"
            _validate_tle_pair(text, line2, line_number, line2_number)
            out.append((name, text, line2))
            i += 2
        else:
            if i + 2 >= len(lines):
                raise ValueError(f"Truncated three-line TLE record at line {line_number}")
            name = text
            line1_number, l1 = lines[i + 1]
            line2_number, l2 = lines[i + 2]
            _validate_tle_pair(l1, l2, line1_number, line2_number)
            out.append((name, l1, l2))
            i += 3
    if not out:
        raise ValueError(f"No TLE records found in {path}")
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

    def reset(self) -> None:
        self.user.reset()
        self.sat.reset()

    def step(self, dt_seconds: float) -> EphemerisState:
        u = self.user.step(dt_seconds)
        s = self.sat.step(dt_seconds)
        return EphemerisState(pos=torch.cat([u.pos, s.pos], dim=0),
                              vel=torch.cat([u.vel, s.vel], dim=0))
