from __future__ import annotations

import enum


class EdgeType(enum.IntEnum):
    """Relations materialized by the satellite graph builders."""

    USER_SAT = 0
    SAT_SAT = 1
