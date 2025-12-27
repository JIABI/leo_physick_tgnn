from __future__ import annotations
import enum

class EdgeType(enum.IntEnum):
    USER_SAT = 0
    SAT_SAT = 1
    # extend: INTRA_BEAM, INTRA_SAT, INTER_SAT, etc.
