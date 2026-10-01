"""Array interfaces shared by physical environment, controllers, and learners."""
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

@dataclass
class Observation:
    epoch: int
    sat_id: np.ndarray
    beam_id: np.ndarray
    feasible: np.ndarray
    z: np.ndarray
    analytic: np.ndarray
    survival: np.ndarray
    prev_sat: np.ndarray
    prev_beam: np.ndarray
    rho: float
    beta: float
    p_feas: np.ndarray
    rst: np.ndarray
    sinr: np.ndarray
    candidate_subband: np.ndarray
    observation_rate_mbps: np.ndarray
    post_join_load: np.ndarray
    previous_subband: np.ndarray
    common_window_s: np.ndarray
    entry_rate: np.ndarray
    visible_satellite_count: np.ndarray
    eph_z: Optional[np.ndarray] = None
    eph_survival: Optional[np.ndarray] = None
    candidate_subband_reason: Optional[np.ndarray] = None
    def to_dict(self): return asdict(self)
    @property
    def exists(self): return self.sat_id>=0
    @property
    def current(self): return (self.sat_id==self.prev_sat[:,None]) & (self.beam_id==self.prev_beam[:,None]) & self.exists

@dataclass
class StepResult:
    next_observation: Optional[Observation]
    reward: np.ndarray
    executed_sat: np.ndarray
    executed_beam: np.ndarray
    executed_subband: np.ndarray
    rate_mbps: np.ndarray
    cost: np.ndarray
    c_exec: np.ndarray
    occupancy: np.ndarray
    attempt: np.ndarray
    failure: np.ndarray
    info: dict
    requested_slots: np.ndarray
    def to_dict(self): return asdict(self)
