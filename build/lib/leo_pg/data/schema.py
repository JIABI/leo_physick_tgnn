from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

@dataclass
class StepGraph:
    t: int
    node_x: torch.Tensor        # [N, F_node]
    edge_index: torch.Tensor    # [2, E] src->dst
    edge_z: torch.Tensor        # [E, F_edge] physics-guided descriptors
    edge_type: torch.Tensor     # [E] long (optional; zeros allowed)
    y: torch.Tensor             # [N, F_out] or task-specific
    meta: Dict[str, Any]

@dataclass
class Episode:
    steps: List[StepGraph]
    meta: Dict[str, Any]

@dataclass
class DatasetBundle:
    episodes: List[Episode]
    meta: Dict[str, Any]
