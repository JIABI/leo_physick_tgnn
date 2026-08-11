from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...sim.state import PolicyDescriptors


INTENSITY_FLOW_HEAD_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class IntensityFlowOutput:
    """Typed prediction returned by :class:`IntensityFlowHead`.

    The auxiliary feasibility logit is deliberately kept outside
    :class:`PolicyDescriptors`: simulator feasibility remains authoritative and
    downstream controllers cannot accidentally treat the learned auxiliary
    prediction as a hard mask.
    """

    policy_descriptors: PolicyDescriptors
    feasibility_logit: torch.Tensor

    def validate(self, edge_count: int, satellite_count: int) -> None:
        self.policy_descriptors.validate(edge_count, satellite_count)
        if self.feasibility_logit.ndim != 1 or self.feasibility_logit.numel() != edge_count:
            raise ValueError(
                "feasibility_logit must have shape "
                f"[{edge_count}], got {tuple(self.feasibility_logit.shape)}"
            )
        if not torch.isfinite(self.feasibility_logit).all():
            raise ValueError("feasibility_logit contains NaN or Inf")

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            **self.policy_descriptors.as_dict(),
            "feasibility_logit": self.feasibility_logit,
        }


class IntensityFlowHead(nn.Module):
    """Predict policy descriptors from node embeddings and candidate edges.

    Candidate-edge representations concatenate the source and destination node
    embeddings with the current edge descriptors. Three separate projections
    produce an unbounded local cue ``gamma``, a non-negative first-violation
    intensity, and an auxiliary feasibility logit. A separate satellite-node
    branch predicts bounded flow.
    """

    def __init__(
        self,
        in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        if self.in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if self.edge_in_dim <= 0:
            raise ValueError("edge_in_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

        self.edge_trunk = nn.Sequential(
            nn.Linear(2 * self.in_dim + self.edge_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.gamma_head = nn.Linear(self.hidden_dim, 1)
        self.intensity_head = nn.Linear(self.hidden_dim, 1)
        self.feasibility_head = nn.Linear(self.hidden_dim, 1)

        self.flow_trunk = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _dimensions(step: Dict[str, Any], node_count: int) -> tuple[int, int]:
        meta = step.get("meta", {})
        if not isinstance(meta, dict) or "K_users" not in meta:
            raise ValueError("intensity_flow head requires step.meta.K_users")
        user_count = int(meta["K_users"])
        satellite_count = int(meta.get("S_sats", node_count - user_count))
        if user_count <= 0 or satellite_count <= 0 or user_count + satellite_count != node_count:
            raise ValueError("step.meta K_users/S_sats are incompatible with node embeddings")
        return user_count, satellite_count

    @staticmethod
    def _candidate_edges(
        step: Dict[str, Any],
        *,
        user_count: int,
        node_count: int,
        device: torch.device,
        edge_in_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "edge_index" not in step:
            raise ValueError("intensity_flow head requires step.edge_index")
        edge_index = torch.as_tensor(step["edge_index"], device=device)
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("step.edge_index must have shape [2,E]")
        if edge_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("step.edge_index must use an integer dtype")
        edge_index = edge_index.long()
        if "edge_z" not in step:
            raise ValueError("intensity_flow head requires step.edge_z")
        edge_z = torch.as_tensor(step["edge_z"], device=device).float()
        if edge_z.ndim != 2 or edge_z.shape != (edge_index.size(1), edge_in_dim):
            raise ValueError(
                "step.edge_z must have shape "
                f"[{edge_index.size(1)},{edge_in_dim}], got {tuple(edge_z.shape)}"
            )
        if not torch.isfinite(edge_z).all():
            raise ValueError("step.edge_z contains NaN or Inf")

        edge_type = step.get("edge_type")
        if edge_type is not None:
            edge_type = torch.as_tensor(edge_type, device=device)
            if edge_type.ndim != 1 or edge_type.numel() != edge_index.size(1):
                raise ValueError("step.edge_type must have one entry per edge")
            candidate_mask = edge_type == 0
            edge_index = edge_index[:, candidate_mask]
            edge_z = edge_z[candidate_mask]

        if edge_index.numel():
            src, dst = edge_index
            if (
                int(src.min()) < 0
                or int(src.max()) >= user_count
                or int(dst.min()) < user_count
                or int(dst.max()) >= node_count
            ):
                raise ValueError(
                    "candidate edges must run from user nodes to satellite nodes"
                )
        return edge_index, edge_z

    def forward(self, emb: torch.Tensor, step: Dict[str, Any] | None = None) -> IntensityFlowOutput:
        if emb.ndim != 2 or emb.size(1) != self.in_dim:
            raise ValueError(
                f"emb must have shape [N,{self.in_dim}], got {tuple(emb.shape)}"
            )
        if step is None or not isinstance(step, dict):
            raise ValueError("intensity_flow head requires a graph step mapping")

        user_count, satellite_count = self._dimensions(step, int(emb.size(0)))
        candidate_edge_index, candidate_edge_z = self._candidate_edges(
            step,
            user_count=user_count,
            node_count=int(emb.size(0)),
            device=emb.device,
            edge_in_dim=self.edge_in_dim,
        )
        src, dst = candidate_edge_index
        edge_hidden = self.edge_trunk(
            torch.cat([emb[src], emb[dst], candidate_edge_z], dim=-1)
        )

        gamma_edge = self.gamma_head(edge_hidden).squeeze(-1)
        intensity_edge = F.softplus(self.intensity_head(edge_hidden).squeeze(-1))
        feasibility_logit = self.feasibility_head(edge_hidden).squeeze(-1)
        flow_node = torch.sigmoid(self.flow_trunk(emb[user_count:]).squeeze(-1))

        output = IntensityFlowOutput(
            policy_descriptors=PolicyDescriptors(
                gamma_edge=gamma_edge,
                intensity_edge=intensity_edge,
                flow_node=flow_node,
            ),
            feasibility_logit=feasibility_logit,
        )
        output.validate(int(candidate_edge_index.size(1)), satellite_count)
        return output
