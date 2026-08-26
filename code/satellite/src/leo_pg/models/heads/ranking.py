"""Edge-level candidate ranking head for the generic TGN registry."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


class RankingHead(nn.Module):
    """Score each candidate edge from its two node embeddings and edge state.

    If an ``edge_type`` vector is present, only type-zero user--resource
    candidate edges are scored; auxiliary graph edges cannot silently enter
    the policy ranking.
    """

    def __init__(
        self,
        in_dim: int = 64,
        edge_in_dim: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.hidden_dim = self.in_dim if hidden_dim is None else int(hidden_dim)
        if min(self.in_dim, self.edge_in_dim, self.hidden_dim) <= 0:
            raise ValueError("ranking dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.net = nn.Sequential(
            nn.Linear(2 * self.in_dim + self.edge_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        emb: torch.Tensor,
        step: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        if emb.ndim != 2 or emb.size(1) != self.in_dim:
            raise ValueError(f"emb must have shape [N,{self.in_dim}]")
        if not isinstance(step, Mapping):
            raise ValueError("ranking head requires the candidate graph step")
        if "edge_index" not in step or "edge_z" not in step:
            raise ValueError("ranking head requires edge_index and edge_z")
        edge_index = torch.as_tensor(step["edge_index"], device=emb.device)
        edge_z = torch.as_tensor(step["edge_z"], device=emb.device).float()
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2,E]")
        edge_index = edge_index.long()
        if edge_z.shape != (edge_index.size(1), self.edge_in_dim):
            raise ValueError(
                f"edge_z must have shape [{edge_index.size(1)},{self.edge_in_dim}]"
            )
        edge_type = step.get("edge_type")
        if edge_type is not None:
            edge_type = torch.as_tensor(edge_type, device=emb.device)
            if edge_type.shape != (edge_index.size(1),):
                raise ValueError("edge_type must have one value per edge")
            candidate = edge_type == 0
            edge_index = edge_index[:, candidate]
            edge_z = edge_z[candidate]
        if edge_index.numel() == 0:
            return emb.new_empty((0,))
        source, destination = edge_index
        if int(source.min()) < 0 or int(destination.min()) < 0:
            raise ValueError("edge indices must be non-negative")
        if int(source.max()) >= emb.size(0) or int(destination.max()) >= emb.size(0):
            raise ValueError("edge index exceeds the node embedding table")
        features = torch.cat((emb[source], emb[destination], edge_z), dim=-1)
        scores = self.net(features).squeeze(-1)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("ranking score contains NaN or Inf")
        return scores


__all__ = ["RankingHead"]
