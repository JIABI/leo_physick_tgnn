from __future__ import annotations
import torch
import torch.nn as nn

class RankingHead(nn.Module):
    """Candidate scoring head scaffold.
    For per-edge ranking you would output scores on edges; here we output node scores as placeholder.
    """
    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.net = nn.Linear(in_dim, 1)

    def forward(self, emb: torch.Tensor, step: dict | None = None) -> torch.Tensor:
        return self.net(emb)
