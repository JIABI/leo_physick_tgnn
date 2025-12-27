from __future__ import annotations
import torch
import torch.nn as nn
from .interface import MessageFunction

class MLPMessage(MessageFunction):
    def __init__(self, mem_dim: int, edge_dim: int, msg_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * mem_dim + edge_dim, msg_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(msg_dim, msg_dim),
        )

    def forward(self, mem_src: torch.Tensor, mem_dst: torch.Tensor,
                z_ij: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([mem_src, mem_dst, z_ij], dim=-1)
        return self.net(x)
