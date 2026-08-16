from __future__ import annotations
import torch
import torch.nn as nn

class Readout(nn.Module):
    def __init__(self, mem_dim: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mem_dim, emb_dim),
            nn.ReLU(),
        )

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        return self.net(mem)
