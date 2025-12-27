from __future__ import annotations
import torch
import torch.nn as nn

class ForecastHead(nn.Module):
    """Node-level regression head (debug default).
    Replace with interference map / load distribution heads as needed.
    """
    def __init__(self, out_dim: int = 1, in_dim: int = 64):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, emb: torch.Tensor, step: dict | None = None) -> torch.Tensor:
        return self.net(emb)
