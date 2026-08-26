from __future__ import annotations
import torch
import torch.nn as nn

class ForecastHead(nn.Module):
    """Generic node-level regression head for the upstream compatibility API.

    Manuscript-aligned models use the typed Intensity--Flow or Snapshot heads;
    this small head remains available for checkpoint and message-operator unit
    tests that exercise the original public package surface.
    """
    def __init__(self, out_dim: int = 1, in_dim: int = 64, output_activation: str = "none"):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)
        self.output_activation = str(output_activation).lower()
        if self.output_activation not in {"none", "sigmoid", "softplus", "tanh"}:
            raise ValueError(f"Unknown forecast output_activation={output_activation!r}")

    def forward(self, emb: torch.Tensor, step: dict | None = None) -> torch.Tensor:
        output = self.net(emb)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(output)
        if self.output_activation == "softplus":
            return torch.nn.functional.softplus(output)
        if self.output_activation == "tanh":
            return torch.tanh(output)
        return output
