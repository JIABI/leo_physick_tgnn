from __future__ import annotations
import torch
import torch.nn as nn

class ForecastHead(nn.Module):
    """Node-level regression head (debug default).
    Replace with interference map / load distribution heads as needed.
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
