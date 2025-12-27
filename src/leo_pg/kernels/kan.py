from __future__ import annotations
import torch
import torch.nn as nn
from .interface import MessageFunction

class KANLinear(nn.Module):
    """Minimal KAN-like layer using per-input 1D piecewise-linear splines."""
    def __init__(self, in_dim: int, out_dim: int, num_knots: int = 16):
        super().__init__()
        assert num_knots >= 2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_knots = num_knots
        grid = torch.linspace(-1.0, 1.0, num_knots)
        self.register_buffer("grid", grid)
        self.values = nn.Parameter(torch.zeros(out_dim, in_dim, num_knots))
        nn.init.uniform_(self.values, a=-0.05, b=0.05)
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        assert D == self.in_dim
        x = torch.clamp(x, -1.0, 1.0)
        grid = self.grid
        right = torch.bucketize(x, grid)
        left = torch.clamp(right - 1, 0, self.num_knots - 2)
        right = left + 1
        g0 = grid[left]
        g1 = grid[right]
        w = (x - g0) / (g1 - g0 + 1e-12)

        v = self.values
        # gather per dim
        dim_idx = torch.arange(D, device=x.device)[None, :]
        v_left = v[:, dim_idx, left]   # [O,B,D]
        v_right = v[:, dim_idx, right] # [O,B,D]
        v_left = v_left.permute(0, 2, 1)   # [O,D,B]
        v_right = v_right.permute(0, 2, 1) # [O,D,B]
        wT = w.transpose(0, 1).unsqueeze(0) # [1,D,B]
        y = (1.0 - wT) * v_left + wT * v_right
        y = y.sum(dim=1).transpose(0, 1)    # [B,O]
        return y + self.skip(x)

class KANMessage(MessageFunction):
    def __init__(self, mem_dim: int, edge_dim: int, msg_dim: int, num_knots: int = 16):
        super().__init__()
        self.kan = KANLinear(2 * mem_dim + edge_dim, msg_dim, num_knots=num_knots)

    def forward(self, mem_src: torch.Tensor, mem_dst: torch.Tensor,
                z_ij: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([mem_src, mem_dst, z_ij], dim=-1)
        x = torch.tanh(x)
        return self.kan(x)
