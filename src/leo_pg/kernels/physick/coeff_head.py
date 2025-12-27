from __future__ import annotations
import torch
import torch.nn as nn
from ..kan import KANLinear

class CoeffHeadKAN(nn.Module):
    """α(z, mem) -> mixing weights over kernel bank.
    Uses KANLinear for spline-based expressivity on low-dimensional physics descriptors.
    """
    def __init__(self, edge_dim: int, mem_dim: int, num_kernels: int, num_knots: int = 16, use_mem: bool = True):
        super().__init__()
        self.use_mem = use_mem
        in_dim = edge_dim + (2 * mem_dim if use_mem else 0)
        self.kan1 = KANLinear(in_dim, max(32, num_kernels), num_knots=num_knots)
        self.act = nn.Tanh()
        self.lin = nn.Linear(max(32, num_kernels), num_kernels)

    def forward(self, z: torch.Tensor, mem_src: torch.Tensor, mem_dst: torch.Tensor) -> torch.Tensor:
        if self.use_mem:
            x = torch.cat([z, mem_src, mem_dst], dim=-1)
        else:
            x = z
        x = torch.tanh(x)
        h = self.act(self.kan1(x))
        a = self.lin(h)
        return a
