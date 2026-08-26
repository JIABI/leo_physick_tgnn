"""Pure-PyTorch edge-conditioned GATv2 layer used by the DA-GWM reference.

The implementation keeps the repository CPU-runnable without compiled PyG
extensions.  When torch-geometric is installed, the factory in ``baselines``
uses its GATv2Conv; both paths share the same multi-head edge-conditioned
attention contract.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PureTorchGATv2Conv(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        heads: int,
        concat: bool,
        dropout: float,
        edge_dim: int,
        add_self_loops: bool = False,
    ) -> None:
        super().__init__()
        if not concat:
            raise ValueError("the manuscript DA-GWM uses concatenated attention heads")
        if add_self_loops:
            raise ValueError("self loops are disabled for the typed candidate graph")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.heads = int(heads)
        self.dropout = float(dropout)
        width = self.heads * self.out_channels
        self.lin_source = nn.Linear(self.in_channels, width, bias=False)
        self.lin_target = nn.Linear(self.in_channels, width, bias=False)
        self.lin_edge = nn.Linear(int(edge_dim), width, bias=False)
        self.attention = nn.Parameter(torch.empty(self.heads, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.attention)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.in_channels:
            raise ValueError(f"x must have shape [N,{self.in_channels}]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_attr.ndim != 2 or edge_attr.size(0) != edge_index.size(1):
            raise ValueError("edge_attr must align one-to-one with edges")
        nodes = int(x.size(0))
        edges = int(edge_index.size(1))
        if edges == 0:
            return x.new_zeros((nodes, self.heads * self.out_channels)) + self.bias
        source, target = edge_index
        source_state = self.lin_source(x).view(nodes, self.heads, self.out_channels)
        target_state = self.lin_target(x).view(nodes, self.heads, self.out_channels)
        edge_state = self.lin_edge(edge_attr).view(edges, self.heads, self.out_channels)
        joint = F.leaky_relu(
            source_state[source] + target_state[target] + edge_state,
            negative_slope=0.2,
        )
        logits = (joint * self.attention).sum(dim=-1)
        messages = source_state[source] + edge_state
        output = x.new_zeros((nodes, self.heads, self.out_channels))
        for head in range(self.heads):
            maximum = x.new_full((nodes,), -torch.inf)
            maximum.scatter_reduce_(
                0, target, logits[:, head], reduce="amax", include_self=True
            )
            unnormalized = torch.exp(logits[:, head] - maximum[target])
            denominator = x.new_zeros(nodes)
            denominator.index_add_(0, target, unnormalized)
            coefficient = unnormalized / denominator[target].clamp_min(1e-12)
            coefficient = F.dropout(
                coefficient, p=self.dropout, training=self.training
            )
            output[:, head].index_add_(
                0, target, coefficient[:, None] * messages[:, head]
            )
        return output.reshape(nodes, -1) + self.bias


__all__ = ["PureTorchGATv2Conv"]

