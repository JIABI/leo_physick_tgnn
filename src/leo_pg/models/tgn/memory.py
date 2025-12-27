from __future__ import annotations
import torch
import torch.nn as nn

class MemoryBank(nn.Module):
    def __init__(self, node_in_dim: int, mem_dim: int):
        super().__init__()
        self.node_enc = nn.Linear(node_in_dim, mem_dim)
        self.gru = nn.GRUCell(input_size=mem_dim, hidden_size=mem_dim)

    def init(self, node_x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.node_enc(node_x))

    def update(self, mem: torch.Tensor, agg_msg: torch.Tensor, node_x: torch.Tensor, inject: float = 0.1) -> torch.Tensor:
        mem_new = self.gru(agg_msg, mem)
        mem_inj = torch.tanh(self.node_enc(node_x))
        return (1.0 - inject) * mem_new + inject * mem_inj
