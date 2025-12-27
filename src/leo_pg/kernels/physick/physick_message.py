from __future__ import annotations
import torch
import torch.nn as nn

from ..interface import MessageFunction
from .kernel_bank import KernelBank
from .coeff_head import CoeffHeadKAN

class PhysiCKMessage(MessageFunction):
    """PhysiCK message: bank + mixing.
      κ = KernelBank(z)  -> [E,M]
      α = CoeffHeadKAN(z, mem_src, mem_dst) -> [E,M]
      s = sum_m α_m κ_m -> [E]
      msg = s * dir(edge_type)  -> [E, msg_dim]
    """
    def __init__(self, mem_dim: int, edge_dim: int, msg_dim: int,
                 edge_type_vocab: int = 8, use_edge_type: bool = True,
                 num_kernels: int = 12, num_knots: int = 16):
        super().__init__()
        self.msg_dim = msg_dim
        self.use_edge_type = use_edge_type
        self.bank = KernelBank(edge_dim=edge_dim, num_kernels=num_kernels)
        self.coeff = CoeffHeadKAN(edge_dim=edge_dim, mem_dim=mem_dim,
                                  num_kernels=num_kernels, num_knots=num_knots, use_mem=True)
        if use_edge_type:
            self.type_emb = nn.Embedding(edge_type_vocab, msg_dim)
        else:
            self.type_emb = None
        self.scale = nn.Linear(1, msg_dim, bias=False)

    def forward(self, mem_src: torch.Tensor, mem_dst: torch.Tensor,
                z_ij: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        kappa = self.bank(z_ij)                 # [E,M]
        alpha = self.coeff(z_ij, mem_src, mem_dst)  # [E,M]
        s = (alpha * kappa).sum(dim=-1, keepdim=True)  # [E,1]
        base = self.scale(s)                    # [E,msg_dim]
        if self.use_edge_type:
            if edge_type is None:
                edge_type = torch.zeros((z_ij.size(0),), dtype=torch.long, device=z_ij.device)
            d = self.type_emb(edge_type)        # [E,msg_dim]
            return base * torch.tanh(d)
        return base
