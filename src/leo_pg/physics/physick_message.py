from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from  leo_pg.kernels.interface import MessageFunction
from .kernel_bank import KernelBank
from .coeff_head import CoeffHeadKAN


class PhysiCKMessage(MessageFunction):
    """PhysiCK message: analytic kernel bank + learned mixing weights.

        κ = KernelBank(z)                     -> [E, M]
        α = softmax(CoeffHeadKAN(...))        -> [E, M]
        s = Σ_m α_m κ_m                       -> [E, 1]
        msg = Linear(s) ⊙ dir(edge_type)      -> [E, msg_dim]   (optional)
    """

    def __init__(
        self,
        mem_dim: int,
        edge_dim: int,
        msg_dim: int,
        edge_type_vocab: int = 8,
        use_edge_type: bool = True,
        num_kernels: int = 12,
        num_knots: int = 16,
        use_mem: bool = True,
        alpha_temperature: float = 1.0,
    ):
        super().__init__()
        self.mem_dim = int(mem_dim)
        self.edge_dim = int(edge_dim)
        self.msg_dim = int(msg_dim)
        self.use_edge_type = bool(use_edge_type)
        self.edge_type_vocab = int(edge_type_vocab)
        self.num_kernels = int(num_kernels)
        self.alpha_temperature = float(alpha_temperature)

        self.bank = KernelBank(edge_dim=self.edge_dim, num_kernels=self.num_kernels)
        self.coeff = CoeffHeadKAN(
            edge_dim=self.edge_dim,
            mem_dim=self.mem_dim,
            num_kernels=self.num_kernels,
            num_knots=num_knots,
            use_mem=use_mem,
        )

        # map scalar interaction strength -> vector message
        self.scale = nn.Linear(1, self.msg_dim, bias=False)

        if self.use_edge_type:
            self.type_emb = nn.Embedding(self.edge_type_vocab, self.msg_dim)

    def forward(
        self,
        mem_src: torch.Tensor,
        mem_dst: torch.Tensor,
        z_ij: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        E = int(z_ij.size(0))
        if E == 0:
            return torch.zeros((0, self.msg_dim), device=z_ij.device, dtype=z_ij.dtype)

        kappa = self.bank(z_ij)  # [E,M]
        logits = self.coeff(z_ij, mem_src, mem_dst)  # [E,M]

        # Stable mixture weights: softmax over kernels (interpretability + boundedness)
        temp = max(self.alpha_temperature, 1e-6)
        alpha = F.softmax(logits / temp, dim=-1)  # [E,M]

        s = (alpha * kappa).sum(dim=-1, keepdim=True)  # [E,1]
        base = self.scale(s)  # [E,msg_dim]

        if self.use_edge_type:
            if edge_type is None:
                edge_type = torch.zeros((E,), dtype=torch.long, device=z_ij.device)
            d = torch.tanh(self.type_emb(edge_type))  # [E,msg_dim]
            return base * d
        return base
