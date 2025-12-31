from __future__ import annotations
import torch
import torch.nn as nn

from ..interface import MessageFunction
from .kernel_bank import KernelBank
from .coeff_head import CoeffHeadKAN, CoeffHeadMLP

class PhysiCKMessage(MessageFunction):
    """PhysiCK message: bank + mixing.
      κ = KernelBank(z)  -> [E,M]
      α = CoeffHeadKAN(z, mem_src, mem_dst) -> [E,M]
      s = sum_m α_m κ_m -> [E]
      msg = s * dir(edge_type)  -> [E, msg_dim]
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
            # Step-1 knob: choose coefficient head implementation.
            coeff_impl: str = "mlp",   # mlp|kan
            # Optional: replace the scalar dot-product mixer with an MLP mixer.
            mix_impl: str = "dot",     # dot|mlp
            coeff_hidden: int = 64,
            coeff_chunk: int = 256,
            mix_hidden: int | None = None,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.msg_dim = msg_dim
        self.use_edge_type = use_edge_type
        self.num_kernels = num_kernels
        self.bank = KernelBank(edge_dim=edge_dim, num_kernels=num_kernels)

        coeff_impl = str(coeff_impl).lower()
        if coeff_impl not in {"mlp", "kan"}:
            raise ValueError(f"coeff_impl must be 'mlp' or 'kan', got {coeff_impl!r}")

        if coeff_impl == "mlp":
            self.coeff = CoeffHeadMLP(
                edge_dim=edge_dim,
                mem_dim=mem_dim,
                M=self.num_kernels,
                hidden=max(64, int(coeff_hidden)),
                dropout=float(dropout),
            )
        else:
            self.coeff = CoeffHeadKAN(
                edge_dim=edge_dim,
                mem_dim=mem_dim,
                M=self.num_kernels,
                hidden=int(coeff_hidden),
                num_knots=int(num_knots),
                coeff_chunk=int(coeff_chunk),
                dropout=float(dropout),
            )

        self.mix_impl = str(mix_impl).lower()
        if self.mix_impl not in {"dot", "mlp"}:
            raise ValueError(f"mix_impl must be 'dot' or 'mlp', got {self.mix_impl!r}")
        if use_edge_type:
            self.type_emb = nn.Embedding(edge_type_vocab, msg_dim)
        else:
            self.type_emb = None
        self.scale = nn.Linear(1, msg_dim, bias=False)

        # Optional MLP mixer (no Python loops; fully batched).
        self.mix_mlp = None
        if self.mix_impl == "mlp":
            in_dim = int(edge_dim + 2 * mem_dim + 2 * self.num_kernels)
            h = int(mix_hidden) if mix_hidden is not None else max(128, msg_dim)
            self.mix_mlp = nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(h, msg_dim),
            )

        print(
            f"[PhysiCK] coeff={type(self.coeff).__name__} mix={self.mix_impl} M={self.num_kernels}",
            flush=True,
        )

    def forward(self, mem_src: torch.Tensor, mem_dst: torch.Tensor,
                z_ij: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        # Compute physics kernels and mixture weights.
        # Keep these in fp32 for stability; downstream can autocast as needed.
        kappa = self.bank(z_ij).float()                    # [E,M]
        alpha = self.coeff(z_ij, mem_src, mem_dst).float() # [E,M]

        if self.mix_impl == "dot":
            s = (alpha * kappa).sum(dim=-1, keepdim=True)  # [E,1]
            base = self.scale(s)                            # [E,msg_dim]
        else:
            feat = torch.cat([z_ij.float(), mem_src.float(), mem_dst.float(), kappa, alpha], dim=-1)
            base = self.mix_mlp(feat)                       # [E,msg_dim]
        if self.use_edge_type:
            if edge_type is None:
                edge_type = torch.zeros((z_ij.size(0),), dtype=torch.long, device=z_ij.device)
            d = self.type_emb(edge_type)        # [E,msg_dim]
            return base * torch.tanh(d)
        return base
