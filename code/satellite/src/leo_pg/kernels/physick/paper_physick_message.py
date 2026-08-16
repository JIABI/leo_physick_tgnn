"""Current-manuscript PhysiCK message operator for Intensity--Flow graphs."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..interface import MessageFunction
from .paper_kernel_bank import PaperVectorKernelBank
from .projection import project_onto_l1_ball


class PaperPhysiCKMessage(MessageFunction):
    """Bounded signed mixture of shared vector-valued learned kernels.

    ``phi_in`` first forms the manuscript edge state from source memory,
    destination memory and the seven-field edge observation. ``phi_desc`` and
    the vector kernels are provided by :class:`PaperVectorKernelBank`.
    """

    def __init__(
        self,
        *,
        mem_dim: int,
        edge_dim: int,
        msg_dim: int,
        edge_type_vocab: int = 8,
        use_edge_type: bool = True,
        num_kernels: int = 16,
        latent_dim: int = 128,
        descriptor_dim: int = 16,
        kernel_hidden_dim: int = 128,
        coeff_hidden_dim: int = 128,
        dropout: float = 0.1,
        projection_radius: float = 1.0,
        operating_clip: float = 5.0,
        use_kernel_bank: bool = True,
    ) -> None:
        super().__init__()
        if float(projection_radius) <= 0:
            raise ValueError("projection_radius must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.mem_dim = int(mem_dim)
        self.edge_dim = int(edge_dim)
        self.msg_dim = int(msg_dim)
        self.num_kernels = int(num_kernels)
        self.latent_dim = int(latent_dim)
        self.projection_radius = float(projection_radius)
        if not isinstance(use_kernel_bank, bool):
            raise TypeError("use_kernel_bank must be boolean")
        self.use_kernel_bank = use_kernel_bank
        in_dim = 2 * self.mem_dim + self.edge_dim

        self.phi_in = nn.Sequential(
            nn.Linear(in_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.latent_dim),
        )
        if self.use_kernel_bank:
            self.bank: PaperVectorKernelBank | None = PaperVectorKernelBank(
                latent_dim=self.latent_dim,
                descriptor_dim=int(descriptor_dim),
                msg_dim=self.msg_dim,
                num_kernels=self.num_kernels,
                hidden_dim=int(kernel_hidden_dim),
                dropout=float(dropout),
                descriptor_clip=float(operating_clip),
            )
            self.coefficient_head: nn.Sequential | None = nn.Sequential(
                nn.Linear(self.latent_dim, int(coeff_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(coeff_hidden_dim), self.num_kernels),
            )
            self.bankless_message: nn.Sequential | None = None
        else:
            self.bank = None
            self.coefficient_head = None
            self.bankless_message = nn.Sequential(
                nn.Linear(self.latent_dim, int(kernel_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(kernel_hidden_dim), self.msg_dim),
            )
        self.output_norm = nn.LayerNorm(self.msg_dim)
        self.type_embedding = (
            nn.Embedding(int(edge_type_vocab), self.msg_dim)
            if use_edge_type
            else None
        )

    def forward(
        self,
        mem_src: torch.Tensor,
        mem_dst: torch.Tensor,
        z_ij: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mem_src.shape != mem_dst.shape:
            raise ValueError("source and destination memory shapes must match")
        if mem_src.ndim != 2 or mem_src.size(1) != self.mem_dim:
            raise ValueError(f"node memories must have shape [E,{self.mem_dim}]")
        if z_ij.shape != (mem_src.size(0), self.edge_dim):
            raise ValueError(f"z_ij must have shape [E,{self.edge_dim}]")
        if not all(
            torch.isfinite(value).all() for value in (mem_src, mem_dst, z_ij)
        ):
            raise ValueError("PhysiCK input contains NaN or Inf")
        latent = self.phi_in(torch.cat((mem_src, mem_dst, z_ij), dim=-1))
        if self.use_kernel_bank:
            assert self.bank is not None and self.coefficient_head is not None
            kernels, _ = self.bank(latent)
            raw_coefficients = self.coefficient_head(latent).float()
            coefficients = project_onto_l1_ball(
                raw_coefficients,
                radius=self.projection_radius,
            ).to(dtype=kernels.dtype)
            message = (coefficients.unsqueeze(-1) * kernels).sum(dim=1)
        else:
            assert self.bankless_message is not None
            message = self.bankless_message(latent)
        if self.type_embedding is not None:
            if edge_type is None:
                edge_type = torch.zeros(
                    z_ij.size(0), dtype=torch.long, device=z_ij.device
                )
            if edge_type.shape != (z_ij.size(0),):
                raise ValueError("edge_type must have shape [E]")
            message = message * torch.tanh(self.type_embedding(edge_type.long()))
        message = self.output_norm(message)
        if not torch.isfinite(message).all():
            raise FloatingPointError("PhysiCK message contains NaN or Inf")
        return message


__all__ = ["PaperPhysiCKMessage"]
