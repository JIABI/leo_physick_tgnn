"""Intensity--Flow PhysiCK kernel bank described by the current manuscript.

The manuscript fixes the operator structure, shared descriptor map, kernel-bank
size, edge-conditioned signed mixing, and L1 projection.  It does not publish a
closed-form list of sixteen kernels.  This implementation therefore realizes
the stated learnable vector-kernel bank directly, with every architectural
choice explicit in configuration and checkpoint state.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PaperVectorKernelBank(nn.Module):
    """Shared ``phi_desc`` plus M vector-valued learned kernels."""

    def __init__(
        self,
        *,
        latent_dim: int,
        descriptor_dim: int,
        msg_dim: int,
        num_kernels: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        descriptor_clip: float = 5.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("descriptor_dim", descriptor_dim),
            ("msg_dim", msg_dim),
            ("num_kernels", num_kernels),
            ("hidden_dim", hidden_dim),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if float(descriptor_clip) <= 0:
            raise ValueError("descriptor_clip must be positive")
        self.latent_dim = int(latent_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.msg_dim = int(msg_dim)
        self.num_kernels = int(num_kernels)
        self.descriptor_clip = float(descriptor_clip)

        self.phi_desc = nn.Sequential(
            nn.Linear(self.latent_dim, self.descriptor_dim),
            nn.Tanh(),
            nn.LayerNorm(self.descriptor_dim),
        )
        self.kernel_network = nn.Sequential(
            nn.Linear(self.descriptor_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_kernels * self.msg_dim),
        )

    def forward(self, latent_edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_edge.ndim != 2 or latent_edge.size(1) != self.latent_dim:
            raise ValueError(
                f"latent_edge must have shape [E,{self.latent_dim}]"
            )
        if not torch.isfinite(latent_edge).all():
            raise ValueError("latent_edge contains NaN or Inf")
        descriptor = self.phi_desc(
            latent_edge.clamp(-self.descriptor_clip, self.descriptor_clip)
        )
        kernels = self.kernel_network(descriptor).reshape(
            latent_edge.size(0), self.num_kernels, self.msg_dim
        )
        if not torch.isfinite(kernels).all():
            raise FloatingPointError("PhysiCK vector kernel bank produced NaN or Inf")
        return kernels, descriptor


__all__ = ["PaperVectorKernelBank"]
