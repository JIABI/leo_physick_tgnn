from __future__ import annotations

import torch

import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from .interface import MessageFunction


class KANLinear(nn.Module):
    """

    Memory-efficient KAN-like layer using per-input 1D piecewise-linear splines.

    Key changes:

      - Avoid building [O,B,D] tensors via advanced indexing.

      - Accumulate over input dims in a loop: per-dim buffer is [O,B].

      - Keep computation in fp32 for stability (and to avoid AMP dtype issues).

    """

    def __init__(self, in_dim: int, out_dim: int, num_knots: int = 16):
        super().__init__()

        assert num_knots >= 2

        self.in_dim = in_dim

        self.out_dim = out_dim

        self.num_knots = num_knots

        grid = torch.linspace(-1.0, 1.0, num_knots)

        self.register_buffer("grid", grid)

        # values: [O, D, K]

        self.values = nn.Parameter(torch.zeros(out_dim, in_dim, num_knots))

        nn.init.uniform_(self.values, a=-0.05, b=0.05)

        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        x: [B, D]

        returns: [B, O] in fp32

        """

        B, D = x.shape

        assert D == self.in_dim, f"KANLinear expected in_dim={self.in_dim}, got {D}"

        # Clamp input to grid range; compute bucket + interpolation weights in fp32

        x_clamped = torch.clamp(x, -1.0, 1.0)

        x_f = x_clamped.float()

        grid_f = self.grid.float()

        # bucketize: right index in [0..K], then left/right in [0..K-1]

        right = torch.bucketize(x_f, grid_f)  # [B,D]

        left = torch.clamp(right - 1, 0, self.num_knots - 2)  # [B,D]

        right = left + 1  # [B,D]

        g0 = grid_f[left]  # [B,D]

        g1 = grid_f[right]  # [B,D]

        w = (x_f - g0) / (g1 - g0 + 1e-12)  # [B,D] fp32

        # Accumulate in fp32

        y = torch.zeros((B, self.out_dim), device=x.device, dtype=torch.float32)

        # values param (fp32), move is implicit with module device

        v = self.values

        # Loop over input dims; per-dim temporary is [O,B]

        for d in range(D):
            idx_l = left[:, d]  # [B]

            idx_r = right[:, d]  # [B]

            wd = w[:, d].unsqueeze(0)  # [1,B]

            vd = v[:, d, :]  # [O,K]

            v_l = vd.index_select(1, idx_l)  # [O,B]

            v_r = vd.index_select(1, idx_r)  # [O,B]

            interp = (1.0 - wd) * v_l + wd * v_r  # [O,B]

            y += interp.transpose(0, 1)  # [B,O]

        # Skip connection in fp32

        y = y + self.skip(x_clamped).float()

        return y


class KANMessage(MessageFunction):
    """

    Message function wrapper.

    Uses checkpointing to avoid storing all spline intermediates for backward,

    which is crucial when E (edge_chunk) and input dim are large.

    """

    def __init__(self, mem_dim: int, edge_dim: int, msg_dim: int, num_knots: int = 16):
        super().__init__()

        self.kan = KANLinear(2 * mem_dim + edge_dim, msg_dim, num_knots=num_knots)

    def forward(

            self,

            mem_src: torch.Tensor,

            mem_dst: torch.Tensor,

            z_ij: torch.Tensor,

            edge_type: torch.Tensor | None = None,

    ) -> torch.Tensor:
        x = torch.cat([mem_src, mem_dst, z_ij], dim=-1)

        x = torch.tanh(x)

        # Checkpoint to reduce memory: recompute KANLinear in backward.

        # PyTorch>=2.0 recommended use_reentrant=False.

        return checkpoint(self.kan, x, use_reentrant=False)




