from __future__ import annotations

import torch

import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from ..kan import KANLinear


class CoeffHeadMLP(nn.Module):
    """

    Stable coefficient head: alpha = MLP([z_ij, mem_src, mem_dst]).

    Returns fp32.

    """

    def __init__(

            self,

            # `z_dim` is the descriptor dimension. Some older call-sites used
            # `edge_dim`; we accept both for robustness.
            z_dim: int | None = None,

            mem_dim: int = 0,

            M: int = 1,

            edge_dim: int | None = None,

            hidden: int = 128,

            dropout: float = 0.0,

            act: str = "relu",

    ):

        super().__init__()

        if z_dim is None:
            z_dim = edge_dim
        if z_dim is None:
            raise ValueError("CoeffHeadMLP requires z_dim (or edge_dim)")

        in_dim = int(z_dim) + 2 * int(mem_dim)

        act = act.lower()

        if act == "relu":

            act_layer = nn.ReLU()

        elif act == "gelu":

            act_layer = nn.GELU()

        else:

            act_layer = nn.Tanh()

        self.net = nn.Sequential(

            nn.Linear(in_dim, int(hidden)),

            act_layer,

            nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity(),

            nn.Linear(int(hidden), int(M)),

        )

    def forward(self, z_ij: torch.Tensor, mem_src: torch.Tensor, mem_dst: torch.Tensor) -> torch.Tensor:

        x = torch.cat([z_ij, mem_src, mem_dst], dim=-1)

        x = torch.tanh(x)

        return self.net(x).float()


class CoeffHeadKAN(nn.Module):
    """

    KAN coefficient head with:

      - autocast disabled (fp32)

      - activation checkpointing

      - edge chunking to cap B

    This is the heavy option; use only after the MLP path runs stably.

    """

    def __init__(

            self,

            z_dim: int | None = None,

            mem_dim: int = 0,

            M: int = 1,

            edge_dim: int | None = None,

            hidden: int = 64,

            num_knots: int = 16,

            dropout: float = 0.0,

            act: str = "relu",

            coeff_chunk: int = 512,

    ):

        super().__init__()

        if z_dim is None:
            z_dim = edge_dim
        if z_dim is None:
            raise ValueError("CoeffHeadKAN requires z_dim (or edge_dim)")

        in_dim = int(z_dim) + 2 * int(mem_dim)

        self.kan1 = KANLinear(in_dim, int(hidden), num_knots=int(num_knots))

        self.kan2 = KANLinear(int(hidden), int(M), num_knots=int(num_knots))

        act = act.lower()

        if act == "relu":

            self.act = nn.ReLU()

        elif act == "gelu":

            self.act = nn.GELU()

        else:

            self.act = nn.Tanh()

        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        self.coeff_chunk = int(coeff_chunk)

    def _kan1(self, t: torch.Tensor) -> torch.Tensor:

        return self.kan1(t)

    def _kan2(self, t: torch.Tensor) -> torch.Tensor:

        return self.kan2(t)

    def forward(self, z_ij: torch.Tensor, mem_src: torch.Tensor, mem_dst: torch.Tensor) -> torch.Tensor:

        x = torch.cat([z_ij, mem_src, mem_dst], dim=-1)

        x = torch.tanh(x)

        E = x.size(0)

        c = max(1, self.coeff_chunk)

        outs = []

        # Force fp32 inside KAN to avoid AMP half intermediates & dtype mismatches

        with torch.autocast("cuda", enabled=False):
            x32 = x.float()

            for s in range(0, E, c):
                xb = x32[s: s + c]

                h = checkpoint(self._kan1, xb, use_reentrant=False)

                h = self.act(h)

                h = self.drop(h)

                a = checkpoint(self._kan2, h, use_reentrant=False)

                outs.append(a)

        return torch.cat(outs, dim=0).float()


# ---- Backwards-compatible alias ----

# Many places import "CoeffHead". Default to the stable MLP variant.

CoeffHead = CoeffHeadMLP

