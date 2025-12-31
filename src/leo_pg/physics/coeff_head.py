from __future__ import annotations

import torch

import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from leo_pg.kernels.kan import KANLinear


class CoeffHeadMLP(nn.Module):
    """

    Stable coefficient head: logits = MLP([z, mem_src, mem_dst]).

    Keeps the same external signature pattern as the legacy CoeffHeadKAN.

    """

    def __init__(

            self,

            edge_dim: int,

            mem_dim: int,

            num_kernels: int,

            use_mem: bool = True,

            hidden: int | None = None,

            dropout: float = 0.0,

            act: str = "relu",

    ):

        super().__init__()

        self.use_mem = bool(use_mem)

        self.num_kernels = int(num_kernels)

        in_dim = int(edge_dim) + (2 * int(mem_dim) if self.use_mem else 0)

        hidden = int(hidden or max(64, self.num_kernels))

        act = act.lower()

        if act == "relu":

            act_layer = nn.ReLU()

        elif act == "gelu":

            act_layer = nn.GELU()

        else:

            act_layer = nn.Tanh()

        self.net = nn.Sequential(

            nn.Linear(in_dim, hidden),

            act_layer,

            nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity(),

            nn.Linear(hidden, self.num_kernels),

        )

    def forward(self, z: torch.Tensor, mem_src: torch.Tensor, mem_dst: torch.Tensor) -> torch.Tensor:

        if self.use_mem:

            x = torch.cat([z, mem_src, mem_dst], dim=-1)

        else:

            x = z

        x = torch.tanh(x)

        return self.net(x).float()


class CoeffHeadKAN(nn.Module):
    """

    KAN coefficient head with:

      - fp32 inside KAN (autocast disabled)

      - activation checkpointing

      - edge chunking (caps B) to reduce peak memory

    Signature is backward-compatible with your original file.

    """

    def __init__(

            self,

            edge_dim: int,

            mem_dim: int,

            num_kernels: int,

            num_knots: int = 16,

            use_mem: bool = True,

            hidden: int | None = None,

            coeff_chunk: int = 512,

            dropout: float = 0.0,

            act: str = "tanh",

    ):

        super().__init__()

        self.use_mem = bool(use_mem)

        self.num_kernels = int(num_kernels)

        self.coeff_chunk = int(coeff_chunk)

        in_dim = int(edge_dim) + (2 * int(mem_dim) if self.use_mem else 0)

        hidden = int(hidden or max(32, self.num_kernels))

        self.kan1 = KANLinear(in_dim, hidden, num_knots=num_knots)

        act = act.lower()

        if act == "relu":

            self.act = nn.ReLU()

        elif act == "gelu":

            self.act = nn.GELU()

        else:

            self.act = nn.Tanh()

        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        self.lin = nn.Linear(hidden, self.num_kernels)

    def _kan1(self, t: torch.Tensor) -> torch.Tensor:

        return self.kan1(t)

    def forward(self, z: torch.Tensor, mem_src: torch.Tensor, mem_dst: torch.Tensor) -> torch.Tensor:

        if self.use_mem:

            x = torch.cat([z, mem_src, mem_dst], dim=-1)

        else:

            x = z

        x = torch.tanh(x)

        E = x.size(0)

        c = max(1, self.coeff_chunk)

        outs = []

        # Run KAN in fp32; avoid AMP half intermediates in spline interpolation.

        with torch.autocast("cuda", enabled=False):

            x32 = x.float()

            for s in range(0, E, c):
                xb = x32[s: s + c]

                h = checkpoint(self._kan1, xb, use_reentrant=False)

                h = self.act(h)

                h = self.drop(h)

                logits = self.lin(h)

                outs.append(logits)

        return torch.cat(outs, dim=0).float()


# Default alias: prefer the stable MLP head for multi-user experiments.

CoeffHead = CoeffHeadMLP


