from __future__ import annotations

from typing import Any, Dict, Literal, Union

import torch
import torch.nn as nn

from ...kernels.mlp import MLPMessage
from ...kernels.kan import KANMessage
from ...kernels.physick.physick_message import PhysiCKMessage
from .memory import MemoryBank
from .readout import Readout

MessageType = Literal["mlp", "kan", "physick"]


class TGN(nn.Module):
    """Temporal Graph Network with pluggable message function and a task head.

    Notes
    -----
    - `cfg` is optional but recommended. It is used to pull:
        * PhysiCK knobs: cfg["physick"] or cfg["model"]["physick"]
        * edge_chunk: cfg["train"]["edge_chunk"]
      Without cfg, sensible defaults are used.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        mem_dim: int,
        msg_dim: int,
        emb_dim: int,
        message_type: str,
        aggregator: str,
        dropout: float,
        use_edge_type: bool,
        edge_type_vocab: int,
        head: nn.Module,
        edge_chunk: int = 50000,
        cfg: Dict[str, Any] | None = None,
    ):
        super().__init__()

        # <<< critical: prevents AttributeError when accessing self.cfg in init
        self.cfg: Dict[str, Any] = cfg or {}

        self.node_in_dim = int(node_in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.mem_dim = int(mem_dim)
        self.msg_dim = int(msg_dim)
        self.emb_dim = int(emb_dim)
        self.aggregator = str(aggregator).lower()

        self.memory = MemoryBank(node_in_dim=self.node_in_dim, mem_dim=self.mem_dim)
        self.readout = Readout(mem_dim=self.mem_dim, emb_dim=self.emb_dim)
        self.head = head

        mt = str(message_type).lower()
        if mt == "mlp":
            self.msg_fn = MLPMessage(self.mem_dim, self.edge_in_dim, self.msg_dim, dropout=float(dropout))
        elif mt == "kan":
            self.msg_fn = KANMessage(self.mem_dim, self.edge_in_dim, self.msg_dim, num_knots=16)
        elif mt == "physick":
            # Optional PhysiCK-specific knobs live under either:
            #   cfg["physick"][...]
            # or
            #   cfg["model"]["physick"][...]
            physick_cfg: Dict[str, Any] = {}
            if isinstance(self.cfg.get("physick"), dict):
                physick_cfg.update(self.cfg.get("physick", {}))
            if isinstance(self.cfg.get("model", {}).get("physick"), dict):
                physick_cfg.update(self.cfg.get("model", {}).get("physick", {}))

            self.msg_fn = PhysiCKMessage(
                mem_dim=self.mem_dim,
                edge_dim=self.edge_in_dim,
                msg_dim=self.msg_dim,
                edge_type_vocab=int(edge_type_vocab),
                use_edge_type=bool(use_edge_type),
                num_kernels=int(physick_cfg.get("num_kernels", 12)),
                num_knots=int(physick_cfg.get("num_knots", 16)),
                coeff_impl=str(physick_cfg.get("coeff_impl", "mlp")),
                mix_impl=str(physick_cfg.get("mix_impl", "dot")),
                coeff_hidden=int(physick_cfg.get("coeff_hidden", 64)),
                coeff_chunk=int(physick_cfg.get("coeff_chunk", 256)),
                mix_hidden=int(physick_cfg.get("mix_hidden", 128)),
                dropout=float(physick_cfg.get("dropout", dropout)),
            )
        else:
            raise ValueError(f"Unknown message_type={message_type}")

        self.edge_chunk = int(edge_chunk)
        self.msg_to_mem = nn.Linear(self.msg_dim, self.mem_dim)

        if self.aggregator not in ("sum", "mean"):
            raise ValueError(f"Unknown aggregator={aggregator} (expected sum|mean)")

    @staticmethod
    def _as_device(device: Union[str, torch.device]) -> torch.device:
        return device if isinstance(device, torch.device) else torch.device(str(device))

    def forward_step(self, step: dict, mem: torch.Tensor | None, device: torch.device):
        """One-step forward. Returns (pred, y, new_mem)."""

        node_x = step["node_x"].to(device).float()
        edge_index = step["edge_index"].to(device).long()
        edge_z = step["edge_z"].to(device).float()
        edge_type = step.get("edge_type", None)
        if edge_type is not None:
            edge_type = edge_type.to(device).long()

        y = step["y"].to(device).float()

        if mem is None:
            mem = self.memory.init(node_x)  # [N,mem_dim]

        N = int(node_x.size(0))
        E = int(edge_index.size(1))

        if E > 0:
            src_all = edge_index[0]
            dst_all = edge_index[1]

            agg = torch.zeros((N, self.msg_dim), device=device, dtype=torch.float32)
            if self.aggregator == "mean":
                counts = torch.zeros((N, 1), device=device, dtype=torch.float32)
            else:
                counts = None

            chunk = max(1, int(self.edge_chunk))

            for start in range(0, E, chunk):
                end = min(E, start + chunk)
                src = src_all[start:end]
                dst = dst_all[start:end]
                z = edge_z[start:end]

                mem_src = mem[src]
                mem_dst = mem[dst]
                et = edge_type[start:end] if edge_type is not None else None

                msg = self.msg_fn(mem_src, mem_dst, z, edge_type=et)  # [e,msg_dim]
                if msg.dtype != agg.dtype:
                    msg = msg.to(dtype=agg.dtype)
                agg.index_add_(0, dst, msg)

                if counts is not None:
                    ones = torch.ones((dst.size(0), 1), device=device, dtype=torch.float32)
                    counts.index_add_(0, dst, ones)

            if counts is not None:
                agg = agg / counts.clamp_min(1.0)

            agg_mem = self.msg_to_mem(agg)  # [N,mem_dim]
        else:
            agg_mem = torch.zeros((N, self.mem_dim), device=device, dtype=torch.float32)

        mem = self.memory.update(mem, agg_mem, node_x, inject=0.1)
        emb = self.readout(mem)
        pred = self.head(emb, step=step)

        return pred, y, mem

    def forward_episode(self, episode: dict, device: torch.device) -> dict:
        """Compatibility path for eval: returns lists of preds/ys over all steps."""
        steps = episode["steps"]
        mem = None
        preds, ys = [], []
        for s in steps:
            pred, y, mem = self.forward_step(s, mem, device)
            preds.append(pred)
            ys.append(y)
        return {"preds": preds, "ys": ys}
