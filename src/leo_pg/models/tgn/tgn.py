from __future__ import annotations
from typing import Literal, Optional

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

    Key implementation detail:
      - To avoid CUDA OOM when E is large, we support edge-chunked message computation
        for sum/mean aggregators.
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
        edge_chunk: int = 50_000,
        inject: float = 0.1,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.mem_dim = mem_dim
        self.msg_dim = msg_dim
        self.emb_dim = emb_dim
        self.aggregator = str(aggregator).lower()
        self.edge_chunk = int(edge_chunk)
        self.inject = float(inject)

        self.memory = MemoryBank(node_in_dim=node_in_dim, mem_dim=mem_dim)
        self.readout = Readout(mem_dim=mem_dim, emb_dim=emb_dim)
        self.head = head

        mt = str(message_type).lower()
        if mt == "mlp":
            self.msg_fn = MLPMessage(mem_dim, edge_in_dim, msg_dim, dropout=dropout)
        elif mt == "kan":
            self.msg_fn = KANMessage(mem_dim, edge_in_dim, msg_dim, num_knots=16)
        elif mt == "physick":
            self.msg_fn = PhysiCKMessage(
                mem_dim,
                edge_in_dim,
                msg_dim,
                edge_type_vocab=edge_type_vocab,
                use_edge_type=use_edge_type,
                num_kernels=12,
                num_knots=16,
            )
        else:
            raise ValueError(f"Unknown message_type={message_type}")

        # Project msg_dim to mem_dim for GRU input
        self.msg_to_mem = nn.Linear(msg_dim, mem_dim)

    @torch.no_grad()
    def _aggregate_chunked_sum_mean(
        self,
        mem: torch.Tensor,              # [N, mem_dim]
        edge_index: torch.Tensor,       # [2, E]
        edge_z: torch.Tensor,           # [E, edge_in_dim]
        edge_type: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Chunked message computation + aggregation for sum/mean.

        Returns:
          agg_msg: [N, msg_dim]
        """
        N = mem.size(0)
        E = edge_index.size(1)
        device = mem.device
        dtype = mem.dtype

        agg = torch.zeros((N, self.msg_dim), device=device, dtype=dtype)

        if self.aggregator == "mean":
            counts = torch.zeros((N, 1), device=device, dtype=dtype)
        else:
            counts = None

        src_all = edge_index[0]
        dst_all = edge_index[1]

        chunk = max(1, int(self.edge_chunk))

        for st in range(0, E, chunk):
            ed = min(E, st + chunk)
            sl = slice(st, ed)

            src = src_all[sl]
            dst = dst_all[sl]
            z = edge_z[sl]
            et = edge_type[sl] if edge_type is not None else None

            mem_src = mem[src]  # [e, mem_dim]
            mem_dst = mem[dst]  # [e, mem_dim]

            msg = self.msg_fn(mem_src, mem_dst, z, edge_type=et)  # [e, msg_dim]
            agg.index_add_(0, dst, msg)

            if counts is not None:
                ones = torch.ones((dst.numel(), 1), device=device, dtype=dtype)
                counts.index_add_(0, dst, ones)

        if counts is not None:
            agg = agg / counts.clamp_min(1.0)

        return agg

    def forward_episode(self, episode: dict, device: torch.device) -> dict:
        """Run through an episode and return predictions per step.

        episode must contain:
          episode["steps"] = list of dicts with keys:
            - node_x: [N, node_in_dim]
            - edge_index: [2, E]
            - edge_z: [E, edge_in_dim]
            - (optional) edge_type: [E]
            - y: task target per step (shape depends on head)
        """
        steps = episode["steps"]
        mem = None
        preds = []
        ys = []

        for s in steps:
            node_x = s["node_x"].to(device).float()
            edge_index = s["edge_index"].to(device).long()
            edge_z = s["edge_z"].to(device).float()

            edge_type = s.get("edge_type", None)
            if edge_type is not None:
                edge_type = edge_type.to(device).long()

            y = s["y"].to(device).float()

            if mem is None:
                mem = self.memory.init(node_x)  # [N, mem_dim]

            # Aggregate messages into nodes (in msg-space), then project to mem-space.
            if edge_index.numel() > 0 and edge_index.size(1) > 0:
                if self.aggregator in ("sum", "mean"):
                    agg_msg = self._aggregate_chunked_sum_mean(
                        mem=mem,
                        edge_index=edge_index,
                        edge_z=edge_z,
                        edge_type=edge_type,
                    )  # [N, msg_dim]
                else:
                    # Fallback: non-chunked path (may OOM for very large E).
                    # If you need attention-style aggregation, implement a chunked version of it.
                    src, dst = edge_index[0], edge_index[1]
                    mem_src = mem[src]
                    mem_dst = mem[dst]
                    msg = self.msg_fn(mem_src, mem_dst, edge_z, edge_type=edge_type)  # [E,msg_dim]

                    # simple sum fallback (to keep behavior defined)
                    agg_msg = torch.zeros((node_x.size(0), self.msg_dim), device=device, dtype=mem.dtype)
                    agg_msg.index_add_(0, dst, msg)

                agg_mem = self.msg_to_mem(agg_msg)  # [N, mem_dim]
            else:
                agg_mem = torch.zeros((node_x.size(0), self.mem_dim), device=device, dtype=mem.dtype)

            mem = self.memory.update(mem, agg_mem, node_x, inject=self.inject)  # [N, mem_dim]
            emb = self.readout(mem)                                             # [N, emb_dim]
            pred = self.head(emb, step=s)

            preds.append(pred)
            ys.append(y)

        return {"preds": preds, "ys": ys}

