from __future__ import annotations
import torch

def aggregate_messages(dst: torch.Tensor, msg: torch.Tensor, N: int, mode: str = "sum") -> torch.Tensor:
    out = torch.zeros((N, msg.size(-1)), device=msg.device, dtype=msg.dtype)
    out.index_add_(0, dst, msg)
    if mode == "mean":
        counts = torch.zeros((N,), device=msg.device, dtype=msg.dtype)
        ones = torch.ones((dst.numel(),), device=msg.device, dtype=msg.dtype)
        counts.index_add_(0, dst, ones)
        out = out / (counts.unsqueeze(-1) + 1e-12)
    return out
