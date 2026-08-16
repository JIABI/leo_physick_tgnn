from __future__ import annotations
import abc
import torch
import torch.nn as nn

class MessageFunction(nn.Module, metaclass=abc.ABCMeta):
    """Unified message function interface.
    forward:
      mem_src: [E, mem_dim]
      mem_dst: [E, mem_dim]
      z_ij:    [E, edge_dim]
      edge_type: [E] optional long
    returns:
      msg: [E, msg_dim]
    """
    @abc.abstractmethod
    def forward(self, mem_src: torch.Tensor, mem_dst: torch.Tensor,
                z_ij: torch.Tensor, edge_type: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
