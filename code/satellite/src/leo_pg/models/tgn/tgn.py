from __future__ import annotations

import copy
import math
from typing import Any, Dict, Literal, Union

import torch
import torch.nn as nn

from ...kernels.mlp import MLPMessage
from ...kernels.kan import KANMessage
from ...kernels.physick.physick_message import PhysiCKMessage
from ...kernels.physick.paper_physick_message import PaperPhysiCKMessage
from .memory import MemoryBank
from .readout import Readout
from .time import HarmonicTimeEncoder
from ...sim.state import PAPER_EDGE_FEATURE_NAMES

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
        model_options = self.cfg.get("model", {})
        if not isinstance(model_options, dict):
            raise TypeError("model configuration must be a mapping")
        temporal_memory = model_options.get("temporal_memory", True)
        if not isinstance(temporal_memory, bool):
            raise TypeError("model.temporal_memory must be boolean")
        self.temporal_memory = temporal_memory
        raw_time_dimension = model_options.get("time_encoding_dim")
        if raw_time_dimension is None:
            self.time_encoding_dim = 0
            self.time_encoder = None
            self.time_to_message = None
        else:
            if isinstance(raw_time_dimension, bool) or int(raw_time_dimension) != raw_time_dimension:
                raise TypeError("model.time_encoding_dim must be a positive integer")
            self.time_encoding_dim = int(raw_time_dimension)
            if self.time_encoding_dim <= 0:
                raise ValueError("model.time_encoding_dim must be a positive integer")
            self.time_encoder = HarmonicTimeEncoder(self.time_encoding_dim)
            self.time_to_message = nn.Linear(
                self.time_encoding_dim,
                self.msg_dim,
                bias=False,
            )
        raw_mask = model_options.get("edge_feature_mask", [])
        if not isinstance(raw_mask, (list, tuple)):
            raise TypeError("model.edge_feature_mask must be a list")
        mask_indices: list[int] = []
        for value in raw_mask:
            if isinstance(value, str):
                if value not in PAPER_EDGE_FEATURE_NAMES:
                    raise ValueError(f"unknown edge feature mask name: {value}")
                index = PAPER_EDGE_FEATURE_NAMES.index(value)
            else:
                if isinstance(value, bool) or int(value) != value:
                    raise TypeError("edge feature mask entries must be names or integers")
                index = int(value)
                if not 0 <= index < self.edge_in_dim:
                    raise ValueError("edge feature mask index is out of range")
            mask_indices.append(index)
        self.edge_feature_mask = tuple(sorted(set(mask_indices)))

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

            projection_radius_value = physick_cfg.get("projection_radius", 1.0)
            projection_radius = None if projection_radius_value is None else float(projection_radius_value)
            implementation = str(
                physick_cfg.get("implementation", "legacy_cox_prototype")
            ).strip().lower()
            if implementation in {"paper", "paper_intensity_flow", "intensity_flow"}:
                if projection_radius is None:
                    raise ValueError(
                        "paper Intensity--Flow PhysiCK requires a finite projection radius"
                    )
                self.msg_fn = PaperPhysiCKMessage(
                    mem_dim=self.mem_dim,
                    edge_dim=self.edge_in_dim,
                    msg_dim=self.msg_dim,
                    edge_type_vocab=int(edge_type_vocab),
                    use_edge_type=bool(use_edge_type),
                    num_kernels=int(physick_cfg.get("num_kernels", 16)),
                    latent_dim=int(physick_cfg.get("latent_dim", self.msg_dim)),
                    descriptor_dim=int(physick_cfg.get("descriptor_dim", 16)),
                    kernel_hidden_dim=int(
                        physick_cfg.get("kernel_hidden_dim", self.msg_dim)
                    ),
                    coeff_hidden_dim=int(
                        physick_cfg.get("coeff_hidden", self.msg_dim)
                    ),
                    dropout=float(physick_cfg.get("dropout", dropout)),
                    projection_radius=projection_radius,
                    operating_clip=float(physick_cfg.get("operating_clip", 5.0)),
                    use_kernel_bank=bool(physick_cfg.get("use_kernel_bank", True)),
                )
            elif implementation in {"legacy", "legacy_cox_prototype"}:
                self.msg_fn = PhysiCKMessage(
                    mem_dim=self.mem_dim,
                    edge_dim=self.edge_in_dim,
                    msg_dim=self.msg_dim,
                    edge_type_vocab=int(edge_type_vocab),
                    use_edge_type=bool(use_edge_type),
                    num_kernels=int(physick_cfg.get("num_kernels", 16)),
                    num_knots=int(physick_cfg.get("num_knots", 16)),
                    coeff_impl=str(physick_cfg.get("coeff_impl", "mlp")),
                    mix_impl=str(physick_cfg.get("mix_impl", "dot")),
                    coeff_hidden=int(physick_cfg.get("coeff_hidden", 64)),
                    coeff_chunk=int(physick_cfg.get("coeff_chunk", 256)),
                    mix_hidden=int(physick_cfg.get("mix_hidden", 128)),
                    dropout=float(physick_cfg.get("dropout", dropout)),
                    projection_radius=projection_radius,
                )
            else:
                raise ValueError(
                    "model.physick.implementation must be paper_intensity_flow or "
                    "legacy_cox_prototype"
                )
        else:
            raise ValueError(f"Unknown message_type={message_type}")

        self.edge_chunk = int(edge_chunk)
        self.msg_to_mem = nn.Linear(self.msg_dim, self.mem_dim)
        configured_layers = self.cfg.get("model", {}).get(
            "message_passing_layers", 1
        )
        if (
            isinstance(configured_layers, bool)
            or int(configured_layers) != configured_layers
            or int(configured_layers) <= 0
        ):
            raise ValueError("model.message_passing_layers must be a positive integer")
        self.num_message_passing_layers = int(configured_layers)
        configured_injection = self.cfg.get("model", {}).get("node_injection", 0.1)
        if isinstance(configured_injection, bool):
            raise TypeError("model.node_injection must be numeric")
        self.node_injection = float(configured_injection)
        if (
            not math.isfinite(self.node_injection)
            or not 0.0 <= self.node_injection <= 1.0
        ):
            raise ValueError("model.node_injection must lie in [0,1]")
        self.extra_msg_fns = nn.ModuleList(
            [copy.deepcopy(self.msg_fn) for _ in range(self.num_message_passing_layers - 1)]
        )
        self.extra_msg_to_mem = nn.ModuleList(
            [
                nn.Linear(self.msg_dim, self.mem_dim)
                for _ in range(self.num_message_passing_layers - 1)
            ]
        )

        if self.aggregator not in ("sum", "mean"):
            raise ValueError(f"Unknown aggregator={aggregator} (expected sum|mean)")

    @staticmethod
    def _as_device(device: Union[str, torch.device]) -> torch.device:
        return device if isinstance(device, torch.device) else torch.device(str(device))

    def predict_step(self, step: dict, mem: torch.Tensor | None, device: torch.device):
        """Run one inference step without reading a supervision target."""
        node_x = step["node_x"].to(device).float()
        edge_index = step["edge_index"].to(device).long()
        edge_z = step["edge_z"].to(device).float()
        if self.edge_feature_mask:
            edge_z = edge_z.clone()
            edge_z[:, list(self.edge_feature_mask)] = 0.0
        edge_type = step.get("edge_type", None)
        if edge_type is not None:
            edge_type = edge_type.to(device).long()

        time_message: torch.Tensor | None = None
        if self.time_encoder is not None:
            if "t" not in step:
                raise ValueError(
                    "step.t is required when model.time_encoding_dim is configured"
                )
            time_encoding = self.time_encoder(
                step["t"],
                device=device,
                dtype=node_x.dtype,
            )
            if self.time_to_message is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("configured time encoder lacks its message projection")
            time_message = self.time_to_message(time_encoding)

        if mem is None or not self.temporal_memory:
            mem = self.memory.init(node_x)  # [N,mem_dim]

        N = int(node_x.size(0))
        E = int(edge_index.size(1))

        message_functions = (self.msg_fn, *tuple(self.extra_msg_fns))
        projections = (self.msg_to_mem, *tuple(self.extra_msg_to_mem))
        src_all = edge_index[0]
        dst_all = edge_index[1]
        chunk = max(1, int(self.edge_chunk))
        for message_function, projection in zip(message_functions, projections):
            if E > 0:
                agg = torch.zeros(
                    (N, self.msg_dim), device=device, dtype=torch.float32
                )
                counts = (
                    torch.zeros((N, 1), device=device, dtype=torch.float32)
                    if self.aggregator == "mean"
                    else None
                )
                for start in range(0, E, chunk):
                    end = min(E, start + chunk)
                    src = src_all[start:end]
                    dst = dst_all[start:end]
                    z = edge_z[start:end]
                    et = edge_type[start:end] if edge_type is not None else None
                    msg = message_function(
                        mem[src], mem[dst], z, edge_type=et
                    )
                    if time_message is not None:
                        msg = msg + time_message.unsqueeze(0)
                    agg.index_add_(0, dst, msg.to(dtype=agg.dtype))
                    if counts is not None:
                        counts.index_add_(
                            0,
                            dst,
                            torch.ones(
                                (dst.size(0), 1),
                                device=device,
                                dtype=torch.float32,
                            ),
                        )
                if counts is not None:
                    agg = agg / counts.clamp_min(1.0)
                agg_mem = projection(agg)
            else:
                agg_mem = torch.zeros(
                    (N, self.mem_dim), device=device, dtype=torch.float32
                )
            mem = self.memory.update(
                mem,
                agg_mem,
                node_x,
                inject=self.node_injection,
            )
        emb = self.readout(mem)
        pred = self.head(emb, step=step)

        return pred, mem

    def forward_step(self, step: dict, mem: torch.Tensor | None, device: torch.device):
        """One supervised step. Returns ``(prediction, target, new_memory)``."""
        pred, mem = self.predict_step(step, mem, device)
        y = step["y"].to(device).float()

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
