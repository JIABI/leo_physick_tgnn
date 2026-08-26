"""Controlled full-capacity paper baselines with a common streaming API.

Every model implements::

    output, state = model.predict_step(step, state, device)

where ``output`` is an
``leo_pg.models.heads.intensity_flow.IntensityFlowOutput`` and ``state`` is
episode-local.  ``step`` follows ``ControlObservation.as_model_step()``:
``node_x`` is ``[N,F_node]``; candidate ``edge_index`` is ``[2,E]``; ``edge_z``
is ``[E,F_edge]``; and ``meta`` provides ``K_users`` and ``S_sats``.  Callers
must reset state to ``None`` between episodes and must move the module to the
same device passed to ``predict_step``.

Implemented systems
-------------------
``LTTRWorldModel`` is a real long-context temporal Transformer, not an alias to
the repository TGN.  Its controlled default is a two-layer edge-conditioned
spatial front-end followed by a six-layer, width-384 Transformer with eight
heads, FFN width 1536 and a 64-step context.  The current-association and dwell
edge channels supply the causal association/action history at every context
step. This targets the manuscript's
approximately 12M-parameter capacity class; exact parameter count depends on
input widths and must be logged by the experiment runner.  Rollout-loss horizon
and scheduled-sampling policy are training choices and are intentionally not
hidden inside the architecture.

``DAGWMWorldModel`` uses edge-conditioned multi-head GATv2 followed by a GRU
node-memory update. It prefers ``torch_geometric.nn.GATv2Conv`` when installed
and otherwise uses the included pure-PyTorch implementation. It exposes
``decision_aware_loss_hook``: a
differentiable soft-rank pairwise loss against the oracle fixed-score ordering.
The hook is additive; the trainer remains responsible for combining it with
the descriptor loss using an explicit coefficient.

``PhysiCKTemporalMixerWorldModel`` is the controlled temporal-mixer ablation.
It keeps the paper TGN's 128-dimensional node memory, 128-dimensional
``PaperPhysiCKMessage`` layers, directed candidate graph, node injection,
readout, and Intensity--Flow head.  Only the GRU update is replaced by an
official S4, Mamba2, or torchaudio Conformer sequence module.  These methods do
not route through the independent edge-conditioned LTT-R graph front-end.
Optional temporal-ablation dependencies raise an actionable error at
construction time and never select a toy fallback.

Factory configuration is read from ``paper_baseline``.  LTT-R and DA-GWM may
also be constructed from a direct baseline mapping.  S4/Mamba2/Conformer must
receive the resolved root mapping because their graph, PhysiCK operator and
head are inherited from root ``model`` and ``head`` rather than duplicated in
the baseline entry.  Supported kinds are ``ltt_r``, ``da_gwm``, ``s4``,
``mamba2`` and ``conformer``.  Explicit configuration assumptions do not claim
to recover a missing publication checkpoint.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..control.policy import normalized_ordinal_rank
from ..kernels.physick.paper_physick_message import PaperPhysiCKMessage
from ..models.heads.intensity_flow import IntensityFlowHead, IntensityFlowOutput
from ..models.tgn.readout import Readout
from ..sim.state import (
    PAPER_EDGE_FEATURE_NAMES,
    PAPER_NODE_FEATURE_NAMES,
    PolicyDescriptors,
)
from .backbones import (
    StreamingTemporalBackbone,
    TransformerTemporalBackbone,
    build_temporal_backbone,
)


class PaperStepPredictor(Protocol):
    """Structural protocol shared by paper baselines and inference adapters."""

    def predict_step(
        self,
        step: Mapping[str, Any],
        state: Any,
        device: torch.device | str,
    ) -> tuple[IntensityFlowOutput, Any]:
        """Predict next-step policy descriptors without consuming a target."""


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _dropout(value: Any) -> float:
    result = _nonnegative("dropout", value)
    if result >= 1.0:
        raise ValueError("dropout must lie in [0, 1)")
    return result


@dataclass(frozen=True)
class _PreparedPaperStep:
    node_x: torch.Tensor
    edge_index: torch.Tensor
    edge_z: torch.Tensor
    user_count: int
    satellite_count: int

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.size(1))


def _prepare_step(
    step: Mapping[str, Any],
    *,
    device: torch.device,
    node_in_dim: int,
    edge_in_dim: int,
) -> _PreparedPaperStep:
    if not isinstance(step, Mapping):
        raise TypeError("step must be a mapping")
    for required in ("node_x", "edge_index", "edge_z", "meta"):
        if required not in step:
            raise ValueError(f"paper baseline step is missing {required!r}")

    node_x = torch.as_tensor(step["node_x"], device=device).float()
    edge_index = torch.as_tensor(step["edge_index"], device=device)
    edge_z = torch.as_tensor(step["edge_z"], device=device).float()
    if node_x.ndim != 2 or node_x.size(1) != node_in_dim:
        raise ValueError(
            f"node_x must have shape [N,{node_in_dim}], got {tuple(node_x.shape)}"
        )
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("edge_index must have an integer dtype")
    edge_index = edge_index.long()
    if edge_z.shape != (edge_index.size(1), edge_in_dim):
        raise ValueError(
            f"edge_z must have shape [{edge_index.size(1)},{edge_in_dim}], "
            f"got {tuple(edge_z.shape)}"
        )
    if not bool(torch.isfinite(node_x).all()) or not bool(torch.isfinite(edge_z).all()):
        raise ValueError("paper baseline inputs must contain only finite values")

    edge_type = step.get("edge_type")
    if edge_type is not None:
        edge_type_tensor = torch.as_tensor(edge_type, device=device)
        if edge_type_tensor.ndim != 1 or edge_type_tensor.numel() != edge_index.size(1):
            raise ValueError("edge_type must have one value per edge")
        candidate = edge_type_tensor == 0
        edge_index = edge_index[:, candidate]
        edge_z = edge_z[candidate]

    meta = step["meta"]
    if not isinstance(meta, Mapping) or "K_users" not in meta:
        raise ValueError("step.meta.K_users is required")
    user_count = _positive_int("step.meta.K_users", meta["K_users"])
    satellite_count = _positive_int(
        "step.meta.S_sats",
        meta.get("S_sats", node_x.size(0) - user_count),
    )
    if user_count + satellite_count != node_x.size(0):
        raise ValueError("step K_users/S_sats do not match node_x")
    if edge_index.numel():
        source, destination = edge_index
        if (
            int(source.min()) < 0
            or int(source.max()) >= user_count
            or int(destination.min()) < user_count
            or int(destination.max()) >= user_count + satellite_count
        ):
            raise ValueError("paper candidate edges must run from users to satellites")
    return _PreparedPaperStep(
        node_x=node_x,
        edge_index=edge_index,
        edge_z=edge_z,
        user_count=user_count,
        satellite_count=satellite_count,
    )


class EdgeConditionedGraphLayer(nn.Module):
    """Bidirectional edge-conditioned update over the fixed candidate graph."""

    def __init__(self, *, model_dim: int, edge_in_dim: int, dropout: float) -> None:
        super().__init__()
        message_dim = 2 * model_dim + edge_in_dim
        self.to_destination = nn.Sequential(
            nn.Linear(message_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.to_source = nn.Sequential(
            nn.Linear(message_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        node_embedding: torch.Tensor,
        edge_index: torch.Tensor,
        edge_z: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return self.norm(node_embedding)
        source, destination = edge_index
        pair = torch.cat(
            (node_embedding[source], node_embedding[destination], edge_z),
            dim=-1,
        )
        aggregate = torch.zeros_like(node_embedding)
        degree = torch.zeros(
            (node_embedding.size(0), 1),
            dtype=node_embedding.dtype,
            device=node_embedding.device,
        )
        aggregate.index_add_(0, destination, self.to_destination(pair))
        aggregate.index_add_(0, source, self.to_source(pair))
        ones = torch.ones(
            (source.numel(), 1),
            dtype=node_embedding.dtype,
            device=node_embedding.device,
        )
        degree.index_add_(0, destination, ones)
        degree.index_add_(0, source, ones)
        aggregate = aggregate / degree.clamp_min(1.0)
        return self.norm(node_embedding + self.dropout(aggregate))


class LongContextGraphWorldModel(nn.Module):
    """Controlled graph front-end plus a real streaming temporal backbone."""

    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        model_dim: int,
        temporal_backbone: StreamingTemporalBackbone,
        spatial_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_dim: int | None = None,
        head_dropout: float | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.node_in_dim = _positive_int("node_in_dim", node_in_dim)
        self.edge_in_dim = _positive_int("edge_in_dim", edge_in_dim)
        self.model_dim = _positive_int("model_dim", model_dim)
        spatial_count = _positive_int("spatial_layers", spatial_layers)
        dropout_value = _dropout(dropout)
        if temporal_backbone.d_model != self.model_dim:
            raise ValueError("temporal backbone d_model must equal model_dim")
        self.cfg = dict(config or {})
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_in_dim, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.spatial = nn.ModuleList(
            [
                EdgeConditionedGraphLayer(
                    model_dim=self.model_dim,
                    edge_in_dim=self.edge_in_dim,
                    dropout=dropout_value,
                )
                for _ in range(spatial_count)
            ]
        )
        self.temporal_backbone = temporal_backbone
        self.head = IntensityFlowHead(
            in_dim=self.model_dim,
            edge_in_dim=self.edge_in_dim,
            hidden_dim=(
                self.model_dim
                if head_hidden_dim is None
                else _positive_int("head_hidden_dim", head_hidden_dim)
            ),
            dropout=(
                dropout_value
                if head_dropout is None
                else _dropout(head_dropout)
            ),
        )

    def predict_step(
        self,
        step: Mapping[str, Any],
        state: torch.Tensor | None,
        device: torch.device | str,
    ) -> tuple[IntensityFlowOutput, torch.Tensor]:
        target_device = torch.device(device)
        prepared = _prepare_step(
            step,
            device=target_device,
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
        )
        embedding = self.node_encoder(prepared.node_x)
        for layer in self.spatial:
            embedding = layer(embedding, prepared.edge_index, prepared.edge_z)
        embedding, next_state = self.temporal_backbone.forward_step(embedding, state)
        output = self.head(embedding, step=dict(step))
        output.validate(prepared.edge_count, prepared.satellite_count)
        return output, next_state


class PhysiCKTemporalMixerWorldModel(nn.Module):
    """Paper PhysiCK graph with an official long-context GRU replacement.

    A recurrent state is a mapping with two entries: ``node_memory`` stores the
    current 128-dimensional node state, while ``layer_histories`` stores one
    bounded message history per paper message-passing layer.  The temporal
    module is shared across layers, just as the default TGN shares one GRU
    update, while histories remain layer-specific so two graph layers do not
    become two control-time samples.
    """

    ARCHITECTURE = "paper_physick_temporal_swap_v1"

    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        mem_dim: int,
        msg_dim: int,
        emb_dim: int,
        temporal_backbone: StreamingTemporalBackbone,
        message_passing_layers: int = 2,
        aggregator: str = "sum",
        dropout: float = 0.1,
        use_edge_type: bool = True,
        edge_type_vocab: int = 8,
        physick_config: Mapping[str, Any] | None = None,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
        node_injection: float = 0.1,
        edge_chunk: int = 50000,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.node_in_dim = _positive_int("node_in_dim", node_in_dim)
        self.edge_in_dim = _positive_int("edge_in_dim", edge_in_dim)
        self.mem_dim = _positive_int("mem_dim", mem_dim)
        self.msg_dim = _positive_int("msg_dim", msg_dim)
        self.emb_dim = _positive_int("emb_dim", emb_dim)
        self.num_message_passing_layers = _positive_int(
            "message_passing_layers", message_passing_layers
        )
        self.edge_type_vocab = _positive_int("edge_type_vocab", edge_type_vocab)
        self.edge_chunk = _positive_int("edge_chunk", edge_chunk)
        if not isinstance(use_edge_type, bool):
            raise TypeError("use_edge_type must be boolean")
        self.use_edge_type = use_edge_type
        self.aggregator = str(aggregator).strip().lower()
        if self.aggregator not in {"sum", "mean"}:
            raise ValueError("aggregator must be 'sum' or 'mean'")
        dropout_value = _dropout(dropout)
        injection_value = _nonnegative("node_injection", node_injection)
        if injection_value > 1.0:
            raise ValueError("node_injection must lie in [0,1]")
        self.node_injection = injection_value
        if not isinstance(temporal_backbone, StreamingTemporalBackbone):
            raise TypeError("temporal_backbone must implement the paper streaming API")
        if temporal_backbone.d_model != self.mem_dim:
            raise ValueError("temporal backbone d_model must equal paper mem_dim")

        if physick_config is not None and not isinstance(physick_config, Mapping):
            raise TypeError("physick_config must be a mapping")
        physick = dict(physick_config or {})
        implementation = str(
            physick.pop("implementation", "paper_intensity_flow")
        ).strip().lower()
        if implementation not in {"paper", "paper_intensity_flow", "intensity_flow"}:
            raise ValueError(
                "temporal-mixer ablations require the paper_intensity_flow "
                "PhysiCK implementation"
            )
        projection_radius = physick.pop("projection_radius", 1.0)
        if projection_radius is None:
            raise ValueError("paper PhysiCK requires a finite projection_radius")
        message = PaperPhysiCKMessage(
            mem_dim=self.mem_dim,
            edge_dim=self.edge_in_dim,
            msg_dim=self.msg_dim,
            edge_type_vocab=self.edge_type_vocab,
            use_edge_type=self.use_edge_type,
            num_kernels=physick.pop("num_kernels", 16),
            latent_dim=physick.pop("latent_dim", self.msg_dim),
            descriptor_dim=physick.pop("descriptor_dim", 16),
            kernel_hidden_dim=physick.pop("kernel_hidden_dim", self.msg_dim),
            coeff_hidden_dim=physick.pop("coeff_hidden", self.msg_dim),
            dropout=physick.pop("dropout", dropout_value),
            projection_radius=projection_radius,
            operating_clip=physick.pop("operating_clip", 5.0),
        )
        if physick:
            raise ValueError(
                "unsupported model.physick fields for paper temporal swap: "
                + ", ".join(sorted(physick))
            )

        self.cfg = dict(config or {})
        self.architecture = self.ARCHITECTURE
        self.temporal_backbone = temporal_backbone
        self.node_encoder = nn.Linear(self.node_in_dim, self.mem_dim)
        self.message_functions = nn.ModuleList(
            [message]
            + [
                copy.deepcopy(message)
                for _ in range(self.num_message_passing_layers - 1)
            ]
        )
        self.message_to_memory = nn.ModuleList(
            [
                nn.Linear(self.msg_dim, self.mem_dim)
                for _ in range(self.num_message_passing_layers)
            ]
        )
        self.readout = Readout(mem_dim=self.mem_dim, emb_dim=self.emb_dim)
        self.head = IntensityFlowHead(
            in_dim=self.emb_dim,
            edge_in_dim=self.edge_in_dim,
            hidden_dim=_positive_int("head_hidden_dim", head_hidden_dim),
            dropout=_dropout(head_dropout),
        )

    def _state(
        self,
        state: Any,
        node_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
        if state is None:
            return node_seed, (None,) * self.num_message_passing_layers
        if not isinstance(state, Mapping):
            raise TypeError(
                "paper PhysiCK temporal state must be a mapping or None"
            )
        if set(state) != {"node_memory", "layer_histories"}:
            raise ValueError(
                "paper PhysiCK temporal state requires node_memory and "
                "layer_histories"
            )
        memory = state["node_memory"]
        histories = state["layer_histories"]
        if not isinstance(memory, torch.Tensor) or memory.shape != node_seed.shape:
            raise ValueError(
                "state.node_memory must match [num_nodes,mem_dim]"
            )
        if memory.device != node_seed.device or memory.dtype != node_seed.dtype:
            raise ValueError("state.node_memory must share node input device/dtype")
        if not bool(torch.isfinite(memory).all()):
            raise ValueError("state.node_memory contains NaN or Inf")
        if (
            not isinstance(histories, (tuple, list))
            or len(histories) != self.num_message_passing_layers
        ):
            raise ValueError(
                "state.layer_histories must contain one history per graph layer"
            )
        normalized: list[torch.Tensor | None] = []
        for history in histories:
            if history is not None and not isinstance(history, torch.Tensor):
                raise TypeError("each temporal layer history must be a tensor or None")
            normalized.append(history)
        return memory, tuple(normalized)

    def _aggregated_message(
        self,
        prepared: _PreparedPaperStep,
        memory: torch.Tensor,
        message_function: PaperPhysiCKMessage,
        projection: nn.Linear,
    ) -> torch.Tensor:
        node_count = int(memory.size(0))
        if prepared.edge_count == 0:
            return torch.zeros(
                (node_count, self.mem_dim),
                device=memory.device,
                dtype=memory.dtype,
            )

        aggregate = torch.zeros(
            (node_count, self.msg_dim),
            device=memory.device,
            dtype=torch.float32,
        )
        counts = (
            torch.zeros(
                (node_count, 1),
                device=memory.device,
                dtype=torch.float32,
            )
            if self.aggregator == "mean"
            else None
        )
        source_all, destination_all = prepared.edge_index
        for start in range(0, prepared.edge_count, self.edge_chunk):
            end = min(prepared.edge_count, start + self.edge_chunk)
            source = source_all[start:end]
            destination = destination_all[start:end]
            edge_type = (
                torch.zeros(
                    end - start,
                    dtype=torch.long,
                    device=memory.device,
                )
                if self.use_edge_type
                else None
            )
            message = message_function(
                memory[source],
                memory[destination],
                prepared.edge_z[start:end],
                edge_type=edge_type,
            )
            aggregate.index_add_(0, destination, message.float())
            if counts is not None:
                counts.index_add_(
                    0,
                    destination,
                    torch.ones(
                        (end - start, 1),
                        dtype=torch.float32,
                        device=memory.device,
                    ),
                )
        if counts is not None:
            aggregate = aggregate / counts.clamp_min(1.0)
        return projection(aggregate)

    def predict_step(
        self,
        step: Mapping[str, Any],
        state: Any,
        device: torch.device | str,
    ) -> tuple[IntensityFlowOutput, dict[str, Any]]:
        target_device = torch.device(device)
        prepared = _prepare_step(
            step,
            device=target_device,
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
        )
        node_seed = torch.tanh(self.node_encoder(prepared.node_x))
        memory, histories = self._state(state, node_seed)
        next_histories: list[torch.Tensor] = []
        for message_function, projection, history in zip(
            self.message_functions,
            self.message_to_memory,
            histories,
        ):
            aggregate = self._aggregated_message(
                prepared,
                memory,
                message_function,
                projection,
            )
            # A GRU consumes both the aggregate and previous memory.  Their
            # parameter-free residual sum is the fixed-width token consumed by
            # every replacement backend; only the temporal module varies.
            mixed, next_history = self.temporal_backbone.forward_step(
                memory + aggregate,
                history,
            )
            memory = (
                (1.0 - self.node_injection) * mixed
                + self.node_injection * node_seed
            )
            if not bool(torch.isfinite(memory).all()):
                raise FloatingPointError(
                    "paper PhysiCK temporal memory contains NaN or Inf"
                )
            next_histories.append(next_history)

        embedding = self.readout(memory)
        output = self.head(embedding, step=dict(step))
        output.validate(prepared.edge_count, prepared.satellite_count)
        return output, {
            "node_memory": memory,
            "layer_histories": tuple(next_histories),
        }


class LTTRWorldModel(LongContextGraphWorldModel):
    """Full long-context Transformer baseline used for LTT-R experiments."""

    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        model_dim: int = 384,
        spatial_layers: int = 2,
        context_length: int = 64,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
        head_hidden_dim: int | None = None,
        head_dropout: float | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        temporal = TransformerTemporalBackbone(
            d_model=model_dim,
            context_length=context_length,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        super().__init__(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            model_dim=model_dim,
            temporal_backbone=temporal,
            spatial_layers=spatial_layers,
            dropout=dropout,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            config=config,
        )


class DecisionAwareRankingLoss(nn.Module):
    """Differentiable soft-rank surrogate for oracle fixed-score ordering.

    Oracle ordinal ranks use the repository's deterministic satellite-id tie
    break.  Predicted ranks use pairwise sigmoid comparisons after explicit
    descriptor scaling, so gradients reach every policy descriptor.  Each
    oracle-best/challenger pair is up-weighted as its oracle score gap approaches
    zero.  This is a loss hook, not a replacement for the typed descriptor loss.
    """

    def __init__(
        self,
        *,
        gamma_weight: float = 1.0,
        load_weight: float = 0.7,
        intensity_weight: float = 0.5,
        gamma_scale: float = 10.0,
        flow_scale: float = 1.0,
        log1p_intensity_scale: float = 1.0,
        soft_rank_temperature: float = 0.1,
        pair_margin: float = 0.0,
        near_tie_strength: float = 1.0,
        near_tie_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma_weight = _nonnegative("gamma_weight", gamma_weight)
        self.load_weight = _nonnegative("load_weight", load_weight)
        self.intensity_weight = _nonnegative("intensity_weight", intensity_weight)
        if self.gamma_weight + self.load_weight + self.intensity_weight <= 0:
            raise ValueError("at least one fixed-score weight must be positive")
        self.gamma_scale = _nonnegative("gamma_scale", gamma_scale, positive=True)
        self.flow_scale = _nonnegative("flow_scale", flow_scale, positive=True)
        self.log1p_intensity_scale = _nonnegative(
            "log1p_intensity_scale", log1p_intensity_scale, positive=True
        )
        self.soft_rank_temperature = _nonnegative(
            "soft_rank_temperature", soft_rank_temperature, positive=True
        )
        self.pair_margin = _nonnegative("pair_margin", pair_margin)
        self.near_tie_strength = _nonnegative(
            "near_tie_strength", near_tie_strength
        )
        self.near_tie_temperature = _nonnegative(
            "near_tie_temperature", near_tie_temperature, positive=True
        )

    def _soft_rank(self, values: torch.Tensor, *, higher_is_better: bool) -> torch.Tensor:
        signed = values if higher_is_better else -values
        comparisons = torch.sigmoid(
            (signed[:, None] - signed[None, :]) / self.soft_rank_temperature
        )
        # The diagonal comparison is 0.5; adding 0.5 recovers the ordinal
        # desirability endpoints 1/n and 1 in the low-temperature limit.
        return (comparisons.sum(dim=1) + 0.5) / float(values.numel())

    def forward(
        self,
        prediction: IntensityFlowOutput,
        oracle: PolicyDescriptors,
        candidate_edge_ids: torch.Tensor,
        persistent_edge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_count = int(candidate_edge_ids.size(0))
        satellite_count = int(oracle.flow_node.numel())
        prediction.validate(edge_count, satellite_count)
        oracle.validate(edge_count, satellite_count)
        if candidate_edge_ids.shape != (edge_count, 2) or candidate_edge_ids.dtype != torch.long:
            raise ValueError("candidate_edge_ids must have shape [E,2] and dtype long")
        tensors = (
            candidate_edge_ids,
            prediction.policy_descriptors.gamma_edge,
            oracle.gamma_edge,
            oracle.flow_node,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("decision-aware loss tensors must share one device")
        if persistent_edge is None:
            persistent = torch.ones(
                edge_count,
                dtype=torch.bool,
                device=candidate_edge_ids.device,
            )
        else:
            if persistent_edge.shape != (edge_count,) or persistent_edge.dtype != torch.bool:
                raise ValueError("persistent_edge must be bool with shape [E]")
            if persistent_edge.device != candidate_edge_ids.device:
                raise ValueError("persistent_edge must share the candidate device")
            persistent = persistent_edge

        predicted = prediction.policy_descriptors
        pair_losses: list[torch.Tensor] = []
        pair_weights: list[torch.Tensor] = []
        for user in torch.unique(candidate_edge_ids[:, 0]).tolist():
            edges = torch.nonzero(
                (candidate_edge_ids[:, 0] == int(user)) & persistent,
                as_tuple=False,
            ).flatten()
            if edges.numel() < 2:
                continue
            satellites = candidate_edge_ids[edges, 1]
            oracle_gamma = oracle.gamma_edge[edges]
            oracle_intensity = oracle.intensity_edge[edges]
            oracle_flow = oracle.flow_node[satellites]
            oracle_score = (
                self.gamma_weight
                * normalized_ordinal_rank(
                    oracle_gamma,
                    satellites,
                    higher_is_better=True,
                )
                + self.load_weight
                * normalized_ordinal_rank(
                    oracle_flow,
                    satellites,
                    higher_is_better=False,
                )
                + self.intensity_weight
                * normalized_ordinal_rank(
                    oracle_intensity,
                    satellites,
                    higher_is_better=False,
                )
            ).detach()

            predicted_score = (
                self.gamma_weight
                * self._soft_rank(
                    predicted.gamma_edge[edges] / self.gamma_scale,
                    higher_is_better=True,
                )
                + self.load_weight
                * self._soft_rank(
                    predicted.flow_node[satellites] / self.flow_scale,
                    higher_is_better=False,
                )
                + self.intensity_weight
                * self._soft_rank(
                    torch.log1p(predicted.intensity_edge[edges])
                    / self.log1p_intensity_scale,
                    higher_is_better=False,
                )
            )
            top_local = torch.argmax(oracle_score)
            challenger = torch.arange(
                edges.numel(), device=edges.device
            ) != top_local
            oracle_gap = (
                oracle_score[top_local] - oracle_score[challenger]
            ).clamp_min(0.0)
            near_weight = 1.0 + self.near_tie_strength * torch.exp(
                -oracle_gap / self.near_tie_temperature
            )
            violation = (
                predicted_score[challenger]
                - predicted_score[top_local]
                + self.pair_margin
            )
            pair_loss = F.softplus(
                violation / self.soft_rank_temperature
            ) * self.soft_rank_temperature
            pair_losses.append(pair_loss)
            pair_weights.append(near_weight)

        if not pair_losses:
            return predicted.gamma_edge.sum() * 0.0
        losses = torch.cat(pair_losses)
        weights = torch.cat(pair_weights).to(dtype=losses.dtype)
        result = (losses * weights).sum() / weights.sum().clamp_min(1e-12)
        if not bool(torch.isfinite(result)):
            raise FloatingPointError("decision-aware ranking loss is NaN or Inf")
        return result


class DAGWMWorldModel(nn.Module):
    """GATv2 decision-aware graph world model with recurrent node memory."""

    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 256,
        gat_layers: int = 4,
        gat_heads: int = 8,
        dropout: float = 0.1,
        head_hidden_dim: int | None = None,
        head_dropout: float | None = None,
        decision_loss_config: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.node_in_dim = _positive_int("node_in_dim", node_in_dim)
        self.edge_in_dim = _positive_int("edge_in_dim", edge_in_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        layer_count = _positive_int("gat_layers", gat_layers)
        head_count = _positive_int("gat_heads", gat_heads)
        dropout_value = _dropout(dropout)
        if self.hidden_dim % head_count != 0:
            raise ValueError("hidden_dim must be divisible by gat_heads")
        try:
            from torch_geometric.nn import GATv2Conv
            self.attention_backend = "torch_geometric"
        except (ImportError, ModuleNotFoundError):
            from .gat import PureTorchGATv2Conv as GATv2Conv

            self.attention_backend = "pure_torch"

        self.cfg = dict(config or {})
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.gat_layers = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim // head_count,
                    heads=head_count,
                    concat=True,
                    dropout=dropout_value,
                    edge_dim=self.edge_in_dim,
                    add_self_loops=False,
                )
                for _ in range(layer_count)
            ]
        )
        self.gat_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(layer_count)]
        )
        self.dropout = nn.Dropout(dropout_value)
        self.memory = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.head = IntensityFlowHead(
            in_dim=self.hidden_dim,
            edge_in_dim=self.edge_in_dim,
            hidden_dim=(
                self.hidden_dim
                if head_hidden_dim is None
                else _positive_int("head_hidden_dim", head_hidden_dim)
            ),
            dropout=(
                dropout_value
                if head_dropout is None
                else _dropout(head_dropout)
            ),
        )
        self.decision_loss = DecisionAwareRankingLoss(
            **dict(decision_loss_config or {})
        )

    def predict_step(
        self,
        step: Mapping[str, Any],
        state: torch.Tensor | None,
        device: torch.device | str,
    ) -> tuple[IntensityFlowOutput, torch.Tensor]:
        target_device = torch.device(device)
        prepared = _prepare_step(
            step,
            device=target_device,
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
        )
        embedding = self.node_encoder(prepared.node_x)
        if prepared.edge_count:
            reverse = torch.stack(
                (prepared.edge_index[1], prepared.edge_index[0]), dim=0
            )
            bidirectional_index = torch.cat((prepared.edge_index, reverse), dim=1)
            bidirectional_edge_z = torch.cat((prepared.edge_z, prepared.edge_z), dim=0)
        else:
            bidirectional_index = prepared.edge_index
            bidirectional_edge_z = prepared.edge_z
        for convolution, norm in zip(self.gat_layers, self.gat_norms):
            update = convolution(
                embedding,
                bidirectional_index,
                edge_attr=bidirectional_edge_z,
            )
            embedding = norm(embedding + self.dropout(F.gelu(update)))

        if state is None:
            previous = torch.zeros_like(embedding)
        else:
            if not isinstance(state, torch.Tensor):
                raise TypeError("DA-GWM state must be a torch.Tensor or None")
            if state.shape != embedding.shape:
                raise ValueError(
                    "DA-GWM state must have shape [num_nodes,hidden_dim] with stable "
                    "node identity"
                )
            if state.device != embedding.device or state.dtype != embedding.dtype:
                raise ValueError("DA-GWM state and current embedding must share device/dtype")
            if not bool(torch.isfinite(state).all()):
                raise ValueError("DA-GWM state contains NaN or Inf")
            previous = state
        next_state = self.memory(embedding, previous)
        output = self.head(next_state, step=dict(step))
        output.validate(prepared.edge_count, prepared.satellite_count)
        return output, next_state

    def decision_aware_loss_hook(
        self,
        prediction: IntensityFlowOutput,
        oracle: PolicyDescriptors,
        candidate_edge_ids: torch.Tensor,
        persistent_edge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the additive decision-aware ranking term for a trainer."""

        return self.decision_loss(
            prediction,
            oracle,
            candidate_edge_ids,
            persistent_edge,
        )


def _baseline_config(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(config, Mapping):
        raise TypeError("baseline configuration must be a mapping")
    nested = config.get("paper_baseline")
    if nested is None:
        return config, config
    if not isinstance(nested, Mapping):
        raise TypeError("paper_baseline must be a mapping")
    return nested, config


def _build_physick_temporal_swap(
    baseline: Mapping[str, Any],
    root: Mapping[str, Any],
    *,
    kind: str,
) -> PhysiCKTemporalMixerWorldModel:
    architecture = str(baseline.get("architecture", "")).strip().lower()
    if architecture != PhysiCKTemporalMixerWorldModel.ARCHITECTURE:
        raise ValueError(
            f"paper_baseline.{kind}.architecture must be "
            f"{PhysiCKTemporalMixerWorldModel.ARCHITECTURE!r}"
        )
    if str(baseline.get("interface", "")).strip().lower() != "intensity_flow":
        raise ValueError(
            f"paper_baseline.{kind}.interface must be 'intensity_flow'"
        )
    if str(baseline.get("message_operator", "")).strip().lower() != "paper_physick":
        raise ValueError(
            f"paper_baseline.{kind}.message_operator must be 'paper_physick'"
        )
    forbidden_overrides = {
        "node_in_dim",
        "edge_in_dim",
        "mem_dim",
        "msg_dim",
        "emb_dim",
        "model_dim",
        "spatial_layers",
        "message_passing_layers",
        "aggregator",
        "dropout",
        "use_edge_type",
        "edge_type_vocab",
        "node_injection",
        "physick",
        "head",
        "head_hidden_dim",
        "head_dropout",
    }.intersection(baseline)
    if forbidden_overrides:
        raise ValueError(
            f"paper_baseline.{kind} must inherit the shared paper model/head; "
            "remove overrides: " + ", ".join(sorted(forbidden_overrides))
        )

    model = root.get("model")
    head = root.get("head")
    if not isinstance(model, Mapping) or not isinstance(head, Mapping):
        raise TypeError("paper temporal swaps require root model and head mappings")
    if str(model.get("message_type", "")).strip().lower() != "physick":
        raise ValueError("paper temporal swaps require model.message_type='physick'")
    physick = model.get("physick")
    if not isinstance(physick, Mapping):
        raise TypeError("paper temporal swaps require model.physick configuration")
    if str(physick.get("implementation", "")).strip().lower() not in {
        "paper",
        "paper_intensity_flow",
        "intensity_flow",
    }:
        raise ValueError(
            "paper temporal swaps require model.physick.implementation="
            "'paper_intensity_flow'"
        )
    if str(head.get("type", "")).strip().lower() != "intensity_flow":
        raise ValueError("paper temporal swaps require head.type='intensity_flow'")

    node_in_dim = _positive_int("model.node_in_dim", model.get("node_in_dim"))
    edge_in_dim = _positive_int("model.edge_in_dim", model.get("edge_in_dim"))
    mem_dim = _positive_int("model.mem_dim", model.get("mem_dim"))
    msg_dim = _positive_int("model.msg_dim", model.get("msg_dim"))
    emb_dim = _positive_int("model.emb_dim", model.get("emb_dim"))
    message_layers = _positive_int(
        "model.message_passing_layers", model.get("message_passing_layers")
    )
    if node_in_dim != len(PAPER_NODE_FEATURE_NAMES):
        raise ValueError("paper temporal swaps require the seven-column node contract")
    if edge_in_dim != len(PAPER_EDGE_FEATURE_NAMES):
        raise ValueError("paper temporal swaps require the seven-column edge contract")
    if (mem_dim, msg_dim, emb_dim) != (128, 128, 128):
        raise ValueError(
            "paper temporal swaps keep mem_dim=msg_dim=emb_dim=128"
        )
    if message_layers != 2:
        raise ValueError("paper temporal swaps keep exactly two message-passing layers")

    temporal = baseline.get("temporal")
    if not isinstance(temporal, Mapping):
        raise TypeError(f"paper_baseline.{kind}.temporal must be a mapping")
    temporal_config = dict(temporal)
    declared = str(temporal_config.get("type", "")).strip().lower()
    if declared != kind:
        raise ValueError(
            f"paper_baseline.kind={kind!r} requires temporal.type={kind!r}"
        )
    temporal_config.setdefault("dropout", model.get("dropout", 0.1))
    backbone = build_temporal_backbone(temporal_config, d_model=mem_dim)

    use_edge_type = model.get("use_edge_type", True)
    if not isinstance(use_edge_type, bool):
        raise TypeError("model.use_edge_type must be boolean")
    train = root.get("train", {})
    if not isinstance(train, Mapping):
        raise TypeError("root train configuration must be a mapping when present")
    edge_chunk = model.get("edge_chunk", train.get("edge_chunk", 50000))
    return PhysiCKTemporalMixerWorldModel(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        mem_dim=mem_dim,
        msg_dim=msg_dim,
        emb_dim=emb_dim,
        temporal_backbone=backbone,
        message_passing_layers=message_layers,
        aggregator=model.get("aggregator", "sum"),
        dropout=model.get("dropout", 0.1),
        use_edge_type=use_edge_type,
        edge_type_vocab=model.get("edge_type_vocab", 8),
        physick_config=physick,
        head_hidden_dim=head.get("hidden_dim", emb_dim),
        head_dropout=head.get("dropout", model.get("dropout", 0.1)),
        node_injection=model.get("node_injection", 0.1),
        edge_chunk=edge_chunk,
        config=root,
    )


def build_paper_baseline(config: Mapping[str, Any]) -> nn.Module:
    """Construct a full baseline from ``paper_baseline`` configuration.

    ``ltt_r`` always builds the controlled independent Transformer baseline.
    ``s4``, ``mamba2`` and ``conformer`` retain the root paper PhysiCK graph and
    Intensity--Flow head and replace only its GRU temporal update. ``da_gwm``
    constructs GATv2 and never aliases to the repository's MLP/TGN path.
    """

    baseline, root = _baseline_config(config)
    kind = str(baseline.get("kind", "")).strip().lower()
    if not kind:
        raise ValueError("paper_baseline.kind is required")

    if kind == "ltt_r":
        node_in_dim = _positive_int("node_in_dim", baseline.get("node_in_dim"))
        edge_in_dim = _positive_int("edge_in_dim", baseline.get("edge_in_dim"))
        dropout = _dropout(baseline.get("dropout", 0.1))
        head_hidden_dim = baseline.get("head_hidden_dim")
        head_dropout = baseline.get("head_dropout")
        temporal = baseline.get("temporal", {})
        if not isinstance(temporal, Mapping):
            raise TypeError("paper_baseline.temporal must be a mapping")
        temporal_type = str(temporal.get("type", "transformer")).lower()
        if temporal_type != "transformer":
            raise ValueError("ltt_r requires temporal.type='transformer'")
        return LTTRWorldModel(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            model_dim=baseline.get("model_dim", 384),
            spatial_layers=baseline.get("spatial_layers", 2),
            context_length=temporal.get("context_length", 64),
            transformer_layers=temporal.get("num_layers", 6),
            transformer_heads=temporal.get("num_heads", 8),
            ffn_dim=temporal.get("ffn_dim", 1536),
            dropout=dropout,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            config=root,
        )
    if kind in {"s4", "mamba2", "conformer"}:
        return _build_physick_temporal_swap(baseline, root, kind=kind)
    if kind == "da_gwm":
        node_in_dim = _positive_int("node_in_dim", baseline.get("node_in_dim"))
        edge_in_dim = _positive_int("edge_in_dim", baseline.get("edge_in_dim"))
        dropout = _dropout(baseline.get("dropout", 0.1))
        head_hidden_dim = baseline.get("head_hidden_dim")
        head_dropout = baseline.get("head_dropout")
        decision_loss = baseline.get("decision_loss", {})
        if not isinstance(decision_loss, Mapping):
            raise TypeError("paper_baseline.decision_loss must be a mapping")
        return DAGWMWorldModel(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=baseline.get("hidden_dim", 256),
            gat_layers=baseline.get("gat_layers", 4),
            gat_heads=baseline.get("gat_heads", 8),
            dropout=dropout,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            decision_loss_config=decision_loss,
            config=root,
        )
    raise ValueError(
        f"unknown paper baseline {kind!r}; expected ltt_r, da_gwm, s4, mamba2, "
        "or conformer"
    )
