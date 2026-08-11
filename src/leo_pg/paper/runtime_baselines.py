"""Trainable runtime-table baselines with the paper Intensity--Flow contract.

The Supplementary Information specifies only the structural substitutions:

``BigMLP``
    Widen only the TGN message MLP to the manuscript runtime table's 2.1M
    capacity class.

``EdgeAttn``
    Replace only the message function by GAT-style edge attention.

``TempTrans``
    Replace only recurrent temporal mixing by a Transformer encoder over each
    node's message sequence.

The shared backbone is fixed by the manuscript: 128-dimensional node memory,
message and readout, two graph-message layers, GRU temporal mixing by default,
and the Paper Intensity--Flow head.  Replacement-module widths, attention heads
and Transformer depth/context are not reported and therefore remain explicit
configuration values.  The runtime table reports 2.1M parameters for BigMLP
and EdgeAttn and 2.4M for TempTrans (rounded to 0.1M).  Factories validate those
released capacity targets and store their provenance; they never call the
current reconstructed PhysiCK checkpoint's smaller parameter count a match.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from leo_pg.models.heads.intensity_flow import IntensityFlowHead, IntensityFlowOutput
from leo_pg.models.tgn.memory import MemoryBank
from leo_pg.models.tgn.readout import Readout
from leo_pg.kernels.mlp import MLPMessage
from leo_pg.sim.state import PolicyDescriptors

from .backbones import TransformerTemporalBackbone
from .baselines import DecisionAwareRankingLoss, _PreparedPaperStep, _prepare_step


RUNTIME_BASELINE_KINDS = ("big_mlp", "edge_attn", "temp_trans")
_SHARED_BACKBONE = "paper_tgn_128d_2layer_intensity_flow_v1"
_RUNTIME_IDENTITIES = {
    "big_mlp": ("paper_tgn_big_message_mlp_v2", "message_operator_only"),
    "edge_attn": ("paper_tgn_edge_attention_v2", "message_operator_only"),
    "temp_trans": (
        "paper_tgn_mlp_temporal_transformer_v2",
        "temporal_mixing_only",
    ),
}


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result != value or result < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _dropout(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must lie in [0, 1)")
    return result


def _unit_interval(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _required(config: Mapping[str, Any], name: str, path: str) -> Any:
    if name not in config:
        raise ValueError(f"{path}.{name} is required")
    return config[name]


def _trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _bind_capacity_provenance(
    model: nn.Module,
    baseline: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """Validate the rounded manuscript-table capacity and persist provenance."""

    target = _positive_int(
        f"{path}.capacity_target_params",
        _required(baseline, "capacity_target_params", path),
    )
    tolerance = _positive_int(
        f"{path}.capacity_tolerance_params",
        _required(baseline, "capacity_tolerance_params", path),
    )
    provenance = str(
        _required(baseline, "capacity_target_provenance", path)
    ).strip()
    if not provenance:
        raise ValueError(f"{path}.capacity_target_provenance must be non-empty")
    actual = _trainable_parameter_count(model)
    if abs(actual - target) > tolerance:
        raise ValueError(
            f"{path} has {actual:,} trainable parameters, outside the released "
            f"capacity target {target:,} +/- {tolerance:,}; adjust only the "
            "replacement module's unreported internal width/depth"
        )
    setattr(model, "capacity_target_params", target)
    setattr(model, "capacity_tolerance_params", tolerance)
    setattr(model, "capacity_target_provenance", provenance)
    setattr(model, "resolved_trainable_parameters", actual)


def _validate_shared_backbone(
    baseline: Mapping[str, Any],
    root: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """Prevent a runtime operator swap from silently changing the TGN core."""

    model = root.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("root model mapping is required for runtime baselines")
    expected = {
        "node_in_dim": int(model.get("node_in_dim", -1)),
        "edge_in_dim": int(model.get("edge_in_dim", -1)),
        "memory_dim": int(model.get("mem_dim", -1)),
        "message_dim": int(model.get("msg_dim", -1)),
        "embedding_dim": int(model.get("emb_dim", -1)),
        "message_passing_layers": int(model.get("message_passing_layers", 1)),
        "aggregator": str(model.get("aggregator", "sum")).strip().lower(),
        "node_injection": float(model.get("node_injection", 0.1)),
    }
    for name, root_value in expected.items():
        baseline_value = _required(baseline, name, path)
        if isinstance(root_value, float):
            equal = math.isclose(
                float(baseline_value), root_value, rel_tol=0.0, abs_tol=1e-12
            )
        elif isinstance(root_value, str):
            equal = str(baseline_value).strip().lower() == root_value
        else:
            equal = int(baseline_value) == root_value
        if not equal:
            raise ValueError(
                f"{path}.{name}={baseline_value!r} changes the shared paper TGN "
                f"value {root_value!r}; runtime controls may replace only their "
                "named operator"
            )
    if expected["memory_dim"] != 128 or expected["message_dim"] != 128:
        raise ValueError("runtime controls require the manuscript 128-d memory/message")
    if expected["embedding_dim"] != 128:
        raise ValueError("runtime controls require the manuscript 128-d readout")
    if expected["message_passing_layers"] != 2:
        raise ValueError("runtime controls require two paper message-passing layers")


def _validate_runtime_identity(
    baseline: Mapping[str, Any],
    kind: str,
    *,
    path: str,
) -> None:
    """Make the replacement scope explicit in configs and saved checkpoints."""

    expected_architecture, expected_scope = _RUNTIME_IDENTITIES[kind]
    architecture = str(_required(baseline, "architecture", path)).strip().lower()
    if architecture != expected_architecture:
        raise ValueError(
            f"{path}.architecture must be {expected_architecture!r}, got "
            f"{architecture!r}"
        )
    shared_backbone = str(
        _required(baseline, "shared_backbone", path)
    ).strip().lower()
    if shared_backbone != _SHARED_BACKBONE:
        raise ValueError(
            f"{path}.shared_backbone must be {_SHARED_BACKBONE!r}, got "
            f"{shared_backbone!r}"
        )
    replacement_scope = str(
        _required(baseline, "replacement_scope", path)
    ).strip().lower()
    if replacement_scope != expected_scope:
        raise ValueError(
            f"{path}.replacement_scope must be {expected_scope!r}, got "
            f"{replacement_scope!r}"
        )


def _aggregator(value: Any) -> str:
    result = str(value).strip().lower()
    if result not in {"sum", "mean"}:
        raise ValueError("aggregator must be 'sum' or 'mean'")
    return result


def _validate_node_state(
    state: Any,
    reference: torch.Tensor,
    *,
    state_dim: int,
    label: str,
) -> torch.Tensor:
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"{label} state must be a torch.Tensor or None")
    expected = (reference.size(0), state_dim)
    if tuple(state.shape) != expected:
        raise ValueError(
            f"{label} state must have shape [num_nodes,{state_dim}] with stable "
            "node identity"
        )
    if state.device != reference.device or state.dtype != reference.dtype:
        raise ValueError(f"{label} state and current embedding must share device/dtype")
    if not bool(torch.isfinite(state).all()):
        raise ValueError(f"{label} state contains NaN or Inf")
    return state


class _DenseMessageNetwork(nn.Module):
    """Configurable edge-message MLP with an explicit linear-layer count."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        linear_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        input_width = _positive_int("message input_dim", input_dim)
        hidden_width = _positive_int("message hidden_dim", hidden_dim)
        output_width = _positive_int("message output_dim", output_dim)
        layer_count = _positive_int("message linear_layers", linear_layers, minimum=2)
        dropout_value = _dropout("message dropout", dropout)
        modules: list[nn.Module] = [
            nn.Linear(input_width, hidden_width),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        ]
        for _ in range(layer_count - 2):
            modules.extend(
                (
                    nn.Linear(hidden_width, hidden_width),
                    nn.ReLU(),
                    nn.Dropout(dropout_value),
                )
            )
        modules.append(nn.Linear(hidden_width, output_width))
        self.network = nn.Sequential(*modules)

    def forward(
        self,
        source_state: torch.Tensor,
        destination_state: torch.Tensor,
        edge_z: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_state.shape != destination_state.shape:
            raise ValueError("source and destination message states must have equal shape")
        if source_state.ndim != 2 or edge_z.ndim != 2:
            raise ValueError("message inputs must be matrices")
        if source_state.size(0) != edge_z.size(0):
            raise ValueError("message inputs must have the same edge count")
        return self.network(
            torch.cat((source_state, destination_state, edge_z), dim=-1)
        )


class RuntimeIntensityFlowBaseline(nn.Module):
    """Common typed head, stateless reset, and optional decision-loss hook."""

    state_contract = "external_episode_state"

    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        embedding_dim: int,
        head_hidden_dim: int,
        head_dropout: float,
        decision_loss_config: Mapping[str, Any] | None,
        config: Mapping[str, Any] | None,
    ) -> None:
        super().__init__()
        self.node_in_dim = _positive_int("node_in_dim", node_in_dim)
        self.edge_in_dim = _positive_int("edge_in_dim", edge_in_dim)
        self.embedding_dim = _positive_int("embedding_dim", embedding_dim)
        self.cfg = copy.deepcopy(dict(config or {}))
        self.head = IntensityFlowHead(
            in_dim=self.embedding_dim,
            edge_in_dim=self.edge_in_dim,
            hidden_dim=_positive_int("head_hidden_dim", head_hidden_dim),
            dropout=_dropout("head_dropout", head_dropout),
        )
        self.decision_loss = (
            None
            if decision_loss_config is None
            else DecisionAwareRankingLoss(**dict(decision_loss_config))
        )

    def reset(self) -> None:
        """No-op: recurrent state is owned by the trainer/provider, not the module."""

        return None

    def _prepare(
        self,
        step: Mapping[str, Any],
        device: torch.device | str,
    ) -> _PreparedPaperStep:
        return _prepare_step(
            step,
            device=torch.device(device),
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
        )

    def _typed_output(
        self,
        embedding: torch.Tensor,
        step: Mapping[str, Any],
        prepared: _PreparedPaperStep,
    ) -> IntensityFlowOutput:
        output = self.head(embedding, step=dict(step))
        if not isinstance(output, IntensityFlowOutput):
            raise TypeError("runtime baseline head must return IntensityFlowOutput")
        output.validate(prepared.edge_count, prepared.satellite_count)
        return output

    def decision_aware_loss_hook(
        self,
        prediction: IntensityFlowOutput,
        oracle: PolicyDescriptors,
        candidate_edge_ids: torch.Tensor,
        persistent_edge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Optional additive hook matching ``PaperTrainer``'s exact signature."""

        if self.decision_loss is None:
            raise ValueError(
                "decision-aware training was requested for a runtime baseline "
                "without decision_loss configuration"
            )
        return self.decision_loss(
            prediction,
            oracle,
            candidate_edge_ids,
            persistent_edge,
        )


class BigMLP(RuntimeIntensityFlowBaseline):
    """Shared Paper TGN with only its two message MLPs widened."""

    architecture = "paper_tgn_big_message_mlp_v2"
    shared_backbone = _SHARED_BACKBONE
    replacement_scope = "message_operator_only"
    state_contract = "Tensor[num_nodes,memory_dim]"

    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        memory_dim: int,
        message_dim: int,
        embedding_dim: int,
        message_passing_layers: int,
        message_hidden_dim: int,
        message_layers: int,
        aggregator: str,
        node_injection: float,
        edge_chunk: int,
        dropout: float,
        head_hidden_dim: int,
        head_dropout: float,
        decision_loss_config: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            embedding_dim=embedding_dim,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            decision_loss_config=decision_loss_config,
            config=config,
        )
        self.memory_dim = _positive_int("memory_dim", memory_dim)
        self.message_dim = _positive_int("message_dim", message_dim)
        self.message_passing_layers = _positive_int(
            "message_passing_layers", message_passing_layers
        )
        if self.memory_dim != 128 or self.message_dim != 128:
            raise ValueError("BigMLP preserves the paper 128-d memory/message widths")
        if self.embedding_dim != 128 or self.message_passing_layers != 2:
            raise ValueError("BigMLP preserves the paper 128-d readout and two graph layers")
        if _positive_int("message_layers", message_layers, minimum=2) != 2:
            raise ValueError(
                "BigMLP widens the two-linear-layer MLP; it does not change MLP depth"
            )
        self.aggregator = _aggregator(aggregator)
        self.node_injection = _unit_interval("node_injection", node_injection)
        self.edge_chunk = _positive_int("edge_chunk", edge_chunk)
        dropout_value = _dropout("dropout", dropout)
        self.memory = MemoryBank(
            node_in_dim=self.node_in_dim,
            mem_dim=self.memory_dim,
        )
        message = _DenseMessageNetwork(
            input_dim=2 * self.memory_dim + self.edge_in_dim,
            hidden_dim=message_hidden_dim,
            output_dim=self.message_dim,
            linear_layers=message_layers,
            dropout=dropout_value,
        )
        self.message_functions = nn.ModuleList(
            [message]
            + [copy.deepcopy(message) for _ in range(self.message_passing_layers - 1)]
        )
        self.message_to_memory = nn.ModuleList(
            [
                nn.Linear(self.message_dim, self.memory_dim)
                for _ in range(self.message_passing_layers)
            ]
        )
        self.readout = Readout(mem_dim=self.memory_dim, emb_dim=self.embedding_dim)

    def _aggregate(
        self,
        prepared: _PreparedPaperStep,
        memory: torch.Tensor,
        message_function: _DenseMessageNetwork,
    ) -> torch.Tensor:
        if prepared.edge_count == 0:
            return memory.new_zeros((memory.size(0), self.message_dim))
        source_all, destination_all = prepared.edge_index
        aggregate = memory.new_zeros((memory.size(0), self.message_dim))
        counts = (
            memory.new_zeros((memory.size(0), 1))
            if self.aggregator == "mean"
            else None
        )
        for start in range(0, prepared.edge_count, self.edge_chunk):
            end = min(prepared.edge_count, start + self.edge_chunk)
            source = source_all[start:end]
            destination = destination_all[start:end]
            message = message_function(
                memory[source],
                memory[destination],
                prepared.edge_z[start:end],
            )
            aggregate.index_add_(0, destination, message)
            if counts is not None:
                counts.index_add_(
                    0,
                    destination,
                    memory.new_ones((end - start, 1)),
                )
        return aggregate if counts is None else aggregate / counts.clamp_min(1.0)

    def predict_step(
        self,
        step: Mapping[str, Any],
        state: torch.Tensor | None,
        device: torch.device | str,
    ) -> tuple[IntensityFlowOutput, torch.Tensor]:
        prepared = self._prepare(step, device)
        memory = (
            self.memory.init(prepared.node_x)
            if state is None
            else _validate_node_state(
                state,
                prepared.node_x,
                state_dim=self.memory_dim,
                label="BigMLP",
            )
        )
        for message_function, projection in zip(
            self.message_functions, self.message_to_memory
        ):
            aggregate = self._aggregate(prepared, memory, message_function)
            memory = self.memory.update(
                memory,
                projection(aggregate),
                prepared.node_x,
                inject=self.node_injection,
            )
        output = self._typed_output(self.readout(memory), step, prepared)
        return output, memory


class _GATStyleMessage(nn.Module):
    """Edge-conditioned multi-head attention over incoming candidate edges."""

    def __init__(
        self,
        *,
        memory_dim: int,
        message_dim: int,
        attention_dim: int,
        edge_in_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.memory_dim = _positive_int("attention memory_dim", memory_dim)
        self.message_dim = _positive_int("attention message_dim", message_dim)
        self.attention_dim = _positive_int("attention hidden_dim", attention_dim)
        self.heads = _positive_int("attention heads", heads)
        if self.attention_dim % self.heads != 0:
            raise ValueError("attention_dim must be divisible by attention heads")
        self.head_dim = self.attention_dim // self.heads
        self.query = nn.Linear(self.memory_dim, self.attention_dim)
        self.key = nn.Linear(self.memory_dim, self.attention_dim)
        self.value = nn.Linear(self.memory_dim, self.attention_dim)
        self.edge_key = nn.Linear(edge_in_dim, self.attention_dim)
        self.edge_value = nn.Linear(edge_in_dim, self.attention_dim)
        self.output = nn.Linear(self.attention_dim, self.message_dim)
        self.attention_dropout = nn.Dropout(_dropout("attention dropout", dropout))

    def forward(
        self,
        memory: torch.Tensor,
        edge_index: torch.Tensor,
        edge_z: torch.Tensor,
    ) -> torch.Tensor:
        node_count = int(memory.size(0))
        if edge_index.size(1) == 0:
            return memory.new_zeros(memory.shape)
        source, destination = edge_index
        edge_count = int(source.numel())
        query = self.query(memory[destination]).view(
            edge_count, self.heads, self.head_dim
        )
        key = (
            self.key(memory[source]) + self.edge_key(edge_z)
        ).view(edge_count, self.heads, self.head_dim)
        value = (
            self.value(memory[source]) + self.edge_value(edge_z)
        ).view(edge_count, self.heads, self.head_dim)
        score = F.leaky_relu(
            (query * key).sum(dim=-1) / math.sqrt(float(self.head_dim)),
            negative_slope=0.2,
        )
        scatter_index = destination.unsqueeze(-1).expand(-1, self.heads)
        maximum = score.new_full((node_count, self.heads), -torch.inf)
        maximum.scatter_reduce_(
            0,
            scatter_index,
            score,
            reduce="amax",
            include_self=True,
        )
        unnormalized = torch.exp(score - maximum[destination])
        denominator = score.new_zeros((node_count, self.heads))
        denominator.index_add_(0, destination, unnormalized)
        attention = unnormalized / denominator[destination].clamp_min(1e-12)
        attention = self.attention_dropout(attention)
        weighted = (attention.unsqueeze(-1) * value).reshape(
            edge_count, self.attention_dim
        )
        aggregate = memory.new_zeros((node_count, self.attention_dim))
        aggregate.index_add_(0, destination, weighted)
        return self.output(aggregate)


class EdgeAttn(RuntimeIntensityFlowBaseline):
    """Shared Paper TGN with only its message function changed to attention."""

    architecture = "paper_tgn_edge_attention_v2"
    shared_backbone = _SHARED_BACKBONE
    replacement_scope = "message_operator_only"
    state_contract = "Tensor[num_nodes,memory_dim]"

    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        memory_dim: int,
        message_dim: int,
        embedding_dim: int,
        message_passing_layers: int,
        attention_hidden_dim: int,
        attention_heads: int,
        aggregator: str,
        node_injection: float,
        dropout: float,
        attention_dropout: float,
        head_hidden_dim: int,
        head_dropout: float,
        decision_loss_config: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            embedding_dim=embedding_dim,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            decision_loss_config=decision_loss_config,
            config=config,
        )
        self.memory_dim = _positive_int("memory_dim", memory_dim)
        self.message_dim = _positive_int("message_dim", message_dim)
        self.message_passing_layers = _positive_int(
            "message_passing_layers", message_passing_layers
        )
        if self.memory_dim != 128 or self.message_dim != 128:
            raise ValueError("EdgeAttn preserves the paper 128-d memory/message widths")
        if self.embedding_dim != 128 or self.message_passing_layers != 2:
            raise ValueError("EdgeAttn preserves the paper 128-d readout and two graph layers")
        self.aggregator = _aggregator(aggregator)
        if self.aggregator != "sum":
            raise ValueError("EdgeAttn's attention-weighted aggregation replaces paper sum")
        head_count = _positive_int("attention_heads", attention_heads)
        attention_width = _positive_int(
            "attention_hidden_dim", attention_hidden_dim
        )
        if attention_width % head_count != 0:
            raise ValueError("attention_hidden_dim must be divisible by attention_heads")
        self.node_injection = _unit_interval("node_injection", node_injection)
        dropout_value = _dropout("dropout", dropout)
        attention_dropout_value = _dropout(
            "attention_dropout", attention_dropout
        )
        if not math.isclose(
            dropout_value, attention_dropout_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "EdgeAttn attention_dropout must equal the shared paper dropout"
            )
        self.memory = MemoryBank(
            node_in_dim=self.node_in_dim,
            mem_dim=self.memory_dim,
        )
        self.attention = nn.ModuleList(
            [
                _GATStyleMessage(
                    memory_dim=self.memory_dim,
                    message_dim=self.message_dim,
                    attention_dim=attention_width,
                    edge_in_dim=self.edge_in_dim,
                    heads=head_count,
                    dropout=attention_dropout_value,
                )
                for _ in range(self.message_passing_layers)
            ]
        )
        self.message_to_memory = nn.ModuleList(
            [
                nn.Linear(self.message_dim, self.memory_dim)
                for _ in range(self.message_passing_layers)
            ]
        )
        self.readout = Readout(mem_dim=self.memory_dim, emb_dim=self.embedding_dim)

    def predict_step(
        self,
        step: Mapping[str, Any],
        state: torch.Tensor | None,
        device: torch.device | str,
    ) -> tuple[IntensityFlowOutput, torch.Tensor]:
        prepared = self._prepare(step, device)
        memory = (
            self.memory.init(prepared.node_x)
            if state is None
            else _validate_node_state(
                state,
                prepared.node_x,
                state_dim=self.memory_dim,
                label="EdgeAttn",
            )
        )
        for attention, projection in zip(
            self.attention, self.message_to_memory
        ):
            aggregate = attention(memory, prepared.edge_index, prepared.edge_z)
            memory = self.memory.update(
                memory,
                projection(aggregate),
                prepared.node_x,
                inject=self.node_injection,
            )
        output = self._typed_output(self.readout(memory), step, prepared)
        return output, memory


class TempTrans(RuntimeIntensityFlowBaseline):
    """Shared Paper TGN/MLP graph with only GRU mixing replaced by Transformer."""

    architecture = "paper_tgn_mlp_temporal_transformer_v2"
    shared_backbone = _SHARED_BACKBONE
    replacement_scope = "temporal_mixing_only"
    state_contract = "{node_memory,layer_histories}"

    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        memory_dim: int,
        message_dim: int,
        embedding_dim: int,
        message_passing_layers: int,
        aggregator: str,
        node_injection: float,
        edge_chunk: int,
        context_length: int,
        transformer_layers: int,
        transformer_heads: int,
        ffn_dim: int,
        dropout: float,
        head_hidden_dim: int,
        head_dropout: float,
        decision_loss_config: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            embedding_dim=embedding_dim,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            decision_loss_config=decision_loss_config,
            config=config,
        )
        self.memory_dim = _positive_int("memory_dim", memory_dim)
        self.message_dim = _positive_int("message_dim", message_dim)
        self.message_passing_layers = _positive_int(
            "message_passing_layers", message_passing_layers
        )
        if self.memory_dim != 128 or self.message_dim != 128:
            raise ValueError("TempTrans preserves the paper 128-d memory/message widths")
        if self.embedding_dim != 128 or self.message_passing_layers != 2:
            raise ValueError("TempTrans preserves the paper 128-d readout and two graph layers")
        self.aggregator = _aggregator(aggregator)
        self.node_injection = _unit_interval("node_injection", node_injection)
        self.edge_chunk = _positive_int("edge_chunk", edge_chunk)
        dropout_value = _dropout("dropout", dropout)
        self.node_encoder = nn.Linear(self.node_in_dim, self.memory_dim)
        message = MLPMessage(
            self.memory_dim,
            self.edge_in_dim,
            self.message_dim,
            dropout=dropout_value,
        )
        self.message_functions = nn.ModuleList(
            [message]
            + [copy.deepcopy(message) for _ in range(self.message_passing_layers - 1)]
        )
        self.message_to_memory = nn.ModuleList(
            [
                nn.Linear(self.message_dim, self.memory_dim)
                for _ in range(self.message_passing_layers)
            ]
        )
        self.temporal = TransformerTemporalBackbone(
            d_model=self.memory_dim,
            context_length=context_length,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            ffn_dim=ffn_dim,
            dropout=dropout_value,
        )
        self.readout = Readout(mem_dim=self.memory_dim, emb_dim=self.embedding_dim)

    def _state(
        self,
        state: Any,
        node_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
        if state is None:
            return node_seed, (None,) * self.message_passing_layers
        if not isinstance(state, Mapping):
            raise TypeError("TempTrans state must be a mapping or None")
        if set(state) != {"node_memory", "layer_histories"}:
            raise ValueError("TempTrans state requires node_memory and layer_histories")
        memory = state["node_memory"]
        histories = state["layer_histories"]
        if (
            not isinstance(memory, torch.Tensor)
            or memory.shape != node_seed.shape
            or memory.device != node_seed.device
            or memory.dtype != node_seed.dtype
            or not bool(torch.isfinite(memory).all())
        ):
            raise ValueError("TempTrans node_memory must match the current node seed")
        if not isinstance(histories, (tuple, list)) or len(histories) != self.message_passing_layers:
            raise ValueError("TempTrans requires one temporal history per graph layer")
        if any(value is not None and not isinstance(value, torch.Tensor) for value in histories):
            raise TypeError("TempTrans layer histories must be tensors or None")
        return memory, tuple(histories)

    def _aggregate(
        self,
        prepared: _PreparedPaperStep,
        memory: torch.Tensor,
        message_function: MLPMessage,
    ) -> torch.Tensor:
        if prepared.edge_count == 0:
            return memory.new_zeros((memory.size(0), self.message_dim))
        source_all, destination_all = prepared.edge_index
        aggregate = memory.new_zeros((memory.size(0), self.message_dim))
        counts = (
            memory.new_zeros((memory.size(0), 1))
            if self.aggregator == "mean"
            else None
        )
        for start in range(0, prepared.edge_count, self.edge_chunk):
            end = min(prepared.edge_count, start + self.edge_chunk)
            source = source_all[start:end]
            destination = destination_all[start:end]
            message = message_function(
                memory[source],
                memory[destination],
                prepared.edge_z[start:end],
            )
            aggregate.index_add_(0, destination, message)
            if counts is not None:
                counts.index_add_(
                    0,
                    destination,
                    memory.new_ones((end - start, 1)),
                )
        return aggregate if counts is None else aggregate / counts.clamp_min(1.0)

    def predict_step(
        self,
        step: Mapping[str, Any],
        state: Any,
        device: torch.device | str,
    ) -> tuple[IntensityFlowOutput, Mapping[str, Any]]:
        prepared = self._prepare(step, device)
        node_seed = torch.tanh(self.node_encoder(prepared.node_x))
        memory, histories = self._state(state, node_seed)
        next_histories: list[torch.Tensor] = []
        for message_function, projection, history in zip(
            self.message_functions,
            self.message_to_memory,
            histories,
        ):
            aggregate = projection(
                self._aggregate(prepared, memory, message_function)
            )
            mixed, next_history = self.temporal.forward_step(
                memory + aggregate,
                history,
            )
            memory = (
                (1.0 - self.node_injection) * mixed
                + self.node_injection * node_seed
            )
            next_histories.append(next_history)
        output = self._typed_output(self.readout(memory), step, prepared)
        return output, {
            "node_memory": memory,
            "layer_histories": tuple(next_histories),
        }


def _baseline_config(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(config, Mapping):
        raise TypeError("runtime baseline configuration must be a mapping")
    nested = config.get("paper_baseline")
    if nested is None:
        return config, config
    if not isinstance(nested, Mapping):
        raise TypeError("paper_baseline must be a mapping")
    return nested, config


def _normalized_kind(value: Any) -> str:
    kind = str(value).strip().lower().replace("-", "_")
    aliases = {
        "bigmlp": "big_mlp",
        "edgeattn": "edge_attn",
        "temptrans": "temp_trans",
    }
    kind = aliases.get(kind, kind)
    if kind not in RUNTIME_BASELINE_KINDS:
        raise ValueError(
            f"unknown runtime baseline {value!r}; expected "
            + ", ".join(RUNTIME_BASELINE_KINDS)
        )
    return kind


def build_runtime_baseline(config: Mapping[str, Any]) -> RuntimeIntensityFlowBaseline:
    """Build one runtime baseline from direct or resolved ``paper_baseline`` config.

    Replacement-module values are intentionally required and labelled as
    estimates.  Shared TGN values are checked against the root paper model.
    Rounded parameter targets and their manuscript provenance are mandatory.
    """

    baseline, root = _baseline_config(config)
    kind = _normalized_kind(_required(baseline, "kind", "paper_baseline"))
    interface = str(
        _required(baseline, "interface", f"paper_baseline.{kind}")
    ).strip().lower()
    if interface != "intensity_flow":
        raise ValueError(f"paper_baseline.{kind}.interface must be 'intensity_flow'")
    path = f"paper_baseline.{kind}"
    _validate_runtime_identity(baseline, kind, path=path)
    _validate_shared_backbone(baseline, root, path=path)
    decision_config = baseline.get("decision_loss")
    if decision_config is not None and not isinstance(decision_config, Mapping):
        raise TypeError(f"{path}.decision_loss must be a mapping or null")
    common = {
        "node_in_dim": _required(baseline, "node_in_dim", path),
        "edge_in_dim": _required(baseline, "edge_in_dim", path),
        "dropout": _required(baseline, "dropout", path),
        "head_hidden_dim": _required(baseline, "head_hidden_dim", path),
        "head_dropout": _required(baseline, "head_dropout", path),
        "decision_loss_config": decision_config,
        "config": root,
    }

    shared_tgn = {
        "memory_dim": _required(baseline, "memory_dim", path),
        "message_dim": _required(baseline, "message_dim", path),
        "embedding_dim": _required(baseline, "embedding_dim", path),
        "message_passing_layers": _required(
            baseline, "message_passing_layers", path
        ),
        "aggregator": _required(baseline, "aggregator", path),
        "node_injection": _required(baseline, "node_injection", path),
    }

    if kind == "big_mlp":
        model: RuntimeIntensityFlowBaseline = BigMLP(
            message_hidden_dim=_required(baseline, "message_hidden_dim", path),
            message_layers=_required(baseline, "message_layers", path),
            edge_chunk=_required(baseline, "edge_chunk", path),
            **shared_tgn,
            **common,
        )
    elif kind == "edge_attn":
        model = EdgeAttn(
            attention_hidden_dim=_required(
                baseline, "attention_hidden_dim", path
            ),
            attention_heads=_required(baseline, "attention_heads", path),
            attention_dropout=_required(baseline, "attention_dropout", path),
            **shared_tgn,
            **common,
        )
    else:
        model = TempTrans(
            edge_chunk=_required(baseline, "edge_chunk", path),
            context_length=_required(baseline, "context_length", path),
            transformer_layers=_required(baseline, "transformer_layers", path),
            transformer_heads=_required(baseline, "transformer_heads", path),
            ffn_dim=_required(baseline, "ffn_dim", path),
            **shared_tgn,
            **common,
        )
    _bind_capacity_provenance(model, baseline, path=path)
    return model


__all__ = [
    "BigMLP",
    "EdgeAttn",
    "RUNTIME_BASELINE_KINDS",
    "RuntimeIntensityFlowBaseline",
    "TempTrans",
    "build_runtime_baseline",
]
