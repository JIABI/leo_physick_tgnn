"""Typed recurrent graph model for the UAV/shared-service platform.

The public inference contract is intentionally smaller than the serialized
training-record contract.  :meth:`UAVDescriptorModel.predict_step` accepts a
live :class:`~leo_pg.uav_shared.state.ServiceObservation` or an already
materialized UAV graph and returns only next-epoch controller descriptors.  It
never looks up ``next_target`` (or any other supervision field).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F
import yaml

from .config import ResolvedUAVSharedConfig, resolve_uav_shared_protocol
from .graph import (
    UAV_EDGE_FEATURE_NAMES,
    UAV_GRAPH_CONTRACT_VERSION,
    UAV_STATION_FEATURE_NAMES,
    UAV_UAV_FEATURE_NAMES,
    observation_to_graph,
    validate_graph,
)
from .protocol import UAV_SHARED_PROTOCOL_VERSION, UAVSharedProtocol
from .state import ServiceObservation, ServicePolicyDescriptors


UAV_MODEL_CONTRACT_VERSION = 1
UAV_MODEL_METHOD = "uav_recurrent_gnn"
_MODEL_ALIASES = {
    "uav": UAV_MODEL_METHOD,
    "uav_gnn": UAV_MODEL_METHOD,
    "recurrent_gnn": UAV_MODEL_METHOD,
    UAV_MODEL_METHOD: UAV_MODEL_METHOD,
}


def _finite_float(value: Any, path: str, *, lower: float | None = None) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be numeric")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"{path} must be finite")
    if lower is not None and result < lower:
        raise ValueError(f"{path} must be >= {lower}")
    return result


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be an integer")
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return result


def _root_mapping(source: Any) -> dict[str, Any]:
    """Read only the primitive configuration mapping needed by the model."""

    if isinstance(source, ResolvedUAVSharedConfig):
        return {}
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    if isinstance(source, (str, PathLike)):
        path = Path(source).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, Mapping):
            raise TypeError("UAV model YAML must contain a mapping")
        return copy.deepcopy(dict(loaded))
    raise TypeError("UAV model source must be a mapping, resolved config, or YAML path")


@dataclass(frozen=True)
class UAVModelConfig:
    """Architecture values frozen into every UAV checkpoint."""

    uav_in_dim: int = len(UAV_UAV_FEATURE_NAMES)
    station_in_dim: int = len(UAV_STATION_FEATURE_NAMES)
    edge_in_dim: int = len(UAV_EDGE_FEATURE_NAMES)
    hidden_dim: int = 128
    message_dim: int = 128
    message_passing_layers: int = 2
    head_hidden_dim: int = 128
    dropout: float = 0.1
    head_dropout: float = 0.1
    intensity_max: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "uav_in_dim",
            "station_in_dim",
            "edge_in_dim",
            "hidden_dim",
            "message_dim",
            "message_passing_layers",
            "head_hidden_dim",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), f"uav_model.{name}"),
            )
        if self.uav_in_dim != len(UAV_UAV_FEATURE_NAMES):
            raise ValueError("uav_model.uav_in_dim disagrees with the graph contract")
        if self.station_in_dim != len(UAV_STATION_FEATURE_NAMES):
            raise ValueError(
                "uav_model.station_in_dim disagrees with the graph contract"
            )
        if self.edge_in_dim != len(UAV_EDGE_FEATURE_NAMES):
            raise ValueError("uav_model.edge_in_dim disagrees with the graph contract")
        for name in ("dropout", "head_dropout"):
            dropout = _finite_float(
                getattr(self, name), f"uav_model.{name}", lower=0.0
            )
            if dropout >= 1.0:
                raise ValueError(f"uav_model.{name} must be in [0,1)")
            object.__setattr__(self, name, dropout)
        intensity_max = _finite_float(
            self.intensity_max, "uav_model.intensity_max", lower=0.0
        )
        if intensity_max <= 0.0:
            raise ValueError("uav_model.intensity_max must be positive")
        object.__setattr__(self, "intensity_max", intensity_max)

    @classmethod
    def from_source(cls, source: Any) -> "UAVModelConfig":
        root = _root_mapping(source)
        pipeline = root.get("uav_pipeline", {})
        if pipeline is None:
            pipeline = {}
        if not isinstance(pipeline, Mapping):
            raise TypeError("uav_pipeline must be a mapping")
        raw = pipeline.get("model", root.get("uav_model", {}))
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError("uav_pipeline.model must be a mapping")
        # Reuse shared paper dimensions only as explicit fallbacks for the
        # currently recovered YAML, which predates a dedicated UAV block.
        shared = root.get("model", {})
        if not isinstance(shared, Mapping):
            shared = {}
        head = root.get("head", {})
        if not isinstance(head, Mapping):
            head = {}
        values = dict(raw)
        return cls(
            uav_in_dim=values.get("uav_in_dim", len(UAV_UAV_FEATURE_NAMES)),
            station_in_dim=values.get(
                "station_in_dim", len(UAV_STATION_FEATURE_NAMES)
            ),
            edge_in_dim=values.get("edge_in_dim", len(UAV_EDGE_FEATURE_NAMES)),
            hidden_dim=values.get("hidden_dim", shared.get("mem_dim", 128)),
            message_dim=values.get("message_dim", shared.get("msg_dim", 128)),
            message_passing_layers=values.get(
                "message_passing_layers",
                shared.get("message_passing_layers", 2),
            ),
            head_hidden_dim=values.get(
                "head_hidden_dim", head.get("hidden_dim", 128)
            ),
            dropout=values.get("dropout", shared.get("dropout", 0.1)),
            head_dropout=values.get(
                "head_dropout", values.get("dropout", head.get("dropout", 0.1))
            ),
            intensity_max=values.get("intensity_max", 3.0),
        )


@dataclass(frozen=True)
class UAVRecurrentState:
    """Persistent hidden state for one UAV episode."""

    uav_hidden: torch.Tensor
    station_hidden: torch.Tensor

    def validate(self, uav_count: int, station_count: int, hidden_dim: int) -> None:
        expected = (
            ("uav_hidden", self.uav_hidden, (uav_count, hidden_dim)),
            (
                "station_hidden",
                self.station_hidden,
                (station_count, hidden_dim),
            ),
        )
        for name, value, shape in expected:
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not value.dtype.is_floating_point or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must be a finite floating tensor")
        if self.uav_hidden.device != self.station_hidden.device:
            raise ValueError("UAV recurrent-state tensors must share one device")

    def detach(self) -> "UAVRecurrentState":
        return UAVRecurrentState(
            uav_hidden=self.uav_hidden.detach(),
            station_hidden=self.station_hidden.detach(),
        )

    def to(self, device: torch.device | str) -> "UAVRecurrentState":
        return UAVRecurrentState(
            uav_hidden=self.uav_hidden.to(device),
            station_hidden=self.station_hidden.to(device),
        )


@dataclass(frozen=True)
class UAVDescriptorOutput:
    """Typed next-epoch descriptor prediction in current-edge row order."""

    eta_edge: torch.Tensor
    intensity_edge: torch.Tensor
    feasible_start_logit_edge: torch.Tensor
    station_flow_node: torch.Tensor

    @property
    def feasibility_logit_edge(self) -> torch.Tensor:
        """Compatibility alias for auxiliary-feasibility consumers."""

        return self.feasible_start_logit_edge

    def validate(self, edge_count: int, station_count: int) -> None:
        for name, value, shape in (
            ("eta_edge", self.eta_edge, (edge_count,)),
            ("intensity_edge", self.intensity_edge, (edge_count,)),
            (
                "feasible_start_logit_edge",
                self.feasible_start_logit_edge,
                (edge_count,),
            ),
            ("station_flow_node", self.station_flow_node, (station_count,)),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not value.dtype.is_floating_point or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must be a finite floating tensor")
        if bool((self.eta_edge < 0.0).any()) or bool((self.eta_edge > 1.0).any()):
            raise ValueError("eta_edge must lie in [0,1]")
        if bool((self.intensity_edge < 0.0).any()):
            raise ValueError("intensity_edge must be non-negative")
        if bool((self.station_flow_node < 0.0).any()) or bool(
            (self.station_flow_node > 1.0).any()
        ):
            raise ValueError("station_flow_node must lie in [0,1]")
        devices = {
            self.eta_edge.device,
            self.intensity_edge.device,
            self.feasible_start_logit_edge.device,
            self.station_flow_node.device,
        }
        if len(devices) != 1:
            raise ValueError("UAV descriptor outputs must share one device")

    def to_policy_descriptors(self) -> ServicePolicyDescriptors:
        """Drop the auxiliary logit before exposing fields to the controller."""

        return ServicePolicyDescriptors(
            eta_edge=self.eta_edge,
            intensity_edge=self.intensity_edge,
            station_flow_node=self.station_flow_node,
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "eta_edge": self.eta_edge,
            "intensity_edge": self.intensity_edge,
            "feasible_start_logit_edge": self.feasible_start_logit_edge,
            "station_flow_node": self.station_flow_node,
        }


class UAVDescriptorHead(nn.Module):
    """Edge eta/intensity/feasibility heads plus a station-flow head."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        head_hidden_dim: int,
        *,
        intensity_max: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = _positive_int(hidden_dim, "head.hidden_dim")
        self.edge_dim = _positive_int(edge_dim, "head.edge_dim")
        self.head_hidden_dim = _positive_int(
            head_hidden_dim, "head.head_hidden_dim"
        )
        self.intensity_max = _finite_float(
            intensity_max, "head.intensity_max", lower=0.0
        )
        if self.intensity_max <= 0.0:
            raise ValueError("head.intensity_max must be positive")
        self.edge_trunk = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + self.edge_dim, self.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.eta = nn.Linear(self.head_hidden_dim, 1)
        self.intensity = nn.Linear(self.head_hidden_dim, 1)
        self.feasibility = nn.Linear(self.head_hidden_dim, 1)
        self.station_flow = nn.Sequential(
            nn.Linear(self.hidden_dim, self.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.head_hidden_dim, 1),
        )

    def forward(
        self,
        uav_hidden: torch.Tensor,
        station_hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embedding: torch.Tensor,
    ) -> UAVDescriptorOutput:
        source, destination = edge_index
        edge_hidden = self.edge_trunk(
            torch.cat(
                (
                    uav_hidden.index_select(0, source),
                    station_hidden.index_select(0, destination),
                    edge_embedding,
                ),
                dim=-1,
            )
        )
        result = UAVDescriptorOutput(
            eta_edge=torch.sigmoid(self.eta(edge_hidden).squeeze(-1)),
            # Bounded construction preserves the policy descriptor contract;
            # log1p is applied only by the supervised loss.
            intensity_edge=self.intensity_max
            * torch.sigmoid(self.intensity(edge_hidden).squeeze(-1)),
            feasible_start_logit_edge=self.feasibility(edge_hidden).squeeze(-1),
            station_flow_node=torch.sigmoid(
                self.station_flow(station_hidden).squeeze(-1)
            ),
        )
        result.validate(int(edge_index.size(1)), int(station_hidden.size(0)))
        return result


class _BipartiteMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, message_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(3 * hidden_dim, message_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.to_uav = nn.Linear(message_dim, hidden_dim)
        self.to_station = nn.Linear(message_dim, hidden_dim)
        self.uav_update = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.station_update = nn.GRUCell(2 * hidden_dim, hidden_dim)

    @staticmethod
    def _mean_aggregate(
        values: torch.Tensor,
        index: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        result = values.new_zeros((count, values.size(1)))
        denominator = values.new_zeros((count, 1))
        if index.numel():
            result.index_add_(0, index, values)
            denominator.index_add_(
                0,
                index,
                values.new_ones((index.numel(), 1)),
            )
        return result / denominator.clamp_min(1.0)

    def forward(
        self,
        uav_input: torch.Tensor,
        station_input: torch.Tensor,
        edge_embedding: torch.Tensor,
        edge_index: torch.Tensor,
        uav_hidden: torch.Tensor,
        station_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        message = self.message(
            torch.cat(
                (
                    uav_hidden.index_select(0, source),
                    station_hidden.index_select(0, destination),
                    edge_embedding,
                ),
                dim=-1,
            )
        )
        to_uav = self._mean_aggregate(
            self.to_uav(message), source, int(uav_hidden.size(0))
        )
        to_station = self._mean_aggregate(
            self.to_station(message), destination, int(station_hidden.size(0))
        )
        next_uav = self.uav_update(
            torch.cat((uav_input, to_uav), dim=-1), uav_hidden
        )
        next_station = self.station_update(
            torch.cat((station_input, to_station), dim=-1), station_hidden
        )
        return next_uav, next_station


class UAVDescriptorModel(nn.Module):
    """Recurrent bipartite graph predictor with a target-free inference API."""

    def __init__(
        self,
        protocol: UAVSharedProtocol,
        config: UAVModelConfig | None = None,
        *,
        method: str = UAV_MODEL_METHOD,
    ) -> None:
        super().__init__()
        if not isinstance(protocol, UAVSharedProtocol):
            raise TypeError("protocol must be a UAVSharedProtocol")
        normalized_method = _MODEL_ALIASES.get(str(method).strip().lower())
        if normalized_method is None:
            raise ValueError(
                f"unsupported UAV model method {method!r}; expected {UAV_MODEL_METHOD}"
            )
        self.protocol = protocol
        self.model_config = config or UAVModelConfig()
        if not math.isclose(
            self.model_config.intensity_max,
            float(protocol.exact.intensity_clip[1]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "uav_pipeline.model.intensity_max must equal the protocol clip maximum"
            )
        self.method = normalized_method
        dimensions = self.model_config
        self.uav_encoder = nn.Sequential(
            nn.Linear(dimensions.uav_in_dim, dimensions.hidden_dim),
            nn.ReLU(),
        )
        self.station_encoder = nn.Sequential(
            nn.Linear(dimensions.station_in_dim, dimensions.hidden_dim),
            nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(dimensions.edge_in_dim, dimensions.hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            [
                _BipartiteMessageLayer(
                    dimensions.hidden_dim,
                    dimensions.message_dim,
                    dimensions.dropout,
                )
                for _ in range(dimensions.message_passing_layers)
            ]
        )
        self.head = UAVDescriptorHead(
            dimensions.hidden_dim,
            dimensions.hidden_dim,
            dimensions.head_hidden_dim,
            intensity_max=dimensions.intensity_max,
            dropout=dimensions.head_dropout,
        )
        self.checkpoint_metadata: dict[str, Any] | None = None

    @property
    def signature(self) -> dict[str, Any]:
        return uav_model_signature(self.protocol, self.model_config, self.method)

    @property
    def fingerprint(self) -> str:
        return uav_model_fingerprint(self.signature)

    def _protocol_for_observation(
        self, observation: ServiceObservation
    ) -> UAVSharedProtocol:
        nominal = self.protocol.exact.nominal_uav_count
        if observation.user_count % nominal != 0:
            raise ValueError(
                "live observation UAV count is not a configured density multiple"
            )
        density = observation.user_count // nominal
        compression = observation.meta.get(
            "capacity_compression", self.protocol.capacity_compression
        )
        # Density/capacity sweeps are inference conditions, not learned tensor
        # shapes.  All exact and estimated physical semantics remain frozen.
        return replace(
            self.protocol,
            episode_seed=observation.episode_seed,
            density_multiplier=density,
            capacity_compression=float(compression),
        )

    def _materialize_graph(
        self, value: ServiceObservation | Mapping[str, Any]
    ) -> dict[str, Any]:
        if isinstance(value, ServiceObservation):
            # Episode seed is a randomness identity, not a model-semantic
            # parameter.  Keep every structural protocol field fixed while
            # materializing graphs from held-out seeds.
            episode_protocol = self._protocol_for_observation(value)
            graph = observation_to_graph(value, episode_protocol)
        elif isinstance(value, Mapping):
            # A full record is tolerated as an adapter convenience, but only
            # its observation member is inspected.  Supervision is never read.
            candidate: Any = value.get("observation", value)
            if isinstance(candidate, ServiceObservation):
                episode_protocol = self._protocol_for_observation(candidate)
                graph = observation_to_graph(candidate, episode_protocol)
            elif isinstance(candidate, Mapping):
                graph = dict(candidate)
            else:
                raise TypeError("record.observation must be a graph or ServiceObservation")
        else:
            raise TypeError("predict_step expects a ServiceObservation or graph mapping")
        validate_graph(graph)
        return graph

    def predict_step(
        self,
        observation_or_graph: ServiceObservation | Mapping[str, Any],
        state: UAVRecurrentState | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[UAVDescriptorOutput, UAVRecurrentState]:
        """Predict ``D_(t+1)`` without reading a training target."""

        graph = self._materialize_graph(observation_or_graph)
        target_device = (
            torch.device(device)
            if device is not None
            else next(self.parameters()).device
        )
        uav_x = torch.as_tensor(graph["uav_x"], device=target_device).float()
        station_x = torch.as_tensor(
            graph["station_x"], device=target_device
        ).float()
        edge_index = torch.as_tensor(
            graph["edge_index"], device=target_device
        ).long()
        edge_attr = torch.as_tensor(
            graph["edge_attr"], device=target_device
        ).float()
        uav_input = self.uav_encoder(uav_x)
        station_input = self.station_encoder(station_x)
        edge_embedding = self.edge_encoder(edge_attr)
        if state is None:
            uav_hidden = uav_input.new_zeros(
                (uav_x.size(0), self.model_config.hidden_dim)
            )
            station_hidden = station_input.new_zeros(
                (station_x.size(0), self.model_config.hidden_dim)
            )
        else:
            if not isinstance(state, UAVRecurrentState):
                raise TypeError("state must be UAVRecurrentState or None")
            state = state.to(target_device)
            state.validate(
                int(uav_x.size(0)),
                int(station_x.size(0)),
                self.model_config.hidden_dim,
            )
            uav_hidden = state.uav_hidden
            station_hidden = state.station_hidden
        for layer in self.layers:
            uav_hidden, station_hidden = layer(
                uav_input,
                station_input,
                edge_embedding,
                edge_index,
                uav_hidden,
                station_hidden,
            )
        prediction = self.head(
            uav_hidden,
            station_hidden,
            edge_index,
            edge_embedding,
        )
        return prediction, UAVRecurrentState(uav_hidden, station_hidden)

    def forward_step(
        self,
        graph: ServiceObservation | Mapping[str, Any],
        state: UAVRecurrentState | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[UAVDescriptorOutput, UAVRecurrentState]:
        return self.predict_step(graph, state, device)

    def forward(
        self,
        graph: ServiceObservation | Mapping[str, Any],
        state: UAVRecurrentState | None = None,
    ) -> tuple[UAVDescriptorOutput, UAVRecurrentState]:
        return self.predict_step(graph, state)


def uav_model_signature(
    protocol: UAVSharedProtocol,
    config: UAVModelConfig,
    method: str = UAV_MODEL_METHOD,
) -> dict[str, Any]:
    normalized_method = _MODEL_ALIASES.get(str(method).strip().lower())
    if normalized_method is None:
        raise ValueError(f"unknown UAV model method {method!r}")
    return {
        "model_contract_version": UAV_MODEL_CONTRACT_VERSION,
        "graph_contract_version": UAV_GRAPH_CONTRACT_VERSION,
        "method": normalized_method,
        "architecture": "bipartite_uav_station_world_model_v1",
        "model_config": asdict(config),
        "uav_feature_names": list(UAV_UAV_FEATURE_NAMES),
        "station_feature_names": list(UAV_STATION_FEATURE_NAMES),
        "edge_feature_names": list(UAV_EDGE_FEATURE_NAMES),
        "descriptor_domains": [
            "edge_eta",
            "edge_feasible_start_intensity",
            "station_flow",
        ],
        "auxiliary_only": ["edge_feasible_start_logit"],
        "intensity_bounds": list(protocol.exact.intensity_clip),
        "structural_protocol": uav_structural_protocol_signature(protocol),
        "structural_protocol_fingerprint": uav_structural_protocol_fingerprint(
            protocol
        ),
    }


def uav_model_fingerprint(signature: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(signature), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def uav_structural_protocol_signature(
    protocol: UAVSharedProtocol,
) -> dict[str, Any]:
    """Return model-relevant environment semantics, excluding episode seed."""

    return {
        "protocol_version": UAV_SHARED_PROTOCOL_VERSION,
        "density_multiplier": protocol.density_multiplier,
        "capacity_compression": protocol.capacity_compression,
        "exact": asdict(protocol.exact),
        "estimated": asdict(protocol.estimated),
    }


def uav_structural_protocol_fingerprint(protocol: UAVSharedProtocol) -> str:
    encoded = json.dumps(
        uav_structural_protocol_signature(protocol),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_uav_model(
    source: Any,
    method: str | None = None,
    *,
    device: torch.device | str | None = None,
) -> UAVDescriptorModel:
    """Build the typed UAV model from repository YAML or a root mapping."""

    protocol = resolve_uav_shared_protocol(source)
    config = UAVModelConfig.from_source(source)
    model = UAVDescriptorModel(
        protocol,
        config,
        method=method or UAV_MODEL_METHOD,
    )
    if device is not None:
        model.to(device)
    return model


def _checkpoint_model_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("UAV checkpoint must be a mapping")
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value  # type: ignore[return-value]
    raise KeyError("UAV checkpoint does not contain a model state dictionary")


def load_uav_model_checkpoint(
    source: Any,
    checkpoint: str | PathLike[str],
    *,
    method: str | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> UAVDescriptorModel:
    """Build, signature-check and restore a best/last UAV checkpoint."""

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("UAV checkpoint payload must be a mapping")
    model = build_uav_model(source, method, device=device)
    checkpoint_contract = payload.get("uav_model_contract_version")
    if checkpoint_contract != UAV_MODEL_CONTRACT_VERSION:
        raise ValueError(
            "UAV model checkpoint contract mismatch: "
            f"{checkpoint_contract!r} != {UAV_MODEL_CONTRACT_VERSION}"
        )
    checkpoint_fingerprint = payload.get("uav_model_fingerprint")
    if checkpoint_fingerprint != model.fingerprint:
        raise ValueError("UAV checkpoint/model fingerprint mismatch")
    model.load_state_dict(_checkpoint_model_state(payload), strict=strict)
    model.checkpoint_metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state_dict", "model_state", "state_dict", "model"}
    }
    model.eval()
    return model


__all__ = [
    "UAVDescriptorHead",
    "UAVDescriptorModel",
    "UAVDescriptorOutput",
    "UAVModelConfig",
    "UAVRecurrentState",
    "UAV_MODEL_CONTRACT_VERSION",
    "UAV_MODEL_METHOD",
    "build_uav_model",
    "load_uav_model_checkpoint",
    "uav_model_fingerprint",
    "uav_model_signature",
    "uav_structural_protocol_fingerprint",
    "uav_structural_protocol_signature",
]
