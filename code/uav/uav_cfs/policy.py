"""Fixed rank policy for the UAV/shared-service protocol.

The policy consumes only the controller-facing descriptor copy.  Reachability,
queue admission, service-start feasibility, motion and energy remain simulator
authority.  Service-start infeasible candidates are removed before ranking in
the primary manuscript protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .state import (
    ReassociationAction,
    ServiceObservation,
    ServicePolicyDescriptors,
)


UAV_POLICY_CONTRACT_VERSION = 1
REPOSITORY_REFERENCE_UAV_SCORE_WEIGHTS = (1.0, 0.5, 0.7)


@dataclass(frozen=True)
class UAVFixedRankPolicyConfig:
    """Explicit fixed-score configuration.

    The weight order is ``eta``, ``intensity`` and ``station_flow`` and follows
    the shared role-position contract (local utility, approach to constraint,
    resource pressure) = (1.0, 0.5, 0.7).
    """

    eta_weight: float
    intensity_weight: float
    flow_weight: float
    hard_feasible_start_mask: bool
    hysteresis: float | None

    def __post_init__(self) -> None:
        weights = (self.eta_weight, self.intensity_weight, self.flow_weight)
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in weights
        ):
            raise ValueError("UAV score weights must be finite and non-negative")
        if sum(float(value) for value in weights) <= 0.0:
            raise ValueError("at least one UAV score weight must be positive")
        if type(self.hard_feasible_start_mask) is not bool:
            raise TypeError("hard_feasible_start_mask must be bool")
        if self.hysteresis is not None and (
            not math.isfinite(float(self.hysteresis))
            or float(self.hysteresis) < 0.0
        ):
            raise ValueError("hysteresis must be finite and non-negative")

    @classmethod
    def from_source(cls, source: Mapping[str, Any]) -> "UAVFixedRankPolicyConfig":
        """Resolve the explicit ``uav_pipeline.policy`` mapping."""

        if not isinstance(source, Mapping):
            raise TypeError("UAV policy source must be a mapping")
        pipeline = source.get("uav_pipeline", source)
        if not isinstance(pipeline, Mapping):
            raise TypeError("uav_pipeline must be a mapping")
        raw = pipeline.get("policy", pipeline)
        if not isinstance(raw, Mapping):
            raise TypeError("uav_pipeline.policy must be a mapping")
        fields = {
            "eta_weight",
            "intensity_weight",
            "flow_weight",
            "hard_feasible_start_mask",
            "hysteresis",
        }
        missing = sorted(fields - set(raw))
        unknown = sorted(set(raw) - fields - {"provenance"})
        if missing or unknown:
            raise ValueError(
                "uav_pipeline.policy schema mismatch; "
                f"missing={missing!r}, unknown={unknown!r}"
            )
        return cls(**{name: raw[name] for name in fields})


@dataclass(frozen=True)
class ServiceCandidateScores:
    eta_rank: torch.Tensor
    intensity_rank: torch.Tensor
    flow_rank: torch.Tensor
    total: torch.Tensor
    eligible: torch.Tensor


def normalized_service_rank(
    values: torch.Tensor,
    station_ids: torch.Tensor,
    *,
    higher_is_better: bool,
) -> torch.Tensor:
    """Return deterministic ordinal desirability ranks in ``(0, 1]``."""

    if values.ndim != 1 or station_ids.ndim != 1:
        raise ValueError("values and station_ids must be one-dimensional")
    if values.numel() != station_ids.numel():
        raise ValueError("values and station_ids must have equal length")
    if station_ids.dtype != torch.long:
        raise TypeError("station_ids must have dtype long")
    if values.device != station_ids.device:
        raise ValueError("values and station_ids must share a device")
    if values.numel() == 0:
        dtype = values.dtype if values.is_floating_point() else torch.float32
        return torch.empty(0, dtype=dtype, device=values.device)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("rank values must be finite")
    if torch.unique(station_ids).numel() != station_ids.numel():
        raise ValueError("station ids must be unique within a UAV candidate set")

    # Stable two-stage sorting realizes (descriptor desirability, station id).
    station_order = torch.argsort(station_ids, stable=True)
    value_order = torch.argsort(
        values.index_select(0, station_order),
        descending=higher_is_better,
        stable=True,
    )
    best_first = station_order.index_select(0, value_order)
    dtype = values.dtype if values.is_floating_point() else torch.float32
    desirability = torch.arange(
        values.numel(),
        0,
        -1,
        dtype=dtype,
        device=values.device,
    ) / float(values.numel())
    ranks = torch.empty(values.numel(), dtype=dtype, device=values.device)
    ranks[best_first] = desirability
    return ranks


def score_service_candidates(
    observation: ServiceObservation,
    config: UAVFixedRankPolicyConfig | None = None,
    descriptors: ServicePolicyDescriptors | None = None,
) -> ServiceCandidateScores:
    """Score the reachable Top-k candidates without selecting an action."""

    observation.validate()
    if config is None:
        raise TypeError("score_service_candidates requires an explicit policy config")
    fields = descriptors or observation.policy_descriptors
    fields.validate(observation.edge_count, observation.station_count)
    device = observation.candidate_edge_ids.device
    tensors = {
        "eta_edge": fields.eta_edge,
        "intensity_edge": fields.intensity_edge,
        "station_flow_node": fields.station_flow_node,
        "feasible_start_edge": observation.sim_descriptors.feasible_start_edge,
        "user_order": observation.user_order,
    }
    mismatched = [name for name, value in tensors.items() if value.device != device]
    if mismatched:
        raise ValueError(
            "UAV policy tensors must share one device; mismatched: "
            + ", ".join(mismatched)
        )

    edge_count = observation.edge_count
    dtype = fields.eta_edge.dtype
    eta_rank = torch.zeros(edge_count, dtype=dtype, device=device)
    intensity_rank = torch.zeros_like(eta_rank)
    flow_rank = torch.zeros_like(eta_rank)
    total = torch.full_like(eta_rank, -torch.inf)
    eligible = torch.zeros(edge_count, dtype=torch.bool, device=device)
    if edge_count == 0:
        return ServiceCandidateScores(
            eta_rank=eta_rank,
            intensity_rank=intensity_rank,
            flow_rank=flow_rank,
            total=total,
            eligible=eligible,
        )

    edge_users = observation.candidate_edge_ids[:, 0]
    edge_stations = observation.candidate_edge_ids[:, 1]
    for user in observation.user_order.tolist():
        user_edges = torch.nonzero(edge_users == user, as_tuple=False).flatten()
        if user_edges.numel() == 0:
            continue
        ranked_edges = user_edges
        if config.hard_feasible_start_mask:
            ranked_edges = ranked_edges[
                observation.sim_descriptors.feasible_start_edge.index_select(
                    0, ranked_edges
                )
            ]
        if ranked_edges.numel() == 0:
            continue
        stations = edge_stations.index_select(0, ranked_edges)
        eta = fields.eta_edge.index_select(0, ranked_edges)
        intensity = fields.intensity_edge.index_select(0, ranked_edges)
        flow = fields.station_flow_node.index_select(0, stations)
        user_eta = normalized_service_rank(
            eta, stations, higher_is_better=True
        ).to(dtype=dtype)
        user_intensity = normalized_service_rank(
            intensity, stations, higher_is_better=False
        ).to(dtype=dtype)
        user_flow = normalized_service_rank(
            flow, stations, higher_is_better=False
        ).to(dtype=dtype)
        user_total = (
            config.eta_weight * user_eta
            + config.intensity_weight * user_intensity
            + config.flow_weight * user_flow
        )
        eta_rank[ranked_edges] = user_eta
        intensity_rank[ranked_edges] = user_intensity
        flow_rank[ranked_edges] = user_flow
        total[ranked_edges] = user_total
        eligible[ranked_edges] = True

    return ServiceCandidateScores(
        eta_rank=eta_rank,
        intensity_rank=intensity_rank,
        flow_rank=flow_rank,
        total=total,
        eligible=eligible,
    )


def _best_edge(
    edge_indices: torch.Tensor,
    score: torch.Tensor,
    station_ids: torch.Tensor,
) -> int:
    by_station = edge_indices.index_select(
        0,
        torch.argsort(
            station_ids.index_select(0, edge_indices),
            stable=True,
        ),
    )
    by_score = torch.argsort(
        score.index_select(0, by_station),
        descending=True,
        stable=True,
    )
    return int(by_station[by_score[0]].item())


class UAVFixedRankPolicy:
    """Stateless fixed reassociation policy used in all paired conditions."""

    def __init__(self, config: UAVFixedRankPolicyConfig) -> None:
        if not isinstance(config, UAVFixedRankPolicyConfig):
            raise TypeError("UAVFixedRankPolicy requires an explicit config")
        self.config = config

    def score(
        self,
        observation: ServiceObservation,
        descriptors: ServicePolicyDescriptors | None = None,
    ) -> ServiceCandidateScores:
        return score_service_candidates(observation, self.config, descriptors)

    def select_action(
        self,
        observation: ServiceObservation,
        descriptors: ServicePolicyDescriptors | None = None,
    ) -> ReassociationAction:
        fields = descriptors or observation.policy_descriptors
        scores = self.score(observation, fields)
        requested = torch.full_like(observation.current_station, -1)
        if observation.edge_count == 0:
            return ReassociationAction(
                observation_id=observation.observation_id,
                requested_station=requested,
            )

        edge_users = observation.candidate_edge_ids[:, 0]
        edge_stations = observation.candidate_edge_ids[:, 1]
        for user in observation.user_order.tolist():
            user_edges = torch.nonzero(edge_users == user, as_tuple=False).flatten()
            ranked_edges = user_edges[scores.eligible.index_select(0, user_edges)]
            if ranked_edges.numel() == 0:
                continue
            best_edge = _best_edge(ranked_edges, scores.total, edge_stations)
            best_station = int(edge_stations[best_edge].item())
            current = int(observation.current_station[user].item())
            if current < 0:
                requested[user] = best_station
                continue
            current_edges = user_edges[
                edge_stations.index_select(0, user_edges) == current
            ]
            if current_edges.numel() != 1 or not bool(
                scores.eligible[current_edges[0]].item()
            ):
                requested[user] = best_station
                continue
            if best_station == current:
                requested[user] = current
                continue
            current_edge = int(current_edges[0].item())
            hysteresis = self.config.hysteresis
            if hysteresis is None:
                hysteresis = 1.0 / float(ranked_edges.numel())
            best_score = float(scores.total[best_edge].item())
            required = float(scores.total[current_edge].item()) + float(hysteresis)
            if best_score >= required or math.isclose(
                best_score, required, rel_tol=1e-7, abs_tol=1e-8
            ):
                requested[user] = best_station
            else:
                requested[user] = current

        action = ReassociationAction(
            observation_id=observation.observation_id,
            requested_station=requested,
        )
        action.validate(observation.user_count, observation.station_count)
        return action

    def __call__(
        self,
        observation: ServiceObservation,
        descriptors: ServicePolicyDescriptors | None = None,
    ) -> ReassociationAction:
        return self.select_action(observation, descriptors)


__all__ = [
    "REPOSITORY_REFERENCE_UAV_SCORE_WEIGHTS",
    "ServiceCandidateScores",
    "UAVFixedRankPolicy",
    "UAVFixedRankPolicyConfig",
    "UAV_POLICY_CONTRACT_VERSION",
    "normalized_service_rank",
    "score_service_candidates",
]
