from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from leo_pg.sim.state import ControlObservation, ServingAction


DEFAULT_SCORE_WEIGHTS = (1.0, 0.4, 0.6)


@dataclass(frozen=True)
class FixedRankPolicyConfig:
    """Configuration for the fixed Intensity--Flow score controller.

    The paper default is the fixed Top-6 rank step ``1/6``. Passing
    ``hysteresis=None`` instead requests an adaptive ``1/k`` margin based on
    the number of candidates actually ranked for that user.
    """

    gamma_weight: float = DEFAULT_SCORE_WEIGHTS[0]
    load_weight: float = DEFAULT_SCORE_WEIGHTS[1]
    intensity_weight: float = DEFAULT_SCORE_WEIGHTS[2]
    hard_feasibility_mask: bool = False
    min_dwell_steps: int = 10
    hysteresis: float | None = 1.0 / 6.0

    def __post_init__(self) -> None:
        weights = (
            self.gamma_weight,
            self.load_weight,
            self.intensity_weight,
        )
        if not all(math.isfinite(float(value)) and value >= 0 for value in weights):
            raise ValueError("score weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one score weight must be positive")
        if isinstance(self.min_dwell_steps, bool) or self.min_dwell_steps < 0:
            raise ValueError("min_dwell_steps must be a non-negative integer")
        if int(self.min_dwell_steps) != self.min_dwell_steps:
            raise ValueError("min_dwell_steps must be a non-negative integer")
        if self.hysteresis is not None and (
            not math.isfinite(float(self.hysteresis)) or self.hysteresis < 0
        ):
            raise ValueError("hysteresis must be finite and non-negative")


@dataclass
class CandidateScores:
    """Per-edge rank components and total controller score.

    Excluded edges have component ranks of zero and a total score of
    ``-inf``. All tensors follow the observation's candidate-edge order.
    """

    gamma_rank: torch.Tensor
    load_rank: torch.Tensor
    intensity_rank: torch.Tensor
    total: torch.Tensor
    eligible: torch.Tensor


def normalized_ordinal_rank(
    values: torch.Tensor,
    satellite_ids: torch.Tensor,
    *,
    higher_is_better: bool,
) -> torch.Tensor:
    """Return deterministic ordinal desirability ranks in ``(0, 1]``.

    The best item receives ``1`` and the worst receives ``1/n``. Equal values
    are ordered by smaller satellite id, so results are independent of the
    incoming candidate-edge order. Ordinal rather than averaged tie ranks are
    intentional: they provide a total, reproducible order for policy use.
    """

    if values.ndim != 1 or satellite_ids.ndim != 1 or values.numel() != satellite_ids.numel():
        raise ValueError("values and satellite_ids must be one-dimensional and equally sized")
    if satellite_ids.dtype != torch.long:
        raise ValueError("satellite_ids must have dtype long")
    if values.numel() == 0:
        dtype = values.dtype if values.is_floating_point() else torch.float32
        return torch.empty(0, dtype=dtype, device=values.device)
    if not torch.isfinite(values).all():
        raise ValueError("rank values must be finite")
    if values.device != satellite_ids.device:
        raise ValueError("values and satellite_ids must be on the same device")
    if torch.unique(satellite_ids).numel() != satellite_ids.numel():
        raise ValueError("satellite ids must be unique within one user's candidate set")

    # Stable two-pass sorting implements the lexicographic key
    # (descriptor desirability, smaller satellite id).
    satellite_order = torch.argsort(satellite_ids, stable=True)
    value_order = torch.argsort(
        values.index_select(0, satellite_order),
        descending=higher_is_better,
        stable=True,
    )
    best_first = satellite_order.index_select(0, value_order)
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


def _validate_policy_tensor_devices(observation: ControlObservation) -> None:
    expected = observation.candidate_edge_index.device
    tensors = {
        "candidate_edge_ids": observation.candidate_edge_ids,
        "sim feasible_edge": observation.sim_descriptors.feasible_edge,
        "policy gamma_edge": observation.policy_descriptors.gamma_edge,
        "policy intensity_edge": observation.policy_descriptors.intensity_edge,
        "policy flow_node": observation.policy_descriptors.flow_node,
        "current_serving": observation.current_serving,
        "hold_steps": observation.hold_steps,
        "user_order": observation.user_order,
    }
    mismatched = [name for name, value in tensors.items() if value.device != expected]
    if mismatched:
        raise ValueError(
            "controller tensors must share one device; mismatched: "
            + ", ".join(mismatched)
        )


def score_candidates(
    observation: ControlObservation,
    config: FixedRankPolicyConfig | None = None,
) -> CandidateScores:
    """Compute fixed per-user ordinal scores without selecting an action."""

    config = config or FixedRankPolicyConfig()
    observation.validate()
    _validate_policy_tensor_devices(observation)

    edge_count = observation.edge_count
    descriptors = observation.policy_descriptors
    dtype = descriptors.gamma_edge.dtype
    if not descriptors.gamma_edge.is_floating_point():
        dtype = torch.float32
    device = observation.candidate_edge_index.device
    gamma_rank = torch.zeros(edge_count, dtype=dtype, device=device)
    load_rank = torch.zeros_like(gamma_rank)
    intensity_rank = torch.zeros_like(gamma_rank)
    total = torch.full_like(gamma_rank, -torch.inf)
    eligible = torch.zeros(edge_count, dtype=torch.bool, device=device)

    edge_users = observation.candidate_edge_ids[:, 0]
    edge_satellites = observation.candidate_edge_ids[:, 1]
    feasible = observation.sim_descriptors.feasible_edge
    for user in observation.user_order.tolist():
        user_edges = torch.nonzero(edge_users == user, as_tuple=False).flatten()
        if user_edges.numel() == 0:
            continue
        all_satellites = edge_satellites.index_select(0, user_edges)
        if torch.unique(all_satellites).numel() != all_satellites.numel():
            raise ValueError(f"user {user} has duplicate candidate satellite ids")
        ranked_edges = user_edges
        if config.hard_feasibility_mask:
            ranked_edges = ranked_edges[feasible.index_select(0, ranked_edges)]
        if ranked_edges.numel() == 0:
            continue

        satellites = edge_satellites.index_select(0, ranked_edges)
        gamma = descriptors.gamma_edge.index_select(0, ranked_edges)
        intensity = descriptors.intensity_edge.index_select(0, ranked_edges)
        flow = descriptors.flow_node.index_select(0, satellites)
        user_gamma_rank = normalized_ordinal_rank(
            gamma,
            satellites,
            higher_is_better=True,
        ).to(dtype=dtype)
        user_load_rank = normalized_ordinal_rank(
            flow,
            satellites,
            higher_is_better=False,
        ).to(dtype=dtype)
        user_intensity_rank = normalized_ordinal_rank(
            intensity,
            satellites,
            higher_is_better=False,
        ).to(dtype=dtype)
        user_total = (
            config.gamma_weight * user_gamma_rank
            + config.load_weight * user_load_rank
            + config.intensity_weight * user_intensity_rank
        )

        gamma_rank[ranked_edges] = user_gamma_rank
        load_rank[ranked_edges] = user_load_rank
        intensity_rank[ranked_edges] = user_intensity_rank
        total[ranked_edges] = user_total
        eligible[ranked_edges] = True

    return CandidateScores(
        gamma_rank=gamma_rank,
        load_rank=load_rank,
        intensity_rank=intensity_rank,
        total=total,
        eligible=eligible,
    )


def _best_edge(
    edge_indices: torch.Tensor,
    total_score: torch.Tensor,
    satellite_ids: torch.Tensor,
) -> int:
    """Return an edge index using score descending, satellite id ascending."""

    if edge_indices.numel() == 0:
        raise ValueError("cannot choose from an empty edge set")
    sat_order = torch.argsort(
        satellite_ids.index_select(0, edge_indices),
        stable=True,
    )
    by_satellite = edge_indices.index_select(0, sat_order)
    score_order = torch.argsort(
        total_score.index_select(0, by_satellite),
        descending=True,
        stable=True,
    )
    return int(by_satellite[score_order[0]].item())


class FixedRankPolicy:
    """Fixed Intensity--Flow rank controller with hard stability rules."""

    def __init__(self, config: FixedRankPolicyConfig | None = None) -> None:
        self.config = config or FixedRankPolicyConfig()

    def score(self, observation: ControlObservation) -> CandidateScores:
        return score_candidates(observation, self.config)

    def select_action(self, observation: ControlObservation) -> ServingAction:
        scores = self.score(observation)
        requested = observation.current_serving.clone()
        edge_users = observation.candidate_edge_ids[:, 0]
        edge_satellites = observation.candidate_edge_ids[:, 1]

        for user in observation.user_order.tolist():
            user_edges = torch.nonzero(edge_users == user, as_tuple=False).flatten()
            ranked_edges = user_edges[scores.eligible.index_select(0, user_edges)]
            current = int(observation.current_serving[user].item())
            if ranked_edges.numel() == 0:
                # The transition layer remains authoritative. With no valid
                # alternative, preserve the current association (including -1).
                continue

            best_edge = _best_edge(ranked_edges, scores.total, edge_satellites)
            best_satellite = int(edge_satellites[best_edge].item())
            if current < 0:
                # Initial admission is not a handover and has no dwell history.
                requested[user] = best_satellite
                continue
            current_edges = user_edges[
                edge_satellites.index_select(0, user_edges) == current
            ]
            if current_edges.numel() != 1 or not bool(
                scores.eligible[current_edges[0]].item()
            ):
                # The fixed policy never bypasses dwell/hysteresis using a
                # simulator feasibility bit. Requesting the existing serving
                # id lets the transition layer apply its authoritative gate;
                # a rejected request becomes disconnected and is admitted as
                # a fresh association at the next decision epoch.
                continue
            if best_satellite == current:
                continue
            if int(observation.hold_steps[user].item()) < self.config.min_dwell_steps:
                continue

            current_edge = int(current_edges[0].item())
            hysteresis = self.config.hysteresis
            if hysteresis is None:
                hysteresis = 1.0 / float(ranked_edges.numel())
            best_score = float(scores.total[best_edge].item())
            required_score = float(scores.total[current_edge].item()) + hysteresis
            # Rank steps such as 1/6 are not exactly representable in float32;
            # keep the protocol's inclusive boundary numerically inclusive.
            if best_score >= required_score or math.isclose(
                best_score,
                required_score,
                rel_tol=1e-7,
                abs_tol=1e-8,
            ):
                requested[user] = best_satellite

        action = ServingAction(
            observation_id=observation.observation_id,
            requested_serving=requested,
        )
        action.validate(observation.user_count, observation.satellite_count)
        return action

    def __call__(self, observation: ControlObservation) -> ServingAction:
        return self.select_action(observation)


def select_action(
    observation: ControlObservation,
    config: FixedRankPolicyConfig | None = None,
) -> ServingAction:
    """Functional entry point for the fixed rank policy."""

    return FixedRankPolicy(config).select_action(observation)
