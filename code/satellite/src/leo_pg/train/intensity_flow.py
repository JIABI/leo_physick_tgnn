from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from leo_pg.models.heads.intensity_flow import IntensityFlowOutput
from leo_pg.sim.state import ControlObservation


@dataclass(frozen=True)
class IntensityFlowTarget:
    """Next-epoch targets aligned to the source epoch's candidate-edge order."""

    gamma_edge: torch.Tensor
    log1p_intensity_edge: torch.Tensor
    feasibility_edge: torch.Tensor
    persistent_edge: torch.Tensor
    flow_node: torch.Tensor

    def validate(self, edge_count: int, satellite_count: int) -> None:
        edge_fields = {
            "gamma_edge": self.gamma_edge,
            "log1p_intensity_edge": self.log1p_intensity_edge,
            "feasibility_edge": self.feasibility_edge,
            "persistent_edge": self.persistent_edge,
        }
        for name, value in edge_fields.items():
            if value.ndim != 1 or value.numel() != edge_count:
                raise ValueError(f"{name} must have shape [{edge_count}]")
        if self.flow_node.ndim != 1 or self.flow_node.numel() != satellite_count:
            raise ValueError(f"flow_node must have shape [{satellite_count}]")
        if self.persistent_edge.dtype != torch.bool:
            raise ValueError("persistent_edge must have dtype bool")
        if self.feasibility_edge.dtype != torch.bool:
            raise ValueError("feasibility_edge must have dtype bool")
        for name, value in (
            ("gamma_edge", self.gamma_edge),
            ("log1p_intensity_edge", self.log1p_intensity_edge),
            ("flow_node", self.flow_node),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if torch.any(self.log1p_intensity_edge < 0):
            raise ValueError("log1p_intensity_edge must be non-negative")


@dataclass(frozen=True)
class IntensityFlowLossWeights:
    """Explicit multi-task weights; no unpublished values are assumed."""

    gamma: float
    intensity: float
    flow: float
    feasibility: float

    def __post_init__(self) -> None:
        values = (self.gamma, self.intensity, self.flow, self.feasibility)
        if not all(math.isfinite(float(value)) and value >= 0 for value in values):
            raise ValueError("loss weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one loss weight must be positive")


@dataclass(frozen=True)
class IntensityFlowLoss:
    total: torch.Tensor
    gamma: torch.Tensor
    intensity: torch.Tensor
    flow: torch.Tensor
    feasibility: torch.Tensor
    persistent_edge_count: int


def build_next_step_target(
    source_observation: ControlObservation,
    next_observation: ControlObservation,
) -> IntensityFlowTarget:
    """Join simulator targets at ``t+1`` onto candidate ids from ``t``.

    Edge predictions are supervised only where local ``(user, satellite)`` ids
    persist into the next geometry-defined candidate set. Non-persistent target
    slots are zero-filled and excluded by ``persistent_edge``. Satellite flow
    remains node-aligned and is supervised for every satellite.
    """

    source_observation.validate()
    next_observation.validate()
    if source_observation.episode_seed != next_observation.episode_seed:
        raise ValueError("source and next observations must belong to one episode")
    if next_observation.epoch != source_observation.epoch + 1:
        raise ValueError("target construction requires consecutive epochs")
    if (
        source_observation.user_count != next_observation.user_count
        or source_observation.satellite_count != next_observation.satellite_count
    ):
        raise ValueError("node identities must be stable across one-step targets")
    source_fields = source_observation.sim_descriptors.policy_fields
    next_fields = next_observation.sim_descriptors.policy_fields
    source_tensors = (
        source_observation.candidate_edge_ids,
        source_fields.gamma_edge,
        source_fields.intensity_edge,
        source_fields.flow_node,
    )
    next_tensors = (
        next_observation.candidate_edge_ids,
        next_fields.gamma_edge,
        next_fields.intensity_edge,
        next_fields.flow_node,
        next_observation.sim_descriptors.feasible_edge,
    )
    if len({value.device for value in (*source_tensors, *next_tensors)}) != 1:
        raise ValueError("source and next target tensors must share one device")
    if (
        source_fields.gamma_edge.dtype != next_fields.gamma_edge.dtype
        or source_fields.intensity_edge.dtype != next_fields.intensity_edge.dtype
        or source_fields.flow_node.dtype != next_fields.flow_node.dtype
    ):
        raise ValueError("source and next descriptor dtypes must match")

    device = source_observation.candidate_edge_ids.device
    next_feasible = next_observation.sim_descriptors.feasible_edge
    gamma = torch.zeros_like(source_observation.sim_descriptors.policy_fields.gamma_edge)
    log_intensity = torch.zeros_like(gamma)
    feasibility = torch.zeros(gamma.numel(), dtype=torch.bool, device=device)
    persistent = torch.zeros_like(feasibility)
    next_lookup = {
        (int(user), int(satellite)): index
        for index, (user, satellite) in enumerate(
            next_observation.candidate_edge_ids.tolist()
        )
    }
    for source_index, edge_id in enumerate(source_observation.candidate_edge_ids.tolist()):
        next_index = next_lookup.get((int(edge_id[0]), int(edge_id[1])))
        if next_index is None:
            continue
        persistent[source_index] = True
        gamma[source_index] = next_fields.gamma_edge[next_index]
        log_intensity[source_index] = torch.log1p(
            next_fields.intensity_edge[next_index]
        )
        feasibility[source_index] = next_feasible[next_index]

    target = IntensityFlowTarget(
        gamma_edge=gamma,
        log1p_intensity_edge=log_intensity,
        feasibility_edge=feasibility,
        persistent_edge=persistent,
        flow_node=next_fields.flow_node.clone(),
    )
    target.validate(source_observation.edge_count, source_observation.satellite_count)
    return target


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    # Keep a differentiable, exactly zero loss when candidate churn leaves no
    # persistent edge; node-flow supervision remains active.
    return values.sum() * 0.0


def intensity_flow_one_step_loss(
    prediction: IntensityFlowOutput,
    target: IntensityFlowTarget,
    weights: IntensityFlowLossWeights,
) -> IntensityFlowLoss:
    """Compute the masked multi-domain one-step objective.

    Gamma uses MSE in its native scale. Intensity uses MSE after ``log1p`` as
    specified by the manuscript. Flow uses satellite-domain MSE. The auxiliary
    feasibility logit uses BCE on exactly the same persistent-edge mask and is
    not part of the controller contract.
    """

    edge_count = prediction.policy_descriptors.gamma_edge.numel()
    satellite_count = prediction.policy_descriptors.flow_node.numel()
    prediction.validate(edge_count, satellite_count)
    target.validate(edge_count, satellite_count)
    tensors = (
        prediction.policy_descriptors.gamma_edge,
        prediction.policy_descriptors.intensity_edge,
        prediction.policy_descriptors.flow_node,
        prediction.feasibility_logit,
        target.gamma_edge,
        target.log1p_intensity_edge,
        target.flow_node,
        target.persistent_edge,
        target.feasibility_edge,
    )
    if len({value.device for value in tensors}) != 1:
        raise ValueError("prediction and target tensors must share one device")

    persistent = target.persistent_edge
    gamma_loss = _masked_mean(
        (prediction.policy_descriptors.gamma_edge - target.gamma_edge).square(),
        persistent,
    )
    intensity_loss = _masked_mean(
        (
            torch.log1p(prediction.policy_descriptors.intensity_edge)
            - target.log1p_intensity_edge
        ).square(),
        persistent,
    )
    flow_loss = F.mse_loss(prediction.policy_descriptors.flow_node, target.flow_node)
    feasibility_pointwise = F.binary_cross_entropy_with_logits(
        prediction.feasibility_logit,
        target.feasibility_edge.to(prediction.feasibility_logit.dtype),
        reduction="none",
    )
    feasibility_loss = _masked_mean(feasibility_pointwise, persistent)
    total = (
        weights.gamma * gamma_loss
        + weights.intensity * intensity_loss
        + weights.flow * flow_loss
        + weights.feasibility * feasibility_loss
    )
    components = (total, gamma_loss, intensity_loss, flow_loss, feasibility_loss)
    if not all(bool(torch.isfinite(value)) for value in components):
        raise FloatingPointError("Intensity--Flow loss contains NaN or Inf")
    return IntensityFlowLoss(
        total=total,
        gamma=gamma_loss,
        intensity=intensity_loss,
        flow=flow_loss,
        feasibility=feasibility_loss,
        persistent_edge_count=int(persistent.sum().item()),
    )
