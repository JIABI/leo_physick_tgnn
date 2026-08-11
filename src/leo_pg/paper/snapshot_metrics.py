"""Metric adapter for Snapshot traces under the shared paper contract.

The common outcome/paired-bootstrap implementation operates on the canonical
closed-loop record type.  Snapshot's feasibility field has the opposite rank
direction from Intensity--Flow risk, so this adapter maps it to the nonnegative
rank-equivalent quantity ``max(margin) - margin``.  The mapping is used only by
the metric engine; it does not alter controller actions or simulator state.
"""

from __future__ import annotations

from typing import Any, Mapping

import math
import torch

from leo_pg.control.policy import FixedRankPolicyConfig
from leo_pg.eval.closed_loop import ClosedLoopResult, ClosedLoopStep
from leo_pg.eval.substitution import SubstitutionMode
from leo_pg.paper.metrics import evaluate_closed_loop_result
from leo_pg.sim.state import PolicyDescriptors, SimulatorDescriptors

from .snapshot import SnapshotOutput, SnapshotPolicyConfig
from .snapshot_evaluation import SnapshotClosedLoopResult


def _rank_equivalent_descriptors(output: SnapshotOutput) -> PolicyDescriptors:
    output.validate(
        int(output.gamma_edge.numel()),
        int(output.admitted_load_node.numel()),
    )
    if output.feasibility_margin_edge.numel():
        ceiling = output.feasibility_margin_edge.max()
        feasibility_risk = ceiling - output.feasibility_margin_edge
    else:
        feasibility_risk = output.feasibility_margin_edge.clone()
    descriptors = PolicyDescriptors(
        gamma_edge=output.gamma_edge.clone(),
        intensity_edge=feasibility_risk,
        flow_node=output.admitted_load_node.clone(),
    )
    descriptors.validate(
        int(output.gamma_edge.numel()),
        int(output.admitted_load_node.numel()),
    )
    return descriptors


def _policy_config(value: SnapshotPolicyConfig) -> FixedRankPolicyConfig:
    return FixedRankPolicyConfig(
        gamma_weight=value.weights.gamma,
        load_weight=value.weights.load,
        intensity_weight=value.weights.feasibility,
        hard_feasibility_mask=value.hard_feasibility_mask,
        min_dwell_steps=value.min_dwell_steps,
        hysteresis=value.hysteresis,
    )


def snapshot_as_canonical_closed_loop(
    result: SnapshotClosedLoopResult,
) -> ClosedLoopResult:
    """Create a lossless outcome and rank-equivalent decision-metric view."""

    is_oracle = result.mode.value == "oracle"
    mode = SubstitutionMode.ORACLE if is_oracle else SubstitutionMode.MODEL
    records: list[ClosedLoopStep] = []
    for record in result.records:
        simulator_fields = _rank_equivalent_descriptors(record.simulator_snapshot)
        controller_fields = _rank_equivalent_descriptors(record.controller_snapshot)
        next_fields = (
            None
            if record.next_model_prediction is None
            else _rank_equivalent_descriptors(record.next_model_prediction)
        )
        records.append(
            ClosedLoopStep(
                observation_id=record.observation_id,
                mode=mode,
                candidate_edge_ids=record.candidate_edge_ids.clone(),
                sim_descriptors=SimulatorDescriptors(
                    policy_fields=simulator_fields,
                    feasible_edge=record.feasible_edge.clone(),
                ),
                model_input_descriptors=controller_fields.clone(),
                model_descriptors=None if is_oracle else controller_fields.clone(),
                next_model_prediction=next_fields,
                policy_descriptors=controller_fields.clone(),
                initialized_edge=record.initialized_edge.clone(),
                feasible_edge=record.feasible_edge.clone(),
                action=record.action,
                execution=record.execution,
            )
        )
    return ClosedLoopResult(
        mode=mode,
        episode_seed=result.episode_seed,
        protocol_version=result.protocol_version,
        protocol_fingerprint=result.protocol_fingerprint,
        provider_fingerprint=result.provider_fingerprint,
        initializer_kind=("oracle" if is_oracle else "constant_snapshot_v1"),
        initializer_fingerprint=result.initializer_fingerprint,
        policy_config=_policy_config(result.policy_config),
        hard_feasibility_mask=result.policy_config.hard_feasibility_mask,
        initial_policy_strategy=("oracle" if is_oracle else "explicit_initializer"),
        new_edge_strategy=("oracle" if is_oracle else "explicit_initializer"),
        records=tuple(records),
        final_serving=result.final_serving.clone(),
        final_flow=result.final_flow.clone(),
    )


def evaluate_snapshot_pair(
    model_result: SnapshotClosedLoopResult,
    oracle_result: SnapshotClosedLoopResult,
    *,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return standard model/oracle rows for one matched Snapshot unit."""

    model = snapshot_as_canonical_closed_loop(model_result)
    oracle = snapshot_as_canonical_closed_loop(oracle_result)
    if model.episode_seed != oracle.episode_seed:
        raise ValueError("Snapshot metric pair must share an episode seed")
    if model.action_count != oracle.action_count:
        raise ValueError("Snapshot metric pair must share a horizon")
    context = {"matched_oracle_result": oracle}
    return {
        "paired": {
            "model": evaluate_closed_loop_result(
                model,
                context=context,
                options=options,
            ),
            "oracle": evaluate_closed_loop_result(
                oracle,
                context=context,
                options=options,
            ),
        },
        "classical": {},
        "snapshot_metric_adapter": {
            "version": 1,
            "feasibility_mapping": "max_margin_minus_margin_for_rank_metrics_only",
            "outcome_contract": "shared_metric_contract_v2",
        },
    }


def snapshot_shrink_jump_ratios(
    result: SnapshotClosedLoopResult,
    *,
    gamma_scale: float,
    feasibility_scale: float,
    load_scale: float,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Return per-user asynchronous shrink/jump ratios for one model trace."""

    scales = (float(gamma_scale), float(feasibility_scale), float(load_scale))
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("Snapshot shrink-jump scales must be finite and positive")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Snapshot shrink-jump epsilon must be positive")
    if result.mode.value != "model":
        raise ValueError("Snapshot shrink-jump ratios require a model trace")
    if len(result.records) < 2:
        raise ValueError("Snapshot shrink-jump ratios require at least two epochs")
    user_count = int(result.records[0].execution.requested_serving.numel())
    norms: list[torch.Tensor] = []
    for record in result.records:
        edge_users = record.candidate_edge_ids[:, 0]
        edge_squared = (
            (record.controller_snapshot.gamma_edge - record.simulator_snapshot.gamma_edge)
            .div(scales[0])
            .square()
            + (
                record.controller_snapshot.feasibility_margin_edge
                - record.simulator_snapshot.feasibility_margin_edge
            )
            .div(scales[1])
            .square()
        )
        per_user = edge_squared.new_zeros(user_count)
        per_user.index_add_(0, edge_users, edge_squared)
        shared_load = (
            (record.controller_snapshot.admitted_load_node
             - record.simulator_snapshot.admitted_load_node)
            .div(scales[2])
            .square()
            .sum()
        )
        norms.append((per_user + shared_load).clamp_min(0.0).sqrt())
    error = torch.stack(norms, dim=0)
    ratios = error[1:] / error[:-1].clamp_min(epsilon)
    switches = torch.stack(
        [
            record.execution.handover_executed
            for record in result.records[:-1]
        ],
        dim=0,
    )
    if switches.shape != ratios.shape:
        raise ValueError("Snapshot shrink-jump switch/error shapes disagree")
    return {
        "contract": "per_user_asynchronous_descriptor_error_ratio_v1",
        "no_switch_ratios": ratios[~switches].detach().cpu().to(torch.float32),
        "switch_ratios": ratios[switches].detach().cpu().to(torch.float32),
        "user_count": user_count,
        "epoch_pairs": int(ratios.size(0)),
        "epsilon": epsilon,
        "scales": {
            "gamma": scales[0],
            "feasibility_margin": scales[1],
            "admitted_load": scales[2],
        },
    }


__all__ = [
    "evaluate_snapshot_pair",
    "snapshot_shrink_jump_ratios",
    "snapshot_as_canonical_closed_loop",
]
