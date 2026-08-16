from __future__ import annotations

import math
from typing import Any, Dict, List

import torch

from ..sim.protocol import validate_allocator_protocol
from .rollout import rollout_episode
from .system_metrics import (
    adjacent_aba_fraction,
    assignment_failure_fraction,
    greedy_assign_from_load,
    load_stats,
)


EVALUATION_MODE = "teacher_forced_frozen_trajectory_diagnostic"

METRIC_CONTRACT = {
    "adjacent_aba": "valid consecutive A-B-A triplets / valid consecutive triplets",
    "assignment_failure": "failed user-time assignments / all user-time assignments",
    "load_var": "mean across time of population variance across satellite utilization",
    "load_peak": "mean across time of maximum satellite utilization",
    "temporal_alignment": "prediction at t-1 is the pre-decision load used for the decision at t",
}


def summarize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for horizon in sorted({int(result["H"]) for result in results}):
        group = [result for result in results if int(result["H"]) == horizon]
        episode_ids = {int(result["episode"]) for result in group}
        summary: Dict[str, Any] = {
            "H": horizon,
            "episodes": len(episode_ids),
            "effective_H_min": min(int(result["effective_H"]) for result in group),
            "effective_H_max": max(int(result["effective_H"]) for result in group),
            "diagnostic_steps_min": min(int(result["diagnostic_steps"]) for result in group),
            "diagnostic_steps_max": max(int(result["diagnostic_steps"]) for result in group),
        }
        for key in group[0]:
            if key in {"H", "effective_H", "diagnostic_steps", "episode"}:
                continue
            values = [float(result[key]) for result in group]
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(f"non-finite diagnostic values for {key}")
            summary[f"{key}_across_episodes"] = float(sum(values) / len(values))
        summaries.append(summary)
    return summaries


@torch.no_grad()
def frozen_diagnostics_for_episode(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    episode: Dict[str, Any],
    device: torch.device,
    horizon: int,
) -> Dict[str, Any]:
    """Teacher-forced predictions plus post-hoc diagnostics on a frozen graph trace."""
    requested_horizon = int(horizon)
    if requested_horizon <= 0:
        raise ValueError("diagnostic horizon must be positive")
    steps: List[Dict[str, Any]] = episode.get("steps", [])
    if not steps:
        raise ValueError("episode contains no steps")
    effective_horizon = min(requested_horizon, len(steps))
    if effective_horizon < 2:
        raise ValueError("frozen diagnostics require at least two episode steps")

    rollout = rollout_episode(
        model,
        episode,
        device=device,
        H=effective_horizon,
        return_preds_cpu=True,
    )
    preds: List[torch.Tensor] = rollout["preds_cpu"]

    meta0 = steps[0].get("meta", {})
    if "K_users" not in meta0 or "S_sats" not in meta0:
        raise ValueError("step metadata must define K_users and S_sats")
    K = int(meta0["K_users"])
    S = int(meta0["S_sats"])

    decision_steps = steps[1:effective_horizon]
    serving_gt = []
    assignment_failures_gt = []
    load_gt = []
    required_meta = {
        "K_users",
        "S_sats",
        "sat_capacity_users",
        "sat_load_pre",
        "allocation_user_order",
        "allocator_protocol",
        "serving_sat",
        "ho_fail",
    }
    for step in decision_steps:
        meta = step.get("meta", {})
        missing = required_meta.difference(meta)
        if missing:
            raise ValueError(f"decision step metadata is missing: {sorted(missing)}")
        if int(meta["K_users"]) != K or int(meta["S_sats"]) != S:
            raise ValueError("K_users/S_sats change within an episode")
        serving = torch.as_tensor(meta["serving_sat"]).long()
        failures = torch.as_tensor(meta["ho_fail"]).bool()
        if serving.shape != (K,) or failures.shape != (K,):
            raise ValueError("serving_sat and ho_fail must each have shape [K_users]")
        if serving.numel() and (int(serving.min()) < -1 or int(serving.max()) >= S):
            raise ValueError("serving_sat contains an invalid local satellite id")
        serving_gt.append(serving)
        assignment_failures_gt.append(failures)
        target = torch.as_tensor(step["y"]).float()
        if target.ndim != 2 or target.size(1) < 1 or target.size(0) != K + S:
            raise ValueError(f"K_users={K} is incompatible with target shape {tuple(target.shape)}")
        load_gt.append(target[K:, 0])

    aba_gt = adjacent_aba_fraction(torch.stack(serving_gt))
    assignment_failure_gt = assignment_failure_fraction(torch.stack(assignment_failures_gt))
    var_gt, peak_gt = load_stats(torch.stack(load_gt))

    serving_pred = []
    assignment_failures_pred = []
    load_pred = []
    for decision_index, (step, pred_cpu) in enumerate(zip(decision_steps, preds[:-1]), start=1):
        meta = step.get("meta", {})
        protocol = validate_allocator_protocol(meta["allocator_protocol"])
        channel_cfg = {
            "base_sinr": protocol["base_sinr"],
            "dist_scale": protocol["dist_scale"],
        }
        sinr_min = protocol["sinr_min"]
        momentum = protocol["load_momentum"]
        recorded_pre_load = torch.as_tensor(meta["sat_load_pre"]).float()
        prior_target = torch.as_tensor(steps[decision_index - 1]["y"])[K:, 0].float()
        if not torch.allclose(recorded_pre_load.cpu(), prior_target.cpu(), atol=1e-6, rtol=1e-5):
            raise ValueError("dataset temporal contract is broken: prior target != current pre-decision load")
        node_x = torch.as_tensor(step["node_x"]).to(device)
        edge_index = torch.as_tensor(step["edge_index"]).long().to(device)
        edge_type = torch.as_tensor(step["edge_type"]).long().to(device)
        sat_load_pred = pred_cpu[K:, 0].to(device).clamp(0.0, 1.0)
        if sat_load_pred.numel() != recorded_pre_load.numel():
            raise ValueError("predicted satellite-load dimension does not match allocator metadata")
        user_order = torch.as_tensor(meta["allocation_user_order"]).long().to(device)
        serving, _, assignment_failure = greedy_assign_from_load(
            node_x=node_x,
            edge_index=edge_index,
            edge_type=edge_type,
            sat_load=sat_load_pred,
            K_users=K,
            sat_capacity_users=int(meta["sat_capacity_users"]),
            channel_cfg=channel_cfg,
            sinr_min=sinr_min,
            user_order=user_order,
        )
        serving_pred.append(serving.cpu())
        assignment_failures_pred.append(assignment_failure.cpu())
        valid_serving = serving[serving >= 0]
        incoming_users = torch.bincount(
            valid_serving,
            minlength=sat_load_pred.numel(),
        ).to(dtype=sat_load_pred.dtype)
        incoming_util = incoming_users / float(max(1, int(meta["sat_capacity_users"])))
        post_decision_load = momentum * sat_load_pred + (1.0 - momentum) * incoming_util
        if not torch.isfinite(post_decision_load).all():
            raise FloatingPointError("post-decision load contains NaN or Inf")
        load_pred.append(post_decision_load.cpu())

    aba_pred = adjacent_aba_fraction(torch.stack(serving_pred))
    assignment_failure_pred = assignment_failure_fraction(torch.stack(assignment_failures_pred))
    var_pred, peak_pred = load_stats(torch.stack(load_pred))

    return {
        "H": requested_horizon,
        "effective_H": effective_horizon,
        "diagnostic_steps": len(decision_steps),
        "mse_mean": float(rollout["mse_mean"]),
        "mse_last": float(rollout["mse_last"]),
        "adjacent_aba_gt": aba_gt,
        "assignment_failure_gt": assignment_failure_gt,
        "load_var_gt": var_gt,
        "load_peak_gt": peak_gt,
        "adjacent_aba_pred": aba_pred,
        "assignment_failure_pred": assignment_failure_pred,
        "load_var_pred": var_pred,
        "load_peak_pred": peak_pred,
    }
