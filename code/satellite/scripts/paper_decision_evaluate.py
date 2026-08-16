"""Held-out next-step re-ranking diagnostics for condition DEC--K.

This is intentionally separate from action-coupled rollout evaluation.  The
model consumes the held-out teacher-forced observation at t, predicts the
controller-facing copy for t+1, and is compared with the simulator target only
on persistent candidate identities.  No optimizer or simulator transition is
invoked by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from leo_pg.paper.dataset import PaperEpisodeDataset
from leo_pg.paper.evaluation import load_frozen_model
from leo_pg.paper.metrics import _kendall_tau, _rank_scores
from leo_pg.paper.models import normalize_paper_method
from leo_pg.paper.training import observation_from_record, target_from_record
from leo_pg.sim.state import PolicyDescriptors
from leo_pg.utils.config import load_cfg


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _evaluate_pair(
    *,
    edge_ids: torch.Tensor,
    persistent: torch.Tensor,
    feasible: torch.Tensor,
    model: PolicyDescriptors,
    oracle: PolicyDescriptors,
    policy_config: Any,
    near_tie_margin: float,
) -> list[dict[str, float]]:
    rows = torch.nonzero(persistent, as_tuple=False).flatten()
    if rows.numel() == 0:
        return []
    ids = edge_ids.index_select(0, rows)
    model_view = PolicyDescriptors(
        gamma_edge=model.gamma_edge.index_select(0, rows),
        intensity_edge=model.intensity_edge.index_select(0, rows),
        flow_node=model.flow_node,
    )
    oracle_view = PolicyDescriptors(
        gamma_edge=oracle.gamma_edge.index_select(0, rows),
        intensity_edge=oracle.intensity_edge.index_select(0, rows),
        flow_node=oracle.flow_node,
    )
    feasible_view = feasible.index_select(0, rows)
    model_scores = _rank_scores(ids, model_view, policy_config, feasible_view)
    oracle_scores = _rank_scores(ids, oracle_view, policy_config, feasible_view)
    output: list[dict[str, float]] = []
    for user in torch.unique(ids[:, 0]).tolist():
        user_rows = torch.nonzero(ids[:, 0] == int(user), as_tuple=False).flatten()
        if policy_config.hard_feasibility_mask:
            user_rows = user_rows[feasible_view.index_select(0, user_rows)]
        if user_rows.numel() == 0:
            continue
        satellites = ids.index_select(0, user_rows)[:, 1]
        satellite_order = torch.argsort(satellites, stable=True)
        ordered = user_rows.index_select(0, satellite_order)
        model_order = ordered[
            torch.argsort(model_scores[ordered], descending=True, stable=True)
        ]
        oracle_order = ordered[
            torch.argsort(oracle_scores[ordered], descending=True, stable=True)
        ]
        model_best = int(model_order[0])
        oracle_best = int(oracle_order[0])
        oracle_gap = math.inf
        if oracle_order.numel() > 1:
            oracle_gap = float(
                (oracle_scores[oracle_order[0]] - oracle_scores[oracle_order[1]]).item()
            )
        output.append(
            {
                "top1": float(model_best == oracle_best),
                "kendall_tau": _kendall_tau(
                    model_scores.index_select(0, user_rows),
                    oracle_scores.index_select(0, user_rows),
                ),
                "regret": max(
                    0.0,
                    float(
                        (oracle_scores[oracle_best] - oracle_scores[model_best]).item()
                    ),
                ),
                "near_tie": float(oracle_gap <= near_tie_margin),
                "near_tie_flip": float(
                    oracle_gap <= near_tie_margin and model_best != oracle_best
                ),
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stress-index", type=int, required=True)
    parser.add_argument("--near-tie-margin", type=float, default=0.05)
    parser.add_argument("--max-episodes", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_episodes <= 0:
        raise ValueError("max-episodes must be positive")
    near_tie_margin = _finite("near_tie_margin", args.near_tie_margin)
    if near_tie_margin < 0:
        raise ValueError("near_tie_margin must be non-negative")

    cfg = load_cfg(args.cfg)
    method = normalize_paper_method(args.method)
    if method.startswith("snapshot_"):
        raise ValueError("DEC--K uses the fixed Intensity--Flow score contract")
    device = torch.device(args.device)
    model, checkpoint = load_frozen_model(
        cfg,
        args.ckpt,
        method=method,
        device=device,
        allow_config_mismatch=args.allow_config_mismatch,
    )
    dataset = PaperEpisodeDataset(args.data, split="test")
    policy_config = __import__(
        "leo_pg.sim.paper_environment", fromlist=["PaperAlignedLEOEnv"]
    ).PaperAlignedLEOEnv(cfg, device="cpu").fixed_policy_config()

    episode_rows: list[dict[str, Any]] = []
    totals = {
        "comparisons": 0,
        "top1": 0.0,
        "kendall_tau": 0.0,
        "regret": 0.0,
        "near_tie": 0,
        "near_tie_flip": 0.0,
    }
    model.eval()
    with torch.no_grad():
        for dataset_index in range(min(len(dataset), args.max_episodes)):
            episode = dataset[dataset_index]
            state = None
            local = {key: 0.0 for key in totals}
            for record in episode["steps"]:
                if record["next_target"] is None:
                    continue
                observation = observation_from_record(record, device)
                prediction, state = model.predict_step(
                    observation.as_model_step(), state, device
                )
                target = target_from_record(record, device)
                oracle = PolicyDescriptors(
                    gamma_edge=target.gamma_edge,
                    intensity_edge=torch.expm1(target.log1p_intensity_edge).clamp_min(0.0),
                    flow_node=target.flow_node,
                )
                pair_rows = _evaluate_pair(
                    edge_ids=observation.candidate_edge_ids,
                    persistent=target.persistent_edge,
                    feasible=target.feasibility_edge,
                    model=prediction.policy_descriptors,
                    oracle=oracle,
                    policy_config=policy_config,
                    near_tie_margin=near_tie_margin,
                )
                for row in pair_rows:
                    local["comparisons"] += 1
                    local["top1"] += row["top1"]
                    local["kendall_tau"] += row["kendall_tau"]
                    local["regret"] += row["regret"]
                    local["near_tie"] += int(row["near_tie"])
                    local["near_tie_flip"] += row["near_tie_flip"]
            comparisons = int(local["comparisons"])
            if comparisons == 0:
                continue
            near_count = int(local["near_tie"])
            episode_rows.append(
                {
                    "run_id": args.run_id,
                    "episode_id": int(episode["episode_id"]),
                    "episode_seed": int(episode["seed"]),
                    "stress_index": int(args.stress_index),
                    "method": method,
                    "comparisons": comparisons,
                    "near_tie_comparisons": near_count,
                    "top1_agreement": local["top1"] / comparisons,
                    "candidate_kendall_tau": local["kendall_tau"] / comparisons,
                    "oracle_score_regret": local["regret"] / comparisons,
                    "near_tie_flip_rate": (
                        local["near_tie_flip"] / near_count if near_count else None
                    ),
                }
            )
            for key in totals:
                totals[key] += local[key]

    comparisons = int(totals["comparisons"])
    if comparisons == 0:
        raise RuntimeError("no persistent held-out decision comparisons were found")
    near_count = int(totals["near_tie"])
    payload = {
        "schema_version": 1,
        "condition_id": "DEC-K",
        "run_id": args.run_id,
        "method": method,
        "stress_index": int(args.stress_index),
        "near_tie_margin": near_tie_margin,
        "checkpoint": checkpoint,
        "contract": {
            "temporal_scale": "held_out_teacher_forced_t_to_t_plus_1",
            "identity_mask": "persistent_candidate_edges_only",
            "score": "fixed_intensity_flow_native_scale",
        },
        "aggregate": {
            "comparisons": comparisons,
            "near_tie_comparisons": near_count,
            "top1_agreement": totals["top1"] / comparisons,
            "candidate_kendall_tau": totals["kendall_tau"] / comparisons,
            "oracle_score_regret": totals["regret"] / comparisons,
            "near_tie_flip_rate": (
                totals["near_tie_flip"] / near_count if near_count else None
            ),
        },
        "episodes": episode_rows,
    }
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] wrote {output} | comparisons={comparisons}")


if __name__ == "__main__":
    main()
