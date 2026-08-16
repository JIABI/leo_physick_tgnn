#!/usr/bin/env python3
"""Select Snapshot score weights on the frozen validation split.

This script evaluates the exact SI grid without retraining the predictor.  For
each validation episode, teacher-forced next-step predictions and aligned
oracle targets are ranked over persistent candidate identities.  The selected
triple minimizes the episode-first mean native oracle-score regret, with a
deterministic lexicographic tie break.  No test episode is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from leo_pg.paper.snapshot import (
    SNAPSHOT_SI_SCORE_GRID,
    SNAPSHOT_SI_SELECTED_SCORE_WEIGHTS,
    SnapshotPolicyConfig,
    SnapshotScoreWeights,
    score_snapshot_candidates,
)
from leo_pg.paper.snapshot_config import resolve_snapshot_pipeline
from leo_pg.paper.snapshot_data import (
    SnapshotEpisodeDataset,
    snapshot_control_from_record,
    snapshot_target_from_record,
)
from leo_pg.paper.snapshot_models import SNAPSHOT_METHODS, build_snapshot_model
from leo_pg.paper.snapshot_training import SnapshotTrainingAdapter
from leo_pg.train.checkpoint import load_ckpt
from leo_pg.utils.config import load_cfg
from leo_pg.utils.device import get_device
from leo_pg.utils.seed import set_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _records(episode: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = episode.get("steps")
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("Snapshot validation episode requires at least two steps")
    if not all(isinstance(item, Mapping) for item in records):
        raise TypeError("Snapshot validation steps must be mappings")
    return records


def _best_edge(
    indices: torch.Tensor,
    total: torch.Tensor,
    satellites: torch.Tensor,
) -> int:
    if indices.numel() == 0:
        raise ValueError("cannot rank an empty candidate set")
    satellite_order = torch.argsort(
        satellites.index_select(0, indices), stable=True
    )
    ordered = indices.index_select(0, satellite_order)
    score_order = torch.argsort(
        total.index_select(0, ordered), descending=True, stable=True
    )
    return int(ordered[score_order[0]].item())


def _transition_regrets(
    control: Any,
    prediction: Any,
    target: Any,
    weights: SnapshotScoreWeights,
) -> list[float]:
    policy = SnapshotPolicyConfig(
        weights=weights,
        hard_feasibility_mask=False,
        min_dwell_steps=0,
        hysteresis=None,
    )
    model_scores = score_snapshot_candidates(control, prediction, policy)
    oracle_scores = score_snapshot_candidates(control, target.as_output(), policy)
    persistent = target.persistent_edge.bool()
    users = control.candidate_edge_ids[:, 0]
    satellites = control.candidate_edge_ids[:, 1]
    values: list[float] = []
    for user in range(control.user_count):
        candidates = torch.nonzero(
            (users == user)
            & persistent
            & model_scores.eligible
            & oracle_scores.eligible,
            as_tuple=False,
        ).flatten()
        if candidates.numel() == 0:
            continue
        model_edge = _best_edge(candidates, model_scores.total, satellites)
        oracle_edge = _best_edge(candidates, oracle_scores.total, satellites)
        regret = float(
            (
                oracle_scores.total[oracle_edge]
                - oracle_scores.total[model_edge]
            )
            .clamp_min(0.0)
            .item()
        )
        values.append(regret)
    return values


def _grid() -> list[SnapshotScoreWeights]:
    return [
        SnapshotScoreWeights(gamma=gamma, feasibility=feasibility, load=load)
        for gamma in SNAPSHOT_SI_SCORE_GRID["gamma"]
        for feasibility in SNAPSHOT_SI_SCORE_GRID["feasibility"]
        for load in SNAPSHOT_SI_SCORE_GRID["load"]
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--method", required=True, choices=SNAPSHOT_METHODS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args(argv)

    root = load_cfg(args.cfg)
    pipeline = resolve_snapshot_pipeline(root, args.method)
    set_seed(int(root.get("seed", 0)))
    requested_device = args.device or str(
        root.get("paper_training", {}).get("device", "cuda")
    )
    device = get_device(requested_device, strict=args.device is not None)
    model = build_snapshot_model(pipeline.resolved_model_config, pipeline.method)
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    checkpoint = load_ckpt(
        str(checkpoint_path), model, map_location=device, strict=True
    )
    if checkpoint.get("paper_method") != pipeline.method:
        raise ValueError("checkpoint paper_method does not match --method")
    if checkpoint.get("paper_interface") != "snapshot_v1":
        raise ValueError("checkpoint is not a Snapshot checkpoint")
    if checkpoint.get("snapshot_pipeline_fingerprint") != pipeline.fingerprint:
        raise ValueError("checkpoint Snapshot pipeline differs from --cfg")
    model.to(device).eval()
    adapter = SnapshotTrainingAdapter(
        model=model,
        loss_weights=pipeline.loss_weights,
        decision_aware_weight=pipeline.effective_decision_aware_weight,
        device=device,
    )
    dataset_path = Path(args.data).expanduser().resolve()
    validation = SnapshotEpisodeDataset(dataset_path, split="val")
    if validation.payload.get("snapshot_method") != pipeline.method:
        raise ValueError("Snapshot validation dataset method does not match --method")
    if validation.payload.get("snapshot_pipeline_fingerprint") != pipeline.fingerprint:
        raise ValueError("Snapshot validation dataset pipeline differs from --cfg")

    candidates = _grid()
    episode_values: list[list[float]] = [[] for _ in candidates]
    episode_seeds: list[int] = []
    with torch.no_grad():
        for episode_index in range(len(validation)):
            episode = validation[episode_index]
            records = _records(episode)
            state: Any = None
            per_grid: list[list[float]] = [[] for _ in candidates]
            for record in records[:-1]:
                control = snapshot_control_from_record(record, device)
                target = snapshot_target_from_record(record, device)
                prediction, state = adapter.predict(control, state)
                for grid_index, weights in enumerate(candidates):
                    per_grid[grid_index].extend(
                        _transition_regrets(control, prediction, target, weights)
                    )
            episode_seeds.append(int(records[0]["snapshot_observation"]["observation_id"][0]))
            for grid_index, values in enumerate(per_grid):
                if not values:
                    raise ValueError(
                        f"validation episode {episode_index} has no rankable persistent candidates"
                    )
                episode_values[grid_index].append(sum(values) / len(values))

    rows: list[dict[str, Any]] = []
    for weights, values in zip(candidates, episode_values):
        mean = sum(values) / len(values)
        rows.append(
            {
                "weights": {
                    "gamma": weights.gamma,
                    "feasibility": weights.feasibility,
                    "load": weights.load,
                },
                "validation_episode_mean_regret": values,
                "validation_regret": mean,
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["validation_regret"],
            row["weights"]["gamma"],
            row["weights"]["feasibility"],
            row["weights"]["load"],
        ),
    )
    reported = SNAPSHOT_SI_SELECTED_SCORE_WEIGHTS.get(pipeline.method)
    payload = {
        "schema_version": 1,
        "artifact_kind": "snapshot_validation_score_weight_selection",
        "paper_version": "v8",
        "method": pipeline.method,
        "selection_contract": (
            "episode_first_mean_native_oracle_score_regret_on_validation_split"
        ),
        "tie_break": "gamma_then_feasibility_then_load_ascending",
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
        "validation_dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "episode_count": len(validation),
            "episode_seeds": episode_seeds,
        },
        "grid": SNAPSHOT_SI_SCORE_GRID,
        "rows": rows,
        "selected": selected,
        "reported_selected": (
            None
            if reported is None
            else {
                "gamma": reported.gamma,
                "feasibility": reported.feasibility,
                "load": reported.load,
            }
        ),
        "selected_matches_reported": (
            False if reported is None else selected["weights"] == {
                "gamma": reported.gamma,
                "feasibility": reported.feasibility,
                "load": reported.load,
            }
        ),
    }
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "selected": selected["weights"]}, indent=2))


if __name__ == "__main__":
    main()
