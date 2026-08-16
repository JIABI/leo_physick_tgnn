#!/usr/bin/env python3
"""Run paired action-coupled Snapshot model/oracle closed-loop evaluation."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from leo_pg.paper.snapshot import SnapshotFixedRankController, SnapshotOutput
from leo_pg.paper.snapshot_config import resolve_snapshot_pipeline
from leo_pg.paper.snapshot_data import SnapshotEpisodeDataset
from leo_pg.paper.snapshot_evaluation import (
    SnapshotClosedLoopResult,
    SnapshotModelProvider,
    run_snapshot_closed_loop,
)
from leo_pg.paper.snapshot_models import SNAPSHOT_METHODS, build_snapshot_model
from leo_pg.paper.snapshot_metrics import (
    evaluate_snapshot_pair,
    snapshot_shrink_jump_ratios,
)
from leo_pg.paper.calibration import nearest_rank_quantile
from leo_pg.paper.snapshot_training import SnapshotTrainingAdapter
from leo_pg.paper.metrics import (
    aggregate_evaluation_units,
    paired_percentile_bootstrap,
)
from leo_pg.sim.paper_environment import PaperAlignedLEOEnv
from leo_pg.train.checkpoint import load_ckpt
from leo_pg.utils.config import load_cfg
from leo_pg.utils.device import get_device
from leo_pg.utils.seed import set_seed


SNAPSHOT_EVALUATION_BUNDLE_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _csv_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            "seeds must be unique non-negative comma-separated integers"
        )
    return seeds


def _positive(name: str, value: Any) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    result = value.detach().cpu().contiguous().clone()
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _snapshot_output(value: SnapshotOutput) -> dict[str, torch.Tensor]:
    return {
        "gamma_edge": _tensor(value.gamma_edge, "snapshot.gamma_edge"),
        "feasibility_margin_edge": _tensor(
            value.feasibility_margin_edge,
            "snapshot.feasibility_margin_edge",
        ),
        "admitted_load_node": _tensor(
            value.admitted_load_node, "snapshot.admitted_load_node"
        ),
    }


def _serialize_result(result: SnapshotClosedLoopResult) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, record in enumerate(result.records):
        records.append(
            {
                "step_index": index,
                "observation_id": list(record.observation_id),
                "candidate_edge_ids": _tensor(
                    record.candidate_edge_ids, "record.candidate_edge_ids"
                ),
                "feasible_edge": _tensor(
                    record.feasible_edge, "record.feasible_edge"
                ),
                "simulator_snapshot": _snapshot_output(record.simulator_snapshot),
                "controller_snapshot": _snapshot_output(record.controller_snapshot),
                "initialized_edge": _tensor(
                    record.initialized_edge, "record.initialized_edge"
                ),
                "next_model_prediction": (
                    None
                    if record.next_model_prediction is None
                    else _snapshot_output(record.next_model_prediction)
                ),
                "action": {
                    "observation_id": list(record.action.observation_id),
                    "requested_serving": _tensor(
                        record.action.requested_serving,
                        "record.action.requested_serving",
                    ),
                },
                "execution": {
                    "observation_id": list(record.execution.observation_id),
                    "requested_serving": _tensor(
                        record.execution.requested_serving,
                        "record.execution.requested_serving",
                    ),
                    "executed_serving": _tensor(
                        record.execution.executed_serving,
                        "record.execution.executed_serving",
                    ),
                    "admitted": _tensor(
                        record.execution.admitted, "record.execution.admitted"
                    ),
                    "failure_reason": _tensor(
                        record.execution.failure_reason,
                        "record.execution.failure_reason",
                    ),
                    "handover_attempted": _tensor(
                        record.execution.handover_attempted,
                        "record.execution.handover_attempted",
                    ),
                    "handover_executed": _tensor(
                        record.execution.handover_executed,
                        "record.execution.handover_executed",
                    ),
                    "flow_before": _tensor(
                        record.execution.flow_before,
                        "record.execution.flow_before",
                    ),
                    "flow_after": _tensor(
                        record.execution.flow_after,
                        "record.execution.flow_after",
                    ),
                },
            }
        )
    return {
        "mode": result.mode.value,
        "episode_seed": result.episode_seed,
        "protocol_version": result.protocol_version,
        "protocol_fingerprint": result.protocol_fingerprint,
        "provider_fingerprint": result.provider_fingerprint,
        "initializer_fingerprint": result.initializer_fingerprint,
        "policy_config": asdict(result.policy_config),
        "action_count": result.action_count,
        "records": records,
        "final_serving": _tensor(result.final_serving, "result.final_serving"),
        "final_flow": _tensor(result.final_flow, "result.final_flow"),
    }


def _restricted_tree(value: Any, *, path: str = "root") -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _restricted_tree(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_restricted_tree(item, path=f"{path}[]") for item in value]
    if isinstance(value, torch.Tensor):
        return _tensor(value, path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    raise TypeError(f"{path} has unsupported type {type(value).__name__}")


def _output_paths(value: str | Path, method: str) -> tuple[Path, Path]:
    requested = Path(value).expanduser()
    if requested.suffix.lower() == ".pt":
        return requested, requested.with_suffix(".manifest.json")
    return (
        requested / f"{method}_paired_snapshot.pt",
        requested / f"{method}_paired_snapshot.manifest.json",
    )


def _atomic_pt(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(_restricted_tree(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(
        _restricted_tree(value),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    try:
        temporary.write_text(encoded + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _episode_seed(base_seed: int, episode_index: int) -> int:
    return int(base_seed) * 1_000_000 + int(episode_index)


def _evaluation_defaults(root: Mapping[str, Any]) -> dict[str, Any]:
    value = root.get("paper_evaluation", {})
    if not isinstance(value, Mapping):
        raise TypeError("paper_evaluation must be a mapping when provided")
    return dict(value)


def _metric_options(
    root: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    near_tie_margin: float,
) -> dict[str, Any]:
    protocol = root.get("paper_protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError("paper_protocol must be a mapping")
    return {
        "outcome": {
            "dt_ctrl": float(protocol["dt_ctrl"]),
            "outage_gamma_db": float(evaluation.get("outage_gamma_db", -3.0)),
            "pingpong_window_seconds": float(
                evaluation.get("pingpong_window_seconds", 2.0)
            ),
            "bandwidth_hz": float(evaluation.get("bandwidth_hz", 1.0)),
            "throughput_load_discount": bool(
                evaluation.get("throughput_load_discount", True)
            ),
            "user_tail_probability": float(
                evaluation.get("user_tail_probability", 0.10)
            ),
        },
        "decision": {
            "near_tie_margin": float(near_tie_margin),
            "descriptor_source": "policy",
        },
    }


def _heldout_one_step_metrics(
    *,
    model: torch.nn.Module,
    dataset: SnapshotEpisodeDataset,
    pipeline: Any,
    device: torch.device,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Evaluate descriptor loss on independent recorded Snapshot episodes."""

    adapter = SnapshotTrainingAdapter(
        model=model,
        loss_weights=pipeline.loss_weights,
        decision_aware_weight=0.0,
        device=device,
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            episode = dataset[index]
            result = adapter.episode_objective(
                episode,
                model_rollin_probability=0.0,
                initializer=None,
            )
            summary = result.summary
            descriptor_total = (
                pipeline.loss_weights.gamma * summary.gamma
                + pipeline.loss_weights.feasibility_margin
                * summary.feasibility_margin
                + pipeline.loss_weights.admitted_load * summary.admitted_load
            )
            rows.append(
                {
                    "episode_index": index,
                    "episode_id": int(episode["episode_id"]),
                    "episode_seed": int(episode["seed"]),
                    "descriptor_loss": float(descriptor_total),
                    "gamma_mse": float(summary.gamma),
                    "feasibility_margin_mse": float(
                        summary.feasibility_margin
                    ),
                    "admitted_load_mse": float(summary.admitted_load),
                    "transitions": int(summary.transitions),
                }
            )
    interval = paired_percentile_bootstrap(
        [row["descriptor_loss"] for row in rows],
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "contract": "episode_unit_mean_of_weighted_descriptor_mse_v1",
        "decision_aware_term_excluded": True,
        "model_rollin_probability": 0.0,
        "loss_weights": asdict(pipeline.loss_weights),
        "units": rows,
        "percentile_interval": interval.as_dict(),
    }


def _aggregate_shrink_jump(
    no_switch: Sequence[torch.Tensor],
    switched: Sequence[torch.Tensor],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    no_switch_values = torch.cat(tuple(no_switch)) if no_switch else torch.empty(0)
    switch_values = torch.cat(tuple(switched)) if switched else torch.empty(0)
    if not no_switch_values.numel() or not switch_values.numel():
        return {
            "status": "insufficient_samples",
            "no_switch_count": int(no_switch_values.numel()),
            "switch_count": int(switch_values.numel()),
        }
    raw_quantiles = config.get("quantiles", (0.95,))
    raw_dwell = config.get("dwell_steps", (10,))
    if not isinstance(raw_quantiles, (list, tuple)) or not raw_quantiles:
        raise ValueError("paper_evaluation.shrink_jump.quantiles must be non-empty")
    if not isinstance(raw_dwell, (list, tuple)) or not raw_dwell:
        raise ValueError("paper_evaluation.shrink_jump.dwell_steps must be non-empty")
    summaries = []
    expansive = float(
        (no_switch_values > 1.0).to(torch.float64).mean().item()
    )
    for probability_value in raw_quantiles:
        probability = float(probability_value)
        if not 0.0 < probability <= 1.0:
            raise ValueError("shrink-jump quantiles must lie in (0,1]")
        alpha = float(nearest_rank_quantile(no_switch_values, probability).item())
        beta = float(nearest_rank_quantile(switch_values, probability).item())
        for dwell_value in raw_dwell:
            dwell = int(dwell_value)
            if dwell < 0 or dwell != dwell_value:
                raise ValueError("shrink-jump dwell values must be non-negative integers")
            summaries.append(
                {
                    "alpha_quantile": alpha,
                    "beta_quantile": beta,
                    "alpha_probability": probability,
                    "beta_probability": probability,
                    "dwell_steps": dwell,
                    "stability_index": beta * (alpha ** dwell),
                    "no_switch_count": int(no_switch_values.numel()),
                    "switch_count": int(switch_values.numel()),
                    "expansive_fraction": expansive,
                    "interpretation": "empirical_diagnostic_not_a_guarantee",
                }
            )
    return {
        "status": "ok",
        "contract": "pooled_per_user_asynchronous_ratios_v1",
        "raw_ratio_location": "paired episode shards",
        "summaries": summaries,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run paired action-coupled Snapshot model/oracle evaluation"
    )
    parser.add_argument("--cfg", required=True, help="Paper YAML configuration")
    parser.add_argument("--method", required=True, choices=SNAPSHOT_METHODS)
    parser.add_argument("--ckpt", required=True, help="Trained Snapshot checkpoint")
    parser.add_argument(
        "--data",
        required=True,
        help="Method-specific formal Snapshot dataset index used for test loss",
    )
    parser.add_argument("--out", required=True, help="Result .pt path or output directory")
    parser.add_argument("--seeds", type=_csv_seeds, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    args = parser.parse_args(argv)

    root = load_cfg(args.cfg)
    pipeline = resolve_snapshot_pipeline(root, args.method)
    defaults = _evaluation_defaults(root)
    configured_seeds = root.get("run_seeds", (root.get("seed", 0),))
    if args.seeds is not None:
        seeds = args.seeds
    else:
        if not isinstance(configured_seeds, (list, tuple)):
            raise TypeError("run_seeds must be a list")
        seeds = tuple(int(seed) for seed in configured_seeds)
        if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("run_seeds must contain unique non-negative integers")
    episodes = _positive(
        "episodes",
        args.episodes
        if args.episodes is not None
        else defaults.get("episodes_per_seed", 1),
    )
    if episodes >= 1_000_000:
        raise ValueError(
            "episodes must be below 1,000,000 to keep the paired seed mapping injective"
        )
    configured_horizon = defaults.get(
        "horizon_steps",
        root.get("paper_protocol", {}).get("horizon_steps"),
    )
    horizon = _positive(
        "horizon",
        args.horizon if args.horizon is not None else configured_horizon,
    )

    set_seed(int(root.get("seed", 0)))
    requested_device = args.device or str(
        root.get("paper_training", {}).get("device", "cuda")
    )
    device = get_device(requested_device, strict=args.device is not None)
    model = build_snapshot_model(pipeline.resolved_model_config, pipeline.method)
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Snapshot checkpoint does not exist: {checkpoint_path}")
    checkpoint_sha = _sha256(checkpoint_path)
    checkpoint = load_ckpt(
        str(checkpoint_path),
        model,
        map_location=device,
        strict=True,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
    if checkpoint.get("paper_method") != pipeline.method:
        raise ValueError("checkpoint paper_method does not match --method")
    if checkpoint.get("paper_interface") != "snapshot_v1":
        raise ValueError("checkpoint is not a Snapshot checkpoint")
    saved_pipeline = checkpoint.get("snapshot_pipeline_fingerprint")
    if saved_pipeline is None and not args.allow_legacy_checkpoint:
        raise ValueError("checkpoint has no Snapshot pipeline fingerprint")
    if saved_pipeline is not None and saved_pipeline != pipeline.fingerprint:
        raise ValueError("checkpoint Snapshot pipeline differs from --cfg")

    provider = SnapshotModelProvider(
        model,
        device=device,
        fingerprint=checkpoint_sha,
    )
    controller = SnapshotFixedRankController(pipeline.policy_config)
    metric_options = _metric_options(
        root,
        defaults,
        near_tie_margin=(
            pipeline.policy_config.hysteresis
            if pipeline.policy_config.hysteresis is not None
            else 1.0 / 6.0
        ),
    )
    dataset_path = Path(args.data).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Snapshot evaluation dataset does not exist: {dataset_path}"
        )
    test_dataset = SnapshotEpisodeDataset(dataset_path, split="test")
    if test_dataset.payload.get("snapshot_method") != pipeline.method:
        raise ValueError("Snapshot test dataset method does not match --method")
    if (
        test_dataset.payload.get("snapshot_pipeline_fingerprint")
        != pipeline.fingerprint
    ):
        raise ValueError("Snapshot test dataset pipeline differs from --cfg")
    one_step_metrics = _heldout_one_step_metrics(
        model=model,
        dataset=test_dataset,
        pipeline=pipeline,
        device=device,
        resamples=int(defaults.get("bootstrap_resamples", 10_000)),
        confidence=float(defaults.get("bootstrap_confidence", 0.95)),
        seed=int(defaults.get("bootstrap_seed", 17)) + 1,
    )
    pt_path, manifest_path = _output_paths(args.out, pipeline.method)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = pt_path.parent / f"{pt_path.stem}_episodes"
    shard_dir.mkdir(parents=True, exist_ok=True)
    pair_entries: list[dict[str, Any]] = []
    pair_manifest: list[dict[str, Any]] = []
    metric_units: list[dict[str, Any]] = []
    no_switch_ratio_units: list[torch.Tensor] = []
    switch_ratio_units: list[torch.Tensor] = []
    shrink_config = defaults.get("shrink_jump", {})
    if not isinstance(shrink_config, Mapping):
        raise TypeError("paper_evaluation.shrink_jump must be a mapping")
    for base_seed in seeds:
        for episode_index in range(episodes):
            episode_seed = _episode_seed(base_seed, episode_index)
            episode_cfg = copy.deepcopy(pipeline.resolved_model_config)
            episode_cfg["seed"] = episode_seed
            protocol = episode_cfg.get("paper_protocol")
            if not isinstance(protocol, dict):
                raise TypeError("paper_protocol must be a mapping")
            protocol["horizon_steps"] = horizon
            model_result = run_snapshot_closed_loop(
                PaperAlignedLEOEnv(episode_cfg, device=device),
                mode="model",
                controller=controller,
                margin_config=pipeline.margin_config,
                feature_config=pipeline.feature_config,
                initial_oracle_admitted_load=(
                    pipeline.initial_oracle_admitted_load
                ),
                descriptor_provider=provider,
                stream_initializer=pipeline.initializer,
            )
            oracle_result = run_snapshot_closed_loop(
                PaperAlignedLEOEnv(episode_cfg, device=device),
                mode="oracle",
                controller=controller,
                margin_config=pipeline.margin_config,
                feature_config=pipeline.feature_config,
                initial_oracle_admitted_load=(
                    pipeline.initial_oracle_admitted_load
                ),
            )
            if model_result.protocol_fingerprint != oracle_result.protocol_fingerprint:
                raise RuntimeError("paired Snapshot conditions changed simulator protocol")
            if model_result.action_count != oracle_result.action_count:
                raise RuntimeError("paired Snapshot conditions have unequal horizons")
            pair_metrics = evaluate_snapshot_pair(
                model_result,
                oracle_result,
                options=metric_options,
            )
            shrink_jump = snapshot_shrink_jump_ratios(
                model_result,
                gamma_scale=pipeline.feature_config.gamma_scale,
                feasibility_scale=1.0,
                load_scale=pipeline.feature_config.admitted_load_scale,
                epsilon=float(shrink_config.get("epsilon", 1e-8)),
            )
            no_switch_ratio_units.append(shrink_jump["no_switch_ratios"])
            switch_ratio_units.append(shrink_jump["switch_ratios"])
            pair_payload = {
                "snapshot_evaluation_pair_version": 2,
                "artifact_kind": "leo_pg.paper.snapshot_paired_episode",
                "method": pipeline.method,
                "base_seed": base_seed,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "model": _serialize_result(model_result),
                "oracle": _serialize_result(oracle_result),
                "metrics": pair_metrics,
                "shrink_jump_audit": shrink_jump,
            }
            shard_name = (
                f"base_{base_seed:08d}_episode_{episode_index:04d}_"
                f"seed_{episode_seed:012d}.pt"
            )
            shard_path = shard_dir / shard_name
            _atomic_pt(shard_path, pair_payload)
            shard_entry = {
                "base_seed": base_seed,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "relative_path": f"{shard_dir.name}/{shard_name}",
                "sha256": _sha256(shard_path),
                "size_bytes": shard_path.stat().st_size,
                "protocol_fingerprint": model_result.protocol_fingerprint,
                "action_count": model_result.action_count,
            }
            pair_entries.append(shard_entry)
            pair_manifest.append(dict(shard_entry))
            metric_units.append(
                {
                    "base_seed": base_seed,
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "metrics": pair_metrics,
                }
            )

    aggregates = aggregate_evaluation_units(
        metric_units,
        options={
            "bootstrap_resamples": int(
                defaults.get("bootstrap_resamples", 10_000)
            ),
            "bootstrap_seed": int(defaults.get("bootstrap_seed", 17)),
            "bootstrap_confidence": float(
                defaults.get("bootstrap_confidence", 0.95)
            ),
            "tail_risk_probability": float(
                defaults.get("tail_risk_probability", 0.10)
            ),
        },
        context={
            "base_seeds": list(seeds),
            "episodes_per_seed": episodes,
        },
    )
    shrink_jump_aggregate = _aggregate_shrink_jump(
        no_switch_ratio_units,
        switch_ratio_units,
        config=shrink_config,
    )

    bundle = {
        "snapshot_evaluation_bundle_version": SNAPSHOT_EVALUATION_BUNDLE_VERSION,
        "artifact_kind": "leo_pg.paper.snapshot_paired_action_coupled",
        "method": pipeline.method,
        "paper_interface": "snapshot_v1",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "checkpoint_schema_version": checkpoint.get(
                "checkpoint_schema_version"
            ),
            "epoch": checkpoint.get("snapshot_trainer_state", {}).get("epoch"),
            "best_epoch": checkpoint.get("snapshot_trainer_state", {}).get(
                "best_epoch"
            ),
        },
        "test_dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "dataset_kind": test_dataset.payload.get("dataset_kind"),
            "snapshot_pipeline_fingerprint": test_dataset.payload.get(
                "snapshot_pipeline_fingerprint"
            ),
        },
        "snapshot_pipeline": pipeline.manifest(),
        "snapshot_pipeline_fingerprint": pipeline.fingerprint,
        "generation": {
            "base_seeds": list(seeds),
            "episodes_per_seed": episodes,
            "episode_seed_formula": "base_seed*1000000+episode_index",
            "horizon_steps": horizon,
            "paired_modes": ["model", "oracle"],
        },
        "storage": {
            "layout": "paired_episode_shards_v1",
            "relative_to": "index_parent",
            "restricted_load": "torch.load(weights_only=True)",
        },
        "paired_episode_shards": pair_entries,
        "one_step_metrics": one_step_metrics,
        "shrink_jump_audit": shrink_jump_aggregate,
        "aggregates": aggregates,
    }
    _atomic_pt(pt_path, bundle)
    manifest = {
        "snapshot_evaluation_bundle_version": SNAPSHOT_EVALUATION_BUNDLE_VERSION,
        "artifact_kind": bundle["artifact_kind"],
        "method": pipeline.method,
        "paper_interface": "snapshot_v1",
        "artifact": {
            "path": str(pt_path.resolve()),
            "sha256": _sha256(pt_path),
            "size_bytes": pt_path.stat().st_size,
            "restricted_load": "torch.load(weights_only=True)",
        },
        "checkpoint": bundle["checkpoint"],
        "test_dataset": bundle["test_dataset"],
        "snapshot_pipeline_fingerprint": pipeline.fingerprint,
        "initializer_fingerprint": pipeline.initializer.fingerprint,
        "policy": asdict(pipeline.policy_config),
        "generation": bundle["generation"],
        "paired_units": pair_manifest,
        "one_step_metrics": {
            "contract": one_step_metrics["contract"],
            "loss_weights": one_step_metrics["loss_weights"],
            "percentile_interval": one_step_metrics["percentile_interval"],
        },
        "shrink_jump_audit": shrink_jump_aggregate,
        "aggregates": aggregates,
    }
    _atomic_json(manifest_path, manifest)
    print(
        f"[OK] method={pipeline.method} pairs={len(pair_entries)} "
        f"trace={pt_path} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
