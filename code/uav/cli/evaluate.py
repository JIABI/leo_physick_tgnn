#!/usr/bin/env python3
"""Run paired UAV descriptor-substitution rollouts and matched inference."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from uav_cfs.config import resolve_uav_shared_config
from uav_cfs.data import (
    UAVEpisodeDataset,
    build_uav_multirun_split_manifest,
    load_uav_dataset,
)
from uav_cfs.environment import UAVSharedServiceEnv
from uav_cfs.evaluation import (
    ConstantUAVPolicyStreamInitializer,
    UAVClosedLoopResult,
    UAVModelProvider,
    UAVSubstitutionMode,
    run_uav_paired_conditions,
)
from uav_cfs.metrics import (
    UAVOutcomeMetricConfig,
    aggregate_matched_uav_metrics,
    evaluate_uav_paired_results,
    evaluate_uav_teacher_forced_one_step_episode,
)
from uav_cfs.model import (
    UAV_MODEL_METHOD,
    load_uav_model_checkpoint,
)
from uav_cfs.policy import (
    UAVFixedRankPolicy,
    UAVFixedRankPolicyConfig,
)
from uav_cfs.training import UAVTrainingConfig
from uav_cfs.runtime import get_device, load_cfg
from uav_cfs.perturbations import UAVStructuralPerturbation
from uav_cfs.experiments import (
    UAVAblation,
    UAVAblationDescriptorProvider,
    ablated_policy_config,
)


UAV_EVALUATION_BUNDLE_VERSION = 2
DEFAULT_MODES = tuple(UAVSubstitutionMode)


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return copy.deepcopy(dict(value))


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("seeds must be unique and non-negative")
    return result


def _csv_modes(value: str) -> tuple[UAVSubstitutionMode, ...]:
    try:
        result = tuple(
            UAVSubstitutionMode.parse(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("modes must be non-empty and unique")
    if UAVSubstitutionMode.ORACLE not in result:
        raise argparse.ArgumentTypeError("paired modes must include oracle")
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    result = value.detach().cpu().contiguous().clone()
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _descriptor(value: Any) -> dict[str, torch.Tensor]:
    return {
        "eta_edge": _tensor(value.eta_edge, "descriptor.eta_edge"),
        "intensity_edge": _tensor(
            value.intensity_edge, "descriptor.intensity_edge"
        ),
        "station_flow_node": _tensor(
            value.station_flow_node, "descriptor.station_flow_node"
        ),
    }


def _serialize_result(result: UAVClosedLoopResult) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, record in enumerate(result.records):
        execution = {
            name: (_tensor(value, f"execution.{name}") if isinstance(value, torch.Tensor) else value)
            for name, value in vars(record.execution).items()
        }
        records.append(
            {
                "step_index": index,
                "observation_id": list(record.observation_id),
                "candidate_edge_ids": _tensor(
                    record.candidate_edge_ids, "record.candidate_edge_ids"
                ),
                "candidate_rank": _tensor(
                    record.candidate_rank, "record.candidate_rank"
                ),
                "feasible_start_edge": _tensor(
                    record.feasible_start_edge, "record.feasible_start_edge"
                ),
                "simulator_descriptors": _descriptor(
                    record.simulator_descriptors
                ),
                "model_input_descriptors": (
                    None
                    if record.model_input_descriptors is None
                    else _descriptor(record.model_input_descriptors)
                ),
                "policy_descriptors": _descriptor(record.policy_descriptors),
                "initialized_edge": _tensor(
                    record.initialized_edge, "record.initialized_edge"
                ),
                "next_model_prediction": (
                    None
                    if record.next_model_prediction is None
                    else _descriptor(record.next_model_prediction)
                ),
                "phase_before": _tensor(record.phase_before, "record.phase_before"),
                "current_station_before": _tensor(
                    record.current_station_before, "record.current_station_before"
                ),
                "action": {
                    "observation_id": list(record.action.observation_id),
                    "requested_station": _tensor(
                        record.action.requested_station,
                        "record.action.requested_station",
                    ),
                },
                "execution": execution,
            }
        )
    return {
        "mode": result.mode.value,
        "episode_seed": result.episode_seed,
        "protocol_version": result.protocol_version,
        "protocol_fingerprint": result.protocol_fingerprint,
        "paired_fingerprint": result.paired_fingerprint,
        "provider_fingerprint": result.provider_fingerprint,
        "initializer_kind": result.initializer_kind,
        "initializer_fingerprint": result.initializer_fingerprint,
        "policy_config": dict(result.policy_config),
        "evaluation_contract_version": result.evaluation_contract_version,
        "action_count": result.action_count,
        "records": records,
        "final_energy_fraction": _tensor(
            result.final_energy_fraction, "result.final_energy_fraction"
        ),
        "final_phase": _tensor(result.final_phase, "result.final_phase"),
        "final_station_flow": _tensor(
            result.final_station_flow, "result.final_station_flow"
        ),
        "final_missions_completed": _tensor(
            result.final_missions_completed, "result.final_missions_completed"
        ),
        "final_failed_service_starts": _tensor(
            result.final_failed_service_starts,
            "result.final_failed_service_starts",
        ),
        "final_reassociations": _tensor(
            result.final_reassociations, "result.final_reassociations"
        ),
    }


def _pure(value: Any, path: str = "root") -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor(value, path)
    if isinstance(value, Mapping):
        return {str(key): _pure(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pure(item, f"{path}[]") for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite value")
        return value
    raise TypeError(f"{path} has unsupported value {type(value).__name__}")


def _json_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_tree(item) for item in value]
    return value


def _atomic_pt(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(_pure(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(
                _json_tree(_pure(value)),
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _output_paths(value: str | Path) -> tuple[Path, Path, Path]:
    requested = Path(value).expanduser()
    if requested.suffix.lower() == ".pt":
        bundle = requested
    else:
        bundle = requested / "uav_paired_evaluation.pt"
    return (
        bundle,
        bundle.with_suffix(".manifest.json"),
        bundle.parent / f"{bundle.stem}_episodes",
    )


def _run_checkpoint_paths(
    specification: str,
    run_seeds: Sequence[int],
) -> dict[int, Path]:
    """Resolve one independently trained checkpoint for every formal run."""

    raw = str(specification).strip()
    if not raw:
        raise ValueError("checkpoint specification cannot be empty")
    seeds = tuple(int(value) for value in run_seeds)
    paths: dict[int, Path] = {}
    declared_sha256: dict[int, str] = {}
    requested = Path(raw).expanduser().resolve()
    if requested.is_file() and requested.suffix.lower() == ".json":
        payload = json.loads(requested.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint manifest root must be a mapping")
        raw_runs: Any = payload.get(
            "checkpoints_by_run",
            payload.get("checkpoints", payload.get("runs")),
        )
        if isinstance(raw_runs, list):
            raw_runs = {
                str(item["run_seed"]): item
                for item in raw_runs
                if isinstance(item, Mapping) and "run_seed" in item
            }
        if not isinstance(raw_runs, Mapping):
            raise ValueError("checkpoint manifest has no per-run checkpoint mapping")
        for run_seed in seeds:
            entry = raw_runs.get(str(run_seed), raw_runs.get(run_seed))
            if isinstance(entry, Mapping):
                digest = entry.get(
                    "checkpoint_sha256", entry.get("best_checkpoint_sha256")
                )
                if digest is not None:
                    if not isinstance(digest, str) or len(digest) != 64:
                        raise ValueError(
                            f"checkpoint manifest digest for run {run_seed} is invalid"
                        )
                    declared_sha256[run_seed] = digest.lower()
                entry = entry.get(
                    "best_checkpoint",
                    entry.get("best", entry.get("path")),
                )
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    f"checkpoint manifest has no best checkpoint for run {run_seed}"
                )
            candidate = Path(entry).expanduser()
            if not candidate.is_absolute():
                candidate = requested.parent / candidate
            paths[run_seed] = candidate.resolve()
    elif "{run_seed}" in raw:
        paths = {
            run_seed: Path(raw.format(run_seed=run_seed)).expanduser().resolve()
            for run_seed in seeds
        }
    else:
        if requested.is_dir():
            for run_seed in seeds:
                candidates = (
                    requested / f"run_{run_seed}" / "best.pt",
                    requested / str(run_seed) / "best.pt",
                    requested / f"run_{run_seed}.pt",
                )
                matches = [candidate for candidate in candidates if candidate.is_file()]
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"expected one checkpoint for run {run_seed} under {requested}; "
                        f"checked {[str(value) for value in candidates]}"
                    )
                paths[run_seed] = matches[0]
        elif len(seeds) == 1:
            paths[seeds[0]] = requested
        else:
            raise ValueError(
                "formal multi-run evaluation requires one checkpoint per run; "
                "pass a directory or a path template containing {run_seed}"
            )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("UAV checkpoints do not exist: " + ", ".join(missing))
    resolved_paths = list(paths.values())
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("formal run seeds cannot share the same checkpoint file")
    for run_seed, expected in declared_sha256.items():
        observed = _file_sha256(paths[run_seed])
        if observed.lower() != expected:
            raise ValueError(
                f"checkpoint SHA-256 mismatch for run {run_seed}: "
                f"manifest={expected}, observed={observed}"
            )
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run paired action-coupled UAV substitution evaluation"
    )
    parser.add_argument("--cfg", required=True, help="Paper protocol YAML")
    parser.add_argument(
        "--ckpt",
        required=True,
        help=(
            "Independent-run checkpoint root, checkpoint_manifest.json, or "
            "path template containing {run_seed}; a single file is accepted "
            "only for a single requested run"
        ),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional generated dataset index; test identities are read from its manifest",
    )
    parser.add_argument("--out", default=None, help="Bundle .pt or output directory")
    parser.add_argument("--method", default=UAV_MODEL_METHOD)
    parser.add_argument("--seeds", type=_csv_ints, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--modes", type=_csv_modes, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--density-multiplier", type=int, default=None)
    parser.add_argument("--capacity-compression", type=float, default=None)
    parser.add_argument(
        "--perturbation-config",
        default=None,
        help="YAML containing a structural_mismatches mapping",
    )
    parser.add_argument(
        "--perturbation-name",
        default="nominal",
        help="nominal or one key in structural_mismatches",
    )
    parser.add_argument(
        "--ablation",
        choices=tuple(item.value for item in UAVAblation),
        default=UAVAblation.FULL.value,
    )
    args = parser.parse_args(argv)

    root = load_cfg(args.cfg)
    pipeline = _mapping(root, "uav_pipeline")
    evaluation_cfg = _mapping(pipeline, "evaluation")
    metrics_cfg = _mapping(pipeline, "metrics")
    initializer_cfg = _mapping(pipeline, "initializer")
    dataset_cfg = _mapping(pipeline, "dataset")
    resolved = resolve_uav_shared_config(root)
    density_multiplier = (
        resolved.protocol.density_multiplier
        if args.density_multiplier is None
        else args.density_multiplier
    )
    capacity_compression = (
        resolved.protocol.capacity_compression
        if args.capacity_compression is None
        else args.capacity_compression
    )
    resolved = replace(
        resolved,
        protocol=replace(
            resolved.protocol,
            density_multiplier=density_multiplier,
            capacity_compression=capacity_compression,
        ),
    )
    if args.perturbation_name == "nominal":
        perturbation = UAVStructuralPerturbation()
    else:
        if args.perturbation_config is None:
            parser.error("non-nominal perturbation requires --perturbation-config")
        perturbation_root = load_cfg(args.perturbation_config)
        mismatch_mapping = perturbation_root.get("structural_mismatches")
        if not isinstance(mismatch_mapping, Mapping):
            raise TypeError("perturbation config requires structural_mismatches")
        selected_mapping = mismatch_mapping.get(args.perturbation_name)
        if not isinstance(selected_mapping, Mapping):
            raise KeyError(
                f"unknown structural perturbation {args.perturbation_name!r}"
            )
        perturbation = UAVStructuralPerturbation.from_mapping(selected_mapping)
    include_one_step_loss = evaluation_cfg.get("include_one_step_loss", True)
    if type(include_one_step_loss) is not bool:
        raise TypeError("uav_pipeline.evaluation.include_one_step_loss must be bool")

    configured_seeds = evaluation_cfg.get(
        "base_seeds",
        dataset_cfg.get(
            "run_seeds", root.get("run_seeds", (root.get("seed", 0),))
        ),
    )
    if args.seeds is not None:
        base_seeds = args.seeds
    else:
        if not isinstance(configured_seeds, (list, tuple)):
            raise TypeError("uav_pipeline.evaluation.base_seeds must be a sequence")
        base_seeds = tuple(int(value) for value in configured_seeds)
        if (
            not base_seeds
            or any(value < 0 for value in base_seeds)
            or len(set(base_seeds)) != len(base_seeds)
        ):
            raise ValueError("evaluation base_seeds must be unique and non-negative")
    if len(base_seeds) != 5:
        raise ValueError(
            "paper evaluation requires exactly five independent training seeds"
        )
    episodes_per_run = _positive_int(
        "episodes_per_seed",
        args.episodes
        if args.episodes is not None
        else evaluation_cfg.get(
            "episodes_per_run", dataset_cfg.get("test_episodes_per_run", 80)
        ),
    )
    if episodes_per_run != 30:
        raise ValueError(
            "paper evaluation requires exactly 30 held-out episodes per run"
        )
    configured_horizon = _positive_int(
        "uav_pipeline.evaluation.horizon_steps",
        evaluation_cfg.get(
            "horizon_steps", resolved.protocol.exact.evaluation_horizon_steps
        ),
    )
    if configured_horizon != resolved.protocol.exact.evaluation_horizon_steps:
        raise ValueError(
            "UAV evaluation horizon must match the manuscript-exact "
            "uav_shared.exact.evaluation_horizon_steps"
        )
    if args.modes is not None:
        modes = args.modes
    else:
        configured_modes = evaluation_cfg.get(
            "modes", [mode.value for mode in DEFAULT_MODES]
        )
        if not isinstance(configured_modes, (list, tuple)):
            raise TypeError("uav_pipeline.evaluation.modes must be a sequence")
        modes = tuple(UAVSubstitutionMode.parse(value) for value in configured_modes)
        if len(set(modes)) != len(modes) or UAVSubstitutionMode.ORACLE not in modes:
            raise ValueError("evaluation modes must be unique and include oracle")

    configured_data = (
        args.data if args.data is not None else dataset_cfg.get("output")
    )
    data_path: Path | None = None
    dataset_index: Mapping[str, Any] | None = None
    if configured_data is not None and str(configured_data).strip():
        data_path = Path(str(configured_data)).expanduser().resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"UAV dataset index does not exist: {data_path}")
        dataset_index = load_uav_dataset(data_path)
        split_manifest = dataset_index["split_manifest"]
        raw_identities = split_manifest.get("episode_identity")
        if not isinstance(raw_identities, list):
            raise ValueError("UAV dataset has no multi-run episode identities")
        available = [
            dict(identity)
            for identity in raw_identities
            if isinstance(identity, Mapping) and identity.get("split") == "test"
        ]
        dataset_runs = tuple(int(value) for value in split_manifest.get("run_seeds", ()))
        if set(base_seeds) - set(dataset_runs):
            raise ValueError("requested evaluation seeds are absent from the dataset")
    else:
        if include_one_step_loss:
            parser.error(
                "held-out one-step loss requires --data or "
                "uav_pipeline.dataset.output"
            )
        split_counts = (
            _positive_int(
                "train_episodes_per_run",
                dataset_cfg.get("train_episodes_per_run", 240),
            ),
            _positive_int(
                "validation_episodes_per_run",
                dataset_cfg.get("validation_episodes_per_run", 40),
            ),
            _positive_int(
                "test_episodes_per_run",
                dataset_cfg.get("test_episodes_per_run", 80),
            ),
        )
        split_seed = int(dataset_cfg.get("split_seed", 1709))
        if split_seed < 0:
            raise ValueError("uav_pipeline.dataset.split_seed must be non-negative")
        split_manifest = build_uav_multirun_split_manifest(
            base_seeds,
            split_counts_per_run=split_counts,
            split_seed=split_seed,
        )
        available = [
            dict(identity)
            for identity in split_manifest["episode_identity"]
            if identity["split"] == "test"
        ]
    selected_identities: list[dict[str, Any]] = []
    for run_seed in base_seeds:
        run_test = [
            identity
            for identity in available
            if int(identity["run_seed"]) == run_seed
        ]
        if len(run_test) < episodes_per_run:
            raise ValueError(
                f"run {run_seed} has {len(run_test)} test identities, "
                f"fewer than requested {episodes_per_run}"
            )
        selected_identities.extend(run_test[:episodes_per_run])

    test_datasets: dict[int, UAVEpisodeDataset] = {}
    test_positions: dict[int, dict[int, int]] = {}
    if include_one_step_loss:
        assert data_path is not None
        for run_seed in base_seeds:
            view = UAVEpisodeDataset(data_path, "test", run_seed=run_seed)
            positions: dict[int, int] = {}
            for position, entry in enumerate(view.entries):
                local_index = int(entry["local_episode_index"])
                if local_index in positions:
                    raise ValueError("duplicate local test identity in UAV dataset")
                positions[local_index] = position
            test_datasets[run_seed] = view
            test_positions[run_seed] = positions

    output_value = args.out if args.out is not None else evaluation_cfg.get("output")
    if output_value is None or not str(output_value).strip():
        parser.error("provide --out or uav_pipeline.evaluation.output")
    bundle_path, manifest_path, shard_dir = _output_paths(str(output_value))
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = _mapping(pipeline, "training")
    requested_device = args.device or str(
        evaluation_cfg.get("device", training_cfg.get("device", "cpu"))
    )
    device = get_device(requested_device, strict=args.device is not None)
    loss_weights = UAVTrainingConfig.from_source(root).loss_weights
    expected_loss_weights = asdict(loss_weights)
    checkpoint_paths = _run_checkpoint_paths(args.ckpt, base_seeds)
    models: dict[int, Any] = {}
    checkpoint_sha: dict[int, str] = {}
    reference_checkpoint_loss_weights: dict[str, float] | None = None
    for run_seed, checkpoint_path in checkpoint_paths.items():
        checkpoint_sha[run_seed] = _file_sha256(checkpoint_path)
        models[run_seed] = load_uav_model_checkpoint(
            root,
            checkpoint_path,
            method=args.method,
            device=device,
            strict=True,
        )
        checkpoint_metadata = getattr(models[run_seed], "checkpoint_metadata", None)
        if not isinstance(checkpoint_metadata, Mapping):
            raise ValueError("UAV checkpoint is missing training-run metadata")
        saved_run_seed = checkpoint_metadata.get("training_run_seed")
        if saved_run_seed != run_seed:
            raise ValueError(
                f"checkpoint training_run_seed={saved_run_seed!r} does not match "
                f"evaluation run {run_seed}"
            )
        if include_one_step_loss:
            saved_training = checkpoint_metadata.get("training_settings")
            if not isinstance(saved_training, Mapping):
                raise ValueError(
                    f"checkpoint for run {run_seed} is missing the "
                    "training_settings mapping required for one-step loss"
                )
            saved_weights = saved_training.get("loss_weights")
            if not isinstance(saved_weights, Mapping):
                raise ValueError(
                    f"checkpoint for run {run_seed} is missing the "
                    "training_settings.loss_weights mapping required for "
                    "one-step loss"
                )
            expected_keys = set(expected_loss_weights)
            saved_keys = set(saved_weights)
            if saved_keys != expected_keys:
                raise ValueError(
                    f"checkpoint loss-weight schema mismatch for run {run_seed}; "
                    f"expected keys {sorted(expected_keys)!r}, got "
                    f"{sorted(saved_keys)!r}"
                )
            normalized_saved_weights: dict[str, float] = {}
            for name in expected_loss_weights:
                value = saved_weights[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(
                        f"checkpoint training_settings.loss_weights.{name} "
                        f"for run {run_seed} must be numeric"
                    )
                normalized_saved_weights[name] = float(value)
            if reference_checkpoint_loss_weights is None:
                reference_checkpoint_loss_weights = normalized_saved_weights
            elif normalized_saved_weights != reference_checkpoint_loss_weights:
                raise ValueError(
                    "one-step loss weights differ across independent training "
                    f"runs; run {run_seed} has {normalized_saved_weights!r}, "
                    f"expected {reference_checkpoint_loss_weights!r}"
                )
            if normalized_saved_weights != expected_loss_weights:
                raise ValueError(
                    f"checkpoint loss weights for run {run_seed} do not match "
                    "the current UAVTrainingConfig; checkpoint="
                    f"{normalized_saved_weights!r}, "
                    f"configuration={expected_loss_weights!r}"
                )

    ablation = UAVAblation.parse(args.ablation)
    policy_config = ablated_policy_config(
        UAVFixedRankPolicyConfig.from_source(root), ablation
    )
    initial_intensity = float(initializer_cfg.get("intensity", 1.5))
    initial_flow = float(initializer_cfg.get("station_flow", 0.0))
    if ablation in {UAVAblation.NO_INTENSITY, UAVAblation.LOCAL_CUE_ONLY}:
        initial_intensity = 0.0
    if ablation in {UAVAblation.NO_FLOW, UAVAblation.LOCAL_CUE_ONLY}:
        initial_flow = 0.0
    initializer = ConstantUAVPolicyStreamInitializer(
        eta=float(initializer_cfg.get("eta", 0.5)),
        intensity=initial_intensity,
        station_flow=initial_flow,
    )
    outcome_config = UAVOutcomeMetricConfig(
        control_dt_s=resolved.protocol.exact.control_dt_s,
        pingpong_window_seconds=float(
            metrics_cfg.get("pingpong_window_seconds", 2.0)
        ),
        tail_probability=float(metrics_cfg.get("tail_probability", 0.10)),
        delay_tail_probability=float(
            metrics_cfg.get("delay_tail_probability", 0.90)
        ),
        fifo_queue_capacity=resolved.protocol.exact.fifo_queue_capacity,
        active_slots_per_station=(
            resolved.protocol.exact.active_slots_per_station
        ),
    )
    resamples = _positive_int(
        "bootstrap_resamples",
        evaluation_cfg.get(
            "bootstrap_resamples", metrics_cfg.get("bootstrap_resamples", 10_000)
        ),
    )
    bootstrap_seed = int(
        evaluation_cfg.get(
            "bootstrap_seed", metrics_cfg.get("bootstrap_seed", 17001)
        )
    )
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")
    confidence = float(
        evaluation_cfg.get(
            "bootstrap_confidence",
            metrics_cfg.get("bootstrap_confidence", 0.95),
        )
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must lie in (0,1)")

    entries: list[dict[str, Any]] = []
    aggregate_units: list[dict[str, Any]] = []
    for evaluation_unit_index, identity in enumerate(selected_identities):
        run_seed = int(identity["run_seed"])
        run_index = int(identity["run_index"])
        local_episode_index = int(identity["local_episode_index"])
        episode_seed = int(identity["episode_seed"])
        paired_episode_id = str(
            identity.get(
                "paired_episode_id",
                f"uav_run_{run_seed}_episode_{local_episode_index}",
            )
        )
        exogenous_sequence_id = int(
            identity.get("exogenous_sequence_id", episode_seed)
        )
        expected_seed = run_seed * 1_000_000 + local_episode_index
        if episode_seed != expected_seed:
            raise ValueError("test identity violates the formal episode seed rule")
        if exogenous_sequence_id != episode_seed:
            raise ValueError(
                "UAV release contract requires exogenous_sequence_id to match "
                "the episode seed that keys all simulator randomness"
            )
        protocol = replace(resolved.protocol, episode_seed=episode_seed)
        environment = UAVSharedServiceEnv(
            protocol, device="cpu", perturbation=perturbation
        )
        def descriptor_provider_factory() -> Any:
            provider: Any = UAVModelProvider(
                models[run_seed],
                device=device,
                fingerprint=checkpoint_sha[run_seed],
                protocol=protocol,
            )
            if ablation not in {UAVAblation.FULL, UAVAblation.NO_GAIN_CONTROL}:
                provider = UAVAblationDescriptorProvider(provider, ablation)
            return provider

        results = run_uav_paired_conditions(
            environment,
            modes=modes,
            policy_factory=lambda: UAVFixedRankPolicy(policy_config),
            descriptor_provider_factory=descriptor_provider_factory,
            stream_initializer_factory=lambda: ConstantUAVPolicyStreamInitializer(
                eta=initializer.eta,
                intensity=initializer.intensity,
                station_flow=initializer.station_flow,
            ),
        )
        metric_rows = evaluate_uav_paired_results(
            results,
            outcome_config=outcome_config,
            near_tie_margin=float(
                metrics_cfg.get("near_tie_margin", 1.0 / 3.0)
            ),
        )
        one_step_metrics: dict[str, Any] | None = None
        if include_one_step_loss:
            position = test_positions[run_seed].get(local_episode_index)
            if position is None:
                raise ValueError("selected test identity has no dataset shard")
            serialized_episode = test_datasets[run_seed][position]
            if int(serialized_episode.get("seed", -1)) != episode_seed:
                raise ValueError("held-out episode shard seed differs from evaluation unit")
            one_step_metrics = evaluate_uav_teacher_forced_one_step_episode(
                models[run_seed],
                serialized_episode,
                protocol=protocol,
                loss_weights=loss_weights,
                device=device,
            ).as_dict()
        metric_payload = {
            "closed_loop": metric_rows,
            "one_step_model": one_step_metrics,
        }
        paired_fingerprint = next(iter(results.values())).paired_fingerprint
        payload = {
            "uav_evaluation_pair_version": 2,
            "artifact_kind": "uav_cfs.paired_substitution_episode",
            "evaluation_unit_index": evaluation_unit_index,
            "run_seed": run_seed,
            "run_index": run_index,
            "local_episode_index": local_episode_index,
            "episode_seed": episode_seed,
            "paired_episode_id": paired_episode_id,
            "exogenous_sequence_id": exogenous_sequence_id,
            "paired_fingerprint": paired_fingerprint,
            "protocol_manifest": protocol.manifest(),
            "structural_perturbation": perturbation.manifest(),
            "ablation": ablation.value,
            "conditions": {
                mode.value: _serialize_result(result)
                for mode, result in results.items()
            },
            "metrics": metric_payload,
        }
        shard_name = (
            f"run_{run_seed:08d}_local_{local_episode_index:04d}_"
            f"seed_{episode_seed:012d}.pt"
        )
        shard_path = shard_dir / shard_name
        _atomic_pt(shard_path, payload)
        entry = {
            "evaluation_unit_index": evaluation_unit_index,
            "run_seed": run_seed,
            "run_index": run_index,
            "local_episode_index": local_episode_index,
            "episode_seed": episode_seed,
            "paired_episode_id": paired_episode_id,
            "exogenous_sequence_id": exogenous_sequence_id,
            "paired_fingerprint": paired_fingerprint,
            "relative_path": f"{shard_dir.name}/{shard_name}",
            "sha256": _file_sha256(shard_path),
            "size_bytes": shard_path.stat().st_size,
            "action_count": next(iter(results.values())).action_count,
        }
        entries.append(entry)
        aggregate_units.append(
            {
                "run_seed": run_seed,
                "local_episode_index": local_episode_index,
                "episode_seed": episode_seed,
                "paired_episode_id": paired_episode_id,
                "exogenous_sequence_id": exogenous_sequence_id,
                "paired_fingerprint": paired_fingerprint,
                "metrics": metric_payload,
            }
        )

    aggregates = aggregate_matched_uav_metrics(
        aggregate_units,
        resamples=resamples,
        confidence=confidence,
        bootstrap_seed=bootstrap_seed,
    )
    bundle = {
        "uav_evaluation_bundle_version": UAV_EVALUATION_BUNDLE_VERSION,
        "artifact_kind": "uav_cfs.paired_action_coupled_evaluation",
        "method": args.method,
        "checkpoints": {
            str(run_seed): {
                "path": (
                    Path(checkpoint_paths[run_seed].parent.name)
                    / checkpoint_paths[run_seed].name
                ).as_posix(),
                "sha256": checkpoint_sha[run_seed],
                "model_fingerprint": models[run_seed].fingerprint,
            }
            for run_seed in base_seeds
        },
        "configuration": {
            "source_file": Path(args.cfg).expanduser().name,
            "operator": models[base_seeds[0]].model_config.operator,
            "base_protocol": resolved.manifest(),
            "structural_perturbation": perturbation.manifest(),
            "ablation": ablation.value,
            "base_seeds": list(base_seeds),
            "episodes_per_run": episodes_per_run,
            "include_one_step_loss": include_one_step_loss,
            "test_identity_source": (
                data_path.name
                if data_path is not None
                else "reconstructed_formal_multirun_split_manifest"
            ),
            "dataset_index_sha256": (
                _file_sha256(data_path) if data_path is not None else None
            ),
            "one_step_loss_weights": asdict(loss_weights),
            "modes": [mode.value for mode in modes],
            "policy": asdict(policy_config),
            "initializer": {
                "kind": initializer.kind,
                "fingerprint": initializer.fingerprint,
                "eta": initializer.eta,
                "intensity": initializer.intensity,
                "station_flow": initializer.station_flow,
            },
            "outcome_metrics": asdict(outcome_config),
        },
        "paired_units": entries,
        "aggregates": aggregates,
    }
    _atomic_pt(bundle_path, bundle)
    manifest = {
        "uav_evaluation_bundle_version": UAV_EVALUATION_BUNDLE_VERSION,
        "artifact_kind": bundle["artifact_kind"],
        "bundle": {
            "path": bundle_path.name,
            "sha256": _file_sha256(bundle_path),
            "size_bytes": bundle_path.stat().st_size,
        },
        "checkpoint_sha256_by_run": {
            str(run_seed): checkpoint_sha[run_seed] for run_seed in base_seeds
        },
        "base_seeds": list(base_seeds),
        "operator": models[base_seeds[0]].model_config.operator,
        "episodes_per_run": episodes_per_run,
        "modes": [mode.value for mode in modes],
        "structural_perturbation": perturbation.manifest(),
        "ablation": ablation.value,
        "paired_units": entries,
        "aggregates": aggregates,
    }
    _atomic_json(manifest_path, manifest)
    print(str(bundle_path))
    print(str(manifest_path))


if __name__ == "__main__":
    main()
