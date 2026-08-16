"""Train the typed UAV world model and save automatic best/last checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from uav_cfs.data import UAVEpisodeDataset
from uav_cfs.model import UAV_MODEL_METHOD, build_uav_model
from uav_cfs.training import UAVTrainer
from uav_cfs.runtime import get_device, load_cfg, set_seed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pipeline(root: Mapping[str, Any]) -> dict[str, Any]:
    value = root.get("uav_pipeline")
    if not isinstance(value, Mapping):
        raise ValueError("configuration requires uav_pipeline")
    return dict(value)


def _configured_data_path(root: Mapping[str, Any], override: str | None) -> Path:
    if override is not None:
        result = Path(override).expanduser()
    else:
        pipeline = _pipeline(root)
        dataset = pipeline.get("dataset")
        if not isinstance(dataset, Mapping):
            raise ValueError("configuration requires uav_pipeline.dataset")
        configured = dataset.get("output")
        if configured is None:
            raise ValueError("provide --data or uav_pipeline.dataset.output")
        result = Path(str(configured)).expanduser()
    if not result.is_file():
        raise FileNotFoundError(f"UAV dataset index does not exist: {result}")
    return result.resolve()


def _apply_objective_overrides(
    root: dict[str, Any],
    *,
    objective: str | None,
    rollout_horizon: int | None,
    feedback_probability: float | None,
) -> dict[str, Any]:
    result = copy.deepcopy(root)
    pipeline = result.get("uav_pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("configuration requires uav_pipeline")
    training = pipeline.get("training")
    if not isinstance(training, dict):
        raise ValueError("configuration requires uav_pipeline.training")
    action = training.get("action_coupled_multistep")
    if not isinstance(action, dict):
        raise ValueError(
            "configuration requires uav_pipeline.training.action_coupled_multistep"
        )
    sampling = training.get("scheduled_sampling")
    if not isinstance(sampling, dict):
        raise ValueError(
            "configuration requires uav_pipeline.training.scheduled_sampling"
        )
    if objective is not None:
        training["objective"] = objective
        if objective in {"action_coupled_multistep", "hybrid"}:
            action["enabled"] = True
        if objective == "one_step_scheduled_sampling":
            sampling["enabled"] = True
    if rollout_horizon is not None:
        if rollout_horizon <= 0:
            raise ValueError("--rollout-horizon must be positive")
        action["rollout_horizon"] = rollout_horizon
        action["enabled"] = True
    if feedback_probability is not None:
        if not 0.0 <= feedback_probability <= 1.0:
            raise ValueError("--feedback-probability must lie in [0,1]")
        action["model_feedback_probability"] = feedback_probability
        action["enabled"] = True
    return result


def _dataset_metadata(
    dataset: UAVEpisodeDataset,
    path: Path,
    *,
    training_run_seed: int,
) -> dict[str, Any]:
    index = dataset.index
    return {
        "index_file": path.name,
        "sha256": _file_sha256(path),
        "dataset_kind": index.get("dataset_kind"),
        "uav_dataset_schema_version": index.get("uav_dataset_schema_version"),
        "uav_graph_contract_version": index.get("graph_contract_version"),
        "uav_target_contract_version": index.get("target_contract_version"),
        "storage": copy.deepcopy(index.get("storage")),
        "split_manifest": copy.deepcopy(index.get("split_manifest")),
        "train_split": "train",
        "validation_split": "val",
        "training_run_seed": int(training_run_seed),
        "model_initialization_seed": int(training_run_seed),
        "optimizer_seed": int(training_run_seed),
        "data_order_seed": int(training_run_seed),
    }


def _action_train_seeds(dataset: UAVEpisodeDataset) -> tuple[int, ...]:
    seeds = []
    seen: set[int] = set()
    for entry in dataset.entries:
        seed = int(entry["seed"])
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    return tuple(seeds)


def _progress(record: Mapping[str, Any]) -> None:
    print("[UAV-EPOCH] " + json.dumps(dict(record), sort_keys=True))


def _run_seeds(index: Mapping[str, Any]) -> tuple[int, ...]:
    manifest = index.get("split_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("UAV dataset index requires split_manifest")
    raw = manifest.get("run_seeds")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "--all-runs requires a multi-run dataset with split_manifest.run_seeds"
        )
    if any(type(seed) is not int or seed < 0 for seed in raw):
        raise ValueError("split_manifest.run_seeds must be non-negative integers")
    if len(set(raw)) != len(raw):
        raise ValueError("split_manifest.run_seeds must be unique")
    if len(raw) != 5:
        raise ValueError(
            "--all-runs publication protocol requires exactly five independent run seeds"
        )
    return tuple(raw)


def _run_config(root: Mapping[str, Any], run_seed: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(root))
    result["seed"] = int(run_seed)
    shared = result.get("uav_shared")
    if not isinstance(shared, dict):
        raise ValueError("configuration requires uav_shared")
    shared["episode_seed"] = int(run_seed)
    pipeline = result.get("uav_pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("configuration requires uav_pipeline")
    training = pipeline.get("training")
    if not isinstance(training, dict):
        raise ValueError("configuration requires uav_pipeline.training")
    training["sampling_seed"] = int(run_seed)
    return result


def _resume_for_run(value: str | None, run_seed: int) -> Path | None:
    if value is None:
        return None
    formatted = Path(value.format(run_seed=run_seed)).expanduser()
    if formatted.is_dir():
        nested = formatted / f"run_{run_seed}" / "last.pt"
        if nested.is_file():
            return nested
        direct = formatted / "last.pt"
        if direct.is_file():
            return direct
    if not formatted.is_file():
        raise FileNotFoundError(f"UAV resume checkpoint does not exist: {formatted}")
    return formatted


def _write_checkpoint_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the protocol-aligned UAV descriptor model. This command "
            "writes best.pt and last.pt automatically."
        )
    )
    parser.add_argument("--cfg", required=True, help="Repository paper YAML")
    parser.add_argument(
        "--data", default=None, help="UAV sharded-dataset index .pt"
    )
    parser.add_argument(
        "--out", required=True, help="Checkpoint run directory or best .pt path"
    )
    parser.add_argument("--resume", default=None, help="Resume a UAV last.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--objective",
        choices=(
            "one_step_teacher_forcing",
            "one_step_scheduled_sampling",
            "action_coupled_multistep",
            "hybrid",
        ),
        default=None,
    )
    parser.add_argument("--rollout-horizon", type=int, default=None)
    parser.add_argument("--feedback-probability", type=float, default=None)
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help="Train exactly one independent run from its own train/val shards",
    )
    run_group.add_argument(
        "--all-runs",
        action="store_true",
        help="Train every independent run listed by the dataset manifest",
    )
    parser.add_argument(
        "--no-verify-hash",
        action="store_true",
        help="Skip per-shard integrity verification while loading",
    )
    args = parser.parse_args(argv)

    root = load_cfg(args.cfg)
    if not isinstance(root, dict):
        raise TypeError("configuration root must be a mapping")
    resolved = _apply_objective_overrides(
        root,
        objective=args.objective,
        rollout_horizon=args.rollout_horizon,
        feedback_probability=args.feedback_probability,
    )
    data_path = _configured_data_path(resolved, args.data)
    verify_hash = not args.no_verify_hash
    unfiltered = UAVEpisodeDataset(
        data_path, "train", verify_hash=verify_hash
    )
    if args.run_seed is not None:
        if args.run_seed < 0:
            parser.error("--run-seed must be non-negative")
        selected_run_seeds = (args.run_seed,)
    else:
        selected_run_seeds = _run_seeds(unfiltered.index)

    output_root = Path(args.out).expanduser()
    if output_root.suffix.lower() == ".pt":
        parser.error("--out must be a directory; runs are stored below run_<seed>/")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for run_seed in selected_run_seeds:
        run_cfg = _run_config(resolved, run_seed)
        pipeline = _pipeline(run_cfg)
        training = pipeline.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("configuration requires uav_pipeline.training")
        train_dataset = UAVEpisodeDataset(
            data_path,
            "train",
            run_seed=run_seed,
            verify_hash=verify_hash,
        )
        validation_dataset = UAVEpisodeDataset(
            data_path,
            "val",
            run_seed=run_seed,
            verify_hash=verify_hash,
        )
        set_seed(run_seed)
        requested_device = args.device or str(training.get("device", "cpu"))
        device = get_device(requested_device, strict=args.device is not None)
        model = build_uav_model(run_cfg, UAV_MODEL_METHOD, device=device)
        run_dir = output_root / f"run_{run_seed}"
        best_path = run_dir / "best.pt"
        last_path = run_dir / "last.pt"
        trainer = UAVTrainer(
            model,
            run_cfg,
            device=device,
            best_checkpoint_path=best_path,
            last_checkpoint_path=last_path,
            dataset_metadata=_dataset_metadata(
                train_dataset,
                data_path,
                training_run_seed=run_seed,
            ),
            progress=_progress,
        )
        resume = _resume_for_run(args.resume, run_seed)
        if resume is not None:
            trainer.resume(resume)
        result = trainer.fit(
            train_dataset,
            validation_dataset,
            action_train_seeds=_action_train_seeds(train_dataset),
        )
        best_sha256 = _file_sha256(best_path)
        last_sha256 = _file_sha256(last_path)
        operator = model.model_config.operator
        config_fingerprint = _mapping_sha256(run_cfg)
        manifest_rows.append(
            {
                "run_id": f"uav_{operator}_run_{run_seed}",
                "run_seed": run_seed,
                "training_run_seed": run_seed,
                "run_directory": run_dir.name,
                "best_checkpoint": (Path(run_dir.name) / best_path.name).as_posix(),
                "last_checkpoint": (Path(run_dir.name) / last_path.name).as_posix(),
                "checkpoint_relative_path": (
                    Path(run_dir.name) / best_path.name
                ).as_posix(),
                "checkpoint_sha256": best_sha256,
                "best_checkpoint_sha256": best_sha256,
                "last_checkpoint_sha256": last_sha256,
                "checkpoint_id": (
                    f"uav_{operator}_run_{run_seed}_best_{best_sha256[:12]}"
                ),
                "model_fingerprint": model.fingerprint,
                "resolved_run_config_sha256": config_fingerprint,
                "config_fingerprint": config_fingerprint,
                "best_epoch": result.best_epoch,
                "best_validation": result.best_validation,
                "model_initialization_seed": run_seed,
                "optimizer_seed": run_seed,
                "data_order_seed": run_seed,
            }
        )
        print(
            f"[OK] run_seed={run_seed} method={UAV_MODEL_METHOD} "
            f"best_epoch={result.best_epoch} "
            f"validation={result.best_validation:.8g} "
            f"best={result.best_checkpoint} last={result.last_checkpoint}"
        )

    manifest_path = output_root / "checkpoint_manifest.json"
    _write_checkpoint_manifest(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "uav_cfs.independent_training_runs",
            "model_method": UAV_MODEL_METHOD,
            "dataset_index_file": data_path.name,
            "dataset_index_sha256": _file_sha256(data_path),
            "run_count": len(manifest_rows),
            "runs": manifest_rows,
        },
    )
    print(f"[OK] checkpoint_manifest={manifest_path}")


if __name__ == "__main__":
    main()
