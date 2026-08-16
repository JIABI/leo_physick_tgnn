#!/usr/bin/env python3
"""Run the complete UAV experiment matrix from frozen checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

import yaml

from uav_cfs.experiments import paper_task_matrix
from uav_cfs.runtime import load_cfg


def _required_free(value: Any, path: str = "root") -> None:
    if value is None:
        raise ValueError(f"unset input at {path}")
    if isinstance(value, str) and "REQUIRED" in value:
        raise ValueError(f"unresolved author input at {path}: {value}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _required_free(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _required_free(item, f"{path}[{index}]")


def _resolve_path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _write_task_config(source: Path, output: Path, *, include_one_step: bool) -> None:
    root = copy.deepcopy(load_cfg(source))
    pipeline = root.setdefault("uav_pipeline", {})
    evaluation = pipeline.setdefault("evaluation", {})
    evaluation["include_one_step_loss"] = bool(include_one_step)
    evaluation["episodes_per_run"] = 30
    evaluation["output"] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = output.parent / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-config", required=True)
    parser.add_argument("--family", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output_root from the release configuration",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    release_path = Path(args.release_config).expanduser().resolve()
    release = load_cfg(release_path)
    _required_free(release)
    base = release_path.parent
    configs = release.get("configs")
    checkpoints = release.get("checkpoints")
    if not isinstance(configs, Mapping) or not isinstance(checkpoints, Mapping):
        raise TypeError("release config requires configs and checkpoints mappings")
    dataset = _resolve_path(base, release.get("dataset_index"), "dataset_index")
    perturbations = _resolve_path(
        base, release.get("perturbation_config"), "perturbation_config"
    )
    output_root = _resolve_path(
        base,
        args.output_root if args.output_root is not None else release.get("output_root"),
        "output_root",
    )
    episodes = int(release.get("episodes_per_run", 30))
    if episodes != 30:
        raise ValueError("paper main summaries require 30 held-out episodes per run")
    selected_families = set(args.family or ())
    selected_tasks = tuple(
        task
        for task in paper_task_matrix()
        if not selected_families or task.family in selected_families
    )
    if not selected_tasks:
        available = sorted({task.family for task in paper_task_matrix()})
        raise ValueError(
            f"no UAV tasks match --family; available families are {available}"
        )
    if (
        not args.dry_run
        and any(task.perturbation_name != "nominal" for task in selected_tasks)
    ):
        _required_free(load_cfg(perturbations), "structural_mismatches")
    commands: list[dict[str, Any]] = []
    evaluate_script = Path(__file__).with_name("evaluate.py").resolve()
    for task in selected_tasks:
        config_value = configs.get(task.config_key)
        checkpoint_value = checkpoints.get(task.checkpoint_key)
        config_path = _resolve_path(base, config_value, f"configs.{task.config_key}")
        checkpoint_path = _resolve_path(
            base, checkpoint_value, f"checkpoints.{task.checkpoint_key}"
        )
        task_dir = output_root / task.task_id
        bundle = task_dir / "evaluation.pt"
        _write_task_config(
            config_path, bundle, include_one_step=task.include_one_step_loss
        )
        resolved_config = task_dir / "resolved_config.yaml"
        command = [
            sys.executable,
            str(evaluate_script),
            "--cfg",
            str(resolved_config),
            "--data",
            str(dataset),
            "--ckpt",
            str(checkpoint_path),
            "--method",
            task.operator,
            "--out",
            str(bundle),
            "--episodes",
            str(episodes),
            "--modes",
            ",".join(task.substitution_modes),
            "--device",
            args.device,
            "--density-multiplier",
            str(task.density_multiplier),
            "--capacity-compression",
            str(task.capacity_compression),
            "--ablation",
            task.ablation,
        ]
        if task.perturbation_name != "nominal":
            command.extend(
                [
                    "--perturbation-config",
                    str(perturbations),
                    "--perturbation-name",
                    task.perturbation_name,
                ]
            )
        commands.append({"task": task.task_id, "command": command})
        if args.dry_run:
            print(shlex.join(command))
        else:
            subprocess.run(command, check=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "command_manifest.json").write_text(
        json.dumps(commands, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
