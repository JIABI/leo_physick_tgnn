#!/usr/bin/env python3
"""Train one typed Snapshot model with atomic best/last checkpoints."""

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

from leo_pg.paper.snapshot import SnapshotFixedRankController
from leo_pg.paper.snapshot_config import (
    SnapshotTrainingSettings,
    canonical_sha256,
    resolve_snapshot_pipeline,
)
from leo_pg.paper.snapshot_data import SnapshotEpisodeDataset
from leo_pg.paper.snapshot_models import SNAPSHOT_METHODS, build_snapshot_model
from leo_pg.paper.snapshot_training import (
    SnapshotLossSummary,
    SnapshotTrainingAdapter,
)
from leo_pg.sim.paper_environment import PaperAlignedLEOEnv
from leo_pg.train.checkpoint import load_ckpt, save_ckpt
from leo_pg.utils.config import load_cfg
from leo_pg.utils.device import get_device
from leo_pg.utils.seed import set_seed


SNAPSHOT_TRAINER_STATE_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_paths(output: str | Path) -> tuple[Path, Path, Path]:
    requested = Path(output).expanduser()
    if requested.suffix.lower() == ".pt":
        run_dir = requested.parent
        if requested.name == "last.pt":
            return run_dir, run_dir / "best.pt", requested
        return run_dir, requested, run_dir / "last.pt"
    return requested, requested / "best.pt", requested / "last.pt"


def _atomic_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    **metadata: Any,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        save_ckpt(str(temporary), model, optimizer, **metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _capture_rng() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            [value.cpu() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng(value: Mapping[str, Any]) -> None:
    cpu = value.get("torch_cpu")
    if not isinstance(cpu, torch.Tensor):
        raise ValueError("resume checkpoint is missing torch CPU RNG state")
    torch.set_rng_state(cpu.cpu())
    cuda = value.get("torch_cuda", [])
    if torch.cuda.is_available() and isinstance(cuda, list) and cuda:
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(state).cpu() for state in cuda]
        )


def _make_optimizer(
    model: torch.nn.Module, settings: SnapshotTrainingSettings
) -> torch.optim.Optimizer:
    optimizer_class = (
        torch.optim.AdamW
        if settings.optimizer_name == "adamw"
        else torch.optim.Adam
    )
    return optimizer_class(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )


def _aggregate_summaries(
    summaries: Sequence[SnapshotLossSummary],
) -> SnapshotLossSummary:
    if not summaries:
        raise ValueError("cannot aggregate an empty Snapshot summary list")
    transitions = sum(item.transitions for item in summaries)
    if transitions <= 0:
        raise ValueError("Snapshot summaries contain no transitions")

    def weighted(name: str) -> float:
        return sum(
            float(getattr(item, name)) * item.transitions for item in summaries
        ) / transitions

    return SnapshotLossSummary(
        total=weighted("total"),
        gamma=weighted("gamma"),
        feasibility_margin=weighted("feasibility_margin"),
        admitted_load=weighted("admitted_load"),
        decision_aware=weighted("decision_aware"),
        transitions=transitions,
        persistent_edges=sum(item.persistent_edges for item in summaries),
        model_rollin_steps=sum(item.model_rollin_steps for item in summaries),
        rollin_opportunities=sum(item.rollin_opportunities for item in summaries),
    )


def _budgeted_seed_passes(
    seeds: Sequence[int],
    *,
    multiplier: float,
    generator: torch.Generator,
) -> tuple[int, ...]:
    """Expand a seed grid to the declared training-budget multiplier."""

    value = float(multiplier)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("training budget_multiplier must be finite and positive")
    if not seeds:
        raise ValueError("training budget requires non-empty seeds")
    complete = int(math.floor(value))
    fractional_count = int(round((value - complete) * len(seeds)))
    expanded = [int(seed) for _ in range(complete) for seed in seeds]
    if fractional_count:
        selection = torch.randperm(len(seeds), generator=generator).tolist()
        expanded.extend(int(seeds[index]) for index in selection[:fractional_count])
    if not expanded:
        selection = torch.randperm(len(seeds), generator=generator).tolist()
        expanded.extend(int(seeds[index]) for index in selection[:1])
    order = torch.randperm(len(expanded), generator=generator).tolist()
    return tuple(expanded[index] for index in order)


def _scheduled_rollin_probability(
    settings: SnapshotTrainingSettings,
    epoch: int,
) -> float:
    if epoch <= settings.rollin_start_epoch:
        return settings.rollin_start_probability
    if epoch >= settings.rollin_end_epoch:
        return settings.rollin_maximum_probability
    span = max(1, settings.rollin_end_epoch - settings.rollin_start_epoch)
    alpha = (epoch - settings.rollin_start_epoch) / span
    return settings.rollin_start_probability + alpha * (
        settings.rollin_maximum_probability
        - settings.rollin_start_probability
    )


def _action_coupled_epoch(
    *,
    adapter: SnapshotTrainingAdapter,
    pipeline: Any,
    seeds: Sequence[int],
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    rollout_horizon: int,
    sequence_batch_size: int,
    device: torch.device,
) -> SnapshotLossSummary:
    """True H-step action-coupled loss used by Snapshot LTT-R."""

    if not seeds:
        raise ValueError("action-coupled Snapshot epoch requires non-empty seeds")
    training = optimizer is not None
    adapter.model.train(training)
    summaries: list[SnapshotLossSummary] = []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for start in range(0, len(seeds), sequence_batch_size):
            batch = tuple(
                int(seed) for seed in seeds[start : start + sequence_batch_size]
            )
            iterators = []
            for seed in batch:
                episode_cfg = copy.deepcopy(pipeline.resolved_model_config)
                episode_cfg["seed"] = seed
                environment = PaperAlignedLEOEnv(episode_cfg, device=device)
                iterators.append(
                    adapter.action_coupled_windows(
                        environment,
                        controller=SnapshotFixedRankController(
                            pipeline.policy_config
                        ),
                        margin_config=pipeline.margin_config,
                        feature_config=pipeline.feature_config,
                        d0_initializer=pipeline.initializer,
                        new_edge_initializer=pipeline.initializer,
                        rollout_horizon=rollout_horizon,
                        model_rollin_probability=1.0,
                    )
                )
            active = list(iterators)
            while active:
                window_results = []
                still_active = []
                for iterator in active:
                    try:
                        window_results.append(next(iterator))
                        still_active.append(iterator)
                    except StopIteration:
                        continue
                active = still_active
                if not window_results:
                    break
                summaries.extend(result.summary for result in window_results)
                if training:
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
                    objective = torch.stack(
                        [result.objective for result in window_results]
                    ).mean()
                    objective.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        adapter.model.parameters(), gradient_clip_norm
                    )
                    if not bool(torch.isfinite(torch.as_tensor(norm))):
                        raise FloatingPointError(
                            "Snapshot rollout gradient is NaN or Inf"
                        )
                    optimizer.step()
    return _aggregate_summaries(summaries)


class _PermutedEpisodeView(Sequence[Mapping[str, Any]]):
    """Lazy permutation; shards are loaded only when the adapter requests one."""

    def __init__(self, dataset: SnapshotEpisodeDataset, order: Sequence[int]) -> None:
        self.dataset = dataset
        self.order = tuple(int(index) for index in order)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self.dataset[item] for item in self.order[index]]
        return self.dataset[self.order[index]]


def _ordered_episodes(
    dataset: SnapshotEpisodeDataset,
    generator: torch.Generator,
) -> _PermutedEpisodeView:
    order = torch.randperm(len(dataset), generator=generator).tolist()
    return _PermutedEpisodeView(dataset, order)


def _dataset_metadata(
    dataset: SnapshotEpisodeDataset,
    path: Path,
    *,
    method: str,
    pipeline_fingerprint: str,
    expected_policy: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dataset.payload
    if payload.get("snapshot_method") != method:
        raise ValueError(
            "formal Snapshot training requires an index generated for the exact "
            f"method; expected {method!r}, got {payload.get('snapshot_method')!r}"
        )
    if payload.get("formal_factorial_training") is not True:
        raise ValueError("Snapshot dataset is not marked formal_factorial_training")
    protocol = payload.get("snapshot_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Snapshot dataset index has no snapshot_protocol")
    saved_policy = protocol.get("policy")
    if not isinstance(saved_policy, Mapping):
        raise ValueError("Snapshot dataset index has no formal behavior policy")
    if canonical_sha256(saved_policy) != canonical_sha256(expected_policy):
        raise ValueError(
            "Snapshot dataset behavior policy differs from the selected method policy"
        )
    saved_pipeline = payload.get("snapshot_pipeline_fingerprint")
    if saved_pipeline != pipeline_fingerprint:
        raise ValueError(
            "Snapshot dataset pipeline fingerprint differs from the training config"
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "snapshot_method": method,
        "dataset_kind": payload.get("dataset_kind"),
        "snapshot_dataset_schema_version": payload.get(
            "snapshot_dataset_schema_version"
        ),
        "snapshot_feature_contract_version": payload.get(
            "snapshot_feature_contract_version"
        ),
        "snapshot_target_contract_version": payload.get(
            "snapshot_target_contract_version"
        ),
        "snapshot_pipeline_fingerprint": saved_pipeline,
        "split_manifest": payload.get("split_manifest"),
    }


def _resume(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    method: str,
    pipeline_fingerprint: str,
    training_fingerprint: str,
    objective_fingerprint: str,
    dataset_sha256: str,
    shuffle_generator: torch.Generator,
    rollin_generator: torch.Generator,
    validation_generator: torch.Generator,
) -> tuple[int, float, int, list[dict[str, Any]]]:
    payload = load_ckpt(
        str(path),
        model,
        optimizer,
        map_location=device,
        strict=True,
    )
    if payload.get("paper_method") != method:
        raise ValueError("resume checkpoint method does not match --method")
    if payload.get("paper_interface") != "snapshot_v1":
        raise ValueError("resume checkpoint is not a Snapshot checkpoint")
    state = payload.get("snapshot_trainer_state")
    if not isinstance(state, Mapping):
        raise ValueError("resume checkpoint has no snapshot_trainer_state")
    if state.get("schema_version") != SNAPSHOT_TRAINER_STATE_VERSION:
        raise ValueError("resume checkpoint trainer-state version is incompatible")
    expected = {
        "method": method,
        "pipeline_fingerprint": pipeline_fingerprint,
        "training_fingerprint": training_fingerprint,
        "objective_fingerprint": objective_fingerprint,
        "dataset_sha256": dataset_sha256,
    }
    for name, value in expected.items():
        if state.get(name) != value:
            raise ValueError(f"resume checkpoint {name} changed")
    _restore_rng(state.get("rng_state", {}))
    for name, generator in (
        ("shuffle_generator_state", shuffle_generator),
        ("rollin_generator_state", rollin_generator),
        ("validation_generator_state", validation_generator),
    ):
        value = state.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"resume checkpoint is missing {name}")
        generator.set_state(value.cpu())
    history = state.get("history")
    if not isinstance(history, list) or not all(
        isinstance(item, Mapping) for item in history
    ):
        raise ValueError("resume checkpoint history is invalid")
    epoch = int(state.get("epoch", 0))
    best_validation = float(state.get("best_validation", math.inf))
    best_epoch = int(state.get("best_epoch", 0))
    if epoch < 1 or not math.isfinite(best_validation) or best_epoch < 1:
        raise ValueError("resume checkpoint progress metadata is invalid")
    return (
        epoch + 1,
        best_validation,
        best_epoch,
        [dict(item) for item in history],
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train one formal Snapshot model with best/last checkpoints"
    )
    parser.add_argument("--cfg", required=True, help="Paper YAML configuration")
    parser.add_argument("--method", required=True, choices=SNAPSHOT_METHODS)
    parser.add_argument("--data", required=True, help="Method-specific dataset index .pt")
    parser.add_argument(
        "--out",
        required=True,
        help="Run directory or selected best .pt path; last.pt is always separate",
    )
    parser.add_argument("--resume", default=None, help="Resume from last.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--training-objective",
        choices=("configured", "teacher_forcing", "scheduled_sampling", "action_coupled"),
        default="configured",
        help="Explicit objective override used by the rollout-aware ladder",
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=None,
        help="Required H for an explicit action_coupled objective",
    )
    parser.add_argument(
        "--budget-multiplier",
        type=float,
        default=None,
        help="Override the declared complete-episode training-pass multiplier",
    )
    args = parser.parse_args(argv)

    root = load_cfg(args.cfg)
    pipeline = resolve_snapshot_pipeline(root, args.method)
    settings = SnapshotTrainingSettings.from_config(root)
    data_path = Path(args.data).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(f"Snapshot dataset does not exist: {data_path}")
    train_dataset = SnapshotEpisodeDataset(data_path, split="train")
    validation_dataset = SnapshotEpisodeDataset(data_path, split="val")
    metadata = _dataset_metadata(
        train_dataset,
        data_path,
        method=pipeline.method,
        pipeline_fingerprint=pipeline.fingerprint,
        expected_policy=asdict(pipeline.policy_config),
    )
    if canonical_sha256(train_dataset.payload) != canonical_sha256(
        validation_dataset.payload
    ):
        raise RuntimeError("train and validation Snapshot indices disagree")
    split_manifest = train_dataset.payload.get("split_manifest")
    if not isinstance(split_manifest, Mapping):
        raise ValueError("Snapshot dataset index has no split manifest")
    split_seeds = split_manifest.get("episode_seeds")
    if not isinstance(split_seeds, Mapping):
        raise ValueError("Snapshot split manifest has no episode seeds")
    action_train_seeds = tuple(int(value) for value in split_seeds.get("train", ()))
    action_validation_seeds = tuple(int(value) for value in split_seeds.get("val", ()))

    baseline = pipeline.resolved_model_config.get("paper_baseline", {})
    baseline_training = (
        baseline.get("training", {}) if isinstance(baseline, Mapping) else {}
    )
    if not isinstance(baseline_training, Mapping):
        raise TypeError("paper_baseline.training must be a mapping")
    objective_kind = str(
        baseline_training.get("objective", "one_step_teacher_forcing")
    ).strip().lower()
    global_training = root.get("paper_training", {})
    if not isinstance(global_training, Mapping):
        raise TypeError("paper_training must be a mapping")
    global_action = global_training.get("action_coupled_multistep", {})
    if not isinstance(global_action, Mapping):
        raise TypeError("paper_training.action_coupled_multistep must be a mapping")
    global_sampling = global_training.get("scheduled_sampling", {})
    if not isinstance(global_sampling, Mapping):
        raise TypeError("paper_training.scheduled_sampling must be a mapping")
    if not baseline_training:
        if global_action.get("enabled") is True:
            objective_kind = "action_coupled_multistep"
        elif global_sampling.get("enabled") is True:
            objective_kind = "one_step_scheduled_sampling"
    objective_overrides = {
        "teacher_forcing": "one_step_teacher_forcing",
        "scheduled_sampling": "one_step_scheduled_sampling",
        "action_coupled": "action_coupled_multistep",
    }
    if args.training_objective != "configured":
        objective_kind = objective_overrides[args.training_objective]
    use_action_coupled_rollout = objective_kind in {
        "action_coupled_multistep",
        "decision_aware_action_coupled",
    }
    supported_objectives = {
        "one_step_teacher_forcing",
        "one_step_scheduled_sampling",
        "action_coupled_multistep",
        "decision_aware_action_coupled",
    }
    if objective_kind not in supported_objectives:
        raise ValueError(f"unsupported Snapshot training objective: {objective_kind}")
    if pipeline.method == "snapshot_ltt_r" and not use_action_coupled_rollout:
        raise ValueError(
            "snapshot_ltt_r requires paper_baselines.ltt_r.training.objective="
            "action_coupled_multistep"
        )
    configured_rollout_horizon = baseline_training.get("rollout_horizon")
    if args.rollout_horizon is not None:
        configured_rollout_horizon = args.rollout_horizon
    if configured_rollout_horizon is None and use_action_coupled_rollout:
        configured_rollout_horizon = global_action.get("rollout_horizon")
    if configured_rollout_horizon is None and use_action_coupled_rollout:
        horizons = global_action.get("horizons")
        if isinstance(horizons, Sequence) and len(horizons) == 1:
            configured_rollout_horizon = horizons[0]
    rollout_horizon = int(configured_rollout_horizon or 1)
    if use_action_coupled_rollout and rollout_horizon <= 1:
        raise ValueError("action-coupled Snapshot rollout_horizon must exceed one")
    if args.rollout_horizon is not None and not use_action_coupled_rollout:
        raise ValueError("--rollout-horizon requires an action_coupled objective")
    budget_multiplier = float(
        args.budget_multiplier
        if args.budget_multiplier is not None
        else baseline_training.get("budget_multiplier", 1.0)
    )
    if not math.isfinite(budget_multiplier) or budget_multiplier <= 0.0:
        raise ValueError("paper_baseline.training.budget_multiplier must be positive")
    sequence_batch_size = int(
        baseline_training.get(
            "rollout_sequence_batch_size",
            max(
                1,
                int(
                    round(
                        settings.batch_size_control_graphs
                        / max(1, rollout_horizon)
                    )
                ),
            ),
        )
    )
    if sequence_batch_size <= 0:
        raise ValueError("rollout_sequence_batch_size must be positive")
    objective_manifest = {
        "kind": objective_kind,
        "cli_objective_override": args.training_objective,
        "rollout_horizon": rollout_horizon,
        "rollout_sequence_batch_size": sequence_batch_size,
        "training_budget_multiplier": budget_multiplier,
        "training_budget_interpretation": (
            "complete_action_coupled_episode_passes_relative_to_snapshot_physick"
            if use_action_coupled_rollout
            else "one_recorded_dataset_pass"
        ),
        "rollout_coverage": (
            "complete_episode_partitioned_into_truncated_windows"
            if use_action_coupled_rollout
            else "recorded_one_step_transitions"
        ),
        "model_feedback_probability": 1.0 if use_action_coupled_rollout else None,
        "scheduled_sampling": (
            {
                "start_probability": settings.rollin_start_probability,
                "maximum_probability": settings.rollin_maximum_probability,
                "start_epoch": settings.rollin_start_epoch,
                "end_epoch": settings.rollin_end_epoch,
                "forced_by_cli": args.training_objective == "scheduled_sampling",
            }
            if objective_kind == "one_step_scheduled_sampling"
            else None
        ),
        "tbptt": (
            f"truncated_action_coupled_window_{rollout_horizon}"
            if use_action_coupled_rollout
            else "one_step"
        ),
    }
    objective_fingerprint = canonical_sha256(objective_manifest)

    set_seed(settings.seed)
    requested_device = args.device or settings.device
    device = get_device(requested_device, strict=args.device is not None)
    model = build_snapshot_model(pipeline.resolved_model_config, pipeline.method)
    model_cfg = getattr(model, "cfg", None)
    if not isinstance(model_cfg, dict):
        raise RuntimeError("Snapshot model factory did not attach resolved config")
    adapter = SnapshotTrainingAdapter(
        model=model,
        loss_weights=pipeline.loss_weights,
        decision_aware_weight=pipeline.effective_decision_aware_weight,
        device=device,
    )
    optimizer = _make_optimizer(model, settings)
    shuffle_generator = torch.Generator(device="cpu")
    rollin_generator = torch.Generator(device="cpu")
    validation_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(settings.seed + 10_001)
    rollin_generator.manual_seed(settings.seed + 20_003)
    validation_generator.manual_seed(settings.seed + 30_007)

    run_dir, best_path, last_path = _checkpoint_paths(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_validation = math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        start_epoch, best_validation, best_epoch, history = _resume(
            resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            method=pipeline.method,
            pipeline_fingerprint=pipeline.fingerprint,
            training_fingerprint=settings.fingerprint,
            objective_fingerprint=objective_fingerprint,
            dataset_sha256=metadata["sha256"],
            shuffle_generator=shuffle_generator,
            rollin_generator=rollin_generator,
            validation_generator=validation_generator,
        )
        if best_epoch > 0 and not best_path.exists():
            raise FileNotFoundError(
                "resume requires the prior best.pt in the selected --out run directory"
            )

    for epoch in range(start_epoch, settings.epochs + 1):
        train_probability = settings.train_rollin_probability(epoch)
        if objective_kind == "one_step_teacher_forcing":
            train_probability = 0.0
        elif objective_kind == "one_step_scheduled_sampling":
            train_probability = _scheduled_rollin_probability(settings, epoch)
        if use_action_coupled_rollout:
            ordered_seeds = _budgeted_seed_passes(
                action_train_seeds,
                multiplier=budget_multiplier,
                generator=shuffle_generator,
            )
            train_summary = _action_coupled_epoch(
                adapter=adapter,
                pipeline=pipeline,
                seeds=ordered_seeds,
                optimizer=optimizer,
                gradient_clip_norm=settings.gradient_clip_norm,
                rollout_horizon=rollout_horizon,
                sequence_batch_size=sequence_batch_size,
                device=device,
            )
            validation_summary = _action_coupled_epoch(
                adapter=adapter,
                pipeline=pipeline,
                seeds=action_validation_seeds,
                optimizer=None,
                gradient_clip_norm=settings.gradient_clip_norm,
                rollout_horizon=rollout_horizon,
                sequence_batch_size=sequence_batch_size,
                device=device,
            )
            train_probability = 1.0
        else:
            train_summary = adapter.train_epoch(
                _ordered_episodes(train_dataset, shuffle_generator),
                optimizer=optimizer,
                clip_grad_norm=settings.gradient_clip_norm,
                batch_size_control_graphs=settings.batch_size_control_graphs,
                model_rollin_probability=train_probability,
                initializer=pipeline.initializer,
                generator=rollin_generator,
            )
            validation_summary = adapter.evaluate_epoch(
                validation_dataset,
                model_rollin_probability=settings.validation_rollin_probability,
                initializer=pipeline.initializer,
                generator=validation_generator,
            )
        selection = float(validation_summary.total)
        if not math.isfinite(selection):
            raise FloatingPointError("Snapshot validation loss is NaN or Inf")
        improved = selection < best_validation
        if improved:
            best_validation = selection
            best_epoch = epoch
        record = {
            "epoch": epoch,
            "train": asdict(train_summary),
            "validation": asdict(validation_summary),
            "train_model_rollin_probability": train_probability,
            "validation_model_rollin_probability": (
                settings.validation_rollin_probability
            ),
            "selection_loss": selection,
            "is_best": improved,
            "objective": objective_manifest,
        }
        history.append(record)
        trainer_state = {
            "schema_version": SNAPSHOT_TRAINER_STATE_VERSION,
            "method": pipeline.method,
            "epoch": epoch,
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "history": history,
            "pipeline_fingerprint": pipeline.fingerprint,
            "training_fingerprint": settings.fingerprint,
            "objective_fingerprint": objective_fingerprint,
            "dataset_sha256": metadata["sha256"],
            "rng_state": _capture_rng(),
            "shuffle_generator_state": shuffle_generator.get_state().cpu(),
            "rollin_generator_state": rollin_generator.get_state().cpu(),
            "validation_generator_state": validation_generator.get_state().cpu(),
        }
        checkpoint_metadata = {
            "config": model_cfg,
            "paper_method": pipeline.method,
            "paper_interface": "snapshot_v1",
            "snapshot_pipeline": pipeline.manifest(),
            "snapshot_pipeline_fingerprint": pipeline.fingerprint,
            "snapshot_training": settings.manifest(),
            "snapshot_training_fingerprint": settings.fingerprint,
            "snapshot_objective": objective_manifest,
            "snapshot_objective_fingerprint": objective_fingerprint,
            "dataset": metadata,
            "snapshot_trainer_state": trainer_state,
        }
        if improved:
            _atomic_checkpoint(
                best_path, model, optimizer, **checkpoint_metadata
            )
        _atomic_checkpoint(last_path, model, optimizer, **checkpoint_metadata)
        print("[EPOCH] " + json.dumps(record, sort_keys=True, separators=(",", ":")))

    if best_epoch < 1 or not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("Snapshot training produced incomplete best/last checkpoints")
    print(
        f"[OK] method={pipeline.method} best_epoch={best_epoch} "
        f"validation={best_validation:.8g} best={best_path} last={last_path}"
    )


if __name__ == "__main__":
    main()
