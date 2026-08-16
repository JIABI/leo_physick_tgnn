#!/usr/bin/env python3
"""Generate sharded, action-coupled UAV/shared-service training data."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from uav_cfs.data import generate_sharded_uav_dataset
from uav_cfs.policy import UAVFixedRankPolicyConfig
from uav_cfs.runtime import load_cfg


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return dict(value)


def _split_ratios(value: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "split ratios must be comma-separated numbers"
        ) from exc
    if len(result) != 3 or any(
        not math.isfinite(item) or item < 0.0 for item in result
    ):
        raise argparse.ArgumentTypeError(
            "split ratios must be three finite non-negative numbers"
        )
    if not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise argparse.ArgumentTypeError("split ratios must sum to one")
    return result  # type: ignore[return-value]


def _split_counts(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "split counts must be comma-separated integers"
        ) from exc
    if len(result) != 3 or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError(
            "split counts must be three non-negative integers"
        )
    return result  # type: ignore[return-value]


def _run_seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "run seeds must be comma-separated integers"
        ) from exc
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError(
            "run seeds must be non-empty and non-negative"
        )
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("run seeds must be unique")
    return result


def _configured_triplet(
    value: Any,
    parser: Any,
    *,
    name: str,
) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return parser(value)
    if isinstance(value, (list, tuple)):
        return parser(",".join(str(item) for item in value))
    raise TypeError(f"{name} must be a string or three-value sequence")


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _finite_non_negative(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _optional_hysteresis(value: str) -> float | None:
    if value.strip().lower() in {"none", "null", "adaptive"}:
        return None
    try:
        return _finite_non_negative("hysteresis", float(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "hysteresis must be non-negative or 'none'"
        ) from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate weights-only-safe UAV graph/target episode shards under "
            "the fixed simulator-oracle behavior policy"
        )
    )
    parser.add_argument("--cfg", required=True, help="Paper protocol YAML")
    parser.add_argument(
        "--out",
        default=None,
        help="Dataset index .pt (overrides uav_pipeline.dataset.output)",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Training decision horizon (formal configuration: 60)",
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split-counts",
        type=_split_counts,
        metavar="TRAIN,VAL,TEST",
        default=None,
    )
    split_group.add_argument(
        "--split-ratios",
        type=_split_ratios,
        metavar="TRAIN,VAL,TEST",
        default=None,
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument(
        "--run-seeds",
        type=_run_seeds,
        default=None,
        metavar="SEED,...",
        help="Formal independent run seeds (overrides uav_pipeline.dataset.run_seeds)",
    )
    parser.add_argument(
        "--split-counts-per-run",
        type=_split_counts,
        default=None,
        metavar="TRAIN,VAL,TEST",
        help="Exact counts inside each formal run",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--eta-weight", type=float, default=None)
    parser.add_argument("--intensity-weight", type=float, default=None)
    parser.add_argument("--flow-weight", type=float, default=None)
    parser.add_argument(
        "--hard-feasible-start-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the behavior-policy hard mask sensitivity switch",
    )
    parser.add_argument(
        "--hysteresis",
        type=_optional_hysteresis,
        default=argparse.SUPPRESS,
        help="Non-negative score margin or 'none' for adaptive 1/K",
    )
    args = parser.parse_args(argv)

    root = load_cfg(args.cfg)
    pipeline = _mapping(root, "uav_pipeline")
    dataset_cfg = _mapping(pipeline, "dataset")

    output_value = args.out if args.out is not None else dataset_cfg.get(
        "output", "artifacts/uav/uav_dataset.pt"
    )
    output = Path(str(output_value)).expanduser()

    configured_counts = _configured_triplet(
        dataset_cfg.get("split_counts"),
        _split_counts,
        name="uav_pipeline.dataset.split_counts",
    )
    configured_ratios = _configured_triplet(
        dataset_cfg.get("split_ratios", (0.8, 0.1, 0.1)),
        _split_ratios,
        name="uav_pipeline.dataset.split_ratios",
    )
    single_run_requested = any(
        value is not None
        for value in (args.episodes, args.split_counts, args.split_ratios)
    )
    if single_run_requested and (
        args.run_seeds is not None or args.split_counts_per_run is not None
    ):
        parser.error(
            "single-run --episodes/--split-* cannot be combined with formal per-run options"
        )
    if single_run_requested:
        split_counts = args.split_counts or configured_counts
        split_ratios = args.split_ratios or configured_ratios or (0.8, 0.1, 0.1)
        configured_episodes = dataset_cfg.get("episodes")
        if args.episodes is not None:
            episode_count: int | None = _positive_int("episodes", args.episodes)
        elif configured_episodes is not None:
            episode_count = _positive_int(
                "uav_pipeline.dataset.episodes", configured_episodes
            )
        elif split_counts is not None:
            episode_count = sum(split_counts)
        else:
            parser.error("single-run generation requires --episodes")
        if split_counts is not None and sum(split_counts) != episode_count:
            parser.error("split counts must sum to the episode count")
        formal_run_seeds = None
        counts_per_run = None
    else:
        configured_run_seeds = dataset_cfg.get("run_seeds")
        if args.run_seeds is not None:
            formal_run_seeds = args.run_seeds
        elif isinstance(configured_run_seeds, (list, tuple)):
            formal_run_seeds = _run_seeds(
                ",".join(str(value) for value in configured_run_seeds)
            )
        elif isinstance(configured_run_seeds, str):
            formal_run_seeds = _run_seeds(configured_run_seeds)
        else:
            parser.error("uav_pipeline.dataset.run_seeds must be configured")
        if len(formal_run_seeds) != 5:
            parser.error(
                "the formal paper dataset requires exactly five independent run seeds"
            )
        configured_per_run = (
            dataset_cfg.get("train_episodes_per_run", 240),
            dataset_cfg.get("validation_episodes_per_run", 40),
            dataset_cfg.get("test_episodes_per_run", 80),
        )
        counts_per_run = args.split_counts_per_run or _split_counts(
            ",".join(str(value) for value in configured_per_run)
        )
        episode_count = len(formal_run_seeds) * sum(counts_per_run)
        split_counts = None
        split_ratios = configured_ratios or (0.8, 0.1, 0.1)

    configured_horizon = dataset_cfg.get(
        "train_horizon_steps",
        dataset_cfg.get("train_horizon", dataset_cfg.get("horizon_steps", 60)),
    )
    horizon = _positive_int(
        "horizon",
        args.horizon if args.horizon is not None else configured_horizon,
    )
    uav_section = _mapping(root, "uav_shared")
    configured_base_seed = dataset_cfg.get(
        "base_seed", uav_section.get("episode_seed", root.get("seed", 7))
    )
    base_seed = _non_negative_int(
        "base_seed",
        args.base_seed if args.base_seed is not None else configured_base_seed,
    )
    configured_split_seed = dataset_cfg.get("split_seed", base_seed)
    split_seed = _non_negative_int(
        "split_seed",
        args.split_seed if args.split_seed is not None else configured_split_seed,
    )

    configured_policy = UAVFixedRankPolicyConfig.from_source(root)
    eta_weight = _finite_non_negative(
        "eta_weight",
        args.eta_weight
        if args.eta_weight is not None
        else configured_policy.eta_weight,
    )
    intensity_weight = _finite_non_negative(
        "intensity_weight",
        args.intensity_weight
        if args.intensity_weight is not None
        else configured_policy.intensity_weight,
    )
    flow_weight = _finite_non_negative(
        "flow_weight",
        args.flow_weight
        if args.flow_weight is not None
        else configured_policy.flow_weight,
    )
    hard_mask_value = (
        args.hard_feasible_start_mask
        if args.hard_feasible_start_mask is not None
        else configured_policy.hard_feasible_start_mask
    )
    if type(hard_mask_value) is not bool:
        raise TypeError("uav_pipeline.policy.hard_feasible_start_mask must be bool")
    if hasattr(args, "hysteresis"):
        hysteresis = args.hysteresis
    else:
        configured_hysteresis = configured_policy.hysteresis
        hysteresis = (
            None
            if configured_hysteresis is None
            else _finite_non_negative("hysteresis", configured_hysteresis)
        )
    policy = replace(
        configured_policy,
        eta_weight=eta_weight,
        intensity_weight=intensity_weight,
        flow_weight=flow_weight,
        hard_feasible_start_mask=hard_mask_value,
        hysteresis=hysteresis,
    )

    payload = generate_sharded_uav_dataset(
        output,
        root,
        episode_count=episode_count,
        horizon_steps=horizon,
        split_ratios=split_ratios,
        split_counts=split_counts,
        split_seed=split_seed,
        base_seed=base_seed,
        policy_config=policy,
        run_seeds=formal_run_seeds,
        split_counts_per_run=counts_per_run,
        device=args.device,
    )
    sizes = {
        split: len(payload["split_manifest"]["episode_indices"][split])
        for split in ("train", "val", "test")
    }
    print(
        f"[OK] UAV data={output} episodes={episode_count} horizon={horizon} "
        f"runs={len(formal_run_seeds) if formal_run_seeds else 1} "
        f"splits={sizes} shards={len(payload['episode_shards'])}"
    )


if __name__ == "__main__":
    main()
