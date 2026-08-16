#!/usr/bin/env python3
"""Generate method-specific formal Snapshot-oracle episode shards."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from leo_pg.paper.snapshot_config import resolve_snapshot_pipeline
from leo_pg.paper.snapshot_data import generate_sharded_snapshot_dataset
from leo_pg.paper.snapshot_models import SNAPSHOT_METHODS
from leo_pg.utils.config import load_cfg


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
            "split ratios must be three finite non-negative values"
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


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return dict(value)


def _positive(name: str, value: Any) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _configured_triplet(
    value: Any,
    parser: Any,
    *,
    path: str,
) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return parser(value)
    if isinstance(value, (list, tuple)):
        return parser(",".join(str(item) for item in value))
    raise TypeError(f"{path} must be a string or three-value list")


def _atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate formal, method-specific Snapshot-oracle shards"
    )
    parser.add_argument("--cfg", required=True, help="Paper YAML configuration")
    parser.add_argument("--method", required=True, choices=SNAPSHOT_METHODS)
    parser.add_argument(
        "--out",
        default=None,
        help="Snapshot dataset index .pt; defaults to output_template for the method",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split-ratios",
        type=_split_ratios,
        metavar="TRAIN,VAL,TEST",
        default=None,
    )
    split_group.add_argument(
        "--split-counts",
        type=_split_counts,
        metavar="TRAIN,VAL,TEST",
        default=None,
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    root = load_cfg(args.cfg)
    pipeline = resolve_snapshot_pipeline(root, args.method)
    snapshot_cfg = _mapping(root, "paper_snapshot")
    dataset_cfg = _mapping(snapshot_cfg, "dataset")
    shared_dataset_cfg = _mapping(root, "paper_dataset")
    output = pipeline.dataset_path(args.out)

    configured_episodes = dataset_cfg.get(
        "episodes", shared_dataset_cfg.get("episodes")
    )
    if args.episodes is None and configured_episodes is None:
        parser.error(
            "provide --episodes or paper_snapshot.dataset.episodes/paper_dataset.episodes"
        )
    episodes = _positive(
        "episodes",
        args.episodes if args.episodes is not None else configured_episodes,
    )
    horizon = args.horizon
    if horizon is None:
        horizon = dataset_cfg.get("horizon_steps")
    if horizon is not None:
        horizon = _positive("horizon", horizon)

    configured_counts = _configured_triplet(
        dataset_cfg.get("split_counts", shared_dataset_cfg.get("split_counts")),
        _split_counts,
        path="Snapshot dataset split_counts",
    )
    configured_ratios = _configured_triplet(
        dataset_cfg.get(
            "split_ratios", shared_dataset_cfg.get("split_ratios", (0.8, 0.1, 0.1))
        ),
        _split_ratios,
        path="Snapshot dataset split_ratios",
    )
    split_counts = args.split_counts
    ratios = args.split_ratios
    if split_counts is None and ratios is None:
        split_counts = configured_counts
        ratios = configured_ratios
    elif split_counts is not None:
        ratios = configured_ratios
    else:
        split_counts = None
    if split_counts is not None and sum(split_counts) != episodes:
        parser.error("split counts must sum to --episodes")
    if ratios is None:
        ratios = (0.8, 0.1, 0.1)

    base_seed = int(root.get("seed", 7) if args.base_seed is None else args.base_seed)
    if base_seed < 0:
        parser.error("--base-seed must be non-negative")
    configured_split_seed = dataset_cfg.get(
        "split_seed", shared_dataset_cfg.get("split_seed")
    )
    split_seed = (
        int(args.split_seed)
        if args.split_seed is not None
        else int(configured_split_seed)
        if configured_split_seed is not None
        else base_seed
    )
    if split_seed < 0:
        parser.error("--split-seed must be non-negative")

    payload = generate_sharded_snapshot_dataset(
        output,
        root,
        episode_count=episodes,
        margin_config=pipeline.margin_config,
        feature_config=pipeline.feature_config,
        policy_config=pipeline.policy_config,
        initial_admitted_load=pipeline.initial_oracle_admitted_load,
        horizon_steps=horizon,
        split_ratios=ratios,
        split_counts=split_counts,
        split_seed=split_seed,
        base_seed=base_seed,
        device=args.device,
    )
    # The shared shard generator already embeds the complete behavior-policy
    # config in every episode.  These index fields make the selected policy cell
    # explicit even when two methods happen to use equal numerical weights.
    payload["snapshot_method"] = pipeline.method
    payload["formal_factorial_training"] = True
    payload["snapshot_pipeline"] = pipeline.manifest()
    payload["snapshot_pipeline_fingerprint"] = pipeline.fingerprint
    _atomic_save(output, payload)
    sizes = {
        split: len(payload["split_manifest"]["episode_indices"][split])
        for split in ("train", "val", "test")
    }
    print(
        f"[OK] method={pipeline.method} data={output} episodes={episodes} "
        f"splits={sizes} shards={len(payload['episode_shards'])}"
    )


if __name__ == "__main__":
    main()
