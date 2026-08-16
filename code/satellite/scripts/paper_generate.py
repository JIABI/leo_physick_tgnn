"""Generate typed action-coupled data for the paper Intensity--Flow core."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Sequence

from leo_pg.paper.dataset import (
    build_paper_dataset,
    generate_sharded_paper_dataset,
    save_paper_dataset,
)
from leo_pg.utils.config import load_cfg


def _split_ratios(value: str) -> tuple[float, float, float]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "split ratios must be three comma-separated numbers"
        ) from exc
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError(
            "split ratios must contain train,val,test values"
        )
    if any(not math.isfinite(item) or item < 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("split ratios must be finite and non-negative")
    if abs(sum(parsed) - 1.0) > 1e-8:
        raise argparse.ArgumentTypeError("split ratios must sum to 1.0")
    return parsed  # type: ignore[return-value]


def _split_counts(value: str) -> tuple[int, int, int]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "split counts must be three comma-separated integers"
        ) from exc
    if len(parsed) != 3 or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "split counts must contain non-negative train,val,test integers"
        )
    return parsed  # type: ignore[return-value]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _paper_data_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    value = cfg.get("paper_dataset", {})
    if not isinstance(value, dict):
        raise TypeError("paper_dataset must be a mapping when provided")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate tensor-only paper-protocol episodes under the fixed oracle "
            "behavior policy"
        )
    )
    parser.add_argument("--cfg", required=True, help="Paper protocol YAML configuration")
    parser.add_argument(
        "--out",
        default=None,
        help="Output index/bundle .pt path (overrides paper_dataset.output)",
    )
    parser.add_argument(
        "--storage-layout",
        choices=("episode_shards", "monolithic"),
        default=None,
        help=(
            "episode_shards writes a lightweight index and one file per episode; "
            "monolithic is retained only for small fixtures"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Episode count (overrides paper_dataset.episodes)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Decision epochs per episode (overrides paper_protocol.horizon_steps)",
    )
    parser.add_argument(
        "--split-ratios",
        type=_split_ratios,
        default=None,
        metavar="TRAIN,VAL,TEST",
        help="Episode split ratios (overrides paper_dataset.split_ratios)",
    )
    parser.add_argument(
        "--split-counts",
        type=_split_counts,
        default=None,
        metavar="TRAIN,VAL,TEST",
        help="Exact split counts; overrides ratios and must sum to --episodes",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Deterministic split seed (default: paper_dataset.split_seed or base seed)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="First episode seed (overrides cfg.seed)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Simulation device; serialized tensors are always moved to CPU",
    )
    args = parser.parse_args(argv)

    cfg = load_cfg(args.cfg)
    data_cfg = _paper_data_config(cfg)
    output_value = args.out or data_cfg.get("output")
    if output_value is None or not str(output_value).strip():
        parser.error("provide --out or paper_dataset.output")
    episode_count = _positive_int(
        "episodes",
        int(args.episodes if args.episodes is not None else data_cfg.get("episodes", 1)),
    )
    horizon = args.horizon
    if horizon is not None:
        _positive_int("horizon", horizon)

    configured_counts = data_cfg.get("split_counts")
    split_counts = args.split_counts
    if split_counts is None and configured_counts is not None:
        if not isinstance(configured_counts, (list, tuple)):
            raise TypeError("paper_dataset.split_counts must be a list")
        split_counts = _split_counts(",".join(str(value) for value in configured_counts))
    if split_counts is not None and sum(split_counts) != episode_count:
        parser.error("split counts must sum to the episode count")

    if args.split_ratios is not None:
        ratios = args.split_ratios
    else:
        configured_ratios = data_cfg.get("split_ratios", (0.8, 0.1, 0.1))
        if isinstance(configured_ratios, str):
            ratios = _split_ratios(configured_ratios)
        elif isinstance(configured_ratios, (list, tuple)):
            ratios = _split_ratios(",".join(str(value) for value in configured_ratios))
        else:
            raise TypeError("paper_dataset.split_ratios must be a string or list")

    configured_base_seed = int(cfg.get("seed", 7))
    base_seed = configured_base_seed if args.base_seed is None else int(args.base_seed)
    if base_seed < 0:
        parser.error("--base-seed must be non-negative")
    configured_split_seed = data_cfg.get("split_seed")
    split_seed = (
        int(args.split_seed)
        if args.split_seed is not None
        else int(configured_split_seed)
        if configured_split_seed is not None
        else base_seed
    )

    storage_layout = args.storage_layout or str(
        data_cfg.get("storage_layout", "episode_shards")
    )
    if storage_layout not in {"episode_shards", "monolithic"}:
        parser.error("paper_dataset.storage_layout must be episode_shards or monolithic")
    output = Path(str(output_value)).expanduser()
    generation_kwargs = {
        "episode_count": episode_count,
        "horizon_steps": horizon,
        "split_ratios": ratios,
        "split_counts": split_counts,
        "split_seed": split_seed,
        "base_seed": base_seed,
        "device": args.device,
    }
    if storage_layout == "episode_shards":
        payload = generate_sharded_paper_dataset(
            output,
            cfg,
            **generation_kwargs,
        )
    else:
        payload = build_paper_dataset(cfg, **generation_kwargs)
        output = save_paper_dataset(output, payload)
    split_manifest = payload["split_manifest"]
    split_sizes = {
        name: len(split_manifest["episode_indices"][name])
        for name in ("train", "val", "test")
    }
    print(
        f"[OK] wrote {output} | episodes={episode_count} "
        f"horizon={payload['generation']['horizon_steps']} "
        f"splits={split_sizes} storage={payload['storage']['layout']}"
    )


if __name__ == "__main__":
    main()
