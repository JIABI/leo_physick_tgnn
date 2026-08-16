"""Generate the explicit matched held-out panel used by DEC--K."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from leo_pg.paper.dataset import generate_sharded_paper_dataset
from leo_pg.utils.config import load_cfg


def _seeds(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("seeds must be non-empty and non-negative")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=_seeds, required=True)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.horizon <= 1:
        raise ValueError("horizon must exceed one for t to t+1 diagnostics")
    if len(args.seeds) != 30:
        raise ValueError("DEC--K requires exactly 30 matched held-out episode seeds")
    cfg = load_cfg(args.cfg)
    output = Path(args.out).expanduser()
    generate_sharded_paper_dataset(
        output,
        cfg,
        episode_count=len(args.seeds),
        horizon_steps=args.horizon,
        split_counts=(0, 0, len(args.seeds)),
        split_seed=args.split_seed,
        episode_seeds=args.seeds,
        device=args.device,
    )
    print(f"[OK] wrote {output} and explicit-seed episode shards")


if __name__ == "__main__":
    main()
