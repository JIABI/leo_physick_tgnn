#!/usr/bin/env python3
"""Write the UAV paper task matrix and, when available, exact episode IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_cfs.experiments import held_out_evaluation_units, task_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-seeds", default=None, help="comma-separated seeds")
    parser.add_argument("--episodes-per-run", type=int, default=30)
    args = parser.parse_args()
    if args.episodes_per_run != 30:
        parser.error("the paper manifest requires exactly 30 episodes per run")
    manifest = task_manifest()
    if args.dataset is not None:
        if args.run_seeds is None:
            parser.error("--dataset requires --run-seeds")
        run_seeds = tuple(int(item) for item in args.run_seeds.split(","))
        if len(run_seeds) != 5 or len(set(run_seeds)) != 5:
            parser.error("the paper manifest requires exactly five distinct run seeds")
        manifest["evaluation_units"] = held_out_evaluation_units(
            args.dataset,
            run_seeds=run_seeds,
            episodes_per_run=args.episodes_per_run,
        )
    else:
        manifest["evaluation_units"] = {
            "status": "not_materialized",
            "required_source": "generated dataset split_manifest.episode_identity",
            "selection": f"first {args.episodes_per_run} prespecified test episodes per run",
        }
    path = Path(args.out).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
