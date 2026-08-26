#!/usr/bin/env python3
"""Create a prespecified matched held-out episode panel without running it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "code"))

from shared.records import read_run_manifests, write_episode_manifests  # noqa: E402
from shared.manifest import validate_run_identity  # noqa: E402
from shared.schemas import EpisodeManifest  # noqa: E402
from shared.seeding import derive_seed, episode_seed_panel  # noqa: E402


PLATFORM_RUNS = {"satellite": 10, "uav": 5}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=sorted(PLATFORM_RUNS), required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--design",
        choices=("fixed_panel", "nested_within_run"),
        default="fixed_panel",
    )
    args = parser.parse_args()
    if args.episodes != 30:
        parser.error("the paper manifest requires exactly 30 episodes per run")

    runs = read_run_manifests(args.runs)
    unexpected_platforms = sorted({run.platform for run in runs} - {args.platform})
    if unexpected_platforms:
        raise ValueError(
            f"run manifest contains platforms other than {args.platform!r}: "
            f"{unexpected_platforms}"
        )
    expected_runs = PLATFORM_RUNS[args.platform]
    run_errors = validate_run_identity(runs, expected_runs=expected_runs)
    if run_errors:
        raise ValueError(
            f"run manifest failed the {args.platform} {expected_runs}-run contract:\n- "
            + "\n- ".join(run_errors)
        )
    rows: list[EpisodeManifest] = []
    fixed = episode_seed_panel(args.root_seed, args.episodes, panel_id=args.panel_id)
    # The same run identity may appear once per compared method in the run
    # manifest. Emit one episode panel per platform/run, not one duplicate panel
    # per method row.
    run_identities = sorted({(run.platform, run.run_id) for run in runs})
    for platform, run_id in run_identities:
        effective_panel_id = (
            args.panel_id
            if args.design == "fixed_panel"
            else f"{args.panel_id}:{run_id}"
        )
        seeds = fixed if args.design == "fixed_panel" else episode_seed_panel(
            derive_seed(args.root_seed, f"nested-run:{platform}:{run_id}"),
            args.episodes,
            panel_id=effective_panel_id,
        )
        for index, seed in enumerate(seeds):
            episode_id = f"{effective_panel_id}-e{index:03d}"
            exogenous_seed = derive_seed(seed, "exogenous")
            rows.append(
                EpisodeManifest(
                    platform=platform,
                    run_id=run_id,
                    split=args.split,
                    episode_id=episode_id,
                    episode_seed=seed,
                    exogenous_sequence_id=f"exo-{exogenous_seed:08x}",
                    exogenous_seed=exogenous_seed,
                    panel_id=effective_panel_id,
                )
            )
    write_episode_manifests(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
