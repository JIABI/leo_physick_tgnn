#!/usr/bin/env python3
"""Regenerate equal-run-weight summaries and paired confidence intervals.

This script consumes run-by-episode records conforming to
``configs/schemas/run_episode_results.schema.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "code"))

from shared.records import read_result_records  # noqa: E402
from shared.integrity import audit_result_records  # noqa: E402
from shared.statistics import (  # noqa: E402
    BootstrapMode,
    paired_hierarchical_bootstrap,
    summarize_condition,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Original run-by-episode CSV")
    parser.add_argument("--analysis-plan", type=Path, required=True, help="JSON file listing summaries and contrasts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--expected-runs", type=int, default=5)
    parser.add_argument("--expected-episodes", type=int, default=30)
    parser.add_argument(
        "--bootstrap-mode",
        choices=[mode.value for mode in BootstrapMode],
        default=BootstrapMode.NESTED_WITHIN_RUN.value,
    )
    args = parser.parse_args()

    records = read_result_records(args.input)
    integrity_errors = audit_result_records(
        records,
        expected_runs=args.expected_runs,
        expected_episodes=args.expected_episodes,
    )
    if integrity_errors:
        raise ValueError(
            "input result integrity audit failed:\n- "
            + "\n- ".join(integrity_errors)
        )
    plan = json.loads(args.analysis_plan.read_text(encoding="utf-8"))
    summaries = [
        summarize_condition(
            records,
            platform=item["platform"],
            condition_id=item["condition_id"],
            cell_id=item["cell_id"],
            metric=item["metric"],
            stress_id=item.get("stress_id", "nominal"),
            split=item.get("split", "test"),
        )
        for item in plan.get("summaries", [])
    ]
    contrasts = [
        paired_hierarchical_bootstrap(
            records,
            platform=item["platform"],
            condition_id=item["condition_id"],
            cell_a=item["cell_a"],
            cell_b=item["cell_b"],
            metric=item["metric"],
            stress_id=item.get("stress_id", "nominal"),
            split=item.get("split", "test"),
            draws=args.draws,
            seed=args.seed + index,
            mode=args.bootstrap_mode,
        ).to_dict()
        for index, item in enumerate(plan.get("contrasts", []))
    ]
    if summaries:
        _write_csv(args.output_dir / "condition_summaries.csv", summaries)
    if contrasts:
        _write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    if not summaries and not contrasts:
        raise ValueError("analysis plan contains neither summaries nor contrasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
