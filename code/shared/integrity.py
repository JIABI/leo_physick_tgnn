"""Integrity checks that fail on pseudo-replication or silent missingness."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schemas import ResultRecord


def audit_result_records(
    records: Iterable[ResultRecord], *, expected_runs: int | None = 5, expected_episodes: int | None = 30
) -> list[str]:
    rows = list(records)
    errors: list[str] = []
    seen: set[tuple[str, ...]] = set()
    groups: dict[tuple[str, str, str, str, str, str, str], set[str]] = defaultdict(set)
    checkpoint_by_run: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)
    runs_by_checkpoint: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)
    exogenous_by_episode: dict[
        tuple[str, str, str, str, str, str, str, str], set[str]
    ] = defaultdict(set)
    panels_by_cell: dict[
        tuple[str, str, str, str, str], dict[str, set[tuple[str, str, str]]]
    ] = defaultdict(lambda: defaultdict(set))
    cell_contracts: dict[str, set[tuple[str, str, str, str, str]]] = defaultdict(set)
    for row in rows:
        if row.key in seen:
            errors.append(f"duplicate result key: {row.key}")
        seen.add(row.key)
        groups[
            (
                row.platform,
                row.condition_id,
                row.cell_id,
                row.stress_id,
                row.split,
                row.metric,
                row.run_id,
            )
        ].add(row.episode_id)
        exogenous_by_episode[
            (
                row.platform,
                row.condition_id,
                row.cell_id,
                row.stress_id,
                row.split,
                row.metric,
                row.run_id,
                row.episode_id,
            )
        ].add(row.exogenous_sequence_id)
        panels_by_cell[
            (
                row.platform,
                row.condition_id,
                row.stress_id,
                row.split,
                row.metric,
            )
        ][row.cell_id].add(
            (row.run_id, row.episode_id, row.exogenous_sequence_id)
        )
        if row.checkpoint_id:
            checkpoint_by_run[
                (
                    row.platform,
                    row.condition_id,
                    row.cell_id,
                    row.stress_id,
                    row.split,
                    row.run_id,
                )
            ].add(row.checkpoint_id)
            runs_by_checkpoint[
                (
                    row.platform,
                    row.condition_id,
                    row.cell_id,
                    row.stress_id,
                    row.split,
                    row.checkpoint_id,
                )
            ].add(row.run_id)
        cell_contracts[row.cell_id].add((row.platform, row.condition_id, row.method, row.interface, row.operator))
    if expected_episodes is not None:
        for group, episodes in sorted(groups.items()):
            if len(episodes) != expected_episodes:
                errors.append(f"{group} has {len(episodes)} episode IDs; expected {expected_episodes}")
    if expected_runs is not None:
        by_condition_metric: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)
        for platform, condition_id, cell_id, stress_id, split, metric, run_id in groups:
            by_condition_metric[(platform, condition_id, cell_id, stress_id, split, metric)].add(run_id)
        for group, runs in sorted(by_condition_metric.items()):
            if len(runs) != expected_runs:
                errors.append(f"{group} has {len(runs)} run IDs; expected {expected_runs}")
    for group, checkpoint_ids in sorted(checkpoint_by_run.items()):
        if len(checkpoint_ids) > 1:
            errors.append(f"{group} maps one run to multiple checkpoints")
    for group, run_ids in sorted(runs_by_checkpoint.items()):
        if len(run_ids) > 1:
            errors.append(
                f"{group} reuses one checkpoint across run IDs: {sorted(run_ids)}"
            )
    for group, exogenous_ids in sorted(exogenous_by_episode.items()):
        if len(exogenous_ids) > 1:
            errors.append(
                f"{group} maps one episode ID to multiple exogenous sequences: "
                f"{sorted(exogenous_ids)}"
            )
    for cell_id, contracts in sorted(cell_contracts.items()):
        if len(contracts) > 1:
            errors.append(f"cell_id {cell_id!r} is not globally unique: {sorted(contracts)}")
    for group, cell_panels in sorted(panels_by_cell.items()):
        if len(cell_panels) < 2:
            continue
        reference_cell = sorted(cell_panels)[0]
        reference = cell_panels[reference_cell]
        for cell_id in sorted(cell_panels):
            if cell_panels[cell_id] != reference:
                missing = sorted(reference - cell_panels[cell_id])
                extra = sorted(cell_panels[cell_id] - reference)
                errors.append(
                    f"{group} does not use one matched episode/exogenous panel: "
                    f"{cell_id!r} differs from {reference_cell!r} "
                    f"(missing={len(missing)}, extra={len(extra)})"
                )
    return errors


def audit_public_paths(root: str | Path) -> list[str]:
    """Find path leaks and hidden files; does not inspect binary contents."""

    base = Path(root)
    errors: list[str] = []
    for path in base.rglob("*"):
        relative = path.relative_to(base)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            errors.append(f"symbolic link is not permitted: {relative.as_posix()}")
        allowed_dotfiles = {".gitignore", ".gitattributes", ".github"}
        if any(
            part.startswith(".") and part not in allowed_dotfiles
            for part in relative.parts
        ):
            errors.append(f"hidden path: {relative.as_posix()}")
        if "__pycache__" in relative.parts or path.suffix.lower() in {".pyc", ".pyo"}:
            errors.append(f"Python cache artifact: {relative.as_posix()}")
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".py"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            # Construct local-home markers so this audit module does not match
            # its own source text while scanning the public release.
            posix_home_marker = "/" + "Users" + "/"
            windows_home_marker = "C:" + "\\" + "Users" + "\\"
            if posix_home_marker in text or windows_home_marker in text:
                errors.append(f"local absolute path embedded in {relative.as_posix()}")
    return errors
