#!/usr/bin/env python3
"""Check source files and a platform-specific matched-episode evidence contract."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib
from pathlib import Path
from typing import Iterable

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "code"))

from shared.integrity import audit_public_paths, audit_result_records  # noqa: E402
from shared.manifest import validate_episode_pairing, validate_run_identity  # noqa: E402
from shared.records import (  # noqa: E402
    read_episode_manifests,
    read_result_records,
    read_run_manifests,
)
from shared.schemas import ResultRecord, RunManifest, require_relative_path  # noqa: E402


RunCheckpointKey = tuple[str, str, str, str, str, str, str, str]
ResultCheckpointKey = tuple[str, str, str, str, str, str]
PLATFORM_RUNS = {"satellite": 10, "uav": 5}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _static_source_errors(root: Path) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    counts = {"python": 0, "json": 0, "yaml": 0, "toml": 0, "csv": 0}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(root).parts:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
                counts["python"] += 1
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                counts["json"] += 1
            elif path.suffix.lower() in {".yaml", ".yml", ".cff"}:
                yaml.safe_load(path.read_text(encoding="utf-8"))
                counts["yaml"] += 1
            elif path.suffix.lower() == ".toml":
                tomllib.loads(path.read_text(encoding="utf-8"))
                counts["toml"] += 1
            elif path.suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.reader(handle))
                blank = [
                    index
                    for index, row in enumerate(rows, 1)
                    if not row or all(not item.strip() for item in row)
                ]
                if blank:
                    errors.append(f"blank CSV rows in {relative}: {blank}")
                counts["csv"] += 1
        except Exception as exc:
            errors.append(
                f"cannot parse {relative}: {type(exc).__name__}: {exc}"
            )
    return errors, counts


def _completed_run_keys(rows: Iterable[RunManifest]) -> set[RunCheckpointKey]:
    return {
        (
            row.platform,
            row.condition_id,
            row.cell_id,
            row.method,
            row.run_id,
            row.checkpoint_id,
            row.config_sha256.lower(),
            row.code_commit,
        )
        for row in rows
        if row.training_status == "complete"
    }


def _pipe_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split("|") if item.strip())


def _checkpoint_manifest_errors(
    payload_root: Path,
    path: Path,
    *,
    expected_run_keys: set[RunCheckpointKey],
) -> tuple[list[str], dict[ResultCheckpointKey, RunCheckpointKey]]:
    """Validate checkpoint payloads and construct exact result bindings.

    A primary checkpoint row is joined to the run manifest on platform,
    condition, cell, method, run, checkpoint, configuration and code identity.
    Result cells and their method labels are enumerated explicitly in two
    aligned pipe-separated columns.
    """

    errors: list[str] = []
    result_bindings: dict[ResultCheckpointKey, RunCheckpointKey] = {}
    required = {
        "platform",
        "condition_id",
        "cell_id",
        "allowed_result_cell_ids",
        "allowed_result_methods",
        "method",
        "run_id",
        "checkpoint_id",
        "relative_path",
        "bytes",
        "sha256",
        "config_sha256",
        "code_commit",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            return [f"checkpoint manifest is missing columns: {sorted(missing)}"], {}
        rows = list(reader)

    checkpoint_ids: set[str] = set()
    seen_paths: set[str] = set()
    manifest_run_keys: set[RunCheckpointKey] = set()
    for index, row in enumerate(rows, 2):
        platform = str(row.get("platform", "")).strip()
        condition_id = str(row.get("condition_id", "")).strip()
        cell_id = str(row.get("cell_id", "")).strip()
        method = str(row.get("method", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        checkpoint_id = str(row.get("checkpoint_id", "")).strip()
        config_sha256 = str(row.get("config_sha256", "")).strip().lower()
        code_commit = str(row.get("code_commit", "")).strip()
        relative = str(row.get("relative_path", "")).strip()
        run_key: RunCheckpointKey = (
            platform,
            condition_id,
            cell_id,
            method,
            run_id,
            checkpoint_id,
            config_sha256,
            code_commit,
        )
        allowed_cells = _pipe_values(str(row.get("allowed_result_cell_ids", "")))
        allowed_methods = _pipe_values(str(row.get("allowed_result_methods", "")))
        if not all(run_key) or not relative:
            errors.append(f"checkpoint manifest row {index} has an empty identity field")
            continue
        if len(config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in config_sha256
        ):
            errors.append(
                f"checkpoint manifest row {index} has an invalid config_sha256"
            )
            continue
        if not allowed_cells or len(allowed_cells) != len(allowed_methods):
            errors.append(
                f"checkpoint manifest row {index} must pair every allowed result "
                "cell with one method"
            )
            continue
        if len(set(zip(allowed_cells, allowed_methods))) != len(allowed_cells):
            errors.append(
                f"checkpoint manifest row {index} repeats an allowed cell/method pair"
            )
        if (cell_id, method) not in set(zip(allowed_cells, allowed_methods)):
            errors.append(
                f"checkpoint manifest row {index} must include its primary "
                "cell_id/method pair in the allowed result bindings"
            )
        if run_key in manifest_run_keys:
            errors.append(f"checkpoint manifest repeats run identity {run_key}")
        manifest_run_keys.add(run_key)
        if checkpoint_id in checkpoint_ids:
            errors.append(f"checkpoint manifest reuses checkpoint_id {checkpoint_id!r}")
        checkpoint_ids.add(checkpoint_id)
        if relative in seen_paths:
            errors.append(f"checkpoint manifest reuses relative_path {relative!r}")
        seen_paths.add(relative)

        for allowed_cell, allowed_method in zip(allowed_cells, allowed_methods):
            result_key: ResultCheckpointKey = (
                platform,
                condition_id,
                allowed_cell,
                allowed_method,
                run_id,
                checkpoint_id,
            )
            previous = result_bindings.get(result_key)
            if previous is not None and previous != run_key:
                errors.append(
                    "checkpoint manifest assigns one result identity to multiple "
                    f"primary checkpoints: {result_key}"
                )
            result_bindings[result_key] = run_key

        try:
            require_relative_path(relative)
        except ValueError as exc:
            errors.append(f"checkpoint manifest row {index}: {exc}")
            continue
        payload = payload_root / relative
        if not payload.is_file():
            errors.append(f"checkpoint payload is absent: {relative}")
            continue
        try:
            expected_bytes = int(str(row.get("bytes", "")))
        except ValueError:
            errors.append(f"checkpoint manifest row {index} has invalid bytes")
            continue
        if payload.stat().st_size != expected_bytes:
            errors.append(f"checkpoint byte count mismatch: {relative}")
        expected_sha = str(row.get("sha256", "")).strip().lower()
        if len(expected_sha) != 64 or _sha256(payload) != expected_sha:
            errors.append(f"checkpoint SHA-256 mismatch: {relative}")

    if manifest_run_keys != expected_run_keys:
        errors.append(
            "checkpoint identities do not exactly match completed run identities "
            f"(missing={len(expected_run_keys - manifest_run_keys)}, "
            f"extra={len(manifest_run_keys - expected_run_keys)})"
        )
    return errors, result_bindings


def _result_checkpoint_binding_errors(
    rows: Iterable[ResultRecord],
    bindings: dict[ResultCheckpointKey, RunCheckpointKey],
) -> list[str]:
    invalid: set[ResultCheckpointKey] = set()
    for row in rows:
        key: ResultCheckpointKey = (
            row.platform,
            row.condition_id,
            row.cell_id,
            row.method,
            row.run_id,
            row.checkpoint_id,
        )
        if key not in bindings:
            invalid.add(key)
    if not invalid:
        return []
    return [
        "result records have no exact checkpoint binding for: "
        f"{sorted(invalid)}"
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--platform", choices=sorted(PLATFORM_RUNS))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Base directory for checkpoint relative_path values",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        help="Optional assertion; must equal the selected platform contract.",
    )
    parser.add_argument("--expected-episodes", type=int, default=30)
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    evidence_inputs = (
        args.runs,
        args.episodes,
        args.results,
        args.checkpoints,
    )
    if any(value is not None for value in evidence_inputs) and args.platform is None:
        parser.error("--platform is required when auditing evidence records")
    expected_runs = None
    if args.platform is not None:
        expected_runs = PLATFORM_RUNS[args.platform]
        if args.expected_runs is not None and args.expected_runs != expected_runs:
            parser.error(
                f"--expected-runs must be {expected_runs} for {args.platform}"
            )
    elif args.expected_runs is not None:
        parser.error("--expected-runs requires --platform")
    root = args.root.expanduser().resolve()
    errors = audit_public_paths(root)
    static_errors, static_counts = _static_source_errors(root)
    errors.extend(static_errors)

    run_rows: list[RunManifest] = []
    if args.runs is not None:
        run_rows = read_run_manifests(args.runs)
        errors.extend(validate_run_identity(run_rows, expected_runs=expected_runs))
        errors.extend(
            f"run manifest platform is {row.platform!r}, expected {args.platform!r}"
            for row in run_rows
            if row.platform != args.platform
        )

    if args.episodes is not None:
        episode_rows = read_episode_manifests(args.episodes)
        errors.extend(
            validate_episode_pairing(
                episode_rows,
                expected_episodes=args.expected_episodes,
            )
        )

    result_rows: list[ResultRecord] = []
    if args.results is not None:
        result_rows = read_result_records(args.results)
        errors.extend(
            audit_result_records(
                result_rows,
                expected_runs=expected_runs,
                expected_episodes=args.expected_episodes,
            )
        )
        errors.extend(
            f"result platform is {row.platform!r}, expected {args.platform!r}"
            for row in result_rows
            if row.platform != args.platform
        )

    if args.checkpoints is not None and args.runs is None:
        errors.append("--checkpoints must be paired with --runs")
    if args.results is not None and args.checkpoints is None:
        errors.append("--results must be paired with --checkpoints")

    bindings: dict[ResultCheckpointKey, RunCheckpointKey] = {}
    if args.checkpoints is not None:
        payload_root = (
            args.checkpoint_root.expanduser().resolve()
            if args.checkpoint_root is not None
            else args.checkpoints.expanduser().resolve().parent
        )
        checkpoint_errors, bindings = _checkpoint_manifest_errors(
            payload_root,
            args.checkpoints.expanduser().resolve(),
            expected_run_keys=_completed_run_keys(run_rows),
        )
        errors.extend(checkpoint_errors)
    if result_rows and bindings:
        errors.extend(_result_checkpoint_binding_errors(result_rows, bindings))

    report = {
        "source_counts": static_counts,
        "run_rows": len(run_rows),
        "result_rows": len(result_rows),
        "checkpoint_result_bindings": len(bindings),
        "errors": errors,
        "ok": not errors,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
