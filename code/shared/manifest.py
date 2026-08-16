"""Hashing and manifest validation for a versioned release."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schemas import EpisodeManifest, RunManifest, require_relative_path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class FileManifestRow:
    relative_path: str
    bytes: int
    sha256: str
    role: str
    provenance_class: str


def build_file_manifest(
    root: str | Path,
    *,
    exclude: Iterable[str] = (),
    role: str = "release_artifact",
    provenance_class: str = "generated_release_file",
) -> list[FileManifestRow]:
    base = Path(root).resolve()
    excluded = set(exclude)
    rows: list[FileManifestRow] = []
    for path in sorted(candidate for candidate in base.rglob("*") if candidate.is_file()):
        relative = path.relative_to(base).as_posix()
        parts = path.relative_to(base).parts
        if (
            relative in excluded
            or any(part.startswith(".") for part in parts)
            or "__pycache__" in parts
            or path.suffix.lower() in {".pyc", ".pyo"}
        ):
            continue
        require_relative_path(relative)
        rows.append(
            FileManifestRow(
                relative_path=relative,
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
                role=role,
                provenance_class=provenance_class,
            )
        )
    return rows


def write_file_manifest(path: str | Path, rows: Iterable[FileManifestRow]) -> None:
    fields = ("relative_path", "bytes", "sha256", "role", "provenance_class")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def validate_run_identity(rows: Iterable[RunManifest], expected_runs: int | None = None) -> list[str]:
    errors: list[str] = []
    rows = list(rows)
    seen: set[tuple[str, str, str, str, str]] = set()
    seed_fields = ("training_seed", "model_init_seed", "optimizer_seed", "data_order_seed")
    groups: dict[tuple[str, str, str, str], list[RunManifest]] = {}
    for row in rows:
        key = (
            row.platform,
            row.condition_id,
            row.cell_id,
            row.method,
            row.run_id,
        )
        if key in seen:
            errors.append(f"duplicate run identity: {key}")
        seen.add(key)
        groups.setdefault(key[:4], []).append(row)
    for group, members in sorted(groups.items()):
        if expected_runs is not None and len(members) != expected_runs:
            errors.append(f"{group} has {len(members)} runs; expected {expected_runs}")
        for field in seed_fields:
            values = [getattr(member, field) for member in members]
            if len(set(values)) != len(values):
                errors.append(f"{group} reuses {field}; independent runs require unique values")
        complete = [member for member in members if member.training_status == "complete"]
        checkpoint_ids = [member.checkpoint_id for member in complete]
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            errors.append(f"{group} reuses a checkpoint across completed training runs")
    return errors


def validate_episode_pairing(rows: Iterable[EpisodeManifest], expected_episodes: int | None = None) -> list[str]:
    errors: list[str] = []
    rows = list(rows)
    groups: dict[tuple[str, str, str, str], list[EpisodeManifest]] = {}
    for row in rows:
        groups.setdefault((row.platform, row.run_id, row.split, row.panel_id), []).append(row)
    for group, members in sorted(groups.items()):
        ids = [member.episode_id for member in members]
        if len(ids) != len(set(ids)):
            errors.append(f"{group} contains duplicate episode IDs")
        if expected_episodes is not None and len(members) != expected_episodes:
            errors.append(f"{group} has {len(members)} episodes; expected {expected_episodes}")
    fixed_panels: dict[
        tuple[str, str, str], dict[str, tuple[int, str, int, str]]
    ] = {}
    for (platform, run_id, split, panel_id), members in sorted(groups.items()):
        signature = {
            member.episode_id: (
                member.episode_seed,
                member.exogenous_sequence_id,
                member.exogenous_seed,
                member.initial_state_id,
            )
            for member in members
        }
        panel_key = (platform, split, panel_id)
        reference = fixed_panels.setdefault(panel_key, signature)
        if signature != reference:
            errors.append(
                f"{panel_key} is not a fixed matched panel across run {run_id!r}"
            )
    return errors
