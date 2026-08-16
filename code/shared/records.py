"""CSV I/O for manifests and run-by-episode results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, TypeVar

from .schemas import EpisodeManifest, ResultRecord, RunManifest

T = TypeVar("T", RunManifest, EpisodeManifest, ResultRecord)


def _write_rows(path: str | Path, fields: tuple[str, ...], rows: Iterable[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_run_manifests(path: str | Path, rows: Iterable[RunManifest]) -> None:
    _write_rows(path, RunManifest.CSV_FIELDS, (row.to_dict() for row in rows))


def write_episode_manifests(path: str | Path, rows: Iterable[EpisodeManifest]) -> None:
    _write_rows(path, EpisodeManifest.CSV_FIELDS, (row.to_dict() for row in rows))


def write_result_records(path: str | Path, rows: Iterable[ResultRecord]) -> None:
    _write_rows(path, ResultRecord.CSV_FIELDS, (row.to_dict() for row in rows))


def _read_dicts(path: str | Path, required_fields: tuple[str, ...]) -> list[dict[str, str]]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required_fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")
        return list(reader)


def read_run_manifests(path: str | Path) -> list[RunManifest]:
    return [RunManifest.from_mapping(row) for row in _read_dicts(path, RunManifest.CSV_FIELDS)]


def read_episode_manifests(path: str | Path) -> list[EpisodeManifest]:
    return [EpisodeManifest.from_mapping(row) for row in _read_dicts(path, EpisodeManifest.CSV_FIELDS)]


def read_result_records(path: str | Path) -> list[ResultRecord]:
    return [ResultRecord.from_mapping(row) for row in _read_dicts(path, ResultRecord.CSV_FIELDS)]
