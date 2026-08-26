from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from controller_facing_state.source_data import verify_source_data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "ControllerFacingState_SourceData_v4.0.0"
    source = root / "SOURCE_DATA"
    extended = source / "EXTENDED_EXPERIMENTS"
    for index in range(38):
        _write_csv(source / f"CORE_{index:02d}.csv", ["value"], [[index]])
    for index in range(16):
        _write_csv(extended / f"EXT_{index:02d}.csv", ["value"], [[index]])
    _write_csv(
        extended / "QC_INVARIANTS.csv",
        ["check_id", "pass", "severity"],
        [["exp4_environment_provenance", 0, "warning"]],
    )

    extended_files = sorted(path.name for path in extended.glob("*.csv"))
    extended_files.append("MANIFEST.csv")
    extended_rows: list[list[object]] = []
    for name in sorted(extended_files):
        if name == "MANIFEST.csv":
            extended_rows.append([name, "", 18, 4])
            continue
        path = extended / name
        with path.open("r", encoding="utf-8", newline="") as handle:
            table = list(csv.reader(handle))
        extended_rows.append([name, _sha256(path), len(table) - 1, len(table[0])])
    _write_csv(
        extended / "MANIFEST.csv",
        ["file_name", "content_sha256", "row_count", "column_count"],
        extended_rows,
    )

    payload = sorted(
        path for path in root.rglob("*") if path.is_file()
    )
    manifest_rows = [
        [path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path)]
        for path in payload
    ]
    _write_csv(
        root / "RELEASE_MANIFEST.csv",
        ["relative_path", "bytes", "sha256"],
        manifest_rows,
    )
    checksummed = sorted(path for path in root.rglob("*") if path.is_file())
    with (root / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for path in checksummed:
            handle.write(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n")
    return root


def test_verifier_requires_closed_manifest_and_checksum_path_sets(tmp_path: Path) -> None:
    root = _package(tmp_path)
    report = verify_source_data(root)
    assert report.passed
    (root / "UNLISTED.txt").write_text("not in either index\n", encoding="utf-8")
    with pytest.raises(ValueError, match="path-set is not closed"):
        verify_source_data(root)


def test_verifier_rejects_manifest_path_escape(tmp_path: Path) -> None:
    root = _package(tmp_path)
    manifest = root / "RELEASE_MANIFEST.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    rows[0]["relative_path"] = "../outside.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="unsafe release manifest path"):
        verify_source_data(root)
