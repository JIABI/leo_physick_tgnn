"""Read-only verification for the Zenodo v4 source-data release."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerificationReport:
    source_data_root: str
    release_files_checked: int
    checksums_checked: int
    core_csv_count: int
    extended_csv_count: int
    extended_manifest_rows: int
    exp4_provenance_warnings: int
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _release_member(root: Path, raw_relative: str, *, source: str) -> Path:
    """Resolve one package member while rejecting path escape and ambiguity."""

    value = str(raw_relative).strip()
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {source} path: {raw_relative!r}")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{source} path escapes the source-data root: {value}") from error
    return target


def verify_source_data(source_data_root: str | Path) -> VerificationReport:
    root = Path(source_data_root).expanduser().resolve()
    required = [
        root / "RELEASE_MANIFEST.csv",
        root / "SHA256SUMS.txt",
        root / "SOURCE_DATA",
        root / "SOURCE_DATA" / "EXTENDED_EXPERIMENTS" / "MANIFEST.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("source-data release is incomplete: " + ", ".join(missing))

    manifest_rows = _rows(root / "RELEASE_MANIFEST.csv")
    manifest_paths = [row["relative_path"] for row in manifest_rows]
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("RELEASE_MANIFEST.csv contains duplicate relative paths")
    for row in manifest_rows:
        target = _release_member(
            root,
            row["relative_path"],
            source="release manifest",
        )
        if not target.is_file():
            raise FileNotFoundError(f"release manifest target is missing: {target}")
        if target.stat().st_size != int(row["bytes"]):
            raise ValueError(f"release manifest byte count mismatch: {target}")
        if _sha256(target) != row["sha256"]:
            raise ValueError(f"release manifest checksum mismatch: {target}")

    checksum_rows = []
    with (root / "SHA256SUMS.txt").open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            digest, separator, relative = line.rstrip("\n").partition("  ")
            try:
                valid_digest = len(digest) == 64 and int(digest, 16) >= 0
            except ValueError:
                valid_digest = False
            if not separator or not valid_digest or not relative:
                raise ValueError(f"malformed SHA256SUMS.txt line {number}")
            target = _release_member(root, relative, source="SHA256SUMS")
            if not target.is_file() or _sha256(target) != digest:
                raise ValueError(f"SHA256SUMS mismatch: {relative}")
            checksum_rows.append(relative)
    if len(checksum_rows) != len(set(checksum_rows)):
        raise ValueError("SHA256SUMS.txt contains duplicate relative paths")

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_manifest_paths = actual_paths - {
        "RELEASE_MANIFEST.csv",
        "SHA256SUMS.txt",
    }
    if set(manifest_paths) != expected_manifest_paths:
        missing_from_manifest = sorted(expected_manifest_paths - set(manifest_paths))
        unbacked_manifest_rows = sorted(set(manifest_paths) - expected_manifest_paths)
        raise ValueError(
            "release manifest path-set is not closed; "
            f"missing={missing_from_manifest}, unbacked={unbacked_manifest_rows}"
        )
    expected_checksum_paths = actual_paths - {"SHA256SUMS.txt"}
    if set(checksum_rows) != expected_checksum_paths:
        missing_from_checksums = sorted(expected_checksum_paths - set(checksum_rows))
        unbacked_checksum_rows = sorted(set(checksum_rows) - expected_checksum_paths)
        raise ValueError(
            "SHA256SUMS path-set is not closed; "
            f"missing={missing_from_checksums}, unbacked={unbacked_checksum_rows}"
        )

    source = root / "SOURCE_DATA"
    core_csv = sorted(source.glob("*.csv"))
    extended_dir = source / "EXTENDED_EXPERIMENTS"
    extended_csv = sorted(extended_dir.glob("*.csv"))
    if len(core_csv) != 38:
        raise ValueError(f"expected 38 top-level core CSVs, found {len(core_csv)}")
    if len(extended_csv) != 18:
        raise ValueError(f"expected 18 extended CSVs, found {len(extended_csv)}")

    extended_manifest = _rows(extended_dir / "MANIFEST.csv")
    if len(extended_manifest) != 18:
        raise ValueError("extended MANIFEST.csv must close exactly 18 files")
    extended_names = [row["file_name"] for row in extended_manifest]
    if len(extended_names) != len(set(extended_names)):
        raise ValueError("extended MANIFEST.csv contains duplicate file names")
    if set(extended_names) != {path.name for path in extended_csv}:
        raise ValueError("extended MANIFEST.csv does not close the 18-file CSV set")
    for row in extended_manifest:
        target = _release_member(
            extended_dir,
            row["file_name"],
            source="extended manifest",
        )
        if not target.is_file():
            raise FileNotFoundError(f"extended manifest target is missing: {target}")
        if row["file_name"] != "MANIFEST.csv" and _sha256(target) != row["content_sha256"]:
            raise ValueError(f"extended checksum mismatch: {target.name}")
        with target.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            data = list(reader)
        if len(data) != int(row["row_count"]) or len(header) != int(row["column_count"]):
            raise ValueError(f"extended shape mismatch: {target.name}")

    qc = _rows(extended_dir / "QC_INVARIANTS.csv")
    warnings = [
        row for row in qc
        if row.get("pass") == "0" and row.get("severity", "").lower() == "warning"
    ]
    if len(warnings) != 1 or warnings[0].get("check_id") != "exp4_environment_provenance":
        raise ValueError("expected exactly the declared EXP4 provenance warning")
    failures = [row for row in qc if row.get("pass") == "0" and row not in warnings]
    if failures:
        raise ValueError(f"extended QC contains {len(failures)} non-warning failures")

    return VerificationReport(
        source_data_root=str(root),
        release_files_checked=len(manifest_rows),
        checksums_checked=len(checksum_rows),
        core_csv_count=len(core_csv),
        extended_csv_count=len(extended_csv),
        extended_manifest_rows=len(extended_manifest),
        exp4_provenance_warnings=len(warnings),
        passed=True,
    )
