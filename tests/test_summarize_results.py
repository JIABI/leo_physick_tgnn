from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import summarize_results
from shared.records import write_result_records
from shared.schemas import ResultRecord


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "ANALYSIS_PLAN_TEMPLATE.json"


def _records(platform: str, run_count: int) -> list[ResultRecord]:
    rows: list[ResultRecord] = []
    for cell_id, offset in (("CELL_A", 1.0), ("CELL_B", 0.0)):
        for run_index in range(run_count):
            for episode_index in range(30):
                rows.append(
                    ResultRecord(
                        platform=platform,
                        condition_id="condition",
                        cell_id=cell_id,
                        method=cell_id.lower(),
                        interface="intensity_flow",
                        operator="mlp",
                        run_id=f"run-{run_index:02d}",
                        checkpoint_id=f"{cell_id}-checkpoint-{run_index:02d}",
                        episode_id=f"episode-{episode_index:02d}",
                        exogenous_sequence_id=f"sequence-{episode_index:02d}",
                        metric="metric",
                        value=offset + run_index + episode_index / 100.0,
                        defined=True,
                        provenance="original_experiment_output",
                    )
                )
    return rows


def _write_plan(path: Path, platform: str) -> None:
    plan = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    plan["platform"] = platform
    plan["estimator"] = (
        "run_first_student_t"
        if platform == "satellite"
        else "run_first_hierarchical_bootstrap"
    )
    plan["summaries"] = [
        {
            "platform": platform,
            "condition_id": "condition",
            "cell_id": "CELL_A",
            "metric": "metric",
        }
    ]
    plan["contrasts"] = [
        {
            "platform": platform,
            "condition_id": "condition",
            "cell_a": "CELL_A",
            "cell_b": "CELL_B",
            "metric": "metric",
        }
    ]
    path.write_text(json.dumps(plan), encoding="utf-8")


def _csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    return rows[0]


def test_satellite_uses_ten_run_student_t_and_paired_df9(tmp_path: Path) -> None:
    input_path = tmp_path / "satellite.csv"
    plan_path = tmp_path / "satellite-plan.json"
    output_dir = tmp_path / "satellite-output"
    write_result_records(input_path, _records("satellite", 10))
    _write_plan(plan_path, "satellite")

    assert summarize_results.main(
        [
            "--platform",
            "satellite",
            "--estimator",
            "run_first_student_t",
            "--input",
            str(input_path),
            "--analysis-plan",
            str(plan_path),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = _csv_row(output_dir / "condition_summaries.csv")
    assert summary["estimator"] == "run_first_student_t"
    assert summary["run_n"] == "10"
    assert summary["degrees_of_freedom"] == "9"
    contrast = _csv_row(output_dir / "paired_contrasts.csv")
    assert contrast["estimator"] == "paired_run_first_student_t"
    assert contrast["run_n"] == "10"
    assert contrast["degrees_of_freedom"] == "9"
    assert float(contrast["estimate"]) == pytest.approx(1.0)


def test_uav_uses_five_run_hierarchical_bootstrap(tmp_path: Path) -> None:
    input_path = tmp_path / "uav.csv"
    plan_path = tmp_path / "uav-plan.json"
    output_dir = tmp_path / "uav-output"
    write_result_records(input_path, _records("uav", 5))
    _write_plan(plan_path, "uav")

    assert summarize_results.main(
        [
            "--platform",
            "uav",
            "--estimator",
            "run_first_hierarchical_bootstrap",
            "--input",
            str(input_path),
            "--analysis-plan",
            str(plan_path),
            "--output-dir",
            str(output_dir),
            "--draws",
            "200",
            "--seed",
            "19",
        ]
    ) == 0

    summary = _csv_row(output_dir / "condition_summaries.csv")
    assert summary["estimator"] == "run_first_hierarchical_bootstrap"
    assert summary["run_n"] == "5"
    assert summary["bootstrap_mode"] == "nested_within_run"
    assert summary["bootstrap_draws"] == "200"
    contrast = _csv_row(output_dir / "paired_contrasts.csv")
    assert contrast["estimator"] == "run_first_hierarchical_bootstrap"
    assert contrast["run_n"] == "5"
    assert contrast["bootstrap_draws"] == "200"
    assert float(contrast["estimate"]) == pytest.approx(1.0)


def test_platform_estimator_and_plan_mismatches_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        summarize_results.main(
            [
                "--platform",
                "satellite",
                "--estimator",
                "run_first_hierarchical_bootstrap",
                "--input",
                str(tmp_path / "missing.csv"),
                "--analysis-plan",
                str(tmp_path / "missing.json"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert error.value.code == 2

    with pytest.raises(SystemExit) as bootstrap_error:
        summarize_results.main(
            [
                "--platform",
                "uav",
                "--estimator",
                "run_first_hierarchical_bootstrap",
                "--bootstrap-mode",
                "fixed_panel",
                "--input",
                str(tmp_path / "missing.csv"),
                "--analysis-plan",
                str(tmp_path / "missing.json"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert bootstrap_error.value.code == 2

    plan_path = tmp_path / "uav-plan.json"
    _write_plan(plan_path, "uav")
    with pytest.raises(ValueError, match="analysis plan platform"):
        summarize_results.main(
            [
                "--platform",
                "satellite",
                "--estimator",
                "run_first_student_t",
                "--input",
                str(tmp_path / "missing.csv"),
                "--analysis-plan",
                str(plan_path),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
