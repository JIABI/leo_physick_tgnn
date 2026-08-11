from __future__ import annotations

import csv
import json

import pytest
import torch

from leo_pg.runtime_profile import (
    RuntimeProfileConfig,
    build_runtime_report,
    checkpoint_identity,
    latency_summary,
    profile_operation,
    write_runtime_report,
)
from scripts.profile_runtime import _evaluation_horizon


class _TinyPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 2)

    def predict_step(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def test_runtime_horizon_comes_from_formal_evaluation_not_training_data() -> None:
    root = {
        "paper_evaluation": {"horizon_steps": 600},
        "uav_pipeline": {
            "evaluation": {"horizon_steps": 120},
            "dataset": {"train_horizon_steps": 60},
        },
    }
    assert _evaluation_horizon(root, "ntn") == 600
    assert _evaluation_horizon(root, "snapshot") == 600
    assert _evaluation_horizon(root, "uav") == 120
    with pytest.raises(ValueError, match="must be 600"):
        _evaluation_horizon(
            {"paper_evaluation": {"horizon_steps": 200}}, "ntn"
        )


def test_latency_summary_preserves_samples_and_reports_percentiles() -> None:
    summary = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert summary["p50"] == 2.5
    assert summary["p90"] == 3.7
    assert summary["samples"] == [1.0, 2.0, 3.0, 4.0]


def test_profile_operation_excludes_prepare_from_repeat_count() -> None:
    calls = {"prepare": 0, "operation": 0, "reset": 0}

    def prepare() -> None:
        calls["prepare"] += 1

    def operation() -> None:
        calls["operation"] += 1

    def reset() -> None:
        calls["reset"] += 1

    result = profile_operation(
        "full_decision_epoch",
        operation,
        device="cpu",
        config=RuntimeProfileConfig(warmup_steps=2, repeats=3),
        timing_contract="test contract",
        prepare_each=prepare,
        reset_after_warmup=reset,
    )
    assert calls == {"prepare": 5, "operation": 5, "reset": 1}
    assert result["latency_ms"]["count"] == 3
    assert result["cuda_synchronized_each_sample"] is False
    assert result["memory"]["reported_peak_kind"] == "cpu_ru_maxrss"


def test_versioned_json_and_csv_report(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights-only-fixture")
    model = _TinyPredictor().eval()
    sample = torch.ones(1, 3)
    profile = profile_operation(
        "model_predict_step",
        lambda: model.predict_step(sample),
        device="cpu",
        config=RuntimeProfileConfig(warmup_steps=0, repeats=1),
        timing_contract="fixed formal graph; predict_step only",
    )
    report = build_runtime_report(
        platform_name="ntn",
        method="tiny",
        model=model,
        checkpoint=checkpoint_identity(checkpoint),
        input_provenance={
            "source_kind": "formal_heldout_episode_shard",
            "synthetic": False,
            "split": "test",
            "episode_index": 7,
            "episode_seed": 17,
        },
        profiles=[profile],
        device="cpu",
        configuration={"file_name": "paper_protocol.yaml", "sha256": "0" * 64},
        evaluation_horizon_steps=600,
        batch_size=1,
    )
    json_path, csv_path = write_runtime_report(
        report,
        json_path=tmp_path / "runtime.json",
        csv_path=tmp_path / "runtime.csv",
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["scientific_contract"]["training_performed"] is False
    assert payload["scientific_contract"]["batch_size"] == 1
    assert payload["scientific_contract"]["si_runtime_table_comparable"] is False
    assert payload["hardware"]["hardware_sha256"]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["profile"] == "model_predict_step"
    assert float(rows[0]["params_m"]) > 0.0
    assert rows[0]["si_runtime_table_comparable"] == "False"
