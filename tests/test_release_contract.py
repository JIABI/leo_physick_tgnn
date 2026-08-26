from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from controller_facing_state.config import (
    DEFAULT_MANUSCRIPT_CONFIG,
    load_experiment_registry,
    load_manuscript_config,
)
from controller_facing_state.demo import run_demo
from shared.statistics import (
    continuity_corrected_ratio_of_rate_ratios,
    holm_adjust,
    paired_geometric_ratio_tost,
    paired_run_first_student_t,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_and_extended_registry() -> None:
    config = load_manuscript_config()
    assert config["roles"]["weights"] == [1.0, 0.5, 0.7]
    assert config["model"]["time_encoding_dim"] == 16
    assert config["satellite"]["independent_reporting_runs"] == 10
    assert config["satellite"]["candidate_generation"]["hard_feasibility_mask"] is True
    assert config["uav"]["candidate_generation"]["hard_feasibility_mask"] is True
    registry = load_experiment_registry()
    assert set(registry["experiments"]) == {"FCT-CL", "EXP1", "EXP2", "EXP3", "EXP4", "EXP5"}
    assert registry["EXP4"]["reproducible_from_release"] is False
    assert len(registry["EXP1"]["selection_rules"]) == 5
    assert len(registry["EXP2"]["cells"]) == 8
    assert len(registry["EXP3"]["arms"]) == 4
    assert len(registry["EXP5"]["variants"]) == 4


def test_platform_configs_match_the_root_contract() -> None:
    root = load_manuscript_config()
    satellite = yaml.safe_load((ROOT / "code/satellite/configs/paper_protocol.yaml").read_text())
    assert len(satellite["run_seeds"]) == 10
    assert satellite["S_sats"] == root["satellite"]["satellites"]
    assert satellite["paper_protocol"]["policy"] == {
        "hard_feasibility_mask": True,
        "gamma_weight": 1.0,
        "load_weight": 0.7,
        "intensity_weight": 0.5,
        "min_dwell_steps": 10,
        "hysteresis": pytest.approx(1.0 / 6.0),
    }
    uav = yaml.safe_load((ROOT / "code/uav/configs/common.yaml").read_text())
    assert uav["uav_pipeline"]["policy"]["hard_feasible_start_mask"] is True
    assert [uav["uav_pipeline"]["policy"][name] for name in ("eta_weight", "intensity_weight", "flow_weight")] == [1.0, 0.5, 0.7]


def test_deterministic_demo_runs_without_training(tmp_path: Path) -> None:
    first = run_demo(DEFAULT_MANUSCRIPT_CONFIG, tmp_path / "a")
    second = run_demo(DEFAULT_MANUSCRIPT_CONFIG, tmp_path / "b")
    assert first == second
    assert first["training_performed"] is False
    assert first["satellite_policy"]["candidate_graph_domain"] == "authorized_post_mask"
    assert set(first["message_operators"]) == {"mlp", "kan", "physick"}
    assert all(value["finite"] for value in first["message_operators"].values())
    assert first["mrg_ppo"]["actor_parameters"] == 731_905
    assert first["mrg_ppo"]["critic_parameters"] == 687_233
    assert first["mrg_ppo"]["finite"] is True
    assert first["experiment_registry"]["EXP2"] == {
        "components_implemented": True,
        "full_historical_rerun_from_release": False,
    }
    assert first["experiment_registry"]["EXP4"] == {
        "components_implemented": False,
        "full_historical_rerun_from_release": False,
    }
    assert json.loads((tmp_path / "a/demo_report.json").read_text()) == first


def test_registered_statistical_primitives() -> None:
    paired = paired_run_first_student_t([2.0, 3.0, 4.0], [1.0, 1.0, 1.0])
    assert paired.estimate == pytest.approx(2.0)
    equivalent = paired_geometric_ratio_tost(
        [1.001, 0.999, 1.002, 0.998],
        [1.0, 1.0, 1.0, 1.0],
    )
    assert equivalent.equivalent
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    value = continuity_corrected_ratio_of_rate_ratios(
        a1_count=709,
        a1_exposure=4_608_000,
        a2_count=49,
        a2_exposure=4_608_000,
        b1_count=252,
        b1_exposure=4_608_000,
        b2_count=27,
        b2_exposure=4_608_000,
    )
    assert value == pytest.approx(1.561056105610561)


def test_da_gwm_and_ltt_r_are_cpu_constructible() -> None:
    from leo_pg.paper.baselines import DAGWMWorldModel, LTTRWorldModel

    da = DAGWMWorldModel(
        node_in_dim=7,
        edge_in_dim=7,
        hidden_dim=256,
        gat_layers=4,
        gat_heads=8,
    )
    assert da.attention_backend in {"pure_torch", "torch_geometric"}
    ltt = LTTRWorldModel(
        node_in_dim=7,
        edge_in_dim=7,
        model_dim=384,
        context_length=64,
        transformer_layers=6,
        transformer_heads=8,
        ffn_dim=1536,
    )
    assert ltt.temporal_backbone.context_length == 64
    assert all(torch.isfinite(parameter).all() for parameter in da.parameters())
