"""Strict loaders for the human-readable manuscript contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUSCRIPT_CONFIG = ROOT / "configs" / "manuscript_v4.yaml"
DEFAULT_EXPERIMENT_REGISTRY = ROOT / "configs" / "experiments_v4.yaml"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"{resolved} must contain a YAML mapping")
    return dict(value)


def load_manuscript_config(path: str | Path = DEFAULT_MANUSCRIPT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(path)
    validate_manuscript_config(config)
    return config


def load_experiment_registry(path: str | Path = DEFAULT_EXPERIMENT_REGISTRY) -> dict[str, Any]:
    registry = _load_yaml(path)
    expected = {"FCT-CL", "EXP1", "EXP2", "EXP3", "EXP4", "EXP5"}
    experiments = {name: registry[name] for name in expected if name in registry}
    missing = sorted(expected - set(experiments))
    if missing:
        raise ValueError(f"experiment registry is missing {missing}")
    validate_experiment_registry(registry)
    registry["experiments"] = experiments
    return registry


def _require(mapping: Mapping[str, Any], path: str, expected: Any) -> None:
    cursor: Any = mapping
    for name in path.split("."):
        if not isinstance(cursor, Mapping) or name not in cursor:
            raise ValueError(f"missing manuscript contract field {path}")
        cursor = cursor[name]
    if cursor != expected:
        raise ValueError(f"{path} must be {expected!r}, got {cursor!r}")


def validate_manuscript_config(config: Mapping[str, Any]) -> None:
    """Reject silent drift in values that determine the published protocol."""

    exact = {
        "software_version": "2.0.0",
        "roles.order": ["local_utility", "approach_to_constraint", "resource_pressure"],
        "roles.weights": [1.0, 0.5, 0.7],
        "satellite.independent_reporting_runs": 10,
        "satellite.users": 128,
        "satellite.satellites": 160,
        "satellite.control_step_s": 0.1,
        "satellite.physical_step_s": 0.001,
        "satellite.candidate_generation.top_k": 6,
        "satellite.candidate_generation.hard_feasibility_mask": True,
        "satellite.switching.minimum_dwell_steps": 10,
        "satellite.switching.hysteresis": 1.0 / 6.0,
        "uav.independent_reporting_runs": 5,
        "uav.control_step_s": 1.0,
        "uav.candidate_generation.top_k": 3,
        "uav.candidate_generation.hard_feasibility_mask": True,
        "model.hidden_dim": 128,
        "model.time_encoding_dim": 16,
        "model.message_passing_layers": 2,
        "model.message_operators.kan.knots": 16,
        "model.message_operators.physick.kernel_count": 16,
        "model.message_operators.physick.deployed_projection_radius": 1.0,
        "statistics.satellite_default.interval": "two_sided_student_t_95",
        "statistics.satellite_default.degrees_of_freedom": 9,
    }
    for path, expected in exact.items():
        _require(config, path, expected)


def validate_experiment_registry(registry: Mapping[str, Any]) -> None:
    """Reject design drift in the executable FCT/EXP1--EXP5 registry."""

    exact = {
        "FCT-CL.design": "interface_by_operator_2x2",
        "FCT-CL.interfaces": ["snapshot", "intensity_flow"],
        "FCT-CL.operators": ["mlp", "physick"],
        "FCT-CL.independent_runs_per_cell": 10,
        "FCT-CL.held_out_episodes_per_run": 30,
        "EXP1.contracts": ["intensity_flow", "snapshot"],
        "EXP1.models": ["mlp", "physick"],
        "EXP1.paired_seeds_per_contract": 20,
        "EXP1.checkpoints_per_model": 50,
        "EXP1.held_out_episodes_per_checkpoint": 30,
        "EXP1.selection_uses_outcomes": False,
        "EXP1.aggregate_equivalence_band": [0.95, 1.05],
        "EXP1.field_equivalence_band": [0.90, 1.10],
        "EXP1.aba_windows_s": [2.0, 2.5, 3.0, 4.0, 6.0, 10.0],
        "EXP2.phases": ["oracle_driven", "frozen_predictor"],
        "EXP2.factors.descriptor": {"D0": "snapshot", "D1": "intensity_flow"},
        "EXP2.factors.staging": {"Stage0": "current_epoch", "Stage1": "causal_staged"},
        "EXP2.factors.score_map": {"S0": "static_weighted", "S1": "hazard_aware"},
        "EXP2.cells": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
        "EXP2.matched_seed_blocks_per_cell": 10,
        "EXP2.held_out_episodes_per_block": 30,
        "EXP2.native_executable_score_bindings": ["D0_x_S0", "D1_x_S1"],
        "EXP2.author_adapter_required_score_bindings": ["D0_x_S1", "D1_x_S0"],
        "EXP2.historical_eight_cell_rerun_from_release": False,
        "EXP3.matched_seed_blocks": 20,
        "EXP3.checkpoints_per_arm": 50,
        "EXP3.held_out_episodes_per_block": 30,
        "EXP3.transitions_per_arm": 400000,
        "EXP3.gradient_updates_per_arm": 31250,
        "EXP3.aba_ratio_of_rate_ratios.correction": "add_0.5_to_each_four_arm_event_count",
        "EXP4.included_in_manuscript": False,
        "EXP4.reproducible_from_release": False,
        "EXP4.provenance_status": "incomplete",
        "EXP5.matched_seed_blocks": 12,
        "EXP5.checkpoints_per_variant": 50,
        "EXP5.held_out_episodes_per_block": 30,
        "EXP5.projected_variant_radius": 1.25,
        "EXP5.deployed_radius": 1.0,
        "EXP5.predictive_loss_equivalence_band": [0.95, 1.05],
    }
    for path, expected in exact.items():
        _require(registry, path, expected)

    rules = registry["EXP1"].get("selection_rules", [])
    if len(rules) != 5 or rules[0] != "minimum_validation_loss":
        raise ValueError("EXP1 must register the five frozen validation-only rules")
    strata = registry["EXP1"].get("near_tie_strata", [])
    if strata != [[0.0, 0.025], [0.025, 0.050], [0.050, 0.100]]:
        raise ValueError("EXP1 near-tie strata do not match the manuscript")
    if set(registry["EXP3"].get("arms", {})) != {"A1", "A2", "B1", "B2"}:
        raise ValueError("EXP3 must contain the four frozen objective/operator arms")
    if registry["EXP5"].get("variants") != [
        "mlp",
        "generic_dynamic_mixture",
        "unprojected_physick",
        "projected_physick",
    ]:
        raise ValueError("EXP5 variants must preserve the ordered mechanism chain")
