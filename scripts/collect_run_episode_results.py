#!/usr/bin/env python3
"""Collect original platform evaluation bundles into the canonical result CSV.

The adapter registry is intentionally closed: unknown artifact kinds, schema
versions, metric containers, cells, and missing shard hashes are rejected. The
collector never accepts manuscript aggregates or display-derived source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "code"))

from shared.integrity import audit_result_records  # noqa: E402
from shared.records import write_result_records  # noqa: E402
from shared.schemas import ResultRecord  # noqa: E402


REGISTRY_VERSION = 1

SATELLITE_OUTCOME_METRICS = {
    "outage_rate",
    "handover_failure_rate",
    "pingpong_rate_user_second",
    "pingpong_fraction_executed_decisions",
    "user_p10_throughput_mean",
    "user_p10_throughput_ratio_to_oracle",
    "active_satellite_load_cv_mean",
    "mean_throughput",
}
SATELLITE_DECISION_METRICS = {
    "top1_agreement",
    "candidate_kendall_tau",
    "near_tie_flip_rate",
    "oracle_score_regret",
}
UAV_OUTCOME_METRICS = {
    "assignment_failure_rate",
    "service_failure_per_attempt",
    "queue_rejection_rate",
    "service_completion_rate",
    "uavs_with_completion_fraction",
    "reassociation_rate_uav_second",
    "pingpong_rate_uav_second",
    "mean_waiting_delay_s",
    "p90_waiting_delay_s",
    "mean_service_delay_s",
    "p90_service_delay_s",
    "energy_depleted_uav_fraction",
    "mean_final_energy_fraction",
    "tail_service_p10_completed_missions",
    "tail_service_ratio_to_oracle",
    "tail_residual_energy_p10",
    "tail_residual_energy_ratio_to_oracle",
    "station_load_cv_mean",
    "station_load_cv_p90",
    "station_flow_cv_mean",
    "station_flow_cv_p90",
    "station_flow_pressure_pearson",
}
UAV_DECISION_METRICS = {
    "top1_rank_agreement",
    "candidate_kendall_tau",
    "near_tie_flip_rate",
    "oracle_action_agreement",
    "oracle_rank_score_regret",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_pt(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read original evaluation bundles") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch with weights_only loading support is required") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} root must be a mapping")
    return payload


def _resolve_path(registry_path: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = registry_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _strict_keys(mapping: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")


def _cell(mapping: Mapping[str, Any], condition: str) -> Mapping[str, str]:
    cells = mapping.get("cells")
    if not isinstance(cells, Mapping) or condition not in cells:
        raise KeyError(f"adapter registry has no cell mapping for {condition!r}")
    value = cells[condition]
    if not isinstance(value, Mapping):
        raise TypeError(f"cell {condition!r} must be a mapping")
    required = {"cell_id", "method", "interface", "operator"}
    if set(value) != required:
        raise ValueError(f"cell {condition!r} requires exactly {sorted(required)}")
    if any(not str(value[key]).strip() for key in required):
        raise ValueError(f"cell {condition!r} has an empty identity field")
    return {key: str(value[key]) for key in required}


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _metric_support(platform: str, section: Mapping[str, Any], metric: str) -> tuple[float | None, float | None, int | None]:
    if platform == "satellite":
        epochs = int(section.get("epochs", 0) or 0)
        users = int(section.get("user_count", 0) or 0)
        if metric == "handover_failure_rate":
            return _number(section.get("handover_failures")), _number(section.get("handover_attempts")), None
        if metric == "pingpong_rate_user_second":
            return (
                _number(section.get("pingpong_events")),
                _number(section.get("user_time_exposure_s")),
                epochs * users or None,
            )
        if metric == "outage_rate":
            denominator = epochs * users
            value = _number(section.get(metric))
            return (None if value is None else value * denominator), (float(denominator) if denominator else None), denominator or None
        if metric in SATELLITE_DECISION_METRICS:
            return None, None, int(section.get("comparisons", 0) or 0) or None
        return None, None, epochs or None
    epochs = int(section.get("epochs", 0) or 0)
    users = int(section.get("uav_count", 0) or 0)
    pairs = {
        "assignment_failure_rate": ("assignment_failures", "assignment_attempts"),
        "service_failure_per_attempt": ("service_admission_failures", "service_admission_attempts"),
        "queue_rejection_rate": ("queue_rejections", None),
        "service_completion_rate": ("service_completions", "service_starts"),
    }
    if metric in pairs:
        numerator_key, denominator_key = pairs[metric]
        numerator = _number(section.get(numerator_key))
        if metric == "queue_rejection_rate":
            denominator = float(int(section.get("queue_rejections", 0)) + int(section.get("queue_admissions", 0)))
        else:
            denominator = _number(section.get(str(denominator_key)))
        return numerator, denominator, None
    if metric in {"reassociation_rate_uav_second", "pingpong_rate_uav_second"}:
        numerator_key = (
            "reassociations"
            if metric == "reassociation_rate_uav_second"
            else "pingpong_events"
        )
        return (
            _number(section.get(numerator_key)),
            _number(section.get("uav_time_exposure_s")),
            epochs * users or None,
        )
    if metric in UAV_DECISION_METRICS:
        support_key = "near_tie_comparisons" if metric == "near_tie_flip_rate" else "comparisons"
        return None, None, int(section.get(support_key, 0) or 0) or None
    if metric in {"uavs_with_completion_fraction", "energy_depleted_uav_fraction", "mean_final_energy_fraction"}:
        return None, float(users) if users else None, users or None
    return None, None, epochs or users or None


def _unit_rows(
    *,
    platform: str,
    entry: Mapping[str, Any],
    condition_name: str,
    metric_row: Mapping[str, Any],
    run_id: str,
    checkpoint_id: str,
    episode_id: str,
    exogenous_id: str,
    stress_id: str,
    split: str,
) -> list[ResultRecord]:
    cell = _cell(entry, condition_name)
    if platform == "satellite":
        _strict_keys(
            metric_row,
            {"outcomes", "decision_fidelity", "classical_oracle_regret"},
            f"satellite metric row {condition_name}",
        )
        allowed = (
            ("outcomes", SATELLITE_OUTCOME_METRICS),
            ("decision_fidelity", SATELLITE_DECISION_METRICS),
            ("classical_oracle_regret", {"oracle_score_regret"}),
        )
    else:
        _strict_keys(
            metric_row,
            {
                "episode_seed",
                "mode",
                "protocol_fingerprint",
                "paired_fingerprint",
                "outcomes",
                "decision_fidelity",
            },
            f"UAV metric row {condition_name}",
        )
        allowed = (("outcomes", UAV_OUTCOME_METRICS), ("decision_fidelity", UAV_DECISION_METRICS))
    rows: list[ResultRecord] = []
    for section_name, metric_names in allowed:
        section = metric_row.get(section_name)
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise TypeError(f"{condition_name}.{section_name} must be a mapping")
        missing_metrics = metric_names - set(section)
        if missing_metrics:
            raise ValueError(
                f"{condition_name}.{section_name} is missing required metrics: "
                f"{sorted(missing_metrics)}"
            )
        for metric in sorted(metric_names):
            value = _number(section.get(metric))
            numerator, denominator, support_n = _metric_support(platform, section, metric)
            defined = value is not None
            if denominator is not None and denominator <= 0:
                defined = False
                value = None
            if metric == "near_tie_flip_rate" and support_n is not None and support_n <= 0:
                defined = False
                value = None
            rows.append(
                ResultRecord(
                    platform=platform,
                    condition_id=str(entry["condition_id"]),
                    cell_id=cell["cell_id"],
                    method=cell["method"],
                    interface=cell["interface"],
                    operator=cell["operator"],
                    run_id=run_id,
                    checkpoint_id=checkpoint_id,
                    episode_id=episode_id,
                    exogenous_sequence_id=exogenous_id,
                    metric=metric,
                    value=value,
                    defined=defined,
                    provenance="original_experiment_output",
                    numerator=numerator,
                    denominator=denominator,
                    support_n=support_n,
                    unit=(
                        "seconds"
                        if metric.endswith("_delay_s")
                        else "events_per_user_second"
                        if metric == "pingpong_rate_user_second"
                        else "events_per_uav_second"
                        if metric in {
                            "reassociation_rate_uav_second",
                            "pingpong_rate_uav_second",
                        }
                        else "dimensionless"
                    ),
                    stress_id=stress_id,
                    split=split,
                    missing_reason="" if defined else "undefined_or_missing_denominator_in_original_bundle",
                )
            )
        if platform == "satellite" and section_name == "outcomes":
            source = next(
                row
                for row in rows
                if row.metric == "user_p10_throughput_ratio_to_oracle"
            )
            rows.append(
                ResultRecord(
                    platform=source.platform,
                    condition_id=source.condition_id,
                    cell_id=source.cell_id,
                    method=source.method,
                    interface=source.interface,
                    operator=source.operator,
                    run_id=source.run_id,
                    checkpoint_id=source.checkpoint_id,
                    episode_id=source.episode_id,
                    exogenous_sequence_id=source.exogenous_sequence_id,
                    metric="tail_service_degradation",
                    value=(None if source.value is None else 1.0 - source.value),
                    defined=source.defined,
                    provenance="derived_from_original_experiment_output",
                    support_n=source.support_n,
                    unit="dimensionless",
                    stress_id=source.stress_id,
                    split=source.split,
                    missing_reason=source.missing_reason,
                )
            )
    return rows


def _collect_satellite(path: Path, entry: Mapping[str, Any]) -> list[ResultRecord]:
    payload = _load_pt(path)
    if payload.get("schema_version") != 2:
        raise ValueError("satellite adapter requires schema_version=2")
    manifest = payload.get("manifest")
    units = payload.get("units")
    if not isinstance(manifest, Mapping) or manifest.get("kind") != "leo_pg_action_coupled_evaluation":
        raise ValueError("satellite artifact kind is not recognized")
    if not isinstance(units, list) or not units:
        raise ValueError("satellite bundle contains no evaluation units")
    run_id = str(entry.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("satellite registry entry requires run_id")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not str(checkpoint.get("sha256", "")).strip():
        raise ValueError("satellite manifest has no checkpoint SHA-256")
    checkpoint_id = "sha256:" + str(checkpoint["sha256"])
    rows: list[ResultRecord] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            raise TypeError("satellite evaluation unit must be a mapping")
        metrics = unit.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("satellite evaluation unit has no original metric rows")
        base_seed = int(unit["base_seed"])
        episode_index = int(unit["episode_index"])
        episode_seed = int(unit["episode_seed"])
        episode_id = f"base-{base_seed}-e{episode_index:03d}"
        for kind in ("paired", "classical"):
            conditions = metrics.get(kind, {})
            if not isinstance(conditions, Mapping):
                raise TypeError(f"satellite metrics.{kind} must be a mapping")
            for name, metric_row in conditions.items():
                if not isinstance(metric_row, Mapping):
                    raise TypeError(f"satellite metric row {kind}/{name} must be a mapping")
                condition_name = f"{kind}/{name}"
                rows.extend(
                    _unit_rows(
                        platform="satellite",
                        entry=entry,
                        condition_name=condition_name,
                        metric_row=metric_row,
                        run_id=run_id,
                        checkpoint_id=checkpoint_id,
                        episode_id=episode_id,
                        exogenous_id=f"episode-seed:{episode_seed}",
                        stress_id=str(entry.get("stress_id", "nominal")),
                        split=str(entry.get("split", "test")),
                    )
                )
    return rows


def _collect_uav(path: Path, entry: Mapping[str, Any]) -> list[ResultRecord]:
    payload = _load_pt(path)
    if payload.get("uav_evaluation_bundle_version") != 2:
        raise ValueError("UAV adapter requires uav_evaluation_bundle_version=2")
    if payload.get("artifact_kind") != "uav_cfs.paired_action_coupled_evaluation":
        raise ValueError("UAV artifact kind is not recognized")
    units = payload.get("paired_units")
    checkpoints = payload.get("checkpoints")
    if not isinstance(units, list) or not units or not isinstance(checkpoints, Mapping):
        raise ValueError("UAV bundle lacks paired units or checkpoint identities")
    rows: list[ResultRecord] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            raise TypeError("UAV paired unit must be a mapping")
        relative = Path(str(unit["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("UAV shard path must remain within the bundle directory")
        shard = (path.parent / relative).resolve()
        if path.parent.resolve() not in shard.parents or not shard.is_file():
            raise FileNotFoundError(shard)
        if _sha256(shard) != str(unit.get("sha256", "")):
            raise ValueError(f"UAV shard hash mismatch: {relative}")
        shard_payload = _load_pt(shard)
        if shard_payload.get("uav_evaluation_pair_version") != 2 or shard_payload.get("artifact_kind") != "uav_cfs.paired_substitution_episode":
            raise ValueError(f"unrecognized UAV episode shard: {relative}")
        metrics_container = shard_payload.get("metrics")
        if not isinstance(metrics_container, Mapping):
            raise ValueError("UAV episode shard has no original metrics")
        closed_loop = metrics_container.get("closed_loop")
        if not isinstance(closed_loop, Mapping):
            raise ValueError("UAV episode shard has no closed_loop metric rows")
        run_seed = int(shard_payload["run_seed"])
        local_episode_index = int(shard_payload["local_episode_index"])
        episode_seed = int(shard_payload["episode_seed"])
        paired_episode_id = str(shard_payload.get("paired_episode_id", "")).strip()
        exogenous_sequence_id = str(shard_payload.get("exogenous_sequence_id", "")).strip()
        if not paired_episode_id or not exogenous_sequence_id:
            raise ValueError("UAV shard lacks paired_episode_id or exogenous_sequence_id")
        if str(unit.get("paired_episode_id", "")).strip() != paired_episode_id:
            raise ValueError(f"UAV bundle/shard paired episode mismatch: {relative}")
        if str(unit.get("exogenous_sequence_id", "")).strip() != exogenous_sequence_id:
            raise ValueError(f"UAV bundle/shard exogenous sequence mismatch: {relative}")
        if exogenous_sequence_id != str(episode_seed):
            raise ValueError(f"UAV exogenous sequence is not keyed by the episode seed: {relative}")
        checkpoint = checkpoints.get(str(run_seed), checkpoints.get(run_seed))
        if not isinstance(checkpoint, Mapping) or not str(checkpoint.get("sha256", "")).strip():
            raise ValueError(f"UAV checkpoint identity is missing for run {run_seed}")
        run_id = f"run-{run_seed}"
        checkpoint_id = "sha256:" + str(checkpoint["sha256"])
        for mode, metric_row in closed_loop.items():
            if not isinstance(metric_row, Mapping):
                raise TypeError(f"UAV metric row {mode!r} must be a mapping")
            rows.extend(
                _unit_rows(
                    platform="uav",
                    entry=entry,
                    condition_name=str(mode),
                    metric_row=metric_row,
                    run_id=run_id,
                    checkpoint_id=checkpoint_id,
                    episode_id=paired_episode_id,
                    exogenous_id=exogenous_sequence_id,
                    stress_id=str(entry.get("stress_id", "nominal")),
                    split=str(entry.get("split", "test")),
                )
            )
        one_step = metrics_container.get("one_step_model")
        one_step_cell = str(entry.get("one_step_cell", "")).strip()
        if one_step is not None:
            if not isinstance(one_step, Mapping) or not one_step_cell:
                raise ValueError("UAV one-step metrics require one_step_cell in the adapter registry")
            cell = _cell(entry, one_step_cell)
            transitions = int(one_step.get("transitions", 0) or 0)
            for name in ("total", "eta", "log1p_intensity", "flow", "feasibility"):
                value = _number(one_step.get(name))
                if value is None:
                    raise ValueError(f"UAV one-step metric {name} is missing or non-finite")
                rows.append(
                    ResultRecord(
                        platform="uav",
                        condition_id=str(entry["condition_id"]),
                        cell_id=cell["cell_id"],
                        method=cell["method"],
                        interface=cell["interface"],
                        operator=cell["operator"],
                        run_id=run_id,
                        checkpoint_id=checkpoint_id,
                        episode_id=paired_episode_id,
                        exogenous_sequence_id=exogenous_sequence_id,
                        metric=f"one_step_{name}",
                        value=value,
                        defined=True,
                        provenance="original_experiment_output",
                        support_n=transitions or None,
                        unit="dimensionless",
                        stress_id=str(entry.get("stress_id", "nominal")),
                        split=str(entry.get("split", "test")),
                    )
                )
    return rows


ADAPTERS: dict[str, Callable[[Path, Mapping[str, Any]], list[ResultRecord]]] = {
    "leo_pg_action_coupled_evaluation_v2": _collect_satellite,
    "uav_cfs_paired_action_coupled_evaluation_v2": _collect_uav,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-runs", type=int, default=5,
        help="Independent training runs required per cell (paper default: 5)",
    )
    parser.add_argument(
        "--expected-episodes", type=int, default=30,
        help="Matched held-out episodes required per run (paper default: 30)",
    )
    args = parser.parse_args()
    registry_path = args.registry.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, Mapping) or registry.get("registry_version") != REGISTRY_VERSION:
        raise ValueError(f"registry_version must equal {REGISTRY_VERSION}")
    _strict_keys(registry, {"registry_version", "bundles"}, "registry")
    bundles = registry.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("registry.bundles must be a non-empty list")
    rows: list[ResultRecord] = []
    for index, raw in enumerate(bundles):
        if not isinstance(raw, Mapping):
            raise TypeError(f"registry bundle {index} must be a mapping")
        _strict_keys(
            raw,
            {"adapter", "path", "condition_id", "run_id", "stress_id", "split", "cells", "one_step_cell"},
            f"registry bundle {index}",
        )
        adapter_name = str(raw.get("adapter", ""))
        adapter = ADAPTERS.get(adapter_name)
        if adapter is None:
            raise ValueError(f"unknown adapter {adapter_name!r}; allowed: {sorted(ADAPTERS)}")
        if not str(raw.get("condition_id", "")).strip():
            raise ValueError(f"registry bundle {index} requires condition_id")
        path = _resolve_path(registry_path, str(raw.get("path", "")))
        rows.extend(adapter(path, raw))
    errors = audit_result_records(
        rows,
        expected_runs=args.expected_runs,
        expected_episodes=args.expected_episodes,
    )
    if errors:
        raise ValueError("result integrity audit failed:\n- " + "\n- ".join(errors))
    write_result_records(args.output, rows)
    receipt = {
        "registry_version": REGISTRY_VERSION,
        "input_registry_sha256": _sha256(registry_path),
        "record_count": len(rows),
        "output": args.output.name,
        "output_sha256": _sha256(args.output),
        "adapters": sorted({str(item["adapter"]) for item in bundles}),
        "deterministic_derived_metrics": {
            "tail_service_degradation": "1 - user_p10_throughput_ratio_to_oracle"
        },
    }
    args.output.with_suffix(args.output.suffix + ".collection.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
