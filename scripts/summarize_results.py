#!/usr/bin/env python3
"""Recompute platform-specific run-first summaries and paired contrasts.

Satellite inference uses ten equal-weight run means with two-sided Student-t
intervals (df=9) and paired Student-t contrasts. UAV inference uses five
independent runs with the registered run-first hierarchical bootstrap.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "code"))

from shared.integrity import audit_result_records  # noqa: E402
from shared.records import read_result_records  # noqa: E402
from shared.schemas import ResultRecord  # noqa: E402
from shared.statistics import (  # noqa: E402
    BootstrapMode,
    paired_hierarchical_bootstrap,
    paired_run_first_student_t,
    run_first_student_t,
)


class Estimator(str, Enum):
    RUN_FIRST_STUDENT_T = "run_first_student_t"
    RUN_FIRST_HIERARCHICAL_BOOTSTRAP = "run_first_hierarchical_bootstrap"


@dataclass(frozen=True)
class PlatformContract:
    platform: str
    estimator: Estimator
    expected_runs: int
    expected_episodes: int
    confidence: float = 0.95
    bootstrap_draws: int | None = None
    bootstrap_seed: int | None = None
    bootstrap_mode: BootstrapMode | None = None


PLATFORM_CONTRACTS: dict[str, PlatformContract] = {
    "satellite": PlatformContract(
        platform="satellite",
        estimator=Estimator.RUN_FIRST_STUDENT_T,
        expected_runs=10,
        expected_episodes=30,
    ),
    "uav": PlatformContract(
        platform="uav",
        estimator=Estimator.RUN_FIRST_HIERARCHICAL_BOOTSTRAP,
        expected_runs=5,
        expected_episodes=30,
        bootstrap_draws=10_000,
        bootstrap_seed=20260815,
        bootstrap_mode=BootstrapMode.NESTED_WITHIN_RUN,
    ),
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute the registered platform-specific run-first estimator."
    )
    parser.add_argument(
        "--platform",
        choices=sorted(PLATFORM_CONTRACTS),
        required=True,
        help="Platform represented by the input records and analysis plan.",
    )
    parser.add_argument(
        "--estimator",
        choices=[value.value for value in Estimator],
        required=True,
        help="Estimator must match the registered platform contract.",
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Original run-by-episode CSV."
    )
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        required=True,
        help="Platform-specific JSON analysis plan.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--draws",
        type=int,
        help="UAV bootstrap draws; defaults to the registered plan value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="UAV bootstrap seed; defaults to the registered plan value.",
    )
    parser.add_argument(
        "--bootstrap-mode",
        choices=[mode.value for mode in BootstrapMode],
        help="UAV bootstrap mode; defaults to nested_within_run.",
    )
    return parser


def _resolve_cli_contract(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> PlatformContract:
    contract = PLATFORM_CONTRACTS[args.platform]
    estimator = Estimator(args.estimator)
    if estimator is not contract.estimator:
        parser.error(
            f"platform {args.platform!r} requires estimator "
            f"{contract.estimator.value!r}, not {estimator.value!r}"
        )
    bootstrap_options = (args.draws, args.seed, args.bootstrap_mode)
    if contract.platform == "satellite" and any(
        value is not None for value in bootstrap_options
    ):
        parser.error("bootstrap options are valid only for the UAV estimator")
    if args.draws is not None and args.draws <= 0:
        parser.error("--draws must be positive")
    if (
        contract.platform == "uav"
        and args.bootstrap_mode is not None
        and BootstrapMode(args.bootstrap_mode) is not contract.bootstrap_mode
    ):
        parser.error(
            "the UAV estimator requires bootstrap mode "
            f"{contract.bootstrap_mode.value!r}"
        )
    return contract


def _load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("analysis plan must contain a JSON object")
    return dict(value)


def _validate_plan(
    plan: Mapping[str, Any], contract: PlatformContract
) -> Mapping[str, Any]:
    if plan.get("schema_version") != 2:
        raise ValueError("analysis plan schema_version must be 2")
    if plan.get("platform") != contract.platform:
        raise ValueError(
            f"analysis plan platform must be {contract.platform!r}, "
            f"got {plan.get('platform')!r}"
        )
    if plan.get("estimator") != contract.estimator.value:
        raise ValueError(
            f"analysis plan estimator must be {contract.estimator.value!r}, "
            f"got {plan.get('estimator')!r}"
        )
    contracts = plan.get("platform_contracts")
    if not isinstance(contracts, Mapping) or contract.platform not in contracts:
        raise ValueError(
            f"analysis plan must define platform_contracts.{contract.platform}"
        )
    active = contracts[contract.platform]
    if not isinstance(active, Mapping):
        raise TypeError(f"platform_contracts.{contract.platform} must be an object")
    exact = {
        "estimator": contract.estimator.value,
        "independent_unit": "training_run",
        "point_estimate": "equal_weight_mean_of_run_means",
        "expected_runs": contract.expected_runs,
        "expected_episodes_per_run": contract.expected_episodes,
        "confidence": contract.confidence,
    }
    if contract.platform == "satellite":
        exact.update(
            {
                "summary_interval": "two_sided_student_t",
                "contrast_interval": "paired_two_sided_student_t",
                "degrees_of_freedom": 9,
            }
        )
    for field, expected in exact.items():
        if active.get(field) != expected:
            raise ValueError(
                f"platform_contracts.{contract.platform}.{field} must be "
                f"{expected!r}, got {active.get(field)!r}"
            )
    if contract.platform == "uav":
        bootstrap = active.get("bootstrap")
        if not isinstance(bootstrap, Mapping):
            raise ValueError("the UAV plan must define its bootstrap contract")
        expected_bootstrap = {
            "mode": BootstrapMode.NESTED_WITHIN_RUN.value,
            "draws": 10_000,
            "seed": 20260815,
            "paired_keys": [
                "platform",
                "condition_id",
                "stress_id",
                "split",
                "run_id",
                "episode_id",
                "exogenous_sequence_id",
            ],
        }
        for field, expected in expected_bootstrap.items():
            if bootstrap.get(field) != expected:
                raise ValueError(
                    f"platform_contracts.uav.bootstrap.{field} must be "
                    f"{expected!r}, got {bootstrap.get(field)!r}"
                )
    for section in ("summaries", "contrasts"):
        items = plan.get(section, [])
        if not isinstance(items, list):
            raise TypeError(f"analysis plan {section} must be a list")
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise TypeError(f"{section}[{index}] must be an object")
            if item.get("platform") != contract.platform:
                raise ValueError(
                    f"{section}[{index}].platform must be {contract.platform!r}"
                )
    return active


def _validate_input_platform(
    records: Iterable[ResultRecord], contract: PlatformContract
) -> list[ResultRecord]:
    rows = list(records)
    platforms = {row.platform for row in rows}
    if platforms != {contract.platform}:
        raise ValueError(
            f"input records must contain only platform {contract.platform!r}; "
            f"found {sorted(platforms)!r}"
        )
    errors = audit_result_records(
        rows,
        expected_runs=contract.expected_runs,
        expected_episodes=contract.expected_episodes,
    )
    if errors:
        raise ValueError(
            "input result integrity audit failed:\n- " + "\n- ".join(errors)
        )
    return rows


def _selected_rows(
    records: Iterable[ResultRecord],
    *,
    platform: str,
    condition_id: str,
    cell_id: str,
    metric: str,
    stress_id: str,
    split: str,
) -> list[ResultRecord]:
    return [
        row
        for row in records
        if row.platform == platform
        and row.condition_id == condition_id
        and row.cell_id == cell_id
        and row.metric == metric
        and row.stress_id == stress_id
        and row.split == split
    ]


def _condition_run_panels(
    records: Iterable[ResultRecord],
    *,
    contract: PlatformContract,
    condition_id: str,
    cell_id: str,
    metric: str,
    stress_id: str,
    split: str,
) -> tuple[dict[str, dict[str, float]], int]:
    selected = _selected_rows(
        records,
        platform=contract.platform,
        condition_id=condition_id,
        cell_id=cell_id,
        metric=metric,
        stress_id=stress_id,
        split=split,
    )
    if not selected:
        raise ValueError(
            f"no records for {contract.platform}, {condition_id}, {cell_id}, {metric}"
        )
    panels: dict[str, dict[str, float]] = defaultdict(dict)
    undefined_n = 0
    for row in selected:
        if not row.defined or row.value is None:
            undefined_n += 1
            continue
        unit = f"{row.episode_id}\x1f{row.exogenous_sequence_id}"
        if unit in panels[row.run_id]:
            raise ValueError(
                f"duplicate episode/exogenous unit for {cell_id}, {metric}, "
                f"{row.run_id}, {unit!r}"
            )
        panels[row.run_id][unit] = float(row.value)
    if len(panels) != contract.expected_runs or any(not panel for panel in panels.values()):
        raise ValueError(
            f"{cell_id}, {metric} must contribute {contract.expected_runs} "
            "defined run means"
        )
    return dict(panels), undefined_n


def _summary_base(
    *,
    contract: PlatformContract,
    condition_id: str,
    cell_id: str,
    metric: str,
    stress_id: str,
    split: str,
    panels: Mapping[str, Mapping[str, float]],
    undefined_n: int,
) -> dict[str, object]:
    run_means = [float(np.mean(list(panel.values()))) for panel in panels.values()]
    return {
        "platform": contract.platform,
        "condition_id": condition_id,
        "cell_id": cell_id,
        "metric": metric,
        "stress_id": stress_id,
        "split": split,
        "estimator": contract.estimator.value,
        "estimate": float(np.mean(run_means)),
        "between_run_sd": float(np.std(run_means, ddof=1)),
        "run_n": len(run_means),
        "defined_episode_n": sum(len(panel) for panel in panels.values()),
        "undefined_episode_n": undefined_n,
    }


def _satellite_summary(
    records: Iterable[ResultRecord],
    *,
    contract: PlatformContract,
    item: Mapping[str, Any],
) -> dict[str, object]:
    stress_id = str(item.get("stress_id", "nominal"))
    split = str(item.get("split", "test"))
    panels, undefined_n = _condition_run_panels(
        records,
        contract=contract,
        condition_id=str(item["condition_id"]),
        cell_id=str(item["cell_id"]),
        metric=str(item["metric"]),
        stress_id=stress_id,
        split=split,
    )
    run_means = [float(np.mean(list(panels[run_id].values()))) for run_id in sorted(panels)]
    interval = run_first_student_t(run_means, confidence=contract.confidence)
    if interval.run_n != 10 or interval.degrees_of_freedom != 9:
        raise ValueError("satellite Student-t inference requires 10 run means and df=9")
    result = _summary_base(
        contract=contract,
        condition_id=str(item["condition_id"]),
        cell_id=str(item["cell_id"]),
        metric=str(item["metric"]),
        stress_id=stress_id,
        split=split,
        panels=panels,
        undefined_n=undefined_n,
    )
    result.update(
        {
            "estimate": interval.estimate,
            "ci_low": interval.ci_low,
            "ci_high": interval.ci_high,
            "confidence": interval.confidence,
            "degrees_of_freedom": interval.degrees_of_freedom,
            "standard_error": interval.standard_error,
        }
    )
    return result


def _bootstrap_condition_summary(
    panels: Mapping[str, Mapping[str, float]],
    *,
    draws: int,
    confidence: float,
    seed: int,
    mode: BootstrapMode,
) -> tuple[float, float, float]:
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    run_ids = sorted(panels)
    point = float(
        np.mean([np.mean(list(panels[run_id].values())) for run_id in run_ids])
    )
    common_units: list[str] = []
    if mode is BootstrapMode.CROSSED_RUN_EPISODE:
        episode_panels = [set(panels[run_id]) for run_id in run_ids]
        if any(panel != episode_panels[0] for panel in episode_panels[1:]):
            raise ValueError(
                "crossed_run_episode requires the same episode IDs in every run"
            )
        common_units = sorted(episode_panels[0])
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled_runs = rng.choice(run_ids, size=len(run_ids), replace=True)
        sampled_common = None
        if mode is BootstrapMode.CROSSED_RUN_EPISODE:
            sampled_common = rng.choice(
                common_units, size=len(common_units), replace=True
            )
        sampled_means: list[float] = []
        for run_id_value in sampled_runs:
            run_id = str(run_id_value)
            panel = panels[run_id]
            if mode is BootstrapMode.FIXED_PANEL:
                selected = list(panel.values())
            elif mode is BootstrapMode.CROSSED_RUN_EPISODE:
                assert sampled_common is not None
                selected = [panel[str(unit)] for unit in sampled_common]
            else:
                units = list(panel)
                sampled_units = rng.choice(units, size=len(units), replace=True)
                selected = [panel[str(unit)] for unit in sampled_units]
            sampled_means.append(float(np.mean(selected)))
        boot[draw] = float(np.mean(sampled_means))
    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(boot, [alpha, 1.0 - alpha])
    return point, float(ci_low), float(ci_high)


def _uav_summary(
    records: Iterable[ResultRecord],
    *,
    contract: PlatformContract,
    item: Mapping[str, Any],
    draws: int,
    seed: int,
    mode: BootstrapMode,
) -> dict[str, object]:
    stress_id = str(item.get("stress_id", "nominal"))
    split = str(item.get("split", "test"))
    panels, undefined_n = _condition_run_panels(
        records,
        contract=contract,
        condition_id=str(item["condition_id"]),
        cell_id=str(item["cell_id"]),
        metric=str(item["metric"]),
        stress_id=stress_id,
        split=split,
    )
    estimate, ci_low, ci_high = _bootstrap_condition_summary(
        panels,
        draws=draws,
        confidence=contract.confidence,
        seed=seed,
        mode=mode,
    )
    result = _summary_base(
        contract=contract,
        condition_id=str(item["condition_id"]),
        cell_id=str(item["cell_id"]),
        metric=str(item["metric"]),
        stress_id=stress_id,
        split=split,
        panels=panels,
        undefined_n=undefined_n,
    )
    result.update(
        {
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence": contract.confidence,
            "bootstrap_draws": draws,
            "bootstrap_mode": mode.value,
            "seed": seed,
        }
    )
    return result


def _paired_run_means(
    records: Iterable[ResultRecord],
    *,
    contract: PlatformContract,
    condition_id: str,
    cell_a: str,
    cell_b: str,
    metric: str,
    stress_id: str,
    split: str,
) -> tuple[list[float], list[float], int, int, int, int]:
    selected_a = _selected_rows(
        records,
        platform=contract.platform,
        condition_id=condition_id,
        cell_id=cell_a,
        metric=metric,
        stress_id=stress_id,
        split=split,
    )
    selected_b = _selected_rows(
        records,
        platform=contract.platform,
        condition_id=condition_id,
        cell_id=cell_b,
        metric=metric,
        stress_id=stress_id,
        split=split,
    )
    raw_a = {
        (row.run_id, row.episode_id, row.exogenous_sequence_id) for row in selected_a
    }
    raw_b = {
        (row.run_id, row.episode_id, row.exogenous_sequence_id) for row in selected_b
    }
    if not raw_a or raw_a != raw_b:
        raise ValueError(
            "paired Student-t requires identical run/episode/exogenous panels"
        )

    def defined_values(
        rows: Iterable[ResultRecord], cell_id: str
    ) -> tuple[dict[tuple[str, str, str], float], int]:
        values: dict[tuple[str, str, str], float] = {}
        undefined = 0
        for row in rows:
            if not row.defined or row.value is None:
                undefined += 1
                continue
            key = (row.run_id, row.episode_id, row.exogenous_sequence_id)
            if key in values:
                raise ValueError(f"duplicate paired record for {cell_id}, {metric}, {key}")
            values[key] = float(row.value)
        return values, undefined

    values_a, undefined_a = defined_values(selected_a, cell_a)
    values_b, undefined_b = defined_values(selected_b, cell_b)
    paired_keys = sorted(set(values_a) & set(values_b))
    if not paired_keys:
        raise ValueError("no defined paired run-episode observations")
    unpaired_n = len(set(values_a) ^ set(values_b))
    by_run_a: dict[str, list[float]] = defaultdict(list)
    by_run_b: dict[str, list[float]] = defaultdict(list)
    for run_id, episode_id, exogenous_id in paired_keys:
        key = (run_id, episode_id, exogenous_id)
        by_run_a[run_id].append(values_a[key])
        by_run_b[run_id].append(values_b[key])
    run_ids = sorted(set(by_run_a) & set(by_run_b))
    if len(run_ids) != contract.expected_runs:
        raise ValueError(
            f"paired contrast must contribute {contract.expected_runs} run pairs"
        )
    means_a = [float(np.mean(by_run_a[run_id])) for run_id in run_ids]
    means_b = [float(np.mean(by_run_b[run_id])) for run_id in run_ids]
    return (
        means_a,
        means_b,
        len(paired_keys),
        undefined_a,
        undefined_b,
        unpaired_n,
    )


def _satellite_contrast(
    records: Iterable[ResultRecord],
    *,
    contract: PlatformContract,
    item: Mapping[str, Any],
) -> dict[str, object]:
    condition_id = str(item["condition_id"])
    cell_a = str(item["cell_a"])
    cell_b = str(item["cell_b"])
    metric = str(item["metric"])
    stress_id = str(item.get("stress_id", "nominal"))
    split = str(item.get("split", "test"))
    means_a, means_b, paired_n, undefined_a, undefined_b, unpaired_n = (
        _paired_run_means(
            records,
            contract=contract,
            condition_id=condition_id,
            cell_a=cell_a,
            cell_b=cell_b,
            metric=metric,
            stress_id=stress_id,
            split=split,
        )
    )
    interval = paired_run_first_student_t(
        means_a, means_b, confidence=contract.confidence
    )
    if interval.run_n != 10 or interval.degrees_of_freedom != 9:
        raise ValueError("satellite paired Student-t requires 10 run pairs and df=9")
    return {
        "platform": contract.platform,
        "condition_id": condition_id,
        "metric": metric,
        "cell_a": cell_a,
        "cell_b": cell_b,
        "contrast": f"{cell_a} - {cell_b}",
        "estimator": "paired_run_first_student_t",
        "estimate": interval.estimate,
        "ci_low": interval.ci_low,
        "ci_high": interval.ci_high,
        "confidence": interval.confidence,
        "run_n": interval.run_n,
        "degrees_of_freedom": interval.degrees_of_freedom,
        "standard_error": interval.standard_error,
        "paired_episode_n": paired_n,
        "undefined_a_n": undefined_a,
        "undefined_b_n": undefined_b,
        "unpaired_n": unpaired_n,
        "stress_id": stress_id,
        "split": split,
    }


def _uav_contrast(
    records: Iterable[ResultRecord],
    *,
    contract: PlatformContract,
    item: Mapping[str, Any],
    draws: int,
    seed: int,
    mode: BootstrapMode,
) -> dict[str, object]:
    result = paired_hierarchical_bootstrap(
        records,
        platform=contract.platform,
        condition_id=str(item["condition_id"]),
        cell_a=str(item["cell_a"]),
        cell_b=str(item["cell_b"]),
        metric=str(item["metric"]),
        stress_id=str(item.get("stress_id", "nominal")),
        split=str(item.get("split", "test")),
        draws=draws,
        confidence=contract.confidence,
        seed=seed,
        mode=mode,
    ).to_dict()
    if result["run_n"] != 5:
        raise ValueError("UAV hierarchical bootstrap requires five independent runs")
    result["estimator"] = contract.estimator.value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    contract = _resolve_cli_contract(parser, args)
    plan = _load_plan(args.analysis_plan)
    active_plan = _validate_plan(plan, contract)
    records = _validate_input_platform(read_result_records(args.input), contract)

    summaries: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    if contract.platform == "satellite":
        summaries = [
            _satellite_summary(records, contract=contract, item=item)
            for item in plan.get("summaries", [])
        ]
        contrasts = [
            _satellite_contrast(records, contract=contract, item=item)
            for item in plan.get("contrasts", [])
        ]
    else:
        bootstrap_plan = active_plan["bootstrap"]
        draws = int(args.draws if args.draws is not None else bootstrap_plan["draws"])
        seed = int(args.seed if args.seed is not None else bootstrap_plan["seed"])
        mode = BootstrapMode(
            args.bootstrap_mode
            if args.bootstrap_mode is not None
            else bootstrap_plan["mode"]
        )
        summaries = [
            _uav_summary(
                records,
                contract=contract,
                item=item,
                draws=draws,
                seed=seed + index,
                mode=mode,
            )
            for index, item in enumerate(plan.get("summaries", []))
        ]
        contrasts = [
            _uav_contrast(
                records,
                contract=contract,
                item=item,
                draws=draws,
                seed=seed + len(summaries) + index,
                mode=mode,
            )
            for index, item in enumerate(plan.get("contrasts", []))
        ]

    if summaries:
        _write_csv(args.output_dir / "condition_summaries.csv", summaries)
    if contrasts:
        _write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    if not summaries and not contrasts:
        raise ValueError("analysis plan contains neither summaries nor contrasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
