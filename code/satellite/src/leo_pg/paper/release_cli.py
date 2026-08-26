"""Publication-study planner for the full satellite implementation.

The planner never invents missing publication inputs.  It can write an
auditable command/configuration plan with explicit blockers, and it refuses to
execute a plan until the author artifact manifest is complete.  Every command
in a ready plan calls the copied full paper data, training or evaluation path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .models import EXPLORATORY_METHODS, PAPER_METHODS


SATELLITE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = SATELLITE_ROOT / "configs" / "paper_protocol.yaml"
DEFAULT_STUDIES = SATELLITE_ROOT / "configs" / "studies.yaml"
DEFAULT_AUTHOR = (
    SATELLITE_ROOT
    / "configs"
    / "author"
    / "paper_artifacts.template.yaml"
)

_MISSING_SENTINELS = {
    "required",
    "to_be_filled",
    "todo",
    "tbd",
    "<required>",
    "[required]",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    study: str
    stage: str
    method: str | None
    run_id: str | None
    condition: str
    command: list[str]
    inputs: list[str]
    outputs: list[str]
    depends_on: list[str]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {resolved}")
    return value


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return (
            not normalized
            or normalized in _MISSING_SENTINELS
            or normalized.startswith("required_")
            or normalized.startswith("required:")
        )
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _unresolved_paths(value: Any, *, prefix: str = "config") -> list[str]:
    """Return paths containing explicit unresolved-value sentinels."""

    unresolved: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            unresolved.extend(_unresolved_paths(child, prefix=f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            unresolved.extend(
                _unresolved_paths(child, prefix=f"{prefix}[{index}]")
            )
    elif isinstance(value, str) and _missing(value):
        unresolved.append(prefix)
    return unresolved


def publication_config_blockers(config: Mapping[str, Any]) -> list[str]:
    """Safety gate for configurations admitted to the publication runner."""

    blockers: list[str] = []
    if bool(config.get("api_fixture_only", False)):
        blockers.append("base configuration is marked api_fixture_only")
    ephemeris = config.get("ephemeris", {})
    if not isinstance(ephemeris, Mapping):
        blockers.append("ephemeris must be a mapping")
    else:
        mode = str(ephemeris.get("mode", "")).strip().lower()
        if mode in {"", "debug", "synthetic", "fixture"}:
            blockers.append(
                "publication execution requires a non-debug author-supplied ephemeris"
            )
        if mode == "skyfield_tle":
            tle_path = ephemeris.get("tle_path")
            if _missing(tle_path):
                blockers.append("ephemeris.tle_path is unresolved")
            elif not Path(str(tle_path)).expanduser().is_file():
                blockers.append("ephemeris.tle_path does not exist")
            elif Path(str(tle_path)).expanduser().resolve() == (
                SATELLITE_ROOT / "data" / "synthetic_fixture_500.tle"
            ).resolve():
                blockers.append(
                    "publication execution cannot use the bundled synthetic TLE fixture"
                )
            if _missing(ephemeris.get("start_time_utc")):
                blockers.append("ephemeris.start_time_utc is unresolved")
    blockers.extend(
        f"unresolved sentinel at {path}" for path in _unresolved_paths(config)
    )
    return sorted(set(blockers))


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return copy.deepcopy(dict(value))


def _runs(author: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = author.get("runs", [])
    if not isinstance(raw, list):
        raise TypeError("author manifest runs must be a list")
    return [dict(item) if isinstance(item, Mapping) else {} for item in raw]


def audit_author_manifest(
    author: Mapping[str, Any],
    *,
    studies: Iterable[str],
) -> list[str]:
    requested = set(studies)
    blockers: list[str] = []
    runs = _runs(author)
    if len(runs) != 10:
        blockers.append("runs must contain exactly ten independent run entries")
    run_ids: list[str] = []
    checkpoint_ids: list[str] = []
    for index, run in enumerate(runs):
        prefix = f"runs[{index}]"
        run_id = run.get("run_id")
        if _missing(run_id):
            blockers.append(f"{prefix}.run_id is missing")
        else:
            run_ids.append(str(run_id))
        checkpoint_id = run.get("checkpoint_id")
        if _missing(checkpoint_id):
            blockers.append(f"{prefix}.checkpoint_id is missing")
        else:
            checkpoint_ids.append(str(checkpoint_id))
        for field in (
            "initialization_seed",
            "optimizer_seed",
            "data_order_seed",
            "validation_sampling_seed",
        ):
            value = run.get(field)
            if value is None:
                blockers.append(f"{prefix}.{field} is missing")
            elif isinstance(value, bool) or int(value) != value or int(value) < 0:
                blockers.append(f"{prefix}.{field} must be a non-negative integer")
    if len(run_ids) != len(set(run_ids)):
        blockers.append("run_id values must be unique")
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        blockers.append("checkpoint_id values must be unique")

    panels = author.get("held_out_panels", {})
    if not isinstance(panels, Mapping):
        blockers.append("held_out_panels must be a mapping")
        panels = {}
    for run_id in run_ids:
        panel = panels.get(run_id, {})
        if not isinstance(panel, Mapping):
            blockers.append(f"held_out_panels.{run_id} must be a mapping")
            continue
        for field in ("episode_seeds", "episode_ids", "exogenous_sequence_ids"):
            values = panel.get(field, [])
            if not isinstance(values, list) or len(values) != 30:
                blockers.append(
                    f"held_out_panels.{run_id}.{field} must contain exactly 30 values"
                )
            elif len(values) != len(set(map(str, values))):
                blockers.append(f"held_out_panels.{run_id}.{field} contains duplicates")

    parameters = author.get("publication_parameters", {})
    if not isinstance(parameters, Mapping):
        blockers.append("publication_parameters must be a mapping")
        parameters = {}
    required_parameters = (
        "tle_training_snapshot",
        "tle_training_start_time_utc",
        "training_ephemeris_overrides",
        "phy_proxy_parameters",
        "flow_phi",
        "cox_baseline_hazard",
        "gamma_covariate_scale_db",
        "elevation_covariate_scale_deg",
        "policy_initializer",
        "snapshot_stream_initializer",
        "snapshot_initial_oracle_admitted_load",
        "multitask_loss_weights",
        "scheduled_sampling_schedule",
        "snapshot_margin_scales",
        "node_injection",
        "validation_episode_count",
        "dataset_split_seed",
        "learned_control_training_parameters",
        "exploratory_backbone_configs",
        "snapshot_learned_control_score_weights",
        "a3_ttt_unit",
        "cho_selected_parameters",
        "load_aware_selected_parameters",
    )
    for field in required_parameters:
        if _missing(parameters.get(field)):
            blockers.append(f"publication_parameters.{field} is missing")
    for field in (
        "training_ephemeris_overrides",
        "phy_proxy_parameters",
        "policy_initializer",
        "snapshot_stream_initializer",
        "multitask_loss_weights",
        "scheduled_sampling_schedule",
        "snapshot_margin_scales",
        "learned_control_training_parameters",
        "exploratory_backbone_configs",
        "snapshot_learned_control_score_weights",
    ):
        value = parameters.get(field)
        if not _missing(value) and not isinstance(value, Mapping):
            blockers.append(f"publication_parameters.{field} must be a mapping")
    for field in ("validation_episode_count", "dataset_split_seed"):
        value = parameters.get(field)
        if not _missing(value) and (
            isinstance(value, bool)
            or int(value) != value
            or int(value) < (1 if field == "validation_episode_count" else 0)
        ):
            blockers.append(
                f"publication_parameters.{field} must be an explicit "
                + ("positive" if field == "validation_episode_count" else "non-negative")
                + " integer"
            )
    nested_requirements = {
        "learned_control_training_parameters": ("ltt_r", "da_gwm"),
        "exploratory_backbone_configs": EXPLORATORY_METHODS,
        "snapshot_learned_control_score_weights": (
            "snapshot_ltt_r",
            "snapshot_da_gwm",
        ),
    }
    for field, keys in nested_requirements.items():
        value = parameters.get(field)
        if isinstance(value, Mapping):
            for key in keys:
                if not isinstance(value.get(key), Mapping):
                    blockers.append(
                        f"publication_parameters.{field}.{key} must be a mapping"
                    )
    ttt_unit = str(parameters.get("a3_ttt_unit", "")).strip().lower()
    if ttt_unit and ttt_unit not in {"steps", "seconds"}:
        blockers.append("publication_parameters.a3_ttt_unit must be steps or seconds")
    for field in ("cho_selected_parameters", "load_aware_selected_parameters"):
        if not _missing(parameters.get(field)) and not isinstance(
            parameters.get(field), Mapping
        ):
            blockers.append(f"publication_parameters.{field} must be a mapping")
    training_tle = parameters.get("tle_training_snapshot")
    if not _missing(training_tle) and not Path(str(training_tle)).expanduser().is_file():
        blockers.append("publication_parameters.tle_training_snapshot does not exist")

    if requested.intersection({"DEC-K", "STRS-K", "LONG-HORIZON-300S"}):
        stress = author.get("stress_index_overrides", {})
        if not isinstance(stress, Mapping):
            blockers.append("stress_index_overrides must be a mapping")
        else:
            needed: set[str] = set()
            if "DEC-K" in requested:
                needed.update({"100", "300", "500"})
            if "STRS-K" in requested:
                needed.update({"100", "150", "200", "300", "400", "500"})
            if "LONG-HORIZON-300S" in requested:
                needed.update({"100", "500"})
            for value in sorted(needed, key=int):
                if not isinstance(stress.get(value), Mapping):
                    blockers.append(
                        f"stress_index_overrides.{value} must be an explicit mapping"
                    )

    if "EPH-REPLAY" in requested:
        replays = author.get("ephemeris_replays", {})
        if not isinstance(replays, Mapping):
            blockers.append("ephemeris_replays must be a mapping")
        else:
            for family in ("starlink", "oneweb", "kuiper", "mixed"):
                entry = replays.get(family, {})
                if not isinstance(entry, Mapping):
                    blockers.append(f"ephemeris_replays.{family} must be a mapping")
                    continue
                for field in ("tle_path", "start_time_utc", "source_url", "retrieved_at"):
                    if _missing(entry.get(field)):
                        blockers.append(f"ephemeris_replays.{family}.{field} is missing")
                tle = entry.get("tle_path")
                if not _missing(tle) and not Path(str(tle)).expanduser().is_file():
                    blockers.append(f"ephemeris_replays.{family}.tle_path does not exist")

    if "ABLATIONS" in requested:
        ablations = author.get("ablation_overrides", {})
        if not isinstance(ablations, Mapping):
            blockers.append("ablation_overrides must be a mapping")
        else:
            for name in (
                "full",
                "without_kernel_bank",
                "without_cox_descriptor",
                "instantaneous_hazard_only",
                "without_thinning",
                "cut_load_sinr",
                "static_flow_ema",
                "broadcast_aggregation",
                "no_temporal_memory",
            ):
                if not isinstance(ablations.get(name), Mapping):
                    blockers.append(f"ablation_overrides.{name} must be a mapping")
    return sorted(set(blockers))


def _apply_publication_parameters(
    config: Mapping[str, Any], author: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    parameters = _mapping(author, "publication_parameters")
    protocol = result.setdefault("paper_protocol", {})
    ephemeris = result.setdefault("ephemeris", {})
    if not _missing(parameters.get("tle_training_snapshot")):
        ephemeris.update(
            {
                "mode": "skyfield_tle",
                "tle_path": str(parameters["tle_training_snapshot"]),
                "start_time_utc": str(parameters["tle_training_start_time_utc"]),
            }
        )
        result.pop("S_sats", None)
    if isinstance(parameters.get("training_ephemeris_overrides"), Mapping):
        result["ephemeris"] = _deep_merge(
            result.get("ephemeris", {}),
            dict(parameters["training_ephemeris_overrides"]),
        )
    if isinstance(parameters.get("phy_proxy_parameters"), Mapping):
        protocol["phy"] = _deep_merge(
            protocol.get("phy", {}), dict(parameters["phy_proxy_parameters"])
        )
    flow = protocol.setdefault("flow", {})
    if not _missing(parameters.get("flow_phi")):
        flow["phi"] = parameters["flow_phi"]
    intensity = protocol.setdefault("intensity", {})
    for target, source in (
        ("baseline_hazard", "cox_baseline_hazard"),
        ("gamma_covariate_scale_db", "gamma_covariate_scale_db"),
        ("elevation_covariate_scale_deg", "elevation_covariate_scale_deg"),
    ):
        if not _missing(parameters.get(source)):
            intensity[target] = parameters[source]
    if isinstance(parameters.get("policy_initializer"), Mapping):
        protocol["policy_initializer"] = copy.deepcopy(
            dict(parameters["policy_initializer"])
        )
    snapshot = result.setdefault("paper_snapshot", {})
    if isinstance(parameters.get("snapshot_stream_initializer"), Mapping):
        snapshot["stream_initializer"] = copy.deepcopy(
            dict(parameters["snapshot_stream_initializer"])
        )
    if not _missing(parameters.get("snapshot_initial_oracle_admitted_load")):
        snapshot["initial_oracle_admitted_load"] = copy.deepcopy(
            parameters["snapshot_initial_oracle_admitted_load"]
        )
    if isinstance(parameters.get("multitask_loss_weights"), Mapping):
        result.setdefault("paper_training", {})["loss_weights"] = copy.deepcopy(
            dict(parameters["multitask_loss_weights"])
        )
    if isinstance(parameters.get("scheduled_sampling_schedule"), Mapping):
        schedule = copy.deepcopy(dict(parameters["scheduled_sampling_schedule"]))
        schedule["enabled"] = True
        result.setdefault("paper_training", {})["scheduled_sampling"] = schedule
    scales = parameters.get("snapshot_margin_scales")
    if isinstance(scales, Mapping):
        result.setdefault("paper_snapshot", {})["margin"] = _deep_merge(
            result.setdefault("paper_snapshot", {}).get("margin", {}), scales
        )
    if not _missing(parameters.get("node_injection")):
        result.setdefault("model", {})["node_injection"] = parameters[
            "node_injection"
        ]
    validation_count = parameters.get("validation_episode_count")
    if not _missing(validation_count):
        count = int(validation_count)
        result.setdefault("paper_dataset", {})["episodes"] = 2000 + count + 30
        result["paper_dataset"]["split_counts"] = [2000, count, 30]
    if not _missing(parameters.get("dataset_split_seed")):
        result.setdefault("paper_dataset", {})["split_seed"] = int(
            parameters["dataset_split_seed"]
        )
    baseline_registry = result.setdefault("paper_baselines", {})
    learned = parameters.get("learned_control_training_parameters")
    if isinstance(learned, Mapping):
        for method in ("ltt_r", "da_gwm"):
            override = learned.get(method)
            if isinstance(override, Mapping):
                baseline_registry[method] = _deep_merge(
                    baseline_registry.get(method, {}),
                    {"training": dict(override)},
                )
    exploratory = parameters.get("exploratory_backbone_configs")
    if isinstance(exploratory, Mapping):
        for method in EXPLORATORY_METHODS:
            override = exploratory.get(method)
            if isinstance(override, Mapping):
                baseline_registry[method] = _deep_merge(
                    baseline_registry.get(method, {}), dict(override)
                )
    snapshot_weights = parameters.get("snapshot_learned_control_score_weights")
    if isinstance(snapshot_weights, Mapping):
        policies = snapshot.setdefault("policies", {})
        for method in ("snapshot_ltt_r", "snapshot_da_gwm"):
            weights = snapshot_weights.get(method)
            if isinstance(weights, Mapping):
                policies.setdefault(method, {})["score_weights"] = copy.deepcopy(
                    dict(weights)
                )
    sweeps = result.setdefault("controller_sweeps", {})
    if not isinstance(sweeps, dict):
        raise TypeError("controller_sweeps must be a mapping")
    if not _missing(parameters.get("a3_ttt_unit")):
        a3_sweep = sweeps.setdefault("a3", {})
        a3_sweep["ttt_unit"] = str(
            parameters["a3_ttt_unit"]
        ).strip().lower()
        selected = a3_sweep.get("selected", {})
        if (
            a3_sweep["ttt_unit"] == "seconds"
            and isinstance(selected, dict)
            and "ttt_steps" in selected
        ):
            selected["ttt_seconds"] = selected.pop("ttt_steps")
    if isinstance(parameters.get("cho_selected_parameters"), Mapping):
        sweeps.setdefault("cho", {})["selected"] = copy.deepcopy(
            dict(parameters["cho_selected_parameters"])
        )
    if isinstance(parameters.get("load_aware_selected_parameters"), Mapping):
        sweeps.setdefault("load_aware_greedy", {})["selected"] = copy.deepcopy(
            dict(parameters["load_aware_selected_parameters"])
        )
    result["release_external_parameter_provenance"] = copy.deepcopy(parameters)
    return result


def _run_config(
    base: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    condition_id: str,
    condition_label: str,
    blockers: Sequence[str],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    init_seed = run.get("initialization_seed")
    optimizer_seed = run.get("optimizer_seed")
    order_seed = run.get("data_order_seed")
    val_seed = run.get("validation_sampling_seed")
    if init_seed is not None:
        result["seed"] = int(init_seed)
    training = result.setdefault("paper_training", {})
    if optimizer_seed is not None:
        training["optimizer_seed"] = int(optimizer_seed)
    if order_seed is not None:
        training["sampling_seed"] = int(order_seed)
    if val_seed is not None:
        training["validation_sampling_seed"] = int(val_seed)
    result["release_run_manifest"] = {
        "condition_id": condition_id,
        "condition_label": condition_label,
        "run_id": run.get("run_id"),
        "source_checkpoint_id": run.get("checkpoint_id"),
        "initialization_seed": init_seed,
        "optimizer_seed": optimizer_seed,
        "optimizer_reinitialized": True,
        "optimizer_state_initialization": "fresh_AdamW_state",
        "data_order_seed": order_seed,
        "validation_sampling_seed": val_seed,
        "execution_ready": not blockers,
        "unresolved_external_inputs": list(blockers),
    }
    return result


class Planner:
    def __init__(
        self,
        *,
        base: Mapping[str, Any],
        author: Mapping[str, Any],
        studies: Mapping[str, Any],
        selected: Sequence[str],
        output_root: Path,
        blockers: Sequence[str],
        device: str,
    ) -> None:
        self.base = _apply_publication_parameters(base, author)
        self.author = author
        self.study_registry = _mapping(studies, "studies")
        self.selected = tuple(selected)
        self.output_root = output_root
        self.blockers = list(blockers)
        self.device = device
        self.tasks: list[Task] = []
        self._task_ids: set[str] = set()
        self._training: dict[tuple[str, str, str], tuple[str, Path, Path, Path]] = {}

    def add(self, task: Task) -> None:
        if task.task_id in self._task_ids:
            raise ValueError(f"duplicate task id: {task.task_id}")
        self._task_ids.add(task.task_id)
        self.tasks.append(task)

    def write_config(
        self,
        study: str,
        run: Mapping[str, Any],
        label: str,
        override: Mapping[str, Any] | None = None,
    ) -> Path:
        config = _run_config(
            self.base,
            run,
            condition_id=study,
            condition_label=label,
            blockers=self.blockers,
        )
        if override:
            config = _deep_merge(config, override)
        path = (
            self.output_root
            / "resolved_configs"
            / study
            / str(run.get("run_id", "missing_run"))
            / f"{label}.yaml"
        )
        _write_yaml(path, config)
        return path

    def ensure_training(
        self,
        *,
        study: str,
        run: Mapping[str, Any],
        method: str,
        label: str,
        override: Mapping[str, Any] | None = None,
        training_objective: str | None = None,
        rollout_horizon: int | None = None,
    ) -> tuple[str, Path, Path, Path]:
        run_id = str(run.get("run_id", "missing_run"))
        cache_key = (run_id, method, f"{study}:{label}")
        if cache_key in self._training:
            return self._training[cache_key]
        normalized = str(method)
        if normalized not in PAPER_METHODS:
            raise ValueError(f"method is not in the publication registry: {method}")
        condition_override = copy.deepcopy(dict(override or {}))
        if training_objective is not None:
            objective = str(training_objective).strip().lower()
            if objective not in {"teacher_forcing", "scheduled_sampling"}:
                raise ValueError(
                    "training_objective must be teacher_forcing or scheduled_sampling"
                )
            condition_override = _deep_merge(
                condition_override,
                {
                    "paper_training": {
                        "objective": {
                            "kind": (
                                "one_step_teacher_forcing"
                                if objective == "teacher_forcing"
                                else "one_step_scheduled_sampling"
                            ),
                            "rollout_horizon": 1,
                        },
                        "scheduled_sampling": {
                            "enabled": objective == "scheduled_sampling"
                        },
                        "action_coupled_multistep": {"enabled": False},
                    }
                },
            )
        if rollout_horizon is not None:
            condition_override = _deep_merge(
                condition_override,
                {
                    "paper_training": {
                        "objective": {
                            "kind": "action_coupled_multistep",
                            "rollout_horizon": int(rollout_horizon),
                        },
                        "action_coupled_multistep": {
                            "enabled": True,
                            "horizons": [int(rollout_horizon)],
                            "rollout_horizon": int(rollout_horizon),
                            "model_feedback_probability": 1.0,
                        },
                        "scheduled_sampling": {"enabled": False},
                    }
                },
            )
        config = self.write_config(study, run, label, condition_override)
        snapshot = normalized.startswith("snapshot_")
        dataset_label = normalized if snapshot else "intensity_flow"
        dataset = (
            self.output_root / "datasets" / study / run_id / f"{label}_{dataset_label}.pt"
        )
        checkpoint = (
            self.output_root / "checkpoints" / study / run_id / label / "checkpoint.pt"
        )
        generate_id = f"{study}:{run_id}:{label}:generate"
        train_id = f"{study}:{run_id}:{label}:train"
        generator = "paper_snapshot_generate.py" if snapshot else "paper_generate.py"
        parameters = self.author.get("publication_parameters", {})
        parameters = parameters if isinstance(parameters, Mapping) else {}
        raw_validation_count = parameters.get("validation_episode_count")
        validation_count = (
            int(raw_validation_count)
            if not _missing(raw_validation_count)
            else 0
        )
        total_episodes = 2000 + validation_count + 30
        generate_command = [
            "python",
            str(SATELLITE_ROOT / "scripts" / generator),
            "--cfg",
            str(config),
            "--out",
            str(dataset),
            "--episodes",
            str(total_episodes),
            "--split-counts",
            f"2000,{validation_count},30",
            "--device",
            self.device,
        ]
        if snapshot:
            generate_command[4:4] = ["--method", normalized]
        self.add(
            Task(
                task_id=generate_id,
                study=study,
                stage="generate",
                method=normalized,
                run_id=run_id,
                condition=label,
                command=generate_command,
                inputs=[str(config)],
                outputs=[str(dataset)],
                depends_on=[],
            )
        )
        trainer = "paper_snapshot_train.py" if snapshot else "paper_train.py"
        self.add(
            Task(
                task_id=train_id,
                study=study,
                stage="train",
                method=normalized,
                run_id=run_id,
                condition=label,
                command=[
                    "python",
                    str(SATELLITE_ROOT / "scripts" / trainer),
                    "--cfg",
                    str(config),
                    "--data",
                    str(dataset),
                    "--method",
                    normalized,
                    "--out",
                    str(checkpoint),
                    "--device",
                    self.device,
                ],
                inputs=[str(config), str(dataset)],
                outputs=[str(checkpoint)],
                depends_on=[generate_id],
            )
        )
        value = (train_id, config, dataset, checkpoint)
        self._training[cache_key] = value
        return value

    def episode_seeds(self, run_id: str) -> list[int]:
        panels = self.author.get("held_out_panels", {})
        panel = panels.get(run_id, {}) if isinstance(panels, Mapping) else {}
        raw = panel.get("episode_seeds", []) if isinstance(panel, Mapping) else []
        return [int(value) for value in raw if value is not None]

    def add_rollout_evaluation(
        self,
        *,
        study: str,
        run: Mapping[str, Any],
        method: str,
        label: str,
        train_task: str,
        config: Path,
        dataset: Path,
        checkpoint: Path,
        modes: Sequence[str],
        horizon_steps: int = 600,
        allow_config_mismatch: bool = False,
        controller_sweep: bool = False,
    ) -> None:
        run_id = str(run.get("run_id", "missing_run"))
        snapshot = method.startswith("snapshot_")
        evaluator = "paper_snapshot_evaluate.py" if snapshot else "paper_evaluate.py"
        output = self.output_root / "evaluation" / study / run_id / f"{label}.pt"
        seeds = self.episode_seeds(run_id)
        command = [
            "python",
            str(SATELLITE_ROOT / "scripts" / evaluator),
            "--cfg",
            str(config),
            "--method",
            method,
            "--ckpt",
            str(checkpoint),
            "--out",
            str(output),
            "--seeds",
            ",".join(map(str, seeds)),
            "--episodes",
            "1",
            "--horizon",
            str(int(horizon_steps)),
            "--device",
            self.device,
        ]
        if snapshot:
            command.extend(["--data", str(dataset)])
        else:
            command.extend(["--modes", ",".join(modes), "--metrics"])
            if allow_config_mismatch:
                command.append("--allow-config-mismatch")
            if controller_sweep:
                command.append("--controller-sweep")
        self.add(
            Task(
                task_id=f"{study}:{run_id}:{label}:evaluate",
                study=study,
                stage="evaluate",
                method=method,
                run_id=run_id,
                condition=label,
                command=command,
                inputs=[str(config), str(dataset), str(checkpoint)],
                outputs=[str(output)],
                depends_on=[train_task],
            )
        )

    def nominal_runs(self) -> list[dict[str, Any]]:
        return _runs(self.author)

    def build(self) -> list[Task]:
        for study in self.selected:
            self._build_study(study)
        return self.tasks

    def _build_study(self, study: str) -> None:
        spec = self.study_registry.get(study)
        if not isinstance(spec, Mapping):
            raise ValueError(f"unknown study: {study}")
        runs = self.nominal_runs()
        methods = [str(item) for item in spec.get("methods", [])]

        if study == "FCT-CL":
            for run in runs:
                for method in methods:
                    label = method
                    train, config, data, ckpt = self.ensure_training(
                        study=study, run=run, method=method, label=label
                    )
                    modes = ["model", "oracle"] if method.startswith("snapshot_") else [
                        "model", "oracle", "gamma", "intensity", "flow", "intensity_flow"
                    ]
                    self.add_rollout_evaluation(
                        study=study,
                        run=run,
                        method=method,
                        label=label,
                        train_task=train,
                        config=config,
                        dataset=data,
                        checkpoint=ckpt,
                        modes=modes,
                    )
            return

        if study in {"DEC-K", "STRS-K"}:
            stress = self.author.get("stress_index_overrides", {})
            stress = stress if isinstance(stress, Mapping) else {}
            train_override = stress.get("100", {})
            train_override = train_override if isinstance(train_override, Mapping) else {}
            for run in runs:
                for method in methods:
                    train, train_cfg, data, ckpt = self.ensure_training(
                        study=study,
                        run=run,
                        method=method,
                        label=f"train_K100_{method}",
                        override=train_override,
                    )
                    for index in spec.get("stress_indices", []):
                        override = stress.get(str(index), {})
                        override = override if isinstance(override, Mapping) else {}
                        eval_cfg = self.write_config(
                            study, run, f"K{index}_{method}_eval", override
                        )
                        run_id = str(run.get("run_id", "missing_run"))
                        if study == "DEC-K":
                            decision_data = (
                                self.output_root
                                / "datasets"
                                / study
                                / run_id
                                / f"K{index}_{method}_held_out.pt"
                            )
                            decision_generate_id = (
                                f"{study}:{run_id}:K{index}_{method}:generate_held_out"
                            )
                            self.add(
                                Task(
                                    task_id=decision_generate_id,
                                    study=study,
                                    stage="generate_held_out_next_step",
                                    method=method,
                                    run_id=run_id,
                                    condition=f"K{index}",
                                    command=[
                                        "python",
                                        str(
                                            SATELLITE_ROOT
                                            / "scripts"
                                            / "paper_decision_generate.py"
                                        ),
                                        "--cfg",
                                        str(eval_cfg),
                                        "--out",
                                        str(decision_data),
                                        "--seeds",
                                        ",".join(map(str, self.episode_seeds(run_id))),
                                        "--horizon",
                                        "200",
                                        "--device",
                                        self.device,
                                    ],
                                    inputs=[str(eval_cfg)],
                                    outputs=[str(decision_data)],
                                    depends_on=[],
                                )
                            )
                            output = (
                                self.output_root
                                / "evaluation"
                                / study
                                / run_id
                                / f"K{index}_{method}.json"
                            )
                            self.add(
                                Task(
                                    task_id=f"{study}:{run_id}:K{index}_{method}:evaluate",
                                    study=study,
                                    stage="evaluate_next_step",
                                    method=method,
                                    run_id=run_id,
                                    condition=f"K{index}",
                                    command=[
                                        "python",
                                        str(SATELLITE_ROOT / "scripts" / "paper_decision_evaluate.py"),
                                        "--cfg", str(eval_cfg),
                                        "--data", str(decision_data),
                                        "--ckpt", str(ckpt),
                                        "--method", method,
                                        "--out", str(output),
                                        "--run-id", run_id,
                                        "--stress-index", str(index),
                                        "--near-tie-margin", str(spec.get("near_tie_margin", 0.05)),
                                        "--max-episodes", "30",
                                        "--device", self.device,
                                        "--allow-config-mismatch",
                                    ],
                                    inputs=[str(eval_cfg), str(decision_data), str(ckpt)],
                                    outputs=[str(output)],
                                    depends_on=[train, decision_generate_id],
                                )
                            )
                        else:
                            self.add_rollout_evaluation(
                                study=study,
                                run=run,
                                method=method,
                                label=f"K{index}_{method}",
                                train_task=train,
                                config=eval_cfg,
                                dataset=data,
                                checkpoint=ckpt,
                                modes=["model", "oracle"],
                                allow_config_mismatch=True,
                            )
            return

        if study == "LONG-HORIZON-300S":
            stress = self.author.get("stress_index_overrides", {})
            stress = stress if isinstance(stress, Mapping) else {}
            train_override = stress.get("100", {})
            train_override = train_override if isinstance(train_override, Mapping) else {}
            evaluation_override = stress.get(str(spec.get("stress_index", 500)), {})
            evaluation_override = (
                evaluation_override
                if isinstance(evaluation_override, Mapping)
                else {}
            )
            horizon_steps = int(spec.get("horizon_steps", 3000))
            for run in runs:
                for method in methods:
                    train, _, data, ckpt = self.ensure_training(
                        study=study,
                        run=run,
                        method=method,
                        label=f"train_K100_{method}",
                        override=train_override,
                    )
                    config = self.write_config(
                        study,
                        run,
                        f"K500_300s_{method}",
                        evaluation_override,
                    )
                    self.add_rollout_evaluation(
                        study=study,
                        run=run,
                        method=method,
                        label=f"K500_300s_{method}",
                        train_task=train,
                        config=config,
                        dataset=data,
                        checkpoint=ckpt,
                        modes=["model", "oracle"],
                        horizon_steps=horizon_steps,
                        allow_config_mismatch=True,
                    )
            return

        if study == "FCT-DWELL":
            for run in runs:
                train, _, data, ckpt = self.ensure_training(
                    study=study, run=run, method="tgn_physick", label="anchor_D10"
                )
                for dwell in spec.get("dwell_steps", []):
                    override = {"paper_protocol": {"policy": {"min_dwell_steps": int(dwell)}}}
                    config = self.write_config(study, run, f"D{dwell}", override)
                    self.add_rollout_evaluation(
                        study=study,
                        run=run,
                        method="tgn_physick",
                        label=f"D{dwell}",
                        train_task=train,
                        config=config,
                        dataset=data,
                        checkpoint=ckpt,
                        modes=["model", "oracle"],
                        allow_config_mismatch=int(dwell) != 10,
                    )
            return

        if study == "EPH-REPLAY":
            replays = self.author.get("ephemeris_replays", {})
            replays = replays if isinstance(replays, Mapping) else {}
            for run in runs:
                for method in methods:
                    train, _, data, ckpt = self.ensure_training(
                        study=study, run=run, method=method, label=f"train_{method}"
                    )
                    for family in spec.get("replay_families", []):
                        entry = replays.get(str(family), {})
                        entry = entry if isinstance(entry, Mapping) else {}
                        override = {
                            "ephemeris": {
                                "mode": "skyfield_tle",
                                "tle_path": entry.get("tle_path"),
                                "start_time_utc": entry.get("start_time_utc"),
                                "source_url": entry.get("source_url"),
                                "retrieved_at": entry.get("retrieved_at"),
                            }
                        }
                        config = self.write_config(
                            study, run, f"{family}_{method}", override
                        )
                        self.add_rollout_evaluation(
                            study=study,
                            run=run,
                            method=method,
                            label=f"{family}_{method}",
                            train_task=train,
                            config=config,
                            dataset=data,
                            checkpoint=ckpt,
                            modes=["model", "oracle"],
                            allow_config_mismatch=True,
                        )
            return

        if study == "ABLATIONS":
            ablations = self.author.get("ablation_overrides", {})
            ablations = ablations if isinstance(ablations, Mapping) else {}
            for run in runs:
                for condition in spec.get("named_conditions", []):
                    override = ablations.get(str(condition), {})
                    override = override if isinstance(override, Mapping) else {}
                    train, config, data, ckpt = self.ensure_training(
                        study=study,
                        run=run,
                        method="tgn_physick",
                        label=str(condition),
                        override=override,
                    )
                    self.add_rollout_evaluation(
                        study=study,
                        run=run,
                        method="tgn_physick",
                        label=str(condition),
                        train_task=train,
                        config=config,
                        dataset=data,
                        checkpoint=ckpt,
                        modes=["model", "oracle"],
                    )
                for exploratory in EXPLORATORY_METHODS:
                    train, config, data, ckpt = self.ensure_training(
                        study=study,
                        run=run,
                        method=exploratory,
                        label=f"exploratory_{exploratory}",
                    )
                    self.add_rollout_evaluation(
                        study=study,
                        run=run,
                        method=exploratory,
                        label=f"exploratory_{exploratory}",
                        train_task=train,
                        config=config,
                        dataset=data,
                        checkpoint=ckpt,
                        modes=["model", "oracle"],
                    )
            return

        if study == "ROLLOUT-AWARE":
            for run in runs:
                for method in methods:
                    for objective in ("teacher_forcing", "scheduled_sampling"):
                        label = f"{method}_{objective}"
                        train, config, data, ckpt = self.ensure_training(
                            study=study,
                            run=run,
                            method=method,
                            label=label,
                            training_objective=objective,
                        )
                        self.add_rollout_evaluation(
                            study=study,
                            run=run,
                            method=method,
                            label=label,
                            train_task=train,
                            config=config,
                            dataset=data,
                            checkpoint=ckpt,
                            modes=["model", "oracle"],
                        )
                    for horizon in spec.get("rollout_horizons", []):
                        label = f"{method}_H{horizon}"
                        train, config, data, ckpt = self.ensure_training(
                            study=study,
                            run=run,
                            method=method,
                            label=label,
                            rollout_horizon=int(horizon),
                        )
                        self.add_rollout_evaluation(
                            study=study,
                            run=run,
                            method=method,
                            label=label,
                            train_task=train,
                            config=config,
                            dataset=data,
                            checkpoint=ckpt,
                            modes=["model", "oracle"],
                        )
            return

        if study == "SCORE-WEIGHT-SENSITIVITY":
            perturbations = spec.get("perturbations", {})
            if not isinstance(perturbations, Mapping):
                raise TypeError("SCORE-WEIGHT-SENSITIVITY perturbations must be a mapping")
            for run in runs:
                train, _, data, ckpt = self.ensure_training(
                    study=study,
                    run=run,
                    method="tgn_physick",
                    label="fixed_checkpoint",
                )
                for label, weights in perturbations.items():
                    if not isinstance(weights, Mapping):
                        raise TypeError(f"score-weight perturbation {label} must be a mapping")
                    config = self.write_config(
                        study,
                        run,
                        str(label),
                        {"paper_protocol": {"policy": dict(weights)}},
                    )
                    self.add_rollout_evaluation(
                        study=study,
                        run=run,
                        method="tgn_physick",
                        label=str(label),
                        train_task=train,
                        config=config,
                        dataset=data,
                        checkpoint=ckpt,
                        modes=["model", "oracle"],
                        allow_config_mismatch=True,
                    )
            return

        if study == "SNAPSHOT-WEIGHT-CONTRACT":
            for run in runs:
                for method in methods:
                    train, config, data, ckpt = self.ensure_training(
                        study=study,
                        run=run,
                        method=method,
                        label=f"{method}_fixed_role_weights",
                    )
                    self.add_rollout_evaluation(
                        study=study,
                        run=run,
                        method=method,
                        label=f"{method}_fixed_role_weights",
                        train_task=train,
                        config=config,
                        dataset=data,
                        checkpoint=ckpt,
                        modes=["model", "oracle"],
                    )
            return

        if study in {"ORACLE-LADDER", "LEARNED-CONTROLS", "CLASSICAL-CONTROLS"}:
            for run in runs:
                for method in methods:
                    train, config, data, ckpt = self.ensure_training(
                        study=study, run=run, method=method, label=method
                    )
                    modes = ["model", "oracle"]
                    if study == "ORACLE-LADDER":
                        modes = ["model", "oracle", "gamma", "intensity", "flow", "intensity_flow"]
                    self.add_rollout_evaluation(
                        study=study,
                        run=run,
                        method=method,
                        label=method,
                        train_task=train,
                        config=config,
                        dataset=data,
                        checkpoint=ckpt,
                        modes=modes,
                        controller_sweep=study == "CLASSICAL-CONTROLS",
                    )
            return
        raise ValueError(f"study has no planner implementation: {study}")


def _selected_studies(value: str, registry: Mapping[str, Any]) -> list[str]:
    available = list(_mapping(registry, "studies"))
    if value.strip().lower() == "all":
        return available
    selected = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError("unknown studies: " + ", ".join(unknown))
    if not selected:
        raise ValueError("select at least one study")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory", help="Print methods and study IDs")
    inventory.add_argument("--studies", default=str(DEFAULT_STUDIES))

    validate = sub.add_parser("validate", help="Audit the author artifact manifest")
    validate.add_argument("--studies", default=str(DEFAULT_STUDIES))
    validate.add_argument("--author", default=str(DEFAULT_AUTHOR))
    validate.add_argument("--select", default="all")

    plan = sub.add_parser("plan", help="Write resolved configs and a dry-run plan")
    plan.add_argument("--config", default=str(DEFAULT_CONFIG))
    plan.add_argument("--studies", default=str(DEFAULT_STUDIES))
    plan.add_argument("--author", default=str(DEFAULT_AUTHOR))
    plan.add_argument("--select", default="all")
    plan.add_argument("--out", required=True)
    plan.add_argument("--device", default="cuda")

    execute = sub.add_parser("run-plan", help="Execute a ready plan")
    execute.add_argument("--plan", required=True)
    execute.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement; without it no command is run",
    )
    return parser


def _inventory(studies_path: str) -> dict[str, Any]:
    studies = _load_yaml(studies_path)
    return {
        "publication_methods": [
            method for method in PAPER_METHODS if method not in EXPLORATORY_METHODS
        ],
        "exploratory_methods": list(EXPLORATORY_METHODS),
        "studies": _mapping(studies, "studies"),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        print(json.dumps(_inventory(args.studies), indent=2, sort_keys=True))
        return

    if args.command == "validate":
        studies = _load_yaml(args.studies)
        selected = _selected_studies(args.select, studies)
        author = _load_yaml(args.author)
        blockers = audit_author_manifest(author, studies=selected)
        print(
            json.dumps(
                {"selected_studies": selected, "execution_ready": not blockers, "blockers": blockers},
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(0 if not blockers else 2)

    if args.command == "plan":
        base = _load_yaml(args.config)
        studies = _load_yaml(args.studies)
        author = _load_yaml(args.author)
        selected = _selected_studies(args.select, studies)
        blockers = audit_author_manifest(author, studies=selected)
        resolved_base = _apply_publication_parameters(base, author)
        blockers.extend(publication_config_blockers(resolved_base))
        blockers = sorted(set(blockers))
        output_root = Path(args.out).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        planner = Planner(
            base=base,
            author=author,
            studies=studies,
            selected=selected,
            output_root=output_root,
            blockers=blockers,
            device=args.device,
        )
        tasks = planner.build()
        payload = {
            "schema_version": 1,
            "paper_version": "current_main_and_si_2026_08_26",
            "satellite_code_root": str(SATELLITE_ROOT),
            "selected_studies": selected,
            "execution_ready": not blockers,
            "blockers": blockers,
            "base_config": str(Path(args.config).expanduser().resolve()),
            "author_manifest": str(Path(args.author).expanduser().resolve()),
            "author_manifest_sha256": _canonical_hash(author),
            "task_count": len(tasks),
            "tasks": [asdict(task) for task in tasks],
        }
        plan_path = output_root / "satellite_plan.json"
        plan_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "plan": str(plan_path),
                    "task_count": len(tasks),
                    "execution_ready": not blockers,
                    "blocker_count": len(blockers),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "run-plan":
        plan_path = Path(args.plan).expanduser().resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not args.execute:
            raise SystemExit("No command was run. Add --execute after inspecting the plan.")
        if not bool(plan.get("execution_ready", False)):
            raise SystemExit(
                "Plan is not execution-ready. Resolve every listed external-artifact blocker."
            )
        tasks = plan.get("tasks", [])
        if not isinstance(tasks, list):
            raise TypeError("plan tasks must be a list")
        config_paths = {
            Path(str(path)).expanduser().resolve()
            for task in tasks
            if isinstance(task, Mapping)
            for path in task.get("inputs", [])
            if str(path).lower().endswith((".yaml", ".yml"))
        }
        active_config_blockers: list[str] = []
        for config_path in sorted(config_paths):
            if not config_path.is_file():
                active_config_blockers.append(f"missing config: {config_path}")
                continue
            for blocker in publication_config_blockers(_load_yaml(config_path)):
                active_config_blockers.append(f"{config_path}: {blocker}")
        if active_config_blockers:
            raise SystemExit(
                "Active publication configuration failed the safety gate:\n- "
                + "\n- ".join(active_config_blockers)
            )
        completed: set[str] = set()
        env = dict(os.environ)
        source = str(SATELLITE_ROOT / "src")
        env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        for raw in tasks:
            if not isinstance(raw, Mapping):
                raise TypeError("every task must be a mapping")
            dependencies = set(map(str, raw.get("depends_on", [])))
            missing_dependencies = dependencies.difference(completed)
            if missing_dependencies:
                raise RuntimeError(
                    f"task {raw.get('task_id')} has unmet dependencies: {sorted(missing_dependencies)}"
                )
            command = list(map(str, raw.get("command", [])))
            if not command:
                raise ValueError(f"task {raw.get('task_id')} has an empty command")
            subprocess.run(command, cwd=SATELLITE_ROOT, env=env, check=True)
            completed.add(str(raw["task_id"]))
        print(json.dumps({"completed_tasks": len(completed)}, indent=2))
        return

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
