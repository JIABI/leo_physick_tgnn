"""Paper-level UAV experiment contracts and intervention adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .data import load_uav_dataset
from .policy import UAVFixedRankPolicyConfig
from .state import ServiceObservation, ServicePolicyDescriptors


class UAVAblation(str, Enum):
    FULL = "full"
    NO_INTENSITY = "no_intensity"
    NO_FLOW = "no_flow"
    NO_GAIN_CONTROL = "no_gain_control"
    LOCAL_CUE_ONLY = "local_cue_only"

    @classmethod
    def parse(cls, value: str | "UAVAblation") -> "UAVAblation":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            raise ValueError(
                f"unknown UAV ablation {value!r}; choose {[item.value for item in cls]}"
            ) from error


def ablated_policy_config(
    config: UAVFixedRankPolicyConfig,
    ablation: UAVAblation | str,
) -> UAVFixedRankPolicyConfig:
    """Remove only the policy coordinate specified by the intervention."""

    selected = UAVAblation.parse(ablation)
    if selected in {UAVAblation.NO_INTENSITY, UAVAblation.LOCAL_CUE_ONLY}:
        config = replace(config, intensity_weight=0.0)
    if selected in {UAVAblation.NO_FLOW, UAVAblation.LOCAL_CUE_ONLY}:
        config = replace(config, flow_weight=0.0)
    return config


def _ablate_descriptors(
    fields: ServicePolicyDescriptors,
    ablation: UAVAblation,
) -> ServicePolicyDescriptors:
    intensity = fields.intensity_edge
    flow = fields.station_flow_node
    if ablation in {UAVAblation.NO_INTENSITY, UAVAblation.LOCAL_CUE_ONLY}:
        intensity = torch.zeros_like(intensity)
    if ablation in {UAVAblation.NO_FLOW, UAVAblation.LOCAL_CUE_ONLY}:
        flow = torch.zeros_like(flow)
    return ServicePolicyDescriptors(
        eta_edge=fields.eta_edge,
        intensity_edge=intensity,
        station_flow_node=flow,
    )


class UAVAblationDescriptorProvider:
    """Apply a nested field intervention on both recursive input and output.

    In ``no_flow`` only the separately exposed Flow coordinate is removed;
    composite Intensity is retained, matching the manuscript contract.
    """

    def __init__(self, provider: Any, ablation: UAVAblation | str) -> None:
        if not hasattr(provider, "predict_next") or not hasattr(provider, "reset"):
            raise TypeError("provider must implement reset and predict_next")
        self.provider = provider
        self.ablation = UAVAblation.parse(ablation)
        payload = (
            str(getattr(provider, "fingerprint", "provider"))
            + ":"
            + self.ablation.value
        ).encode("utf-8")
        self._fingerprint = hashlib.sha256(payload).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def reset(self) -> None:
        self.provider.reset()

    def predict_next(self, observation: ServiceObservation) -> ServicePolicyDescriptors:
        staged = observation.with_policy_descriptors(
            _ablate_descriptors(observation.policy_descriptors, self.ablation)
        )
        prediction = self.provider.predict_next(staged)
        return _ablate_descriptors(prediction, self.ablation)


@dataclass(frozen=True)
class UAVExperimentTask:
    """One checkpoint family evaluated under one simulator condition."""

    task_id: str
    family: str
    operator: str
    checkpoint_key: str
    config_key: str
    density_multiplier: int = 1
    capacity_compression: float = 1.0
    perturbation_name: str = "nominal"
    ablation: str = "full"
    substitution_modes: tuple[str, ...] = (
        "model",
        "oracle",
        "eta",
        "intensity",
        "flow",
        "intensity_flow",
    )
    include_one_step_loss: bool = False


def paper_task_matrix() -> tuple[UAVExperimentTask, ...]:
    """Enumerate every UAV experiment family reported in main/SI v8."""

    tasks: list[UAVExperimentTask] = []
    operators = ("mlp", "kan", "physick")
    for operator in operators:
        tasks.append(
            UAVExperimentTask(
                task_id=f"uav_nominal_{operator}",
                family="operator_comparison_and_decision_diagnostics",
                operator=operator,
                checkpoint_key=operator,
                config_key=operator,
                substitution_modes=("model", "oracle"),
                include_one_step_loss=True,
            )
        )
        for multiplier in (1, 2, 3, 4, 5):
            tasks.append(
                UAVExperimentTask(
                    task_id=f"uav_density_m{multiplier}_{operator}",
                    family="density_multiplier_sweep",
                    operator=operator,
                    checkpoint_key=operator,
                    config_key=operator,
                    density_multiplier=multiplier,
                    substitution_modes=("model", "oracle"),
                )
            )
        for compression in (1.0, 0.75, 0.5):
            token = str(compression).replace(".", "p")
            tasks.append(
                UAVExperimentTask(
                    task_id=f"uav_capacity_chi{token}_{operator}",
                    family="capacity_compression_sweep",
                    operator=operator,
                    checkpoint_key=operator,
                    config_key=operator,
                    capacity_compression=compression,
                    substitution_modes=("model", "oracle"),
                )
            )
        for mismatch in (
            "service_time_inflation",
            "queue_law_swap",
            "station_memory_shift",
            "threshold_tightening",
        ):
            tasks.append(
                UAVExperimentTask(
                    task_id=f"uav_mismatch_{mismatch}_{operator}",
                    family="structural_mismatch",
                    operator=operator,
                    checkpoint_key=operator,
                    config_key=operator,
                    perturbation_name=mismatch,
                    substitution_modes=("model", "oracle"),
                )
            )
    tasks.append(
        UAVExperimentTask(
            task_id="uav_substitution_ladder_nominal_physick",
            family="oracle_partial_substitution",
            operator="physick",
            checkpoint_key="physick",
            config_key="physick",
            include_one_step_loss=False,
        )
    )
    tasks.append(
        UAVExperimentTask(
            task_id="uav_substitution_ladder_high_stress_physick",
            family="oracle_partial_substitution",
            operator="physick",
            checkpoint_key="physick",
            config_key="physick",
            density_multiplier=5,
            include_one_step_loss=False,
        )
    )
    for ablation in UAVAblation:
        checkpoint = (
            "physick_no_gain_control"
            if ablation is UAVAblation.NO_GAIN_CONTROL
            else f"physick_{ablation.value}"
        )
        if ablation is UAVAblation.FULL:
            checkpoint = "physick"
        tasks.append(
            UAVExperimentTask(
                task_id=f"uav_ablation_{ablation.value}_high_stress",
                family="component_removal",
                operator="physick",
                checkpoint_key=checkpoint,
                config_key=checkpoint,
                density_multiplier=5,
                ablation=ablation.value,
                substitution_modes=("model", "oracle"),
            )
        )
    identifiers = [task.task_id for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("paper UAV task identifiers are not unique")
    return tuple(tasks)


def held_out_evaluation_units(
    dataset_index: str | Path,
    *,
    run_seeds: Sequence[int],
    episodes_per_run: int = 30,
) -> list[dict[str, Any]]:
    """Read the exact matched run-by-episode identities from a dataset index."""

    payload = load_uav_dataset(Path(dataset_index))
    split = payload.get("split_manifest")
    if not isinstance(split, Mapping):
        raise ValueError("dataset index is missing split_manifest")
    identities = split.get("episode_identity")
    if not isinstance(identities, list):
        raise ValueError("dataset index is missing episode_identity rows")
    units: list[dict[str, Any]] = []
    for run_seed in run_seeds:
        rows = [
            row
            for row in identities
            if isinstance(row, Mapping)
            and row.get("split") == "test"
            and int(row.get("run_seed", -1)) == int(run_seed)
        ]
        if len(rows) < episodes_per_run:
            raise ValueError(
                f"run {run_seed} has only {len(rows)} held-out episodes"
            )
        for row in rows[:episodes_per_run]:
            units.append(
                {
                    "run_seed": int(row["run_seed"]),
                    "run_index": int(row["run_index"]),
                    "local_episode_index": int(row["local_episode_index"]),
                    "episode_seed": int(row["episode_seed"]),
                    "paired_episode_id": str(
                        row.get(
                            "paired_episode_id",
                            (
                                f"uav_run_{int(row['run_seed'])}_episode_"
                                f"{int(row['local_episode_index'])}"
                            ),
                        )
                    ),
                    "exogenous_sequence_id": int(
                        row.get("exogenous_sequence_id", row["episode_seed"])
                    ),
                }
            )
    return units


def task_manifest() -> dict[str, Any]:
    tasks = [asdict(task) for task in paper_task_matrix()]
    encoded = json.dumps(tasks, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "platform": "uav_shared_service",
        "task_count": len(tasks),
        "task_matrix_sha256": hashlib.sha256(encoded).hexdigest(),
        "tasks": tasks,
    }


__all__ = [
    "UAVAblation",
    "UAVAblationDescriptorProvider",
    "UAVExperimentTask",
    "ablated_policy_config",
    "held_out_evaluation_units",
    "paper_task_matrix",
    "task_manifest",
]
