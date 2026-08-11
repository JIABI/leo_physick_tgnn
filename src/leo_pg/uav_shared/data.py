"""Tensor-only action-coupled datasets for the UAV/shared-service platform.

Formal generation writes one episode per shard and a lightweight index.  No
environment, policy, protocol, or target dataclass is serialized, so every
artifact can be read with ``torch.load(..., weights_only=True)``.  The behavior
trajectory is closed loop: at epoch ``t`` the fixed UAV oracle reads the
simulator descriptor copy, its requested action is committed, and the resulting
state produces the graph and typed target for ``t+1``.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from enum import Enum
import hashlib
import math
from pathlib import Path, PurePosixPath
import random
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .config import ConfigSource, resolve_uav_shared_config
from .environment import UAVSharedServiceEnv
from .graph import (
    UAV_EDGE_FEATURE_NAMES,
    UAV_GRAPH_CONTRACT_VERSION,
    UAV_STATION_FEATURE_NAMES,
    UAV_TARGET_CONTRACT_VERSION,
    UAV_UAV_FEATURE_NAMES,
    UAVTarget,
    build_next_step_target,
    observation_to_graph,
    validate_graph,
)
from .policy import (
    UAVFixedRankPolicy,
    UAVFixedRankPolicyConfig,
    UAV_POLICY_CONTRACT_VERSION,
)
from .protocol import UAV_SHARED_PROTOCOL_VERSION, UAVSharedProtocol
from .state import (
    ReassociationAction,
    ServiceExecutionResult,
    ServiceFailureReason,
    ServiceObservation,
)


UAV_DATASET_SCHEMA_VERSION = 1
UAV_DATASET_KIND = "leo_pg.uav_shared.oracle_action_coupled"
UAV_EPISODE_SHARD_VERSION = 1
UAV_SHARDED_LAYOUT = "episode_shards_v1"
UAV_SPLIT_NAMES = ("train", "val", "test")
UAV_RUN_SEED_STRIDE = 1_000_000


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _cpu_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.detach().to(device="cpu").contiguous().clone()
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _pure_tree(value: Any, *, path: str = "root") -> Any:
    """Clone into the object types accepted by ``weights_only=True``."""

    if isinstance(value, torch.Tensor):
        return _cpu_tensor(value, name=path)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string key {key!r}")
            result[key] = _pure_tree(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _pure_tree(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Enum):
        return _pure_tree(value.value, path=path)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    raise TypeError(
        f"{path} has unsupported type {type(value).__name__}; UAV datasets "
        "allow only dict/list/tensor/primitive values"
    )


def _validate_pure_tree(value: Any, *, path: str = "root") -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{path} tensor must reside on CPU")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{path} contains NaN or Inf")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string key {key!r}")
            _validate_pure_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_pure_tree(item, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise TypeError(f"{path} is not a weights-only-safe payload tree")


def _serialize_observation(
    observation: ServiceObservation,
    *,
    protocol: UAVSharedProtocol,
) -> dict[str, Any]:
    graph = observation_to_graph(observation, protocol)
    return _pure_tree(graph, path="observation")


def _serialize_action(
    action: ReassociationAction,
    *,
    user_count: int,
    station_count: int,
) -> dict[str, Any]:
    action.validate(user_count, station_count)
    return {
        "observation_id": [
            int(action.observation_id[0]),
            int(action.observation_id[1]),
        ],
        "requested_station": _cpu_tensor(
            action.requested_station, name="action.requested_station"
        ),
    }


def _serialize_execution(
    execution: ServiceExecutionResult,
    *,
    user_count: int,
    station_count: int,
) -> dict[str, Any]:
    execution.validate(user_count, station_count)
    result: dict[str, Any] = {
        "observation_id": [
            int(execution.observation_id[0]),
            int(execution.observation_id[1]),
        ]
    }
    for name in (
        "requested_station",
        "target_before",
        "target_after",
        "assignment_attempted",
        "assignment_accepted",
        "reassociation_executed",
        "failure_reason",
        "queue_admitted",
        "service_started",
        "service_completed",
        "energy_depleted",
        "energy_before",
        "energy_after",
        "station_flow_before",
        "station_flow_after",
        "queue_length_before",
        "queue_length_after",
        "active_slots_before",
        "active_slots_after",
    ):
        result[name] = _cpu_tensor(
            getattr(execution, name), name=f"execution.{name}"
        )
    return result


def _serialize_target(
    target: UAVTarget,
    *,
    source_observation: ServiceObservation,
    next_observation: ServiceObservation,
) -> dict[str, Any]:
    target.validate(source_observation.edge_count, source_observation.station_count)
    result = {
        "contract_version": UAV_TARGET_CONTRACT_VERSION,
        "source_observation_id": [
            source_observation.episode_seed,
            source_observation.epoch,
        ],
        "target_observation_id": [
            next_observation.episode_seed,
            next_observation.epoch,
        ],
    }
    for name, value in target.as_dict().items():
        if name == "contract_version":
            continue
        result[name] = _cpu_tensor(value, name=f"target.{name}")
    return result


def _terminal_payload(environment: UAVSharedServiceEnv) -> dict[str, Any]:
    snapshot = environment.snapshot()
    state = snapshot.state
    protocol = environment.protocol
    return {
        "epoch": int(state.epoch),
        "uav_position": _cpu_tensor(
            state.uav_position, name="terminal.uav_position"
        ),
        "energy_fraction": _cpu_tensor(
            (state.energy_units / protocol.estimated.nominal_battery_units).clamp(0, 1),
            name="terminal.energy_fraction",
        ),
        "phase": _cpu_tensor(state.phase, name="terminal.phase"),
        "target_station": _cpu_tensor(
            state.target_station, name="terminal.target_station"
        ),
        "station_flow": _cpu_tensor(
            state.station_flow, name="terminal.station_flow"
        ),
        "active_uav": _cpu_tensor(state.active_uav, name="terminal.active_uav"),
        "service_remaining_s": _cpu_tensor(
            state.service_remaining_s, name="terminal.service_remaining_s"
        ),
        "fifo_queues": [list(queue) for queue in state.fifo_queues],
        "service_visit_count": _cpu_tensor(
            state.service_visit_count, name="terminal.service_visit_count"
        ),
        "missions_completed": _cpu_tensor(
            state.missions_completed, name="terminal.missions_completed"
        ),
        "failed_service_starts": _cpu_tensor(
            state.failed_service_starts, name="terminal.failed_service_starts"
        ),
        "reassociations": _cpu_tensor(
            state.reassociations, name="terminal.reassociations"
        ),
    }


def generate_uav_oracle_episode(
    source: ConfigSource,
    *,
    episode_id: int,
    seed: int,
    horizon_steps: int | None = None,
    policy_config: UAVFixedRankPolicyConfig | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Generate one closed-loop episode under simulator-oracle descriptors."""

    if type(episode_id) is not int or episode_id < 0:
        raise ValueError("episode_id must be a non-negative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    resolved = resolve_uav_shared_config(source)
    protocol = replace(resolved.protocol, episode_seed=seed)
    requested_horizon = protocol.horizon_steps if horizon_steps is None else horizon_steps
    if (
        type(requested_horizon) is not int
        or not 1 <= requested_horizon <= protocol.horizon_steps
    ):
        raise ValueError(
            f"horizon_steps must lie in [1,{protocol.horizon_steps}]"
        )
    policy_config = policy_config or UAVFixedRankPolicyConfig()
    if not isinstance(policy_config, UAVFixedRankPolicyConfig):
        raise TypeError("policy_config must be a UAVFixedRankPolicyConfig")
    environment = UAVSharedServiceEnv(protocol, device=device)
    policy = UAVFixedRankPolicy(policy_config)
    observation = environment.reset_control()
    records: list[dict[str, Any]] = []

    for step_index in range(requested_horizon):
        # This explicit copy documents behavior-policy authority even though a
        # fresh environment observation initially contains equal policy fields.
        oracle_observation = observation.with_policy_descriptors(
            observation.sim_descriptors.policy_fields
        )
        action = policy.select_action(
            oracle_observation,
            oracle_observation.sim_descriptors.policy_fields,
        )
        next_observation, execution, environment_done = environment.step_action(action)
        episode_done = environment_done or step_index + 1 == requested_horizon
        target_payload = None
        if not episode_done:
            if next_observation is None:
                raise RuntimeError(
                    "UAV environment omitted a non-terminal next observation"
                )
            target_payload = _serialize_target(
                build_next_step_target(oracle_observation, next_observation),
                source_observation=oracle_observation,
                next_observation=next_observation,
            )
        records.append(
            {
                "step_index": step_index,
                "observation": _serialize_observation(
                    oracle_observation, protocol=protocol
                ),
                "action": _serialize_action(
                    action,
                    user_count=protocol.uav_count,
                    station_count=protocol.station_count,
                ),
                "execution": _serialize_execution(
                    execution,
                    user_count=protocol.uav_count,
                    station_count=protocol.station_count,
                ),
                "next_target": target_payload,
                "done": bool(episode_done),
                "environment_done": bool(environment_done),
            }
        )
        if episode_done:
            break
        if next_observation is None:
            raise RuntimeError("missing next observation before the episode boundary")
        observation = next_observation

    episode = {
        "episode_id": episode_id,
        "seed": seed,
        "protocol_version": UAV_SHARED_PROTOCOL_VERSION,
        "protocol_fingerprint": protocol.fingerprint,
        "graph_contract_version": UAV_GRAPH_CONTRACT_VERSION,
        "target_contract_version": UAV_TARGET_CONTRACT_VERSION,
        "horizon_steps": requested_horizon,
        "environment_horizon_steps": protocol.horizon_steps,
        "truncated": requested_horizon < protocol.horizon_steps,
        "selected_condition": {
            "density_multiplier": protocol.density_multiplier,
            "capacity_compression": protocol.capacity_compression,
            "uav_count": protocol.uav_count,
            "station_count": protocol.station_count,
        },
        "behavior_policy": {
            "name": "uav_fixed_rank_oracle_v1",
            "policy_contract_version": UAV_POLICY_CONTRACT_VERSION,
            "descriptor_source": "simulator_oracle",
            "config": _pure_tree(
                asdict(policy_config), path="behavior_policy.config"
            ),
        },
        "protocol_manifest": _pure_tree(
            protocol.manifest(), path="protocol_manifest"
        ),
        "steps": records,
        "terminal": _terminal_payload(environment),
    }
    validate_uav_episode(episode, expected_index=episode_id, expected_seed=seed)
    return episode


def _split_counts_from_ratios(
    episode_count: int,
    ratios: Sequence[float],
) -> list[int]:
    if len(ratios) != 3:
        raise ValueError("split ratios must contain train,val,test values")
    normalized = [float(value) for value in ratios]
    if any(not math.isfinite(value) or value < 0.0 for value in normalized):
        raise ValueError("split ratios must be finite and non-negative")
    if not math.isclose(sum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("split ratios must sum to one")
    raw = [episode_count * value for value in normalized]
    counts = [math.floor(value) for value in raw]
    remainder = episode_count - sum(counts)
    order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def build_uav_split_manifest(
    episode_count: int,
    *,
    base_seed: int,
    split_seed: int,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build stable, disjoint episode-level train/validation/test splits."""

    if type(episode_count) is not int or episode_count <= 0:
        raise ValueError("episode_count must be a positive integer")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a non-negative integer")
    if counts is None:
        count_values = _split_counts_from_ratios(episode_count, ratios)
        ratio_values = [float(value) for value in ratios]
        assignment = "deterministic_episode_shuffle_largest_remainder_v1"
    else:
        if len(counts) != 3 or any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("split counts must be three non-negative integers")
        count_values = list(counts)
        if sum(count_values) != episode_count:
            raise ValueError("split counts must sum to episode_count")
        ratio_values = [value / episode_count for value in count_values]
        assignment = "deterministic_episode_shuffle_exact_counts_v1"
    indices = list(range(episode_count))
    random.Random(split_seed).shuffle(indices)
    train_end = count_values[0]
    val_end = train_end + count_values[1]
    split_indices = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }
    return {
        "split_seed": split_seed,
        "assignment": assignment,
        "ratios": {
            name: ratio_values[index]
            for index, name in enumerate(UAV_SPLIT_NAMES)
        },
        "counts": {
            name: count_values[index]
            for index, name in enumerate(UAV_SPLIT_NAMES)
        },
        "episode_indices": {
            name: list(split_indices[name]) for name in UAV_SPLIT_NAMES
        },
        "episode_ids": {
            name: list(split_indices[name]) for name in UAV_SPLIT_NAMES
        },
        "episode_seeds": {
            name: [base_seed + index for index in split_indices[name]]
            for name in UAV_SPLIT_NAMES
        },
    }


def build_uav_multirun_split_manifest(
    run_seeds: Sequence[int],
    *,
    split_counts_per_run: Sequence[int],
    split_seed: int,
) -> dict[str, Any]:
    """Build the formal five-run split with fixed counts inside every run.

    Episode simulation seeds use the collision-safe published rule
    ``run_seed * 1_000_000 + local_episode_index``.  The local indices are
    shuffled only for split assignment; global shard order remains run-major
    and then local-index-major so identities do not depend on the shuffle.
    """

    if not isinstance(run_seeds, Sequence) or isinstance(run_seeds, (str, bytes)):
        raise TypeError("run_seeds must be a sequence of integers")
    normalized_run_seeds = list(run_seeds)
    if not normalized_run_seeds:
        raise ValueError("run_seeds cannot be empty")
    if any(type(value) is not int or value < 0 for value in normalized_run_seeds):
        raise ValueError("run_seeds must contain non-negative integers")
    if len(set(normalized_run_seeds)) != len(normalized_run_seeds):
        raise ValueError("run_seeds must be unique")
    if (
        len(split_counts_per_run) != 3
        or any(type(value) is not int or value < 0 for value in split_counts_per_run)
    ):
        raise ValueError(
            "split_counts_per_run must contain three non-negative integers"
        )
    per_run_counts = list(split_counts_per_run)
    per_run_total = sum(per_run_counts)
    if per_run_total <= 0:
        raise ValueError("split_counts_per_run must have a positive sum")
    if per_run_total >= UAV_RUN_SEED_STRIDE:
        raise ValueError(
            "episodes per run must be smaller than the run-seed stride"
        )
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a non-negative integer")

    identities: list[dict[str, Any]] = []
    aggregate_indices = {name: [] for name in UAV_SPLIT_NAMES}
    aggregate_seeds = {name: [] for name in UAV_SPLIT_NAMES}
    per_run: list[dict[str, Any]] = []
    for run_index, run_seed in enumerate(normalized_run_seeds):
        local_indices = list(range(per_run_total))
        # This formula is part of the manifest, so split membership remains
        # reproducible without depending on process-global RNG state.
        run_split_seed = split_seed + UAV_RUN_SEED_STRIDE * run_seed + run_index
        random.Random(run_split_seed).shuffle(local_indices)
        train_end = per_run_counts[0]
        val_end = train_end + per_run_counts[1]
        local_by_split = {
            "train": local_indices[:train_end],
            "val": local_indices[train_end:val_end],
            "test": local_indices[val_end:],
        }
        split_by_local = {
            local_index: split
            for split, values in local_by_split.items()
            for local_index in values
        }
        run_global = {name: [] for name in UAV_SPLIT_NAMES}
        run_episode_seeds = {name: [] for name in UAV_SPLIT_NAMES}
        for local_index in range(per_run_total):
            global_index = len(identities)
            episode_seed = run_seed * UAV_RUN_SEED_STRIDE + local_index
            split = split_by_local[local_index]
            identity = {
                "episode_index": global_index,
                "episode_id": global_index,
                "episode_seed": episode_seed,
                "run_index": run_index,
                "run_seed": run_seed,
                "local_episode_index": local_index,
                "split": split,
            }
            identities.append(identity)
            aggregate_indices[split].append(global_index)
            aggregate_seeds[split].append(episode_seed)
            run_global[split].append(global_index)
            run_episode_seeds[split].append(episode_seed)
        per_run.append(
            {
                "run_index": run_index,
                "run_seed": run_seed,
                "split_seed": run_split_seed,
                "counts": {
                    name: per_run_counts[index]
                    for index, name in enumerate(UAV_SPLIT_NAMES)
                },
                "local_episode_indices": {
                    name: list(local_by_split[name]) for name in UAV_SPLIT_NAMES
                },
                "episode_indices": run_global,
                "episode_seeds": run_episode_seeds,
            }
        )

    total = len(identities)
    aggregate_counts = {
        name: per_run_counts[index] * len(normalized_run_seeds)
        for index, name in enumerate(UAV_SPLIT_NAMES)
    }
    return {
        "split_seed": split_seed,
        "assignment": "deterministic_per_run_shuffle_exact_counts_v1",
        "run_seed_stride": UAV_RUN_SEED_STRIDE,
        "episode_seed_rule": "run_seed_times_1000000_plus_local_episode_index",
        "run_seeds": normalized_run_seeds,
        "split_counts_per_run": {
            name: per_run_counts[index]
            for index, name in enumerate(UAV_SPLIT_NAMES)
        },
        "counts": aggregate_counts,
        "ratios": {
            name: aggregate_counts[name] / total for name in UAV_SPLIT_NAMES
        },
        "episode_indices": aggregate_indices,
        "episode_ids": {
            name: list(aggregate_indices[name]) for name in UAV_SPLIT_NAMES
        },
        "episode_seeds": aggregate_seeds,
        "per_run": per_run,
        "episode_identity": identities,
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_sharded_uav_dataset(
    path: str | Path,
    source: ConfigSource,
    *,
    episode_count: int | None = None,
    horizon_steps: int | None = None,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    split_counts: Sequence[int] | None = None,
    split_seed: int | None = None,
    base_seed: int | None = None,
    policy_config: UAVFixedRankPolicyConfig | None = None,
    run_seeds: Sequence[int] | None = None,
    split_counts_per_run: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Write a lightweight index plus one action-coupled episode per shard."""

    resolved = resolve_uav_shared_config(source)
    resolved_base_seed = (
        resolved.protocol.episode_seed if base_seed is None else base_seed
    )
    if type(resolved_base_seed) is not int or resolved_base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    resolved_split_seed = (
        resolved_base_seed if split_seed is None else split_seed
    )
    if type(resolved_split_seed) is not int or resolved_split_seed < 0:
        raise ValueError("split_seed must be a non-negative integer")
    policy_config = policy_config or UAVFixedRankPolicyConfig()
    if not isinstance(policy_config, UAVFixedRankPolicyConfig):
        raise TypeError("policy_config must be a UAVFixedRankPolicyConfig")
    multi_run = run_seeds is not None or split_counts_per_run is not None
    if multi_run:
        if run_seeds is None or split_counts_per_run is None:
            raise ValueError(
                "run_seeds and split_counts_per_run must be provided together"
            )
        if split_counts is not None:
            raise ValueError(
                "aggregate split_counts cannot be combined with per-run counts"
            )
        split_manifest = build_uav_multirun_split_manifest(
            run_seeds,
            split_counts_per_run=split_counts_per_run,
            split_seed=resolved_split_seed,
        )
        identities = list(split_manifest["episode_identity"])
        derived_episode_count = len(identities)
        if episode_count is not None and episode_count != derived_episode_count:
            raise ValueError(
                "episode_count does not match run_seeds × split_counts_per_run"
            )
        episode_count = derived_episode_count
        generation_mode = "formal_per_run_exact_counts_v1"
    else:
        if type(episode_count) is not int or episode_count <= 0:
            raise ValueError("episode_count must be a positive integer")
        split_manifest = build_uav_split_manifest(
            episode_count,
            base_seed=resolved_base_seed,
            split_seed=resolved_split_seed,
            ratios=split_ratios,
            counts=split_counts,
        )
        split_by_index: dict[int, str] = {}
        for split in UAV_SPLIT_NAMES:
            for episode_index in split_manifest["episode_indices"][split]:
                split_by_index[int(episode_index)] = split
        identities = [
            {
                "episode_index": episode_index,
                "episode_id": episode_index,
                "episode_seed": resolved_base_seed + episode_index,
                "run_index": 0,
                "run_seed": resolved_base_seed,
                "local_episode_index": episode_index,
                "split": split_by_index[episode_index],
            }
            for episode_index in range(episode_count)
        ]
        generation_mode = "single_run_compatibility_v1"

    output = Path(path).expanduser()
    if output.suffix != ".pt":
        raise ValueError("UAV dataset index path must use a .pt suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_directory_name = f"{output.stem}_episodes"
    shard_root = output.parent / shard_directory_name
    shard_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    effective_horizons: set[int] = set()
    fingerprints: set[str] = set()

    for identity in identities:
        episode_index = int(identity["episode_index"])
        episode_seed = int(identity["episode_seed"])
        episode = generate_uav_oracle_episode(
            resolved,
            episode_id=episode_index,
            seed=episode_seed,
            horizon_steps=horizon_steps,
            policy_config=policy_config,
            device=device,
        )
        episode["run_index"] = int(identity["run_index"])
        episode["run_seed"] = int(identity["run_seed"])
        episode["local_episode_index"] = int(identity["local_episode_index"])
        validate_uav_episode(
            episode,
            expected_index=episode_index,
            expected_seed=episode_seed,
        )
        shard_payload = {
            "uav_episode_shard_version": UAV_EPISODE_SHARD_VERSION,
            "uav_dataset_schema_version": UAV_DATASET_SCHEMA_VERSION,
            "dataset_kind": UAV_DATASET_KIND,
            "episode": episode,
        }
        _validate_pure_tree(shard_payload, path="uav_episode_shard")
        filename = f"episode_{episode_index:06d}.pt"
        shard_path = shard_root / filename
        _atomic_torch_save(shard_path, shard_payload)
        entries.append(
            {
                "episode_index": episode_index,
                "episode_id": episode_index,
                "seed": episode_seed,
                "split": str(identity["split"]),
                "run_index": int(identity["run_index"]),
                "run_seed": int(identity["run_seed"]),
                "local_episode_index": int(identity["local_episode_index"]),
                "relative_path": (
                    Path(shard_directory_name) / filename
                ).as_posix(),
                "size_bytes": int(shard_path.stat().st_size),
                "sha256": _file_sha256(shard_path),
                "horizon_steps": int(episode["horizon_steps"]),
                "protocol_fingerprint": str(episode["protocol_fingerprint"]),
            }
        )
        effective_horizons.add(int(episode["horizon_steps"]))
        fingerprints.add(str(episode["protocol_fingerprint"]))

    sorted_horizons = sorted(effective_horizons)
    index: dict[str, Any] = {
        "uav_dataset_schema_version": UAV_DATASET_SCHEMA_VERSION,
        "dataset_kind": UAV_DATASET_KIND,
        "protocol_version": UAV_SHARED_PROTOCOL_VERSION,
        "graph_contract_version": UAV_GRAPH_CONTRACT_VERSION,
        "target_contract_version": UAV_TARGET_CONTRACT_VERSION,
        "feature_contract": {
            "uav_feature_names": list(UAV_UAV_FEATURE_NAMES),
            "station_feature_names": list(UAV_STATION_FEATURE_NAMES),
            "edge_feature_names": list(UAV_EDGE_FEATURE_NAMES),
        },
        "target_contract": {
            "alignment": "source_stable_candidate_id_at_t_to_simulator_target_at_t_plus_1",
            "edge_mask": "persistent_edge",
            "eta_domain": "raw_[0,1]",
            "intensity_domain": "log1p",
            "station_flow_domain": "station_node_[0,1]",
            "feasibility_role": "auxiliary_target_and_controller_authority",
        },
        "failure_reason_codes": {
            reason.name.lower(): int(reason) for reason in ServiceFailureReason
        },
        "source_configuration": _pure_tree(
            resolved.manifest(), path="source_configuration"
        ),
        "generation": {
            "episode_count": episode_count,
            "generation_mode": generation_mode,
            "base_seed": resolved_base_seed,
            "episode_seed_rule": split_manifest.get(
                "episode_seed_rule", "base_seed_plus_episode_id"
            ),
            "run_seeds": (
                list(split_manifest.get("run_seeds", []))
                if multi_run
                else [resolved_base_seed]
            ),
            "split_counts_per_run": (
                dict(split_manifest["split_counts_per_run"])
                if multi_run
                else None
            ),
            "horizon_steps": (
                sorted_horizons[0]
                if len(sorted_horizons) == 1
                else sorted_horizons
            ),
            "device": str(torch.device(device)),
            "behavior_policy": "uav_fixed_rank_oracle_v1",
            "behavior_policy_config": _pure_tree(
                asdict(policy_config), path="generation.behavior_policy_config"
            ),
        },
        "storage": {
            "layout": UAV_SHARDED_LAYOUT,
            "episode_count": episode_count,
            "relative_shard_directory": shard_directory_name,
            "integrity": "size_bytes_and_sha256",
            "weights_only_compatible": True,
        },
        "split_manifest": split_manifest,
        "protocol_fingerprints": sorted(fingerprints),
        "episode_shards": entries,
    }
    save_uav_dataset(output, index)
    return index


def validate_uav_episode(
    episode: Mapping[str, Any],
    *,
    expected_index: int | None = None,
    expected_seed: int | None = None,
) -> None:
    """Validate episode chronology, physical edge ids, and target alignment."""

    if not isinstance(episode, Mapping):
        raise TypeError("UAV episode must be a mapping")
    episode_id = episode.get("episode_id")
    seed = episode.get("seed")
    if type(episode_id) is not int or episode_id < 0:
        raise ValueError("episode_id must be a non-negative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("episode seed must be a non-negative integer")
    if expected_index is not None and episode_id != expected_index:
        raise ValueError("episode_id does not match its stable index")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("episode seed does not match its shard entry")
    fingerprint = episode.get("protocol_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("episode requires a 64-character protocol fingerprint")
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("UAV episode must contain decision steps")
    if episode.get("horizon_steps") != len(steps):
        raise ValueError("episode horizon_steps must equal its record count")
    station_count: int | None = None
    for step_index, record in enumerate(steps):
        if not isinstance(record, Mapping) or record.get("step_index") != step_index:
            raise ValueError("step_index must be contiguous from zero")
        observation = record.get("observation")
        action = record.get("action")
        execution = record.get("execution")
        if not all(isinstance(value, Mapping) for value in (observation, action, execution)):
            raise TypeError("step observation/action/execution must be mappings")
        validate_graph(observation)
        expected_id = [seed, step_index]
        if any(
            value.get("observation_id") != expected_id
            for value in (observation, action, execution)
        ):
            raise ValueError("step observation/action/execution ids are misaligned")
        if not torch.equal(
            action["requested_station"], execution["requested_station"]
        ):
            raise ValueError("execution request differs from the committed action")
        station_count = int(observation["station_count"])
        target = record.get("next_target")
        is_last = step_index + 1 == len(steps)
        if is_last:
            if target is not None or record.get("done") is not True:
                raise ValueError("last episode record must be terminal without a target")
        else:
            if not isinstance(target, Mapping) or record.get("done") is not False:
                raise ValueError("non-terminal records require a next target")
            if target.get("contract_version") != UAV_TARGET_CONTRACT_VERSION:
                raise ValueError("next target contract version mismatch")
            if target.get("source_observation_id") != expected_id:
                raise ValueError("target source id is misaligned")
            if target.get("target_observation_id") != [seed, step_index + 1]:
                raise ValueError("target observation id is not t+1")
            typed_target = UAVTarget(
                eta_edge=target["eta_edge"],
                log1p_intensity_edge=target["log1p_intensity_edge"],
                feasibility_edge=target["feasibility_edge"],
                persistent_edge=target["persistent_edge"],
                station_flow_node=target["station_flow_node"],
                source_stable_candidate_id=target["source_stable_candidate_id"],
            )
            typed_target.validate(
                int(observation["candidate_edge_ids"].size(0)), station_count
            )
            if not torch.equal(
                typed_target.source_stable_candidate_id,
                observation["stable_candidate_id"],
            ):
                raise ValueError("target source ids differ from graph candidate ids")
    terminal = episode.get("terminal")
    if not isinstance(terminal, Mapping) or terminal.get("epoch") != len(steps):
        raise ValueError("terminal epoch must equal the committed transition count")
    _validate_pure_tree(dict(episode), path="episode")


def _validate_split_manifest(
    manifest: Mapping[str, Any],
    *,
    episode_count: int,
) -> None:
    if not isinstance(manifest, Mapping):
        raise TypeError("split_manifest must be a mapping")
    seen: set[int] = set()
    for split in UAV_SPLIT_NAMES:
        values = manifest.get("episode_indices", {}).get(split)
        if not isinstance(values, list) or any(type(value) is not int for value in values):
            raise ValueError(f"split {split} episode indices must be integer lists")
        overlap = seen.intersection(values)
        if overlap:
            raise ValueError(f"episode split overlap detected: {sorted(overlap)}")
        seen.update(values)
        if manifest.get("counts", {}).get(split) != len(values):
            raise ValueError(f"split {split} count does not match its indices")
    if seen != set(range(episode_count)):
        raise ValueError("split manifest must cover every episode exactly once")


def _safe_relative_shard_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("shard relative_path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("shard relative_path must stay below the index directory")
    return path


def validate_uav_dataset_index(index: Mapping[str, Any]) -> None:
    """Validate the lightweight index without loading any episode tensors."""

    if not isinstance(index, Mapping):
        raise TypeError("UAV dataset index must be a mapping")
    if index.get("uav_dataset_schema_version") != UAV_DATASET_SCHEMA_VERSION:
        raise ValueError("UAV dataset schema version mismatch")
    if index.get("dataset_kind") != UAV_DATASET_KIND:
        raise ValueError("unexpected UAV dataset kind")
    storage = index.get("storage")
    if not isinstance(storage, Mapping) or storage.get("layout") != UAV_SHARDED_LAYOUT:
        raise ValueError("UAV dataset must use the episode-sharded layout")
    episode_count = storage.get("episode_count")
    if type(episode_count) is not int or episode_count <= 0:
        raise ValueError("storage episode_count must be positive")
    entries = index.get("episode_shards")
    if not isinstance(entries, list) or len(entries) != episode_count:
        raise ValueError("episode_shards count does not match storage metadata")
    split_manifest = index.get("split_manifest")
    _validate_split_manifest(split_manifest, episode_count=episode_count)
    split_membership = {
        int(episode_index): split
        for split in UAV_SPLIT_NAMES
        for episode_index in split_manifest["episode_indices"][split]
    }
    relative_paths: set[str] = set()
    for expected_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError("episode shard entries must be mappings")
        if entry.get("episode_index") != expected_index or entry.get("episode_id") != expected_index:
            raise ValueError("episode shard entries must be in stable index order")
        if entry.get("split") not in UAV_SPLIT_NAMES:
            raise ValueError("episode shard has an invalid split")
        if entry.get("split") != split_membership[expected_index]:
            raise ValueError("episode shard split differs from the split manifest")
        for name in ("run_index", "run_seed", "local_episode_index", "seed"):
            if type(entry.get(name)) is not int or entry[name] < 0:
                raise ValueError(f"episode shard {name} must be non-negative")
        path = str(_safe_relative_shard_path(entry.get("relative_path")))
        if path in relative_paths:
            raise ValueError("episode shard relative paths must be unique")
        relative_paths.add(path)
        if type(entry.get("size_bytes")) is not int or entry["size_bytes"] <= 0:
            raise ValueError("episode shard size_bytes must be positive")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("episode shard requires a SHA-256 digest")
    _validate_pure_tree(dict(index), path="uav_dataset_index")


def save_uav_dataset(path: str | Path, index: Mapping[str, Any]) -> Path:
    """Validate and atomically save a UAV dataset index."""

    output = Path(path).expanduser()
    if output.suffix != ".pt":
        raise ValueError("UAV dataset index path must use a .pt suffix")
    validate_uav_dataset_index(index)
    _atomic_torch_save(output, index)
    return output


def load_uav_dataset(path: str | Path) -> dict[str, Any]:
    """Load and validate a tensor-safe UAV dataset index."""

    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("UAV dataset index must deserialize to a dict")
    validate_uav_dataset_index(payload)
    return payload


def load_uav_episode_shard(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    expected_index: int | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Load one episode shard, optionally enforcing its index integrity data."""

    source = Path(path).expanduser().resolve()
    if expected_size_bytes is not None and source.stat().st_size != expected_size_bytes:
        raise ValueError(f"UAV episode shard size mismatch: {source}")
    if expected_sha256 is not None and _file_sha256(source) != expected_sha256:
        raise ValueError(f"UAV episode shard SHA-256 mismatch: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("UAV episode shard must deserialize to a dict")
    if payload.get("uav_episode_shard_version") != UAV_EPISODE_SHARD_VERSION:
        raise ValueError("UAV episode shard version mismatch")
    if payload.get("uav_dataset_schema_version") != UAV_DATASET_SCHEMA_VERSION:
        raise ValueError("UAV episode dataset schema mismatch")
    if payload.get("dataset_kind") != UAV_DATASET_KIND:
        raise ValueError("unexpected UAV episode dataset kind")
    episode = payload.get("episode")
    validate_uav_episode(
        episode,
        expected_index=expected_index,
        expected_seed=expected_seed,
    )
    return episode


class UAVEpisodeDataset(Dataset):
    """Lazy split view over episode shards referenced by a validated index."""

    def __init__(
        self,
        index_path: str | Path,
        split: str,
        *,
        run_seed: int | None = None,
        verify_hash: bool = True,
    ) -> None:
        if split not in UAV_SPLIT_NAMES:
            raise ValueError(f"split must be one of {UAV_SPLIT_NAMES}")
        if type(verify_hash) is not bool:
            raise TypeError("verify_hash must be bool")
        if run_seed is not None and (type(run_seed) is not int or run_seed < 0):
            raise ValueError("run_seed must be a non-negative integer or None")
        self.index_path = Path(index_path).expanduser().resolve()
        self.index = load_uav_dataset(self.index_path)
        self.split = split
        self.run_seed = run_seed
        self.verify_hash = verify_hash
        self.entries = [
            entry
            for entry in self.index["episode_shards"]
            if entry["split"] == split
            and (run_seed is None or entry.get("run_seed") == run_seed)
        ]
        if run_seed is not None and not self.entries:
            raise ValueError(
                f"dataset has no {split!r} episodes for run_seed={run_seed}"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        relative = _safe_relative_shard_path(entry["relative_path"])
        shard_path = self.index_path.parent.joinpath(*relative.parts).resolve()
        try:
            shard_path.relative_to(self.index_path.parent)
        except ValueError as exc:
            raise ValueError("episode shard resolves outside the index directory") from exc
        return load_uav_episode_shard(
            shard_path,
            expected_sha256=entry["sha256"] if self.verify_hash else None,
            expected_size_bytes=entry["size_bytes"] if self.verify_hash else None,
            expected_index=entry["episode_index"],
            expected_seed=entry["seed"],
        )


__all__ = [
    "UAV_DATASET_KIND",
    "UAV_DATASET_SCHEMA_VERSION",
    "UAV_EPISODE_SHARD_VERSION",
    "UAV_SHARDED_LAYOUT",
    "UAV_SPLIT_NAMES",
    "UAV_RUN_SEED_STRIDE",
    "UAVEpisodeDataset",
    "build_uav_multirun_split_manifest",
    "build_uav_split_manifest",
    "generate_sharded_uav_dataset",
    "generate_uav_oracle_episode",
    "load_uav_dataset",
    "load_uav_episode_shard",
    "save_uav_dataset",
    "validate_uav_dataset_index",
    "validate_uav_episode",
]
