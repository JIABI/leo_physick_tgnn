"""Typed, tensor-only datasets for the paper-aligned Intensity--Flow protocol.

This module is deliberately separate from :mod:`leo_pg.data`, whose schema and
generator describe the legacy frozen-trajectory load-forecasting prototype.
Paper datasets record action-coupled transitions from
:class:`~leo_pg.sim.paper_environment.PaperAlignedLEOEnv` and expose explicit
``t -> t+1`` Intensity--Flow targets.

Only dictionaries, lists, tensors, and primitive scalar values are written to
disk.  Consequently, bundles can be loaded with ``torch.load(...,
weights_only=True)`` and do not depend on importing Python dataclasses during
deserialization.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from enum import Enum
import hashlib
import math
from pathlib import Path, PurePosixPath
import random
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from leo_pg.control.policy import FixedRankPolicy
from leo_pg.data.splits import split_episodes
from leo_pg.sim.paper_environment import PAPER_PROTOCOL_VERSION, PaperAlignedLEOEnv
from leo_pg.sim.state import (
    ControlObservation,
    ExecutionResult,
    FailureReason,
    PAPER_EDGE_FEATURE_NAMES,
    PAPER_FEATURE_CONTRACT_VERSION,
    PAPER_NODE_FEATURE_NAMES,
    PolicyDescriptors,
    ServingAction,
)
from leo_pg.train.intensity_flow import IntensityFlowTarget, build_next_step_target


PAPER_DATASET_SCHEMA_VERSION = 1
PAPER_DATASET_KIND = "leo_pg.paper.oracle_action_coupled"
PAPER_TARGET_CONTRACT_VERSION = 1
PAPER_SPLIT_NAMES = ("train", "val", "test")
PAPER_MONOLITHIC_LAYOUT = "monolithic_v1"
PAPER_SHARDED_LAYOUT = "episode_shards_v1"
PAPER_EPISODE_SHARD_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _cpu_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    """Detach, validate, and clone a tensor into the portable CPU payload."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.detach().to(device="cpu").contiguous().clone()
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _pure_tree(value: Any, *, path: str = "root") -> Any:
    """Convert supported values to the restricted paper-bundle object graph.

    Unsupported objects are rejected instead of being stringified silently.
    In particular, a callable ``flow.phi`` must be replaced by a named built-in
    mode or represented by separately released parameters before generation.
    """

    if isinstance(value, torch.Tensor):
        return _cpu_tensor(value, name=path)
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key: {key!r}")
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
        f"{path} has unsupported type {type(value).__name__}; paper bundles allow "
        "only dict/list/tensor/primitive values"
    )


def _validate_pure_tree(value: Any, *, path: str = "root") -> None:
    """Assert the saved object graph is already restricted and finite.

    This validator intentionally does not clone tensors.  Paper bundles can be
    large, so save-time validation must not transiently duplicate the complete
    dataset in memory.
    """

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{path} tensor must be on CPU before serialization")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{path} contains NaN or Inf")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key: {key!r}")
            _validate_pure_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_pure_tree(item, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    raise TypeError(
        f"{path} has unsupported type {type(value).__name__}; saved paper bundles "
        "allow only dict/list/tensor/primitive values"
    )


def serialize_policy_descriptors(descriptors: PolicyDescriptors) -> Dict[str, torch.Tensor]:
    """Serialize the three policy descriptor domains without storage aliasing."""

    return {
        "gamma_edge": _cpu_tensor(descriptors.gamma_edge, name="gamma_edge"),
        "intensity_edge": _cpu_tensor(
            descriptors.intensity_edge,
            name="intensity_edge",
        ),
        "flow_node": _cpu_tensor(descriptors.flow_node, name="flow_node"),
    }


def serialize_observation(observation: ControlObservation) -> Dict[str, Any]:
    """Serialize one complete controller observation and exact model input.

    The graph tensors are taken from :meth:`ControlObservation.as_model_step`,
    so ``node_x`` and ``edge_z`` contain the policy-facing descriptor copy that
    a TGN actually consumes.  Simulator-authoritative descriptors are stored in
    a separate nested mapping for target construction and audit.
    """

    observation.validate()
    model_step = observation.as_model_step()
    return {
        "observation_id": [observation.episode_seed, observation.epoch],
        "epoch": observation.epoch,
        "user_count": observation.user_count,
        "satellite_count": observation.satellite_count,
        "node_x": _cpu_tensor(model_step["node_x"], name="observation.node_x"),
        "edge_index": _cpu_tensor(
            model_step["edge_index"],
            name="observation.edge_index",
        ),
        "edge_z": _cpu_tensor(model_step["edge_z"], name="observation.edge_z"),
        "edge_type": _cpu_tensor(
            model_step["edge_type"],
            name="observation.edge_type",
        ),
        "candidate_edge_ids": _cpu_tensor(
            observation.candidate_edge_ids,
            name="observation.candidate_edge_ids",
        ),
        "elevation_deg": _cpu_tensor(
            observation.elevation_deg,
            name="observation.elevation_deg",
        ),
        "sim_descriptors": {
            **serialize_policy_descriptors(observation.sim_descriptors.policy_fields),
            "feasible_edge": _cpu_tensor(
                observation.sim_descriptors.feasible_edge,
                name="observation.sim_descriptors.feasible_edge",
            ),
        },
        "policy_descriptors": serialize_policy_descriptors(
            observation.policy_descriptors
        ),
        "current_serving": _cpu_tensor(
            observation.current_serving,
            name="observation.current_serving",
        ),
        "hold_steps": _cpu_tensor(
            observation.hold_steps,
            name="observation.hold_steps",
        ),
        "user_order": _cpu_tensor(
            observation.user_order,
            name="observation.user_order",
        ),
        "meta": _pure_tree(observation.meta, path="observation.meta"),
    }


def serialize_action(action: ServingAction) -> Dict[str, Any]:
    """Serialize the policy request committed for an observation."""

    return {
        "observation_id": [int(action.observation_id[0]), int(action.observation_id[1])],
        "requested_serving": _cpu_tensor(
            action.requested_serving,
            name="action.requested_serving",
        ),
    }


def serialize_execution(execution: ExecutionResult) -> Dict[str, Any]:
    """Serialize the simulator-authoritative result of one committed action."""

    return {
        "observation_id": [
            int(execution.observation_id[0]),
            int(execution.observation_id[1]),
        ],
        "requested_serving": _cpu_tensor(
            execution.requested_serving,
            name="execution.requested_serving",
        ),
        "executed_serving": _cpu_tensor(
            execution.executed_serving,
            name="execution.executed_serving",
        ),
        "admitted": _cpu_tensor(execution.admitted, name="execution.admitted"),
        "failure_reason": _cpu_tensor(
            execution.failure_reason,
            name="execution.failure_reason",
        ),
        "handover_attempted": _cpu_tensor(
            execution.handover_attempted,
            name="execution.handover_attempted",
        ),
        "handover_executed": _cpu_tensor(
            execution.handover_executed,
            name="execution.handover_executed",
        ),
        "flow_before": _cpu_tensor(
            execution.flow_before,
            name="execution.flow_before",
        ),
        "flow_after": _cpu_tensor(
            execution.flow_after,
            name="execution.flow_after",
        ),
    }


def serialize_target(
    target: IntensityFlowTarget,
    *,
    source_observation: ControlObservation,
    next_observation: ControlObservation,
) -> Dict[str, Any]:
    """Serialize one typed ``t -> t+1`` training target."""

    target.validate(source_observation.edge_count, source_observation.satellite_count)
    return {
        "contract_version": PAPER_TARGET_CONTRACT_VERSION,
        "source_observation_id": [
            source_observation.episode_seed,
            source_observation.epoch,
        ],
        "target_observation_id": [
            next_observation.episode_seed,
            next_observation.epoch,
        ],
        "gamma_edge": _cpu_tensor(target.gamma_edge, name="target.gamma_edge"),
        "log1p_intensity_edge": _cpu_tensor(
            target.log1p_intensity_edge,
            name="target.log1p_intensity_edge",
        ),
        "feasibility_edge": _cpu_tensor(
            target.feasibility_edge,
            name="target.feasibility_edge",
        ),
        "persistent_edge": _cpu_tensor(
            target.persistent_edge,
            name="target.persistent_edge",
        ),
        "flow_node": _cpu_tensor(target.flow_node, name="target.flow_node"),
    }


def _oracle_policy_observation(observation: ControlObservation) -> ControlObservation:
    """Create an explicit oracle policy copy for behavior-data generation."""

    return observation.with_policy_descriptors(
        observation.sim_descriptors.policy_fields
    )


def generate_oracle_episode(
    cfg: Mapping[str, Any],
    *,
    episode_id: int,
    seed: int,
    horizon_steps: int | None = None,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Generate one action-coupled episode under the fixed oracle policy.

    Every record contains the pre-decision observation, requested action,
    executed transition, and (except at the terminal epoch) the typed target at
    the next decision epoch.  The final post-action flow and association state
    are retained even though no further observation is constructed.
    """

    if isinstance(episode_id, bool) or int(episode_id) != episode_id or episode_id < 0:
        raise ValueError("episode_id must be a non-negative integer")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    episode_cfg = copy.deepcopy(dict(cfg))
    episode_cfg["seed"] = int(seed)
    if horizon_steps is not None:
        if (
            isinstance(horizon_steps, bool)
            or int(horizon_steps) != horizon_steps
            or horizon_steps <= 0
        ):
            raise ValueError("horizon_steps must be a positive integer")
        protocol = episode_cfg.setdefault("paper_protocol", {})
        if not isinstance(protocol, dict):
            raise TypeError("paper_protocol must be a mapping")
        protocol["horizon_steps"] = int(horizon_steps)

    environment = PaperAlignedLEOEnv(episode_cfg, device=device)
    policy_config = environment.fixed_policy_config()
    policy = FixedRankPolicy(policy_config)
    observation = environment.reset_control()
    records = []

    while True:
        oracle_observation = _oracle_policy_observation(observation)
        action = policy.select_action(oracle_observation)
        next_observation, execution, done = environment.step_action(action)
        target = None
        if next_observation is not None:
            target = serialize_target(
                build_next_step_target(oracle_observation, next_observation),
                source_observation=oracle_observation,
                next_observation=next_observation,
            )
        records.append(
            {
                "step_index": len(records),
                "observation": serialize_observation(oracle_observation),
                "action": serialize_action(action),
                "execution": serialize_execution(execution),
                "next_target": target,
                "done": bool(done),
            }
        )
        if done:
            break
        if next_observation is None:
            raise RuntimeError("paper environment omitted a non-terminal observation")
        observation = next_observation

    return {
        "episode_id": int(episode_id),
        "seed": int(seed),
        "protocol_version": PAPER_PROTOCOL_VERSION,
        "protocol_fingerprint": environment.protocol_fingerprint,
        "feature_contract_version": PAPER_FEATURE_CONTRACT_VERSION,
        "horizon_steps": environment.horizon_steps,
        "behavior_policy": {
            "name": "fixed_rank_oracle_v1",
            "descriptor_source": "simulator_oracle",
            "config": _pure_tree(asdict(policy_config), path="behavior_policy.config"),
        },
        "steps": records,
        "terminal": {
            "epoch": int(environment.epoch),
            "serving": _cpu_tensor(
                environment.current_serving,
                name="terminal.serving",
            ),
            "flow": _cpu_tensor(environment.flow, name="terminal.flow"),
        },
    }


def build_split_manifest(
    episodes: Sequence[Mapping[str, Any]],
    *,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    counts: Sequence[int] | None = None,
    seed: int = 7,
) -> Dict[str, Any]:
    """Build deterministic, mutually exclusive split indices and identities."""

    if isinstance(seed, bool):
        raise TypeError("split seed must be an integer")
    indices = list(range(len(episodes)))
    if counts is None:
        split_indices = split_episodes(indices, ratios=ratios, seed=int(seed))
        ratio_values = [float(value) for value in ratios]
        count_values = [len(split_indices[name]) for name in PAPER_SPLIT_NAMES]
        assignment = "deterministic_episode_shuffle_v1"
    else:
        if len(counts) != 3:
            raise ValueError("counts must contain train,val,test values")
        count_values = [int(value) for value in counts]
        if any(value < 0 for value in count_values):
            raise ValueError("split counts must be non-negative")
        if sum(count_values) != len(indices):
            raise ValueError("split counts must sum to the episode count")
        random.Random(int(seed)).shuffle(indices)
        train_end = count_values[0]
        val_end = train_end + count_values[1]
        split_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }
        ratio_values = [value / len(indices) for value in count_values]
        assignment = "deterministic_episode_shuffle_exact_counts_v1"
    return {
        "split_seed": int(seed),
        "ratios": {
            name: ratio_values[index]
            for index, name in enumerate(PAPER_SPLIT_NAMES)
        },
        "counts": {
            name: count_values[index]
            for index, name in enumerate(PAPER_SPLIT_NAMES)
        },
        "assignment": assignment,
        "episode_indices": {
            name: [int(index) for index in split_indices[name]]
            for name in PAPER_SPLIT_NAMES
        },
        "episode_ids": {
            name: [int(episodes[index]["episode_id"]) for index in split_indices[name]]
            for name in PAPER_SPLIT_NAMES
        },
        "episode_seeds": {
            name: [int(episodes[index]["seed"]) for index in split_indices[name]]
            for name in PAPER_SPLIT_NAMES
        },
    }


def _dataset_payload_header(
    cfg: Mapping[str, Any],
    *,
    generation: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    storage: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build metadata shared by monolithic bundles and sharded indices."""

    return {
        "paper_dataset_schema_version": PAPER_DATASET_SCHEMA_VERSION,
        "dataset_kind": PAPER_DATASET_KIND,
        "paper_protocol_version": PAPER_PROTOCOL_VERSION,
        "feature_contract_version": PAPER_FEATURE_CONTRACT_VERSION,
        "target_contract_version": PAPER_TARGET_CONTRACT_VERSION,
        "feature_contract": {
            "node_feature_names": list(PAPER_NODE_FEATURE_NAMES),
            "edge_feature_names": list(PAPER_EDGE_FEATURE_NAMES),
        },
        "target_contract": {
            "alignment": "source_candidate_ids_at_t_to_simulator_targets_at_t_plus_1",
            "edge_mask": "persistent_edge",
            "gamma_domain": "raw",
            "intensity_domain": "log1p",
            "flow_domain": "satellite_node",
            "feasibility_role": "auxiliary",
        },
        "failure_reason_codes": {
            reason.name.lower(): int(reason)
            for reason in FailureReason
        },
        "source_config": _pure_tree(cfg, path="source_config"),
        "generation": _pure_tree(generation, path="generation"),
        "storage": _pure_tree(storage, path="storage"),
        "split_manifest": _pure_tree(split_manifest, path="split_manifest"),
    }


def _validate_generation_request(
    cfg: Mapping[str, Any],
    *,
    episode_count: int,
    split_seed: int | None,
    base_seed: int | None,
) -> tuple[int, int]:
    if (
        isinstance(episode_count, bool)
        or int(episode_count) != episode_count
        or episode_count <= 0
    ):
        raise ValueError("episode_count must be a positive integer")
    configured_seed = cfg.get("seed", 7)
    resolved_seed = int(configured_seed if base_seed is None else base_seed)
    if resolved_seed < 0:
        raise ValueError("base_seed must be non-negative")
    resolved_split_seed = resolved_seed if split_seed is None else int(split_seed)
    if resolved_split_seed < 0:
        raise ValueError("split_seed must be non-negative")
    return resolved_seed, resolved_split_seed


def build_paper_dataset(
    cfg: Mapping[str, Any],
    *,
    episode_count: int,
    horizon_steps: int | None = None,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    split_counts: Sequence[int] | None = None,
    split_seed: int | None = None,
    base_seed: int | None = None,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Generate a monolithic payload for small fixtures and compatibility.

    Formal paper generation must use :func:`generate_sharded_paper_dataset` so
    episode tensors are never accumulated in one Python object.
    """

    resolved_seed, resolved_split_seed = _validate_generation_request(
        cfg,
        episode_count=episode_count,
        split_seed=split_seed,
        base_seed=base_seed,
    )

    episodes = [
        generate_oracle_episode(
            cfg,
            episode_id=index,
            seed=resolved_seed + index,
            horizon_steps=horizon_steps,
            device=device,
        )
        for index in range(int(episode_count))
    ]
    split_manifest = build_split_manifest(
        episodes,
        ratios=split_ratios,
        counts=split_counts,
        seed=resolved_split_seed,
    )
    effective_horizons = sorted(
        {int(episode["horizon_steps"]) for episode in episodes}
    )
    payload = _dataset_payload_header(
        cfg,
        generation={
            "episode_count": int(episode_count),
            "base_seed": resolved_seed,
            "episode_seed_rule": "base_seed_plus_episode_id",
            "horizon_steps": effective_horizons[0]
            if len(effective_horizons) == 1
            else effective_horizons,
            "device": str(torch.device(device)),
            "behavior_policy": "fixed_rank_oracle_v1",
        },
        storage={
            "layout": PAPER_MONOLITHIC_LAYOUT,
            "episode_count": int(episode_count),
        },
        split_manifest=split_manifest,
    )
    payload["episodes"] = episodes
    validate_paper_dataset_payload(payload)
    return payload


def generate_sharded_paper_dataset(
    path: str | Path,
    cfg: Mapping[str, Any],
    *,
    episode_count: int,
    horizon_steps: int | None = None,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    split_counts: Sequence[int] | None = None,
    split_seed: int | None = None,
    base_seed: int | None = None,
    episode_seeds: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Generate and save a lightweight index plus one file per episode.

    Only one generated episode is resident at a time.  The returned mapping is
    the lightweight index; it contains no episode tensors.
    """

    resolved_seed, resolved_split_seed = _validate_generation_request(
        cfg,
        episode_count=episode_count,
        split_seed=split_seed,
        base_seed=base_seed,
    )
    explicit_seeds: list[int] | None = None
    if episode_seeds is not None:
        if not isinstance(episode_seeds, Sequence) or isinstance(
            episode_seeds, (str, bytes)
        ):
            raise TypeError("episode_seeds must be a sequence of integers")
        explicit_seeds = []
        for index, value in enumerate(episode_seeds):
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(
                    f"episode_seeds[{index}] must be a non-negative integer"
                )
            explicit_seeds.append(int(value))
        if len(explicit_seeds) != int(episode_count):
            raise ValueError("episode_seeds length must equal episode_count")
        if len(explicit_seeds) != len(set(explicit_seeds)):
            raise ValueError("episode_seeds must not contain duplicates")
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_directory_name = f"{output.stem}_episodes"
    shard_root = output.parent / shard_directory_name
    shard_root.mkdir(parents=True, exist_ok=True)

    resolved_episode_seeds = (
        explicit_seeds
        if explicit_seeds is not None
        else [resolved_seed + index for index in range(int(episode_count))]
    )
    identities = [
        {"episode_id": index, "seed": resolved_episode_seeds[index]}
        for index in range(int(episode_count))
    ]
    split_manifest = build_split_manifest(
        identities,
        ratios=split_ratios,
        counts=split_counts,
        seed=resolved_split_seed,
    )
    split_by_index: Dict[int, str] = {}
    for split in PAPER_SPLIT_NAMES:
        for episode_index in split_manifest["episode_indices"][split]:
            split_by_index[int(episode_index)] = split

    shard_entries = []
    effective_horizons = set()
    for episode_index in range(int(episode_count)):
        episode = generate_oracle_episode(
            cfg,
            episode_id=episode_index,
            seed=resolved_episode_seeds[episode_index],
            horizon_steps=horizon_steps,
            device=device,
        )
        validate_paper_episode(
            episode,
            expected_index=episode_index,
            expected_seed=resolved_episode_seeds[episode_index],
        )
        shard_payload = {
            "paper_episode_shard_version": PAPER_EPISODE_SHARD_VERSION,
            "paper_dataset_schema_version": PAPER_DATASET_SCHEMA_VERSION,
            "dataset_kind": PAPER_DATASET_KIND,
            "episode": episode,
        }
        _validate_pure_tree(shard_payload, path="paper_episode_shard")
        filename = f"episode_{episode_index:06d}.pt"
        shard_path = shard_root / filename
        temporary_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        torch.save(shard_payload, temporary_path)
        temporary_path.replace(shard_path)
        relative_path = (Path(shard_directory_name) / filename).as_posix()
        shard_entries.append(
            {
                "episode_index": episode_index,
                "episode_id": int(episode["episode_id"]),
                "seed": int(episode["seed"]),
                "split": split_by_index[episode_index],
                "relative_path": relative_path,
                "size_bytes": int(shard_path.stat().st_size),
                "sha256": _file_sha256(shard_path),
                "horizon_steps": int(episode["horizon_steps"]),
                "protocol_fingerprint": str(episode["protocol_fingerprint"]),
            }
        )
        effective_horizons.add(int(episode["horizon_steps"]))

    sorted_horizons = sorted(effective_horizons)
    payload = _dataset_payload_header(
        cfg,
        generation={
            "episode_count": int(episode_count),
            "base_seed": None if explicit_seeds is not None else resolved_seed,
            "episode_seed_rule": (
                "explicit_seed_manifest"
                if explicit_seeds is not None
                else "base_seed_plus_episode_id"
            ),
            "episode_seeds": (
                resolved_episode_seeds if explicit_seeds is not None else None
            ),
            "horizon_steps": sorted_horizons[0]
            if len(sorted_horizons) == 1
            else sorted_horizons,
            "device": str(torch.device(device)),
            "behavior_policy": "fixed_rank_oracle_v1",
        },
        storage={
            "layout": PAPER_SHARDED_LAYOUT,
            "episode_count": int(episode_count),
            "relative_shard_directory": shard_directory_name,
            "integrity": "size_bytes_and_sha256",
        },
        split_manifest=split_manifest,
    )
    payload["episode_shards"] = shard_entries
    save_paper_dataset(output, payload)
    return payload


def validate_paper_episode(
    episode: Mapping[str, Any],
    *,
    expected_index: int | None = None,
    expected_seed: int | None = None,
) -> None:
    """Validate one monolithic episode or independently stored shard."""

    if not isinstance(episode, Mapping):
        raise TypeError("each paper episode must be a mapping")
    episode_id = episode.get("episode_id")
    seed = episode.get("seed")
    if (
        isinstance(episode_id, bool)
        or not isinstance(episode_id, int)
        or episode_id < 0
    ):
        raise ValueError("episode_id must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("episode seed must be a non-negative integer")
    if expected_index is not None and episode_id != expected_index:
        raise ValueError("episode_id must equal its stable top-level index")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("episode seed does not match its index entry")
    fingerprint = episode.get("protocol_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("each episode requires a 64-character protocol fingerprint")
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("each paper episode must contain decision steps")
    if int(episode.get("horizon_steps", -1)) != len(steps):
        raise ValueError("episode horizon_steps must equal the number of records")
    for step_index, record in enumerate(steps):
        if not isinstance(record, Mapping):
            raise TypeError("each paper step record must be a mapping")
        if int(record.get("step_index", -1)) != step_index:
            raise ValueError("step_index must be contiguous from zero")
        observation = record.get("observation")
        action = record.get("action")
        execution = record.get("execution")
        if not all(
            isinstance(value, Mapping)
            for value in (observation, action, execution)
        ):
            raise TypeError("step observation/action/execution must be mappings")
        expected_id = [seed, step_index]
        ids = (
            observation.get("observation_id"),
            action.get("observation_id"),
            execution.get("observation_id"),
        )
        if any(value != expected_id for value in ids):
            raise ValueError("observation, action, and execution ids must agree")
        done = record.get("done")
        if not isinstance(done, bool) or done != (step_index == len(steps) - 1):
            raise ValueError("only the last paper step may have done=True")
        target = record.get("next_target")
        if done and target is not None:
            raise ValueError("terminal paper step must not contain a next target")
        if not done:
            if not isinstance(target, Mapping):
                raise ValueError("non-terminal paper step requires a typed next target")
            if target.get("source_observation_id") != expected_id:
                raise ValueError("target source observation id is misaligned")
            if target.get("target_observation_id") != [seed, step_index + 1]:
                raise ValueError("target observation id is not t+1")


def _validate_split_manifest(
    manifest: Any,
    *,
    episode_ids: Sequence[int],
    episode_seeds: Sequence[int],
) -> Dict[int, str]:
    if not isinstance(manifest, Mapping):
        raise ValueError("paper dataset requires a split_manifest mapping")
    assignments = manifest.get("episode_indices")
    if not isinstance(assignments, Mapping):
        raise ValueError("split_manifest requires episode_indices")
    manifest_ids = manifest.get("episode_ids")
    manifest_seeds = manifest.get("episode_seeds")
    if not isinstance(manifest_ids, Mapping) or not isinstance(
        manifest_seeds, Mapping
    ):
        raise ValueError("split_manifest requires episode_ids and episode_seeds")
    flattened = []
    split_by_index: Dict[int, str] = {}
    for split in PAPER_SPLIT_NAMES:
        indices = assignments.get(split)
        if not isinstance(indices, list):
            raise ValueError(f"split {split!r} must contain a list of episode indices")
        if len(indices) != len(set(indices)):
            raise ValueError(f"split {split!r} contains duplicate episode indices")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(episode_ids)
            for index in indices
        ):
            raise ValueError(f"split {split!r} contains an invalid episode index")
        if manifest_ids.get(split) != [episode_ids[index] for index in indices]:
            raise ValueError(f"split {split!r} episode_ids do not match episode_indices")
        if manifest_seeds.get(split) != [episode_seeds[index] for index in indices]:
            raise ValueError(f"split {split!r} episode_seeds do not match episode_indices")
        for index in indices:
            split_by_index[index] = split
        flattened.extend(indices)
    if sorted(flattened) != list(range(len(episode_ids))):
        raise ValueError("train/val/test splits must cover every episode exactly once")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or any(
        counts.get(split) != len(assignments[split]) for split in PAPER_SPLIT_NAMES
    ):
        raise ValueError("split_manifest counts do not match episode_indices")
    return split_by_index


def _storage_layout(payload: Mapping[str, Any]) -> str:
    storage = payload.get("storage")
    if storage is None and isinstance(payload.get("episodes"), list):
        return PAPER_MONOLITHIC_LAYOUT
    if not isinstance(storage, Mapping):
        raise ValueError("paper dataset requires a storage mapping")
    layout = storage.get("layout")
    if layout not in {PAPER_MONOLITHIC_LAYOUT, PAPER_SHARDED_LAYOUT}:
        raise ValueError(f"unsupported paper dataset storage layout: {layout!r}")
    return str(layout)


def _validated_relative_shard_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("episode shard relative_path must be a non-empty string")
    if "\\" in value:
        raise ValueError("episode shard relative_path must use POSIX separators")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or value != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(
            "episode shard relative_path must stay below the index directory"
        )
    return relative


def validate_paper_dataset_payload(payload: Mapping[str, Any]) -> None:
    """Validate a legacy monolithic bundle or lightweight sharded index."""

    if not isinstance(payload, Mapping):
        raise TypeError("paper dataset payload must be a mapping")
    if payload.get("paper_dataset_schema_version") != PAPER_DATASET_SCHEMA_VERSION:
        raise ValueError(
            "unsupported paper dataset schema version: "
            f"{payload.get('paper_dataset_schema_version')!r}"
        )
    if payload.get("dataset_kind") != PAPER_DATASET_KIND:
        raise ValueError(
            f"unexpected paper dataset kind: {payload.get('dataset_kind')!r}"
        )
    layout = _storage_layout(payload)
    episode_ids: list[int] = []
    episode_seeds: list[int] = []

    if layout == PAPER_MONOLITHIC_LAYOUT:
        episodes = payload.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("monolithic paper dataset must contain episodes")
        if "episode_shards" in payload:
            raise ValueError("monolithic paper dataset cannot contain episode_shards")
        for expected_index, episode in enumerate(episodes):
            validate_paper_episode(episode, expected_index=expected_index)
            episode_ids.append(int(episode["episode_id"]))
            episode_seeds.append(int(episode["seed"]))
    else:
        if "episodes" in payload:
            raise ValueError("sharded paper index cannot embed episodes")
        entries = payload.get("episode_shards")
        if not isinstance(entries, list) or not entries:
            raise ValueError("sharded paper index must contain episode_shards")
        relative_paths = []
        for expected_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise TypeError("each episode shard index entry must be a mapping")
            episode_index = entry.get("episode_index")
            episode_id = entry.get("episode_id")
            seed = entry.get("seed")
            if episode_index != expected_index or episode_id != expected_index:
                raise ValueError("shard episode_index and episode_id must be contiguous")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("shard seed must be a non-negative integer")
            relative = _validated_relative_shard_path(entry.get("relative_path"))
            relative_paths.append(relative.as_posix())
            size_bytes = entry.get("size_bytes")
            if (
                isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes <= 0
            ):
                raise ValueError("shard size_bytes must be a positive integer")
            sha256 = entry.get("sha256")
            if not isinstance(sha256, str) or len(sha256) != 64:
                raise ValueError("shard sha256 must contain 64 hexadecimal characters")
            try:
                int(sha256, 16)
            except ValueError as exc:
                raise ValueError(
                    "shard sha256 must contain 64 hexadecimal characters"
                ) from exc
            if entry.get("split") not in PAPER_SPLIT_NAMES:
                raise ValueError("shard split must be train, val, or test")
            horizon = entry.get("horizon_steps")
            if (
                isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or horizon <= 0
            ):
                raise ValueError("shard horizon_steps must be a positive integer")
            fingerprint = entry.get("protocol_fingerprint")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                raise ValueError("shard requires a 64-character protocol fingerprint")
            episode_ids.append(int(episode_id))
            episode_seeds.append(seed)
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError("episode shard relative paths must be unique")

    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode ids must be unique")
    if len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("episode seeds must be unique")
    split_by_index = _validate_split_manifest(
        payload.get("split_manifest"),
        episode_ids=episode_ids,
        episode_seeds=episode_seeds,
    )
    if layout == PAPER_SHARDED_LAYOUT:
        storage = payload["storage"]
        if storage.get("episode_count") != len(episode_ids):
            raise ValueError("storage episode_count does not match episode_shards")
        shard_directory = _validated_relative_shard_path(
            storage.get("relative_shard_directory")
        )
        for index, entry in enumerate(payload["episode_shards"]):
            if entry["split"] != split_by_index[index]:
                raise ValueError("shard split does not match split_manifest")
            relative = _validated_relative_shard_path(entry["relative_path"])
            if relative.parent != shard_directory:
                raise ValueError(
                    "episode shard path does not match relative_shard_directory"
                )
    generation = payload.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("paper dataset requires generation metadata")
    if generation.get("episode_count") != len(episode_ids):
        raise ValueError("generation episode_count does not match the index")


def save_paper_dataset(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Validate and save a restricted-unpickler-compatible paper bundle."""

    if not isinstance(payload, dict):
        raise TypeError("paper dataset payload must be a dictionary")
    _validate_pure_tree(payload, path="paper_dataset")
    validate_paper_dataset_payload(payload)
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output)
    return output


def load_paper_dataset(path: str | Path) -> Dict[str, Any]:
    """Load and validate a paper bundle with PyTorch's restricted unpickler."""

    payload = torch.load(
        Path(path).expanduser(),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("paper dataset file must contain a dictionary")
    validate_paper_dataset_payload(payload)
    return payload


class PaperEpisodeDataset(Dataset):
    """Select one split and lazily load one sharded episode per access."""

    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        *,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        if split not in PAPER_SPLIT_NAMES:
            raise ValueError(f"split must be one of: {', '.join(PAPER_SPLIT_NAMES)}")
        self.index_path = Path(path).expanduser()
        if payload is None:
            self.payload = load_paper_dataset(self.index_path)
        else:
            validate_paper_dataset_payload(payload)
            self.payload = payload
        self.storage_layout = _storage_layout(self.payload)
        self.split = split
        self.indices = list(
            self.payload["split_manifest"]["episode_indices"][split]
        )
        self._verified_shards: set[int] = set()
        if not self.indices:
            raise ValueError(
                f"paper dataset split {split!r} is empty; generate more episodes or "
                "change the requested split ratios"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("paper dataset index must be an integer")
        if index < 0:
            index += len(self.indices)
        if not 0 <= index < len(self.indices):
            raise IndexError("paper dataset index out of range")
        episode_index = self.indices[index]
        if self.storage_layout == PAPER_MONOLITHIC_LAYOUT:
            return self.payload["episodes"][episode_index]

        entry = self.payload["episode_shards"][episode_index]
        relative = _validated_relative_shard_path(entry["relative_path"])
        index_root = self.index_path.parent.resolve()
        shard_path = index_root.joinpath(*relative.parts).resolve()
        try:
            shard_path.relative_to(index_root)
        except ValueError as exc:
            raise ValueError("episode shard resolves outside the index directory") from exc
        if not shard_path.is_file():
            raise FileNotFoundError(f"paper episode shard does not exist: {shard_path}")
        if shard_path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"paper episode shard size mismatch: {shard_path}")
        verify_integrity = episode_index not in self._verified_shards
        if verify_integrity:
            if _file_sha256(shard_path) != entry["sha256"]:
                raise ValueError(f"paper episode shard SHA-256 mismatch: {shard_path}")

        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        if not isinstance(shard, dict):
            raise TypeError("paper episode shard must contain a dictionary")
        if shard.get("paper_episode_shard_version") != PAPER_EPISODE_SHARD_VERSION:
            raise ValueError("unsupported paper episode shard version")
        if shard.get("paper_dataset_schema_version") != PAPER_DATASET_SCHEMA_VERSION:
            raise ValueError("episode shard dataset schema does not match its index")
        if shard.get("dataset_kind") != PAPER_DATASET_KIND:
            raise ValueError("episode shard dataset kind does not match its index")
        episode = shard.get("episode")
        if not isinstance(episode, dict):
            raise TypeError("paper episode shard requires an episode dictionary")
        validate_paper_episode(
            episode,
            expected_index=episode_index,
            expected_seed=int(entry["seed"]),
        )
        if int(episode["horizon_steps"]) != entry["horizon_steps"]:
            raise ValueError("episode shard horizon does not match its index")
        if episode["protocol_fingerprint"] != entry["protocol_fingerprint"]:
            raise ValueError("episode shard protocol fingerprint does not match its index")
        if verify_integrity:
            self._verified_shards.add(episode_index)
        return episode
