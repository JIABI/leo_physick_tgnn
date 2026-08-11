from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from leo_pg.data.io import load_bundle
from leo_pg.data.schema import DATASET_SCHEMA_VERSION
from leo_pg.sim.protocol import validate_allocator_protocol


REQUIRED_STEP_KEYS = {"t", "node_x", "edge_index", "edge_z", "edge_type", "y", "meta"}
INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


def _check_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def verify_bundle(path: str) -> Dict[str, Any]:
    payload = load_bundle(path)
    schema_version = payload.get("dataset_schema_version") if isinstance(payload, dict) else None
    if schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Dataset schema version {schema_version!r} is unsupported; expected "
            f"{DATASET_SCHEMA_VERSION}"
        )
    if not isinstance(payload, dict) or "episodes" not in payload:
        raise ValueError("Expected a dict dataset bundle containing 'episodes'")
    episodes = payload["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Dataset must contain at least one episode")

    step_count = 0
    edge_counts: List[int] = []
    for episode_index, episode in enumerate(episodes):
        steps = episode.get("steps") if isinstance(episode, dict) else None
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Episode {episode_index} has no steps")
        previous_satellite_target = None
        expected_dimensions = None
        for step_index, step in enumerate(steps):
            missing = REQUIRED_STEP_KEYS.difference(step)
            if missing:
                raise ValueError(f"Episode {episode_index}, step {step_index} missing keys: {sorted(missing)}")
            node_x = torch.as_tensor(step["node_x"])
            edge_index = torch.as_tensor(step["edge_index"])
            edge_z = torch.as_tensor(step["edge_z"])
            edge_type = torch.as_tensor(step["edge_type"])
            target = torch.as_tensor(step["y"])
            meta = step["meta"]
            if not isinstance(meta, dict):
                raise ValueError("step.meta must be a mapping")
            if node_x.ndim != 2:
                raise ValueError(f"node_x must be rank 2, got {tuple(node_x.shape)}")
            if edge_index.ndim != 2 or edge_index.size(0) != 2:
                raise ValueError(f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}")
            if edge_index.dtype not in INTEGER_DTYPES:
                raise ValueError(f"edge_index must use an integer dtype, got {edge_index.dtype}")
            edge_count = int(edge_index.size(1))
            if edge_count and (int(edge_index.min()) < 0 or int(edge_index.max()) >= node_x.size(0)):
                raise ValueError(
                    f"Episode {episode_index}, step {step_index} has edge indices outside "
                    f"[0,{node_x.size(0) - 1}]"
                )
            if edge_z.ndim != 2 or edge_z.size(0) != edge_count:
                raise ValueError("edge_z row count must equal edge_index edge count")
            if edge_type.ndim != 1 or edge_type.numel() != edge_count:
                raise ValueError("edge_type length must equal edge_index edge count")
            if edge_type.dtype not in INTEGER_DTYPES:
                raise ValueError(f"edge_type must use an integer dtype, got {edge_type.dtype}")
            if target.ndim != 2 or target.size(0) != node_x.size(0) or target.size(1) < 1:
                raise ValueError("y must have shape [node_count, target_dim>=1]")
            required_meta = {
                "K_users",
                "S_sats",
                "sat_capacity_users",
                "sat_load_pre",
                "allocation_user_order",
                "allocator_protocol",
                "serving_sat",
                "ho_fail",
            }
            missing_meta = required_meta.difference(meta)
            if missing_meta:
                raise ValueError(f"step.meta missing protocol fields: {sorted(missing_meta)}")
            user_count = int(meta["K_users"])
            satellite_count = int(meta["S_sats"])
            if user_count < 0 or satellite_count <= 0 or user_count + satellite_count != node_x.size(0):
                raise ValueError(
                    f"K_users + S_sats must equal node count, got {user_count} + "
                    f"{satellite_count} != {node_x.size(0)}"
                )
            dimensions = (user_count, satellite_count)
            if expected_dimensions is None:
                expected_dimensions = dimensions
            elif dimensions != expected_dimensions:
                raise ValueError("K_users/S_sats change within an episode")
            supported_edge_types = (edge_type == 0) | (edge_type == 1)
            if edge_count and not bool(torch.all(supported_edge_types)):
                invalid_types = torch.unique(edge_type[~supported_edge_types]).tolist()
                raise ValueError(f"edge_type contains unsupported values: {invalid_types}")
            user_sat = edge_type == 0
            if torch.any(user_sat):
                user_sat_edges = edge_index[:, user_sat]
                if torch.any(user_sat_edges[0] >= user_count) or torch.any(
                    user_sat_edges[1] < user_count
                ):
                    raise ValueError(
                        "USER_SAT edges must run from a user node to a satellite node"
                    )
            sat_sat = edge_type == 1
            if torch.any(sat_sat):
                sat_sat_edges = edge_index[:, sat_sat]
                if torch.any(sat_sat_edges < user_count):
                    raise ValueError("SAT_SAT edges must connect two satellite nodes")
            capacity = int(meta["sat_capacity_users"])
            if capacity <= 0:
                raise ValueError("sat_capacity_users must be positive")
            pre_load = torch.as_tensor(meta["sat_load_pre"]).float()
            order = torch.as_tensor(meta["allocation_user_order"]).long()
            serving = torch.as_tensor(meta["serving_sat"]).long()
            failures = torch.as_tensor(meta["ho_fail"]).bool()
            if pre_load.shape != (satellite_count,):
                raise ValueError("sat_load_pre must have shape [S_sats]")
            if order.shape != (user_count,) or not torch.equal(
                torch.sort(order).values,
                torch.arange(user_count),
            ):
                raise ValueError("allocation_user_order must be a permutation of [0,K_users)")
            if serving.shape != (user_count,) or failures.shape != (user_count,):
                raise ValueError("serving_sat and ho_fail must have shape [K_users]")
            if serving.numel() and (int(serving.min()) < -1 or int(serving.max()) >= satellite_count):
                raise ValueError("serving_sat contains an invalid local satellite id")
            validate_allocator_protocol(meta["allocator_protocol"])
            _check_finite(pre_load, "sat_load_pre")
            if previous_satellite_target is not None and not torch.allclose(
                pre_load,
                previous_satellite_target,
                atol=1e-6,
                rtol=1e-5,
            ):
                raise ValueError("prior satellite target does not match current sat_load_pre")
            previous_satellite_target = target[user_count:, 0].float()
            _check_finite(node_x, "node_x")
            _check_finite(edge_z, "edge_z")
            _check_finite(target, "y")
            edge_counts.append(edge_count)
            step_count += 1

    split_names = ("train", "val", "test")
    present_splits = [split for split in split_names if split in payload]
    if present_splits and len(present_splits) != len(split_names):
        raise ValueError("Explicit split bundles must contain train, val, and test keys")

    def episode_key(episode: Dict[str, Any]) -> tuple[str, Any]:
        meta = episode.get("meta", {}) if isinstance(episode, dict) else {}
        if "episode" in meta:
            return ("episode", meta["episode"])
        return ("object", id(episode))

    episode_keys = [episode_key(episode) for episode in episodes]
    if len(set(episode_keys)) != len(episode_keys):
        raise ValueError("Top-level episodes contain duplicate episode identifiers")
    if present_splits:
        split_keys = {
            split: [episode_key(episode) for episode in payload[split]]
            for split in split_names
        }
        for split, keys in split_keys.items():
            if len(keys) != len(set(keys)):
                raise ValueError(f"Split {split!r} contains duplicate episodes")
            unknown = set(keys).difference(episode_keys)
            if unknown:
                raise ValueError(f"Split {split!r} contains episodes absent from top-level episodes")
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1:]:
                if set(split_keys[left]).intersection(split_keys[right]):
                    raise ValueError(f"Splits {left!r} and {right!r} are not mutually exclusive")
        covered = set().union(*(set(keys) for keys in split_keys.values()))
        if covered != set(episode_keys):
            raise ValueError("Explicit splits do not cover every top-level episode exactly once")

    split_sizes = {split: len(payload[split]) for split in present_splits}
    first = episodes[0]["steps"][0]
    summary = {
        "path": str(Path(path)),
        "episodes": len(episodes),
        "steps": step_count,
        "split_sizes": split_sizes,
        "node_shape": tuple(torch.as_tensor(first["node_x"]).shape),
        "edge_feature_dim": int(torch.as_tensor(first["edge_z"]).size(-1)),
        "target_shape": tuple(torch.as_tensor(first["y"]).shape),
        "edge_count_min": min(edge_counts),
        "edge_count_max": max(edge_counts),
        "edge_count_mean": sum(edge_counts) / len(edge_counts),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a leo_physick_tgn dataset bundle")
    parser.add_argument("path", help="Path to a torch .pt dataset bundle")
    args = parser.parse_args()
    summary = verify_bundle(args.path)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("[OK] dataset validation passed")


if __name__ == "__main__":
    main()
