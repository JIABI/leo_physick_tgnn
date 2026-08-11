from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch

from leo_pg.utils.config import load_cfg
from leo_pg.utils.seed import set_seed
from leo_pg.utils.device import get_device
from leo_pg.sim.environment import MultiUserLEOEnv
from leo_pg.data.io import save_bundle
from leo_pg.data.schema import DATASET_SCHEMA_VERSION
from leo_pg.data.splits import split_episodes


def _to_cpu_tree(x: Any) -> Any:
    """Recursively move tensors to CPU for safe torch.save portability."""
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_to_cpu_tree(v) for v in x]
        return type(x)(t) if not isinstance(x, tuple) else tuple(t)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--device", type=str, default=None, help="Override device (cpu|cuda)")
    ap.add_argument(
        "--split-ratios",
        type=str,
        default="0.8,0.1,0.1",
        help="Comma-separated train,val,test episode ratios",
    )
    ap.add_argument("--split-seed", type=int, default=None, help="Episode split seed (default: cfg.seed)")
    args = ap.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    cfg = load_cfg(args.cfg)
    set_seed(int(cfg.get("seed", 7)))
    device = get_device(args.device or cfg.get("device", "cuda"), strict=args.device is not None)

    episodes = []
    for ep in range(int(args.episodes)):
        # Construct a fresh environment for every episode. Resetting only the
        # counters would otherwise leave the ephemeris advanced from the prior
        # episode and make the episodes neither independent nor true resets.
        episode_cfg = copy.deepcopy(cfg)
        episode_cfg["seed"] = int(cfg.get("seed", 7)) + ep
        env = MultiUserLEOEnv(episode_cfg, device=device)
        step = env.reset()
        steps = []
        horizon = int(cfg.get("T", 200))
        for step_index in range(horizon):
            meta = _to_cpu_tree(step.get("meta", {}))
            steps.append({
                "t": int(step["t"]),
                "node_x": step["node_x"].detach().cpu(),
                "edge_index": step["edge_index"].detach().cpu(),
                "edge_z": step["edge_z"].detach().cpu(),
                "edge_type": step["edge_type"].detach().cpu(),
                "y": step["y"].detach().cpu(),
                "meta": meta,
            })
            if step_index + 1 < horizon:
                step = env.step()
        episodes.append({"steps": steps, "meta": {"episode": ep, "seed": episode_cfg["seed"]}})

    ratios = tuple(float(value.strip()) for value in args.split_ratios.split(",") if value.strip())
    split_seed = int(args.split_seed if args.split_seed is not None else cfg.get("seed", 7))
    splits = split_episodes(episodes, ratios=ratios, seed=split_seed)
    payload = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "episodes": episodes,
        **splits,
        "meta": {
            "generator_cfg": cfg,
            "episodes": len(episodes),
            "split_ratios": ratios,
            "split_seed": split_seed,
            "split_sizes": {name: len(values) for name, values in splits.items()},
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_bundle(args.out, payload)
    sizes = ", ".join(f"{name}={len(values)}" for name, values in splits.items())
    print(f"[OK] Saved dataset to {args.out} with {len(episodes)} episode(s): {sizes}.")

if __name__ == "__main__":
    main()
