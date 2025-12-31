from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from _cfg import load_cfg
from leo_pg.utils.seed import set_seed
from leo_pg.utils.device import get_device
from leo_pg.sim.environment import MultiUserLEOEnv
from leo_pg.data.io import save_bundle


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
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(int(cfg.get("seed", 7)))
    device = get_device(args.device or cfg.get("device", "cuda"))

    env = MultiUserLEOEnv(cfg, device=device)

    episodes = []
    for ep in range(int(args.episodes)):
        step = env.reset()
        steps = []
        for _ in range(int(cfg.get("T", 200))):
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
            step = env.step()
        episodes.append({"steps": steps, "meta": {"episode": ep}})

    payload = {"episodes": episodes, "meta": {"generator_cfg": cfg, "episodes": len(episodes)}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_bundle(args.out, payload)
    print(f"[OK] Saved dataset to {args.out} with {len(episodes)} episode(s).")

if __name__ == "__main__":
    main()
