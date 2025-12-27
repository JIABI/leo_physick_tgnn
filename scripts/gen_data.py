from __future__ import annotations
import argparse
from pathlib import Path
import torch

from _cfg import load_cfg
from leo_pg.utils.seed import set_seed
from leo_pg.sim.environment import MultiUserLEOEnv
from leo_pg.data.io import save_bundle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(int(cfg.get("seed", 7)))
    device = torch.device("cpu")
    env = MultiUserLEOEnv(cfg, device=device)
    print(f"[CFG] ephemeris.mode={cfg.get('ephemeris',{}).get('mode','debug')}")
    # In TLE mode, env.S is overridden by len(TLE)
    print(f"[ENV] K_users={env.K} S_sats={env.S} N={env.N}")

    episodes = []
    for ep in range(args.episodes):
        step = env.reset()
        steps = []
        for t in range(int(cfg.get("T", 200))):
            if t > 0:
                step = env.step()
            # move to cpu tensors for saving
            steps.append({
                "t": int(step["t"]),
                "node_x": step["node_x"].detach().cpu(),
                "edge_index": step["edge_index"].detach().cpu(),
                "edge_z": step["edge_z"].detach().cpu(),
                "edge_type": step["edge_type"].detach().cpu(),
                "y": step["y"].detach().cpu(),
                "meta": step.get("meta", {}),
            })
        episodes.append({"steps": steps, "meta": {"episode": ep}})

    payload = {"episodes": episodes, "meta": {"generator_cfg": cfg, "episodes": len(episodes)}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_bundle(args.out, payload)
    print(f"[OK] Saved dataset to {args.out} with {len(episodes)} episode(s).")

if __name__ == "__main__":
    main()
