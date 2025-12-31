"""Plot rollout curves for mlp/kan/physick across horizons.

This script expects checkpoints saved by scripts/train.py (best.pt/last.pt).
It is robust to older checkpoint schemas via leo_pg.train.checkpoint.load_ckpt.
"""
from __future__ import annotations
import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from leo_pg.models.registry import build_model
from leo_pg.sim.environment import MultiUserLEOEnv
from leo_pg.train.checkpoint import load_ckpt
from leo_pg.train.rollout import rollout_episode

def _deep_update(d, u):
    for k,v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k]=v
    return d

def _resolve_run_name(cfg, message_type: str):
    rn = str(cfg["train"].get("run_name","run_{message_type}"))
    rn = rn.replace("$message_type$", message_type).replace("{message_type}", message_type)
    if ("$message_type$" not in str(cfg["train"].get("run_name","")) and "{message_type}" not in str(cfg["train"].get("run_name",""))):
        # avoid accidental overwrite across methods
        if not rn.endswith(f"_{message_type}"):
            rn = f"{rn}_{message_type}"
    return rn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--Hs", required=True, help="comma-separated horizons, e.g. 20,50,100,600")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--which", default="best", choices=["best","last"])
    ap.add_argument("--metric", default="mse_mean", choices=["mse_mean","mse_last"])
    ap.add_argument("--out", default="rollout_curves.png")
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["train"].get("device","cpu")
    Hs = [int(x) for x in args.Hs.split(",") if x.strip()]

    methods = ["mlp","kan","physick"]
    ys = {m: [] for m in methods}

    # Build a single env/dataset episode (the .pt may hold 1 episode; we use env to interpret)
    # For plotting, we load the episode from the pt directly to keep it simple.
    pt_path = cfg["data"]["path"]
    obj = torch.load(pt_path, map_location="cpu")
    episode = obj["episodes"][0]

    for m in methods:
        cfg_m = dict(cfg)
        cfg_m = _deep_update(cfg_m, {"model": {"message_type": m}})
        run_name = _resolve_run_name(cfg_m, m)
        ckpt_path = os.path.join(cfg_m["train"].get("save_dir","runs"), run_name, f"{args.which}.pt")

        model = build_model(cfg_m).to(device)
        load_ckpt(ckpt_path, model, opt=None, map_location=device, strict=True)

        for H in Hs:
            out = rollout_episode(model, episode, device=device, H=H)
            ys[m].append(float(out[args.metric]))

    # plot
    plt.figure()
    for m in methods:
        plt.plot(Hs, ys[m], marker="o", label=m)
    plt.xlabel("Horizon H (steps)")
    plt.ylabel(args.metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
