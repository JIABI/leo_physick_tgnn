from __future__ import annotations

import argparse

import os

from pathlib import Path

from typing import Any, Dict, Optional

import torch

from torch.utils.data import DataLoader

from _cfg import load_cfg

from leo_pg.utils.seed import set_seed

from leo_pg.utils.device import get_device

from leo_pg.data.dataset import TemporalEpisodeDataset

from leo_pg.data.collate import collate_episode

from leo_pg.models import build_model

from leo_pg.train import Trainer

from leo_pg.train.checkpoint import save_ckpt


def _resolve_run_name(cfg: Dict[str, Any], message_type: str, mode: str) -> str:
    tr = cfg.setdefault("train", {})

    rn = str(tr.get("run_name", "tgn_run"))

    # Replace placeholders

    rn = rn.replace("$message_type$", message_type).replace("{message_type}", message_type)

    rn = rn.replace("$mode$", mode).replace("{mode}", mode)

    # If no placeholder was used, append message_type to avoid collisions

    raw = str(tr.get("run_name", ""))

    if ("$message_type$" not in raw) and ("{message_type}" not in raw) and (not rn.endswith(f"_{message_type}")):
        rn = f"{rn}_{message_type}"

    if ("$mode$" not in raw) and ("{mode}" not in raw) and (f"_{mode}_" not in rn) and (not rn.endswith(f"_{mode}")):

        # only append mode if it's non-empty and not already present

        if mode:
            rn = f"{rn}_{mode}"

    tr["run_name"] = rn

    return rn


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cfg", type=str, required=True, help="Path to YAML config")

    ap.add_argument("--message", type=str, default=None, help="Override message_type: mlp|kan|physick")

    ap.add_argument("--data", type=str, default=None, help="Override data path (pt)")

    ap.add_argument("--mode", type=str, default=None, help="Override mode label for run_name (single|multi), optional")

    args = ap.parse_args()

    cfg: Dict[str, Any] = load_cfg(args.cfg)

    # Overrides: message_type

    if args.message is not None:
        cfg.setdefault("model", {})

        cfg["model"]["message_type"] = args.message

    # Overrides: data path priority = --data > env DATA_PATH > cfg.data.path

    env_data = os.environ.get("DATA_PATH")

    data_path = args.data or env_data or cfg.get("data", {}).get("path")

    if not data_path:
        raise ValueError("No data path provided. Set cfg.data.path or pass --data or export DATA_PATH=...")

    cfg.setdefault("data", {})

    cfg["data"]["path"] = data_path

    # Mode label (only for naming)

    env_mode = os.environ.get("MODE")

    mode = args.mode or env_mode or str(cfg.get("train", {}).get("mode", "")).strip()

    # Seed/device

    set_seed(int(cfg.get("seed", 7)))

    device = get_device(str(cfg.get("train", {}).get("device", "cuda")))

    # DataLoader

    split = str(cfg["data"].get("split", "train"))

    ds = TemporalEpisodeDataset(cfg["data"]["path"], split=split)

    dl = DataLoader(

        ds,

        batch_size=int(cfg["data"].get("batch_size", 1)),

        shuffle=True,

        num_workers=int(cfg["data"].get("num_workers", 0)),

        collate_fn=collate_episode,

    )

    # Model

    model = build_model(cfg).to(device)

    # Run directory

    save_dir = Path(cfg.get("train", {}).get("save_dir", "runs"))

    msg = str(cfg.get("model", {}).get("message_type", "mlp"))

    run_name = _resolve_run_name(cfg, msg, mode)

    run_dir = save_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=True)

    # Print the *actual* data path and run info

    print(f"[RUN] mode={mode or '-'} message_type={msg}")

    print(
        f"[DATA] path={cfg['data']['path']} split={split} batch_size={cfg['data'].get('batch_size', 1)} workers={cfg['data'].get('num_workers', 0)}")

    print(f"[OUT]  dir={run_dir}")

    # Trainer

    tr = Trainer(

        model=model,

        device=device,

        lr=float(cfg["train"]["lr"]),

        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),

        clip_grad_norm=float(cfg["train"].get("clip_grad_norm", 1.0)),

        log_every=int(cfg["train"].get("log_every", 20)),

    )

    # Train

    task = str(cfg.get("train", {}).get("task", "one_step"))

    epochs = int(cfg.get("train", {}).get("epochs", 10))

    if task == "one_step":

        tr.train_one_step(dl, epochs=epochs)

    elif task == "rollout_teacher_forcing":

        horizon = int(cfg.get("train", {}).get("horizon", cfg.get("eval", {}).get("rollout_horizon", 30)))

        multistep_loss_cfg = cfg.get("train", {}).get("multistep_loss", None)

        tr.train_rollout_teacher_forcing(dl, epochs=epochs, horizon=horizon, multistep_loss_cfg=multistep_loss_cfg)

    else:

        raise ValueError(f"Unknown train.task={task}")

    # Save last checkpoint (always)

    save_ckpt(str(run_dir / "last.pt"), model=model, opt=tr.opt, epoch=epochs)

    print(f"[CKPT] saved: {run_dir / 'last.pt'}")


if __name__ == "__main__":
    main()


