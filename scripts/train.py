from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from _cfg import load_cfg
from leo_pg.utils.seed import set_seed
from leo_pg.utils.device import get_device
from leo_pg.data.dataset import TemporalEpisodeDataset
from leo_pg.data.collate import collate_episode
from leo_pg.models import build_model
from leo_pg.train import Trainer
from leo_pg.train.metrics import mse
from leo_pg.train.rollout import rollout_teacher_forcing


def _default_ckpt_dir(cfg: Dict) -> Path:
    save_dir = cfg.get("train", {}).get("save_dir", "runs")
    run_name = cfg.get("train", {}).get("run_name", None)
    if not run_name:
        msg = cfg.get("model", {}).get("message_type", "model")
        run_name = f"{msg}"
    return Path(save_dir) / str(run_name)


def save_checkpoint(path: Path, model: torch.nn.Module, cfg: Dict, metric_name: str, metric_value: float, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "metric_name": str(metric_name),
        "metric_value": float(metric_value),
        "cfg": cfg,
    }
    torch.save(payload, str(path))


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> Dict:
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return ckpt


@torch.no_grad()
def eval_one_step_mse(model: torch.nn.Module, dl: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> float:
    model.eval()
    vals = []
    for bi, episode in enumerate(dl):
        out = model.forward_episode(episode, device=device)
        pred = out["preds"][-1]
        y = out["ys"][-1]
        vals.append(float(mse(pred, y)))
        if max_batches is not None and (bi + 1) >= max_batches:
            break
    return float(sum(vals) / max(1, len(vals)))


@torch.no_grad()
def eval_rollout_teacher_forcing_mse(model: torch.nn.Module, dl: DataLoader, device: torch.device, H: int, max_batches: Optional[int] = None) -> float:
    model.eval()
    vals = []
    for bi, episode in enumerate(dl):
        r = rollout_teacher_forcing(model, episode, device=device, H=H)
        vals.append(float(r["mse_mean"]))
        if max_batches is not None and (bi + 1) >= max_batches:
            break
    return float(sum(vals) / max(1, len(vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--message", type=str, default=None, help="override message_type: mlp|kan|physick")
    ap.add_argument("--ckpt_dir", type=str, default=None, help="override checkpoint output directory")
    ap.add_argument("--eval_max_batches", type=int, default=None, help="limit val batches for faster selection")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.message is not None:
        cfg["model"]["message_type"] = args.message

    set_seed(int(cfg.get("seed", 7)))
    device = get_device(cfg["train"]["device"])

    # Datasets
    train_split = cfg.get("data", {}).get("train_split", "train")
    val_split = cfg.get("data", {}).get("val_split", "val")

    ds_train = TemporalEpisodeDataset(cfg["data"]["path"], split=train_split)
    dl_train = DataLoader(
        ds_train,
        batch_size=int(cfg["data"].get("batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=collate_episode,
    )

    ds_val = TemporalEpisodeDataset(cfg["data"]["path"], split=val_split)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_episode)

    model = build_model(cfg).to(device)

    tr = Trainer(
        model=model,
        device=device,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        clip_grad_norm=float(cfg["train"]["clip_grad_norm"]),
        log_every=int(cfg["train"].get("log_every", 20)),
    )

    task = cfg["train"].get("task", "one_step")
    epochs = int(cfg["train"]["epochs"])
    rollout_H = int(cfg.get("eval", {}).get("rollout_horizon", 30))

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else _default_ckpt_dir(cfg)
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    best_metric = float("inf")
    best_epoch = -1

    # Train epoch-by-epoch to enable best checkpointing without modifying Trainer internals.
    for ep in range(1, epochs + 1):
        if task == "one_step":
            tr.train_one_step(dl_train, epochs=1)
            val = eval_one_step_mse(model, dl_val, device=device, max_batches=args.eval_max_batches)
            metric_name = "val_one_step_mse"
        elif task == "rollout_teacher_forcing":
            tr.train_rollout_teacher_forcing(dl_train, epochs=1, horizon=rollout_H)
            val = eval_rollout_teacher_forcing_mse(model, dl_val, device=device, H=rollout_H, max_batches=args.eval_max_batches)
            metric_name = f"val_rollout_mse_mean@{rollout_H}"
        else:
            raise ValueError(f"Unknown train.task={task}")

        # Save last every epoch (lightweight + useful for debugging)
        save_checkpoint(last_path, model=model, cfg=cfg, metric_name=metric_name, metric_value=val, epoch=ep)

        if val < best_metric:
            best_metric = val
            best_epoch = ep
            save_checkpoint(best_path, model=model, cfg=cfg, metric_name=metric_name, metric_value=val, epoch=ep)

        print(f"[CKPT] epoch={ep:03d} {metric_name}={val:.6g} best={best_metric:.6g} (epoch {best_epoch})  dir={ckpt_dir}")

    print(f"[DONE] best checkpoint: {best_path}  ({metric_name}={best_metric:.6g} at epoch {best_epoch})")


if __name__ == "__main__":
    main()
