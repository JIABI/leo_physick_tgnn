from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from _cfg import load_cfg
from leo_pg.utils.seed import set_seed
from leo_pg.utils.device import get_device
from leo_pg.data.dataset import TemporalEpisodeDataset
from leo_pg.data.collate import collate_episode
from leo_pg.models import build_model
from leo_pg.train.rollout import rollout_teacher_forcing


def _default_ckpt_path(cfg: Dict) -> Optional[Path]:
    p = cfg.get("eval", {}).get("ckpt_path", None) or cfg.get("train", {}).get("ckpt_path", None)
    if p:
        return Path(p)
    save_dir = cfg.get("train", {}).get("save_dir", "runs")
    run_name = cfg.get("train", {}).get("run_name", None)
    if run_name:
        cand = Path(save_dir) / str(run_name) / "best.pt"
        if cand.exists():
            return cand
    return None


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> Dict:
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return ckpt


@torch.no_grad()
def eval_rollout_mean_mse(model: torch.nn.Module, dl: DataLoader, device: torch.device, H: int, max_batches: Optional[int] = None) -> float:
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
    ap.add_argument("--H", type=int, default=30)
    ap.add_argument("--message", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None, help="path to checkpoint .pt (best.pt/last.pt)")
    ap.add_argument("--split", type=str, default=None, help="override split: train|val|test")
    ap.add_argument("--max_batches", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.message is not None:
        cfg["model"]["message_type"] = args.message

    set_seed(int(cfg.get("seed", 7)))
    device = get_device(cfg["train"]["device"])

    split = args.split if args.split is not None else cfg["data"].get("split", "val")
    ds = TemporalEpisodeDataset(cfg["data"]["path"], split=split)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_episode)

    model = build_model(cfg).to(device)

    ckpt_path = Path(args.ckpt) if args.ckpt else _default_ckpt_path(cfg)
    if ckpt_path is not None:
        ckpt = load_checkpoint(ckpt_path, model=model, device=device)
        print(f"[LOAD] ckpt={ckpt_path} epoch={ckpt.get('epoch', '?')} metric={ckpt.get('metric_name','?')}={ckpt.get('metric_value','?')}")
    else:
        print("[LOAD] no checkpoint provided/found; rolling out randomly initialized model")

    val = eval_rollout_mean_mse(model, dl, device=device, H=args.H, max_batches=args.max_batches)
    print(f"[ROLLOUT] split={split} H={args.H} mse_mean={val:.6f}")


if __name__ == "__main__":
    main()
