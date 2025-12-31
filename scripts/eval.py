from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from _cfg import load_cfg
from leo_pg.utils.seed import set_seed
from leo_pg.utils.device import get_device
from leo_pg.data.dataset import TemporalEpisodeDataset
from leo_pg.data.collate import collate_episode
from leo_pg.models import build_model
from leo_pg.train.checkpoint import load_ckpt
from leo_pg.train.system_metrics import pingpong_rate, ho_failure_rate, load_stats, greedy_assign_from_load


def _apply_placeholders(s: str, kv: Dict[str, str]) -> str:
    for k, v in kv.items():
        s = s.replace(f"${k}$", v).replace(f"{{{k}}}", v)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--message", type=str, default=None)
    ap.add_argument("--split", type=str, default=None, help="Override cfg.data.split")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--which", type=str, default="best", choices=["best", "last"])
    ap.add_argument("--H", type=int, default=None, help="Also compute system metrics over first H steps (default: cfg.eval.rollout_horizon)")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.message:
        cfg.setdefault("model", {})["message_type"] = args.message
    if args.split:
        cfg.setdefault("data", {})["split"] = args.split

    set_seed(int(cfg.get("seed", 7)))
    device = get_device(cfg.get("train", {}).get("device", "cuda"))

    msg = str(cfg.get("model", {}).get("message_type", "mlp"))
    run_name = str(cfg.get("train", {}).get("run_name", f"run_{msg}"))
    run_name = _apply_placeholders(run_name, {"message_type": msg, "message": msg, "mode": str(cfg.get("mode", ""))})

    save_dir = str(cfg.get("train", {}).get("save_dir", "runs"))
    ckpt_path = Path(args.ckpt) if args.ckpt else (Path(save_dir) / run_name / f"{args.which}.pt")

    ds = TemporalEpisodeDataset(cfg["data"]["path"], split=cfg["data"].get("split", "test"))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_episode)

    model = build_model(cfg).to(device)
    if ckpt_path.exists():
        payload = load_ckpt(str(ckpt_path), model=model, opt=None, map_location=device)
        print(f"[LOAD] ckpt={ckpt_path} epoch={payload.get('epoch','?')}")
    else:
        print(f"[WARN] checkpoint not found: {ckpt_path}")

    model.eval()

    mse_last_list = []
    with torch.no_grad():
        for episode in dl:
            out = model.forward_episode(episode, device=device)
            pred_last = out["preds"][-1]
            y_last = out["ys"][-1]
            mse_last_list.append(torch.mean((pred_last - y_last) ** 2).item())

    mean_mse_last = float(sum(mse_last_list) / max(1, len(mse_last_list)))
    print(f"[EVAL] split={cfg['data'].get('split','test')} mean_mse(last_step)={mean_mse_last:.6e}")

    # Optional system metrics over horizon H
    H = args.H if args.H is not None else int(cfg.get("eval", {}).get("rollout_horizon", 30))
    if H > 0:
        with torch.no_grad():
            episode = next(iter(dl))
            steps = episode["steps"]
            meta0 = steps[0].get("meta", {})
            K = int(meta0.get("K_users", cfg.get("K_users", 1)))
            cap = int(meta0.get("sat_capacity_users", cfg.get("load", {}).get("sat_capacity_users", 10)))

            out = model.forward_episode(episode, device=device)
            preds = out["preds"][:H]
            # ground-truth sequences
            serving_gt = []
            ho_fail_gt = []
            load_gt = []
            for t in range(min(H, len(steps))):
                m = steps[t].get("meta", {})
                if "serving_sat" in m:
                    serving_gt.append(torch.as_tensor(m["serving_sat"]).long())
                if "ho_fail" in m:
                    ho_fail_gt.append(torch.as_tensor(m["ho_fail"]).bool())
                y = torch.as_tensor(steps[t]["y"]).float()
                load_gt.append(y[K:, 0])

            ping_gt = pingpong_rate(torch.stack(serving_gt, 0)) if serving_gt else 0.0
            hof_gt = ho_failure_rate(torch.stack(ho_fail_gt, 0)) if ho_fail_gt else 0.0
            var_gt, peak_gt = load_stats(torch.stack(load_gt, 0)) if load_gt else (0.0, 0.0)

            # predicted sequences via greedy reconstruction
            channel_cfg = cfg.get("channel", {})
            sinr_min = float(channel_cfg.get("sinr_min", -1.0))
            serving_pred = []
            ho_fail_pred = []
            load_pred = []
            for t in range(min(H, len(steps))):
                node_x = torch.as_tensor(steps[t]["node_x"]).to(device)
                ei = torch.as_tensor(steps[t]["edge_index"]).long().to(device)
                et = torch.as_tensor(steps[t]["edge_type"]).long().to(device)
                sat_load_pred = preds[t][K:, 0]
                serv_t, _, hof_t = greedy_assign_from_load(
                    node_x=node_x, edge_index=ei, edge_type=et, sat_load=sat_load_pred,
                    K_users=K, sat_capacity_users=cap, channel_cfg=channel_cfg, sinr_min=sinr_min
                )
                serving_pred.append(serv_t.detach().cpu())
                ho_fail_pred.append(hof_t.detach().cpu())
                load_pred.append(sat_load_pred.detach().cpu())

            ping_pred = pingpong_rate(torch.stack(serving_pred, 0)) if serving_pred else 0.0
            hof_pred = ho_failure_rate(torch.stack(ho_fail_pred, 0)) if ho_fail_pred else 0.0
            var_pred, peak_pred = load_stats(torch.stack(load_pred, 0)) if load_pred else (0.0, 0.0)

            print(f"[SYS@H={H}] gt: ping={ping_gt:.4f} hof={hof_gt:.4f} var={var_gt:.4e} peak={peak_gt:.4e}")
            print(f"[SYS@H={H}] pred: ping={ping_pred:.4f} hof={hof_pred:.4f} var={var_pred:.4e} peak={peak_pred:.4e}")

if __name__ == "__main__":
    main()
