from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from _cfg import load_cfg
from leo_pg.utils.seed import set_seed
from leo_pg.utils.device import get_device
from leo_pg.data.dataset import TemporalEpisodeDataset
from leo_pg.data.collate import collate_episode
from leo_pg.models import build_model
from leo_pg.train.checkpoint import load_ckpt
from leo_pg.train.system_metrics import (
    pingpong_rate,
    ho_failure_rate,
    load_stats,
    greedy_assign_from_load,
)


def _apply_placeholders(s: str, kv: Dict[str, str]) -> str:
    for k, v in kv.items():
        s = s.replace(f"${k}$", v).replace(f"{{{k}}}", v)
    return s


def _auto_ckpt_path(save_dir: str, run_name: str, which: str) -> Path:
    return Path(save_dir) / run_name / (f"{which}.pt")


def _rollout_metrics_for_episode(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    episode: Dict[str, Any],
    device: torch.device,
    H: int,
) -> Dict[str, float]:
    """Teacher-forcing rollout metrics for one episode."""
    steps: List[Dict[str, Any]] = episode["steps"]
    H = min(H, len(steps))
    if H <= 0:
        return {}

    # Forward full episode once; slice outputs to H
    out = model.forward_episode(episode, device=device)
    preds = out["preds"][:H]
    ys = out["ys"][:H]

    # Determine K and S from meta (preferred), else from cfg
    meta0 = steps[0].get("meta", {})
    K = int(meta0.get("K_users", cfg.get("K_users", cfg.get("head", {}).get("user_count", 1))))
    S = int(meta0.get("S_sats", cfg.get("S_sats", cfg.get("head", {}).get("sat_count", 0))))
    cap = int(meta0.get("sat_capacity_users", cfg.get("load", {}).get("sat_capacity_users", 10)))

    # ---- MSE metrics on last_step and mean over horizon ----
    mse_t = []
    for t in range(H):
        pred = preds[t]
        y = ys[t]
        mse_t.append(torch.mean((pred - y) ** 2).item())
    mse_mean = float(sum(mse_t) / max(1, len(mse_t)))
    mse_last = float(mse_t[-1])

    # ---- System metrics (ground truth) ----
    serving_gt = []
    ho_fail_gt = []
    load_gt = []

    for t in range(H):
        st = steps[t]
        m = st.get("meta", {})
        if "serving_sat" in m:
            serving_gt.append(torch.as_tensor(m["serving_sat"]).long())
        if "ho_fail" in m:
            ho_fail_gt.append(torch.as_tensor(m["ho_fail"]).bool())
        # y stores sat_load on sat nodes
        y = torch.as_tensor(st["y"]).float()
        load_gt.append(y[K:, 0])

    if len(serving_gt) > 0:
        serving_gt_tk = torch.stack(serving_gt, dim=0)  # [H,K]
        ping_gt = pingpong_rate(serving_gt_tk)
    else:
        ping_gt = 0.0
    if len(ho_fail_gt) > 0:
        ho_fail_gt_tk = torch.stack(ho_fail_gt, dim=0)
        hof_gt = ho_failure_rate(ho_fail_gt_tk)
    else:
        hof_gt = 0.0
    if len(load_gt) > 0:
        load_gt_ts = torch.stack(load_gt, dim=0)  # [H,S]
        var_gt, peak_gt = load_stats(load_gt_ts)
    else:
        var_gt, peak_gt = 0.0, 0.0

    # ---- System metrics (predicted): reconstruct serving decisions from predicted sat_load ----
    channel_cfg = cfg.get("channel", {})
    sinr_min = float(channel_cfg.get("sinr_min", -1.0))

    serving_pred = []
    ho_fail_pred = []
    load_pred = []
    prev_serving = None

    for t in range(H):
        st = steps[t]
        node_x = torch.as_tensor(st["node_x"]).to(device)
        edge_index = torch.as_tensor(st["edge_index"]).long().to(device)
        edge_type = torch.as_tensor(st["edge_type"]).long().to(device)

        pred = preds[t]
        sat_load_pred = pred[K:, 0].detach()

        serv_t, _, hof_t = greedy_assign_from_load(
            node_x=node_x,
            edge_index=edge_index,
            edge_type=edge_type,
            sat_load=sat_load_pred,
            K_users=K,
            sat_capacity_users=cap,
            channel_cfg=channel_cfg,
            sinr_min=sinr_min,
        )
        # handover can be derived if needed; pingpong uses serving seq only
        serving_pred.append(serv_t.detach().cpu())
        ho_fail_pred.append(hof_t.detach().cpu())
        load_pred.append(sat_load_pred.detach().cpu())

        prev_serving = serv_t

    serving_pred_tk = torch.stack(serving_pred, dim=0)
    ho_fail_pred_tk = torch.stack(ho_fail_pred, dim=0)
    load_pred_ts = torch.stack(load_pred, dim=0)

    ping_pred = pingpong_rate(serving_pred_tk)
    hof_pred = ho_failure_rate(ho_fail_pred_tk)
    var_pred, peak_pred = load_stats(load_pred_ts)

    return {
        "H": float(H),
        "mse_mean": mse_mean,
        "mse_last": mse_last,
        "pingpong_gt": ping_gt,
        "ho_fail_gt": hof_gt,
        "load_var_gt": var_gt,
        "load_peak_gt": peak_gt,
        "pingpong_pred": ping_pred,
        "ho_fail_pred": hof_pred,
        "load_var_pred": var_pred,
        "load_peak_pred": peak_pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--message", type=str, default=None, help="Override cfg.model.message_type (mlp|kan|physick)")
    ap.add_argument("--split", type=str, default="test", help="train|val|test")
    ap.add_argument("--Hs", type=str, default="30", help="Comma-separated horizons, e.g. 20,50,100,1000")
    ap.add_argument("--ckpt", type=str, default=None, help="Checkpoint path. If omitted, auto-load from runs/<run_name>/{best,last}.pt")
    ap.add_argument("--which", type=str, default="best", choices=["best", "last"])
    ap.add_argument("--out", type=str, default=None, help="Write metrics JSON to this path (default: runs/<run_name>/rollout_metrics.json)")
    ap.add_argument("--max_eps", type=int, default=1, help="How many episodes to evaluate")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.message:
        cfg.setdefault("model", {})["message_type"] = args.message

    set_seed(int(cfg.get("seed", 7)))
    device = get_device(cfg.get("train", {}).get("device", "cuda"))

    # Resolve run_name placeholders
    msg = str(cfg.get("model", {}).get("message_type", "mlp"))
    run_name = str(cfg.get("train", {}).get("run_name", f"run_{msg}"))
    run_name = _apply_placeholders(run_name, {"message_type": msg, "message": msg, "mode": str(cfg.get("mode", ""))})

    save_dir = str(cfg.get("train", {}).get("save_dir", "runs"))
    ckpt_path = Path(args.ckpt) if args.ckpt else _auto_ckpt_path(save_dir, run_name, args.which)

    # Dataset
    data_path = str(cfg.get("data", {}).get("path"))
    ds = TemporalEpisodeDataset(data_path, split=args.split)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_episode)

    # Model
    model = build_model(cfg).to(device)
    if ckpt_path.exists():
        payload = load_ckpt(str(ckpt_path), model=model, opt=None, map_location=device)
        ep = payload.get("epoch", "?")
        mn = payload.get("metric_name", "?")
        mv = payload.get("metric_value", "?")
        print(f"[LOAD] ckpt={ckpt_path} epoch={ep} {mn}={mv}")
    else:
        print(f"[WARN] checkpoint not found: {ckpt_path} (evaluating random init model)")

    model.eval()

    Hs = [int(x) for x in args.Hs.split(",") if x.strip()]
    results = []
    with torch.no_grad():
        for ei, episode in enumerate(dl):
            if args.max_eps and ei >= args.max_eps:
                break
            for H in Hs:
                r = _rollout_metrics_for_episode(cfg, model, episode, device=device, H=H)
                results.append(r)
                print(f"[ROLLOUT] H={H} mse_mean={r['mse_mean']:.6e} mse_last={r['mse_last']:.6e} "
                      f"ping_pred={r['pingpong_pred']:.4f} hof_pred={r['ho_fail_pred']:.4f} "
                      f"load_var_pred={r['load_var_pred']:.4e} load_peak_pred={r['load_peak_pred']:.4e}")
            break  # by default only first episode

    out_path = Path(args.out) if args.out else (Path(save_dir) / run_name / "rollout_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"run_name": run_name, "message_type": msg, "data_path": data_path, "results": results}, f, indent=2)
    print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()
