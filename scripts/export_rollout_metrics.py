from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from leo_pg.models.registry import build_model
from leo_pg.train.checkpoint import load_ckpt
from leo_pg.train.rollout import rollout_episode


def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def _resolve_run_name(cfg: Dict[str, Any], message_type: str) -> str:
    tr = cfg.setdefault("train", {})
    rn = str(tr.get("run_name", "run_{message_type}"))
    rn = rn.replace("$message_type$", message_type).replace("{message_type}", message_type)

    if ("$message_type$" not in str(tr.get("run_name", "")) and "{message_type}" not in str(tr.get("run_name", ""))):
        if not rn.endswith(f"_{message_type}"):
            rn = f"{rn}_{message_type}"
    return rn


@torch.no_grad()
def _infer_serving_sequence(
    episode: Dict[str, Any],
    preds: List[torch.Tensor],
    K: int,
    S: int,
    risk_th: float = 0.7,
    w_rate: float = 1.0,
    w_cost: float = 0.2,
    w_load: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Infer per-user serving satellite index sequence from:
      - edge candidates in episode steps
      - edge_z features: [P_risk, log1p(ET0), log1p(Lambda), rate, H_norm, load]
      - predicted sat load from preds[t] (sat nodes)
    Returns:
      serving: LongTensor [T, K] in [0..S-1]
      ho_fail: BoolTensor [T, K]
    """
    steps = episode["steps"]
    T = min(len(steps), len(preds))

    serving = torch.full((T, K), -1, dtype=torch.long)
    ho_fail = torch.zeros((T, K), dtype=torch.bool)
    prev = torch.full((K,), -1, dtype=torch.long)

    for t in range(T):
        step = steps[t]
        edge_index = step["edge_index"]  # [2, E]
        edge_z = step["edge_z"]          # [E, 6]

        if edge_index.numel() == 0:
            if t > 0:
                serving[t] = serving[t - 1]
            ho_fail[t].fill_(True)
            continue

        src_u = edge_index[0].long()           # [E]
        dst_global = edge_index[1].long()      # [E] in [K..K+S-1]
        dst_s = (dst_global - K).clamp(min=0, max=S - 1)

        P_risk = edge_z[:, 0].float()
        rate = edge_z[:, 3].float()
        H_norm = edge_z[:, 4].float()

        pred_t = preds[t].float()              # CPU tensor
        sat_load_pred = pred_t[K:, 0].float()  # [S]
        load_e = sat_load_pred[dst_s]          # [E]

        util = w_rate * rate - w_cost * H_norm - w_load * load_e

        for u in range(K):
            mask = (src_u == u)
            if not torch.any(mask):
                if t > 0:
                    serving[t, u] = serving[t - 1, u]
                else:
                    serving[t, u] = 0
                ho_fail[t, u] = True
                prev[u] = serving[t, u]
                continue

            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            safe = idx[P_risk[idx] <= risk_th]

            if safe.numel() == 0:
                ho_fail[t, u] = True
                best_i = idx[torch.argmax(util[idx])]
            else:
                best_i = safe[torch.argmax(util[safe])]

            chosen = int(dst_s[best_i].item())

            # hysteresis
            if prev[u] >= 0:
                prev_sat = int(prev[u].item())
                cand_prev = idx[dst_s[idx] == prev_sat]
                if cand_prev.numel() > 0:
                    j = cand_prev[0]
                    if (P_risk[j] <= risk_th) and (util[j] >= 0.95 * util[best_i]):
                        chosen = prev_sat

            serving[t, u] = chosen
            prev[u] = chosen

    return serving, ho_fail


def _pingpong_rate(serving: torch.Tensor) -> float:
    T, K = serving.shape
    if T < 3:
        return float("nan")
    a = serving[:-2]
    b = serving[1:-1]
    c = serving[2:]
    ping = (a == c) & (a != b)
    return float(ping.float().mean().item())


def _load_stats_from_serving(serving: torch.Tensor, S: int) -> Tuple[float, float]:
    T, K = serving.shape
    vars_: List[float] = []
    peaks_: List[float] = []
    for t in range(T):
        counts = torch.bincount(serving[t].clamp(min=0), minlength=S).float()
        load = counts / max(1.0, float(K))
        vars_.append(float(load.var(unbiased=False).item()))
        peaks_.append(float(load.max().item()))
    return float(sum(vars_) / max(1, len(vars_))), float(sum(peaks_) / max(1, len(peaks_)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="configs/default.yaml")
    ap.add_argument("--Hs", required=True, help="comma-separated, e.g. 20,50,100,600")
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--methods", default="mlp,kan,physick")
    ap.add_argument("--out_dir", default=None, help="if set, write jsons here; otherwise write into each run dir")
    ap.add_argument("--risk_th", type=float, default=0.7)

    # NEW: export on CPU to avoid GPU OOM (recommended)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"], help="override device (recommended: cpu)")

    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or str(cfg.get("train", {}).get("device", "cuda"))
    device = torch.device(device_str)

    Hs = [int(x) for x in args.Hs.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    split = args.split or cfg.get("data", {}).get("split", "train")

    # load episode
    pt_path = cfg["data"]["path"]
    obj = torch.load(pt_path, map_location="cpu")
    episode = obj["episodes"][0]

    meta0 = episode["steps"][0].get("meta", {})
    K = int(meta0.get("K_users", cfg.get("head", {}).get("user_count", 1)))
    S = int(meta0.get("S_sats", cfg.get("head", {}).get("sat_count", 200)))

    y0 = episode["steps"][0].get("y", None)
    if y0 is not None and y0.ndim == 2:
        N = int(y0.shape[0])
        if N >= K:
            S = int(N - K)

    save_root = Path(args.out_dir) if args.out_dir else None

    for m in methods:
        cfg_m = _deep_update(dict(cfg), {"model": {"message_type": m}, "data": {"split": split}})
        run_name = _resolve_run_name(cfg_m, m)
        run_dir = Path(cfg_m.get("train", {}).get("save_dir", "runs")) / run_name
        ckpt_path = run_dir / f"{args.which}.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = build_model(cfg_m).to(device)
        load_ckpt(str(ckpt_path), model, opt=None, map_location=device, strict=True)
        model.eval()

        results: List[Dict[str, Any]] = []

        for H in Hs:
            # IMPORTANT: single pass rollout that optionally returns preds on CPU.
            out = rollout_episode(model, episode, device=device, H=H, return_preds_cpu=True)

            mse_mean = float(out["mse_mean"])
            mse_last = float(out["mse_last"])

            preds_cpu = out.get("preds_cpu", [])
            if len(preds_cpu) == 0:
                # If preds not returned for any reason, still write MSEs and NaN system metrics.
                pingpong = float("nan")
                ho_fail_rate = float("nan")
                load_var = float("nan")
                load_peak = float("nan")
            else:
                serving, ho_fail = _infer_serving_sequence(
                    episode=episode,
                    preds=preds_cpu,
                    K=K,
                    S=S,
                    risk_th=float(args.risk_th),
                )
                pingpong = _pingpong_rate(serving)
                ho_fail_rate = float(ho_fail.float().mean().item())
                load_var, load_peak = _load_stats_from_serving(serving, S=S)

            results.append(
                {
                    "H": int(H),
                    "mse_mean": mse_mean,
                    "mse_last": mse_last,
                    "pingpong_pred": float(pingpong),
                    "ho_fail_pred": float(ho_fail_rate),
                    "load_var_pred": float(load_var),
                    "load_peak_pred": float(load_peak),
                }
            )

            print(f"[OK] {m} H={H} mse_mean={mse_mean:.3e} mse_last={mse_last:.3e}")

        blob = {
            "meta": {
                "method": m,
                "split": split,
                "which": args.which,
                "Hs": Hs,
                "data_path": pt_path,
                "K_users": K,
                "S_sats": S,
                "risk_th": float(args.risk_th),
                "device": str(device),
            },
            "results": results,
        }

        out_path = (save_root / f"{m}_rollout_metrics.json") if save_root else (run_dir / "rollout_metrics.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            json.dump(blob, f, indent=2)

        print(f"[WRITE] {out_path}")


if __name__ == "__main__":
    main()


