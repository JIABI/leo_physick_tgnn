from __future__ import annotations

import argparse
from pathlib import Path

from leo_pg.data.dataset import TemporalEpisodeDataset
from leo_pg.models import build_model
from leo_pg.train.checkpoint import load_ckpt
from leo_pg.train.frozen_diagnostics import frozen_diagnostics_for_episode
from leo_pg.utils.config import load_cfg
from leo_pg.utils.device import get_device
from leo_pg.utils.run_name import resolve_run_name
from leo_pg.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction error and frozen-trajectory prototype diagnostics"
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--message", default=None)
    parser.add_argument("--data", default=None, help="Override cfg.data.path")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--mode", default=None, help="Run-name mode used during training")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--which", default="last", choices=["best", "last"])
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--max_eps", type=int, default=0, help="0 evaluates the full split")
    args = parser.parse_args()
    if args.max_eps < 0:
        parser.error("--max_eps must be non-negative")

    cfg = load_cfg(args.cfg)
    if args.message:
        cfg.setdefault("model", {})["message_type"] = args.message
    if args.data:
        cfg.setdefault("data", {})["path"] = args.data
    split = args.split

    horizon = args.H if args.H is not None else int(cfg.get("eval", {}).get("rollout_horizon", 30))
    if horizon < 2:
        parser.error("--H must be at least 2 for temporally aligned diagnostics")

    set_seed(int(cfg.get("seed", 7)))
    device = get_device(
        args.device or cfg.get("train", {}).get("device", "cuda"),
        strict=args.device is not None,
    )
    message = str(cfg.get("model", {}).get("message_type", "mlp"))
    run_name = resolve_run_name(cfg, message_type=message, mode=args.mode)
    save_dir = Path(cfg.get("train", {}).get("save_dir", "runs"))
    ckpt_path = Path(args.ckpt) if args.ckpt else save_dir / run_name / f"{args.which}.pt"

    dataset = TemporalEpisodeDataset(str(cfg["data"]["path"]), split=split)
    model = build_model(cfg).to(device)
    if ckpt_path.exists():
        payload = load_ckpt(
            str(ckpt_path),
            model=model,
            opt=None,
            map_location=device,
            allow_config_mismatch=args.allow_config_mismatch,
            allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        )
        print(f"[LOAD] ckpt={ckpt_path} epoch={payload.get('epoch', '?')}")
    elif not args.allow_random_init:
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first, pass --ckpt, "
            "or explicitly use --allow-random-init for a diagnostic run."
        )
    else:
        print(f"[WARN] checkpoint not found: {ckpt_path}; using random initialization by request")
    model.eval()

    episode_count = len(dataset) if args.max_eps == 0 else min(args.max_eps, len(dataset))
    results = [
        frozen_diagnostics_for_episode(cfg, model, dataset[index], device, horizon)
        for index in range(episode_count)
    ]

    def mean(key: str) -> float:
        return sum(float(result[key]) for result in results) / len(results)

    print(
        f"[EVAL] split={split} episodes={episode_count} "
        f"mean_mse_last={mean('mse_last'):.6e} mean_mse_horizon={mean('mse_mean'):.6e}"
    )
    print(
        f"[FROZEN-DIAG@H={horizon}] adjacent_aba_gt={mean('adjacent_aba_gt'):.4f} "
        f"adjacent_aba_pred={mean('adjacent_aba_pred'):.4f} "
        f"assignment_failure_gt={mean('assignment_failure_gt'):.4f} "
        f"assignment_failure_pred={mean('assignment_failure_pred'):.4f}"
    )


if __name__ == "__main__":
    main()
