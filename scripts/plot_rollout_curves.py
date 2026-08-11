"""Plot teacher-forced prediction error across horizons and held-out episodes."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt

from leo_pg.data.dataset import TemporalEpisodeDataset
from leo_pg.models.registry import build_model
from leo_pg.train.checkpoint import load_ckpt
from leo_pg.train.rollout import rollout_episode
from leo_pg.utils.config import load_cfg
from leo_pg.utils.device import get_device
from leo_pg.utils.run_name import resolve_run_name


def _positive_horizons(value: str) -> list[int]:
    horizons = list(dict.fromkeys(int(item) for item in value.split(",") if item.strip()))
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise argparse.ArgumentTypeError("--Hs must contain one or more positive integers")
    return horizons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--Hs", required=True, type=_positive_horizons)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--which", default="last", choices=["best", "last"])
    parser.add_argument("--metric", default="mse_mean", choices=["mse_mean", "mse_last"])
    parser.add_argument("--methods", default="mlp,kan,physick")
    parser.add_argument("--max_eps", type=int, default=0, help="0 uses the full split")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--out", default="teacher_forced_horizon_curves.png")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.max_eps < 0:
        parser.error("--max_eps must be non-negative")

    cfg = load_cfg(args.cfg)
    device = get_device(
        args.device or cfg.get("train", {}).get("device", "cuda"),
        strict=args.device is not None,
    )
    dataset = TemporalEpisodeDataset(str(cfg["data"]["path"]), split=args.split)
    episode_count = len(dataset) if args.max_eps == 0 else min(args.max_eps, len(dataset))
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    curves = {method: [] for method in methods}

    for method in methods:
        method_cfg = copy.deepcopy(cfg)
        method_cfg.setdefault("model", {})["message_type"] = method
        run_name = resolve_run_name(method_cfg, message_type=method)
        run_dir = Path(method_cfg.get("train", {}).get("save_dir", "runs")) / run_name
        ckpt_path = run_dir / f"{args.which}.pt"
        model = build_model(method_cfg).to(device)
        load_ckpt(
            str(ckpt_path),
            model,
            opt=None,
            map_location=device,
            strict=True,
            allow_config_mismatch=args.allow_config_mismatch,
            allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        )
        model.eval()

        for horizon in args.Hs:
            values = [
                float(
                    rollout_episode(
                        model,
                        dataset[index],
                        device=device,
                        H=horizon,
                    )[args.metric]
                )
                for index in range(episode_count)
            ]
            curves[method].append(sum(values) / len(values))

    for method in methods:
        plt.plot(args.Hs, curves[method], marker="o", label=method)
    plt.xlabel("Teacher-forced horizon (steps)")
    plt.ylabel(f"Mean {args.metric} across {args.split} episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[WRITE] {args.out}")


if __name__ == "__main__":
    main()
