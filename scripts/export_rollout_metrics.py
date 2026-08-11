"""Export canonical frozen-trajectory diagnostics for several message functions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from leo_pg.data.dataset import TemporalEpisodeDataset
from leo_pg.models.registry import build_model
from leo_pg.train.checkpoint import load_ckpt
from leo_pg.train.frozen_diagnostics import (
    EVALUATION_MODE,
    METRIC_CONTRACT,
    frozen_diagnostics_for_episode,
    summarize_results,
)
from leo_pg.utils.config import load_cfg
from leo_pg.utils.device import get_device
from leo_pg.utils.run_name import resolve_run_name
from leo_pg.utils.seed import set_seed


def _positive_horizons(value: str) -> list[int]:
    horizons = list(dict.fromkeys(int(item) for item in value.split(",") if item.strip()))
    if not horizons or any(horizon < 2 for horizon in horizons):
        raise argparse.ArgumentTypeError("--Hs must contain one or more integers >= 2")
    return horizons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one canonical diagnostic JSON per message-function model"
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--Hs", required=True, type=_positive_horizons)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--which", default="last", choices=["best", "last"])
    parser.add_argument("--methods", default="mlp,kan,physick")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--max_eps", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.max_eps < 0:
        parser.error("--max_eps must be non-negative")

    cfg = load_cfg(args.cfg)
    set_seed(int(cfg.get("seed", 7)))
    device = get_device(
        args.device or cfg.get("train", {}).get("device", "cuda"),
        strict=args.device is not None,
    )
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    if not methods:
        parser.error("--methods must contain at least one method")

    dataset = TemporalEpisodeDataset(str(cfg["data"]["path"]), split=args.split)
    episode_count = len(dataset) if args.max_eps == 0 else min(args.max_eps, len(dataset))
    save_root = Path(args.out_dir) if args.out_dir else None

    for method in methods:
        method_cfg = copy.deepcopy(cfg)
        method_cfg.setdefault("model", {})["message_type"] = method
        run_name = resolve_run_name(method_cfg, message_type=method)
        run_dir = Path(method_cfg.get("train", {}).get("save_dir", "runs")) / run_name
        ckpt_path = run_dir / f"{args.which}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = build_model(method_cfg).to(device)
        payload = load_ckpt(
            str(ckpt_path),
            model,
            opt=None,
            map_location=device,
            strict=True,
            allow_config_mismatch=args.allow_config_mismatch,
            allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        )
        model.eval()
        results = []
        for episode_index in range(episode_count):
            episode = dataset[episode_index]
            for horizon in args.Hs:
                result = frozen_diagnostics_for_episode(
                    method_cfg, model, episode, device, horizon
                )
                result["episode"] = episode_index
                results.append(result)

        blob = {
            "run_name": run_name,
            "message_type": method,
            "checkpoint": str(ckpt_path),
            "checkpoint_schema_version": payload.get("checkpoint_schema_version", 1),
            "data_path": str(method_cfg["data"]["path"]),
            "dataset_schema_version": dataset.schema_version,
            "split": args.split,
            "evaluation_mode": EVALUATION_MODE,
            "metric_contract": METRIC_CONTRACT,
            "results": results,
            "summary": summarize_results(results),
        }
        out_path = (
            save_root / f"{method}_rollout_metrics.json"
            if save_root is not None
            else run_dir / "rollout_metrics.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(blob, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[WRITE] {out_path} ({episode_count} episodes, {len(args.Hs)} horizons)")


if __name__ == "__main__":
    main()
