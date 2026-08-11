from __future__ import annotations

import argparse
import json
from pathlib import Path

from leo_pg.utils.config import load_cfg
from leo_pg.data.dataset import TemporalEpisodeDataset
from leo_pg.models import build_model
from leo_pg.train.checkpoint import load_ckpt
from leo_pg.train.frozen_diagnostics import (
    EVALUATION_MODE,
    METRIC_CONTRACT,
    frozen_diagnostics_for_episode,
    summarize_results,
)
from leo_pg.utils.device import get_device
from leo_pg.utils.run_name import resolve_run_name
from leo_pg.utils.seed import set_seed


# Compatibility for tests and downstream users of the original helper name.
_summarize_results = summarize_results


def _parse_horizons(value: str) -> list[int]:
    horizons = list(dict.fromkeys(int(item) for item in value.split(",") if item.strip()))
    if not horizons or any(horizon < 2 for horizon in horizons):
        raise argparse.ArgumentTypeError("--Hs must contain one or more integers >= 2")
    return horizons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run teacher-forced, frozen-trajectory prototype diagnostics"
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--message", default=None, help="Override mlp|kan|physick")
    parser.add_argument("--data", default=None, help="Override cfg.data.path")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--mode", default=None, help="Run-name mode used during training")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--Hs", type=_parse_horizons, default=_parse_horizons("30"))
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--which", default="last", choices=["best", "last"])
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max_eps", type=int, default=1, help="0 evaluates the full split")
    args = parser.parse_args()
    if args.max_eps < 0:
        parser.error("--max_eps must be non-negative")

    cfg = load_cfg(args.cfg)
    if args.message:
        cfg.setdefault("model", {})["message_type"] = args.message
    if args.data:
        cfg.setdefault("data", {})["path"] = args.data

    set_seed(int(cfg.get("seed", 7)))
    device = get_device(
        args.device or cfg.get("train", {}).get("device", "cuda"),
        strict=args.device is not None,
    )
    message = str(cfg.get("model", {}).get("message_type", "mlp"))
    run_name = resolve_run_name(cfg, message_type=message, mode=args.mode)
    save_dir = Path(cfg.get("train", {}).get("save_dir", "runs"))
    ckpt_path = Path(args.ckpt) if args.ckpt else save_dir / run_name / f"{args.which}.pt"

    dataset = TemporalEpisodeDataset(str(cfg["data"]["path"]), split=args.split)
    model = build_model(cfg).to(device)
    checkpoint_schema = None
    if ckpt_path.exists():
        payload = load_ckpt(
            str(ckpt_path),
            model=model,
            opt=None,
            map_location=device,
            allow_config_mismatch=args.allow_config_mismatch,
            allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        )
        checkpoint_schema = payload.get("checkpoint_schema_version", 1)
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
    results = []
    for episode_index in range(episode_count):
        episode = dataset[episode_index]
        for horizon in args.Hs:
            result = frozen_diagnostics_for_episode(cfg, model, episode, device, horizon)
            result["episode"] = episode_index
            results.append(result)
            print(
                f"[DIAG] episode={episode_index} H={horizon} effective_H={result['effective_H']} "
                f"mse_mean={result['mse_mean']:.6e} "
                f"adjacent_aba_pred={result['adjacent_aba_pred']:.4f} "
                f"assignment_failure_pred={result['assignment_failure_pred']:.4f}"
            )

    out_path = Path(args.out) if args.out else save_dir / run_name / "rollout_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "run_name": run_name,
        "message_type": message,
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
        "checkpoint_schema_version": checkpoint_schema,
        "data_path": str(cfg["data"]["path"]),
        "dataset_schema_version": dataset.schema_version,
        "split": args.split,
        "evaluation_mode": EVALUATION_MODE,
        "metric_contract": METRIC_CONTRACT,
        "results": results,
        "summary": summarize_results(results),
    }
    out_path.write_text(json.dumps(blob, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
