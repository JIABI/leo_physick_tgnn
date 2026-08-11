"""Train a typed paper model and save automatic best/last checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from leo_pg.paper.dataset import PaperEpisodeDataset
from leo_pg.paper.models import (
    PAPER_METHODS,
    build_paper_model,
    normalize_paper_method,
    resolved_model_config,
)
from leo_pg.paper.training import PaperTrainer
from leo_pg.utils.config import load_cfg
from leo_pg.utils.device import get_device
from leo_pg.utils.seed import set_seed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _paper_training_mapping(cfg: Dict[str, Any]) -> Dict[str, Any]:
    name = "paper_train" if "paper_train" in cfg else "paper_training"
    value = cfg.get(name)
    if not isinstance(value, dict):
        raise ValueError("configuration requires paper_training (or paper_train)")
    return value


def _dataset_path(cfg: Dict[str, Any], override: str | None) -> Path:
    if override is not None:
        path = Path(override).expanduser()
    else:
        training = _paper_training_mapping(cfg)
        configured = training.get("dataset_path")
        if configured is None:
            paper_dataset = cfg.get("paper_dataset", {})
            if isinstance(paper_dataset, dict):
                configured = paper_dataset.get("output")
        if configured is None or not str(configured).strip():
            raise ValueError(
                "provide --data, paper_training.dataset_path, or paper_dataset.output"
            )
        path = Path(str(configured)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"paper dataset does not exist: {path}")
    return path


def _checkpoint_paths(output: str) -> tuple[Path, Path, Path]:
    """Return run directory, selected-best path, and rolling-last path."""

    requested = Path(output).expanduser()
    if requested.suffix.lower() == ".pt":
        run_dir = requested.parent
        if requested.name == "last.pt":
            return run_dir, run_dir / "best.pt", requested
        return run_dir, requested, run_dir / "last.pt"
    return requested, requested / "best.pt", requested / "last.pt"


def _dataset_metadata(dataset: PaperEpisodeDataset, path: Path) -> Dict[str, Any]:
    payload = dataset.payload
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "paper_dataset_schema_version": payload["paper_dataset_schema_version"],
        "dataset_kind": payload["dataset_kind"],
        "paper_protocol_version": payload["paper_protocol_version"],
        "feature_contract_version": payload["feature_contract_version"],
        "target_contract_version": payload["target_contract_version"],
        "storage_layout": dataset.storage_layout,
        "split_manifest": payload["split_manifest"],
    }


def _progress(record: Dict[str, Any]) -> None:
    print("[EPOCH] " + json.dumps(record, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the paper Intensity--Flow objective with typed records and "
            "optional action-coupled scheduled sampling"
        )
    )
    parser.add_argument("--cfg", required=True, help="Formal paper YAML configuration")
    parser.add_argument(
        "--data",
        default=None,
        help="Paper dataset index/bundle .pt path (overrides the configuration)",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Paper method: " + ", ".join(PAPER_METHODS),
    )
    parser.add_argument(
        "--message",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Selected-best .pt path, or a run directory. A separate last.pt is "
            "updated after every epoch."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume optimizer, AMP, RNG, and trainer state from a paper checkpoint",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args(argv)

    if args.method is not None and args.message is not None:
        parser.error("use --method; do not pass --method and --message together")
    raw_method = args.method
    if raw_method is None and args.message is not None:
        raw_method = args.message
    if raw_method is None:
        parser.error("--method is required")
    try:
        method = normalize_paper_method(raw_method)
    except ValueError as exc:
        parser.error(str(exc))
    if method.startswith("snapshot_"):
        parser.error(
            "Snapshot methods use the isolated Snapshot data contract; call "
            "scripts/paper_snapshot_train.py or scripts/paper_protocol.sh"
        )

    root_cfg: Dict[str, Any] = load_cfg(args.cfg)
    resolved_cfg = resolved_model_config(root_cfg, method)
    training_cfg = _paper_training_mapping(resolved_cfg)
    run_dir, best_path, last_path = _checkpoint_paths(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    if "paper_train" in resolved_cfg:
        training_cfg["save_dir"] = str(run_dir)
    else:
        training_cfg["output_dir"] = str(run_dir)

    data_path = _dataset_path(resolved_cfg, args.data)
    train_dataset = PaperEpisodeDataset(data_path, split="train")
    validation_dataset = PaperEpisodeDataset(
        data_path,
        split="val",
        payload=train_dataset.payload,
    )
    if train_dataset.payload is not validation_dataset.payload:
        raise RuntimeError("train and validation datasets do not share one index payload")

    seed = int(resolved_cfg.get("seed", 0))
    if seed < 0:
        raise ValueError("seed must be non-negative")
    set_seed(seed)
    requested_device = args.device or str(training_cfg.get("device", "cuda"))
    device = get_device(requested_device, strict=args.device is not None)

    model = build_paper_model(resolved_cfg, method)
    # The factory binds method-specific model/baseline semantics to model.cfg.
    # Save exactly that resolved mapping so checkpoint signature verification
    # cannot compare a baseline against the unresolved registry root.
    model_cfg = getattr(model, "cfg", None)
    if not isinstance(model_cfg, dict):
        raise RuntimeError("paper model factory did not attach a resolved config")
    resolved_cfg = model_cfg
    metadata = _dataset_metadata(train_dataset, data_path)
    metadata["train_split"] = "train"
    metadata["validation_split"] = "val"
    metadata["paper_method"] = method

    trainer = PaperTrainer(
        model=model,
        config=resolved_cfg,
        device=device,
        dataset_metadata=metadata,
        progress=_progress,
        best_checkpoint_path=best_path,
        last_checkpoint_path=last_path,
    )
    if args.resume is not None:
        trainer.resume(args.resume)
    result = trainer.fit(train_dataset, validation_dataset)
    print(
        f"[OK] method={method} best_epoch={result.best_epoch} "
        f"validation={result.best_validation:.8g} best={result.best_checkpoint} "
        f"last={result.last_checkpoint}"
    )


if __name__ == "__main__":
    main()
