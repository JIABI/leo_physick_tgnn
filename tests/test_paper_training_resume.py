from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from leo_pg.paper.training import (
    PaperTrainer,
    _legacy_full_training_plan_fingerprint,
)


class _TinyPaperModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.cfg = copy.deepcopy(config)
        self.paper_method = str(config["paper_method"])

    def predict_step(self, step, state, device=None):  # pragma: no cover - contract only
        raise AssertionError("resume tests must not execute model inference")


def _config(output: Path, *, epochs: int = 5) -> dict:
    return {
        "paper_method": "tgn_mlp",
        "model": {
            "node_in_dim": 7,
            "edge_in_dim": 7,
            "mem_dim": 8,
            "msg_dim": 8,
            "emb_dim": 8,
            "message_type": "mlp",
            "aggregator": "sum",
            "dropout": 0.0,
            "use_edge_type": True,
            "edge_type_vocab": 8,
            "message_passing_layers": 2,
        },
        "head": {"type": "intensity_flow", "hidden_dim": 8, "dropout": 0.0},
        "paper_train": {
            "epochs": epochs,
            "save_dir": str(output),
            "optimizer": {"name": "adamw", "lr": 5e-4, "weight_decay": 1e-4},
            "clip_grad_norm": 1.0,
            "batch_size_control_graphs": 32,
            "amp": {"enabled": False, "dtype": "float16"},
            "sampling_seed": 17,
            "learning_rate_multiplier": {"kind": "constant", "value": 1.0},
            "loss_weights": {
                "gamma": 0.01,
                "intensity": 1.0,
                "flow": 1.0,
                "feasibility": 0.1,
            },
            "decision_aware_weight": {"kind": "constant", "value": 0.0},
            "objectives": {
                "one_step": {
                    "enabled": True,
                    "weight": {"kind": "constant", "value": 1.0},
                    "scheduled_sampling": {"kind": "constant", "value": 0.0},
                    "tbptt_steps": 1,
                    "new_edge_source": "teacher",
                },
                "action_coupled": {"enabled": False},
            },
            "validation": {
                "every_epochs": 1,
                "selection_metric": "one_step",
                "one_step_model_probability": 0.0,
                "action_coupled_model_probability": 0.0,
                "one_step_weight": 1.0,
                "action_coupled_weight": 0.0,
                "sampling_seed": 29,
            },
        },
    }


def _dataset(path: Path, *, sha256: str = "a" * 64) -> dict:
    return {
        "path": str(path),
        "sha256": sha256,
        "paper_dataset_schema_version": 1,
        "dataset_kind": "paper_episode_dataset",
        "paper_protocol_version": 1,
        "feature_contract_version": 1,
        "target_contract_version": 1,
        "storage_layout": "episode_shards",
        "split_manifest": {
            "episode_seeds": {"train": [101], "val": [202], "test": [303]}
        },
    }


def _trainer(config: dict, dataset: dict, output: Path) -> PaperTrainer:
    return PaperTrainer(
        model=_TinyPaperModel(config),
        config=config,
        device="cpu",
        dataset_metadata=dataset,
        best_checkpoint_path=output / "best.pt",
        last_checkpoint_path=output / "last.pt",
    )


def _checkpoint(tmp_path: Path) -> tuple[Path, dict, dict]:
    original = tmp_path / "original"
    config = _config(original, epochs=5)
    dataset = _dataset(original / "dataset.pt")
    trainer = _trainer(config, dataset, original)
    trainer.global_step = 7
    trainer.best_validation = 0.25
    trainer.best_epoch = 2
    trainer._save_checkpoint(original / "best.pt", epoch=2, validation_metric=0.25)
    path = original / "last.pt"
    trainer._save_checkpoint(path, epoch=3, validation_metric=0.25)
    return path, config, dataset


def test_resume_allows_epoch_extension_output_relocation_and_dataset_relocation(
    tmp_path: Path,
) -> None:
    checkpoint, config, dataset = _checkpoint(tmp_path)
    resumed_config = copy.deepcopy(config)
    resumed_config["paper_train"]["epochs"] = 8
    resumed_config["paper_train"]["save_dir"] = str(tmp_path / "relocated")
    relocated_dataset = copy.deepcopy(dataset)
    relocated_dataset["path"] = str(tmp_path / "relocated" / "same-dataset.pt")
    trainer = _trainer(resumed_config, relocated_dataset, tmp_path / "new-checkpoints")

    trainer.resume(checkpoint)

    assert trainer.start_epoch == 4
    assert trainer.global_step == 7
    assert trainer.best_checkpoint_path == tmp_path / "new-checkpoints" / "best.pt"
    assert trainer.last_checkpoint_path == tmp_path / "new-checkpoints" / "last.pt"
    assert trainer.best_checkpoint_path.is_file()


def test_resume_rejects_epoch_budget_below_completed_epoch(tmp_path: Path) -> None:
    checkpoint, config, dataset = _checkpoint(tmp_path)
    config["paper_train"]["epochs"] = 2
    trainer = _trainer(config, dataset, tmp_path / "shortened")

    with pytest.raises(ValueError, match="below the completed checkpoint epoch"):
        trainer.resume(checkpoint)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("optimizer", "training plan changed"),
        ("objective", "training plan changed"),
        ("batch", "training plan changed"),
        ("seed", "training plan changed"),
        ("model", "model semantics changed"),
        ("data", "different dataset"),
        ("environment", "action-environment semantics changed"),
    ),
)
def test_resume_rejects_semantic_changes(
    tmp_path: Path, change: str, message: str
) -> None:
    checkpoint, config, dataset = _checkpoint(tmp_path)
    if change == "optimizer":
        config["paper_train"]["optimizer"]["lr"] = 1e-3
    elif change == "objective":
        config["paper_train"]["objectives"]["one_step"]["tbptt_steps"] = 2
    elif change == "batch":
        config["paper_train"]["batch_size_control_graphs"] = 16
    elif change == "seed":
        config["paper_train"]["sampling_seed"] = 18
    elif change == "model":
        config["model"]["aggregator"] = "mean"
    elif change == "data":
        dataset["sha256"] = "b" * 64
    elif change == "environment":
        config["paper_protocol"] = {"policy_initializer": {"kind": "changed"}}
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(change)
    trainer = _trainer(config, dataset, tmp_path / f"changed-{change}")

    with pytest.raises(ValueError, match=message):
        trainer.resume(checkpoint)


@pytest.mark.parametrize("fingerprint_kind", ("semantic", "full"))
def test_resume_migrates_legacy_plan_fingerprint(
    tmp_path: Path, fingerprint_kind: str
) -> None:
    checkpoint, config, dataset = _checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["trainer_state"]
    state["trainer_schema_version"] = 3
    if fingerprint_kind == "full":
        state["training_plan_fingerprint"] = _legacy_full_training_plan_fingerprint(
            state["training_plan"]
        )
    for name in (
        "training_plan_fingerprint_version",
        "immutable_training_plan",
        "immutable_training_plan_fingerprint",
        "resumable_training_fields",
        "dataset_semantics",
        "dataset_semantic_fingerprint",
        "action_environment_fingerprint",
    ):
        state.pop(name, None)
    legacy = checkpoint.parent / f"legacy-v3-{fingerprint_kind}.pt"
    torch.save(payload, legacy)

    config["paper_train"]["epochs"] = 9
    config["paper_train"]["save_dir"] = str(tmp_path / "legacy-relocated")
    trainer = _trainer(config, dataset, tmp_path / "legacy-output")
    trainer.resume(legacy)

    assert trainer.start_epoch == 4


def test_resume_rejects_tampered_legacy_fingerprint_without_mutating_model(
    tmp_path: Path,
) -> None:
    checkpoint, config, dataset = _checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["trainer_state"]
    state["trainer_schema_version"] = 3
    state["training_plan_fingerprint"] = "0" * 64
    state.pop("training_plan_fingerprint_version", None)
    tampered = tmp_path / "tampered-v3.pt"
    torch.save(payload, tampered)
    trainer = _trainer(config, dataset, tmp_path / "tampered-output")
    before = trainer.model.weight.detach().clone()

    with pytest.raises(ValueError, match="fingerprint is invalid"):
        trainer.resume(tampered)

    assert torch.equal(trainer.model.weight.detach(), before)
