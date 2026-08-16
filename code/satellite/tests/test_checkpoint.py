import pytest
import torch

from leo_pg.models.registry import build_model
from leo_pg.train.checkpoint import load_ckpt, save_ckpt


def _cfg(kernels: int):
    return {
        "model": {
            "node_in_dim": 6,
            "edge_in_dim": 6,
            "mem_dim": 8,
            "msg_dim": 8,
            "emb_dim": 8,
            "message_type": "physick",
            "aggregator": "sum",
            "dropout": 0.0,
            "use_edge_type": True,
            "edge_type_vocab": 8,
            "physick": {"num_kernels": kernels, "projection_radius": 1.0},
        },
        "head": {"type": "forecast", "out_dim": 1},
    }


def test_checkpoint_reports_architecture_mismatch(tmp_path):
    path = tmp_path / "legacy.pt"
    legacy_cfg = _cfg(12)
    save_ckpt(str(path), model=build_model(legacy_cfg), epoch=1, config=legacy_cfg)
    with pytest.raises(RuntimeError, match="legacy PhysiCK checkpoints used 12 kernels"):
        load_ckpt(str(path), model=build_model(_cfg(16)), map_location=torch.device("cpu"))


def test_checkpoint_rejects_same_shape_semantic_mismatch(tmp_path):
    path = tmp_path / "semantic.pt"
    saved_cfg = _cfg(16)
    save_ckpt(str(path), model=build_model(saved_cfg), config=saved_cfg)
    changed_cfg = _cfg(16)
    changed_cfg["model"]["aggregator"] = "mean"
    changed_cfg["model"]["physick"]["projection_radius"] = 0.25
    with pytest.raises(RuntimeError, match="model signature"):
        load_ckpt(str(path), model=build_model(changed_cfg), map_location=torch.device("cpu"))


def test_checkpoint_rejects_conflicting_duplicate_model_states(tmp_path):
    path = tmp_path / "conflicting.pt"
    cfg = _cfg(16)
    model = build_model(cfg)
    preferred = model.state_dict()
    legacy = {name: value.clone() for name, value in preferred.items()}
    floating_name = next(name for name, value in legacy.items() if value.is_floating_point())
    legacy[floating_name].view(-1)[0] += 1.0
    torch.save({"model_state": preferred, "model": legacy}, path)
    with pytest.raises(ValueError, match="conflicting model states"):
        load_ckpt(str(path), model=build_model(cfg), map_location=torch.device("cpu"))


def test_checkpoint_rejects_unsigned_legacy_semantics_by_default(tmp_path):
    path = tmp_path / "unsigned_legacy.pt"
    cfg = _cfg(16)
    torch.save({"model": build_model(cfg).state_dict()}, path)
    with pytest.raises(RuntimeError, match="no model semantic signature"):
        load_ckpt(str(path), model=build_model(cfg), map_location=torch.device("cpu"))
    load_ckpt(
        str(path),
        model=build_model(cfg),
        map_location=torch.device("cpu"),
        allow_legacy_checkpoint=True,
    )
