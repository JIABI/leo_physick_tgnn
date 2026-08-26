from pathlib import Path

from leo_pg.utils.config import load_cfg


def test_partial_model_config_inherits_default():
    root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(str(root / "configs/model/tgn_physick.yaml"))
    assert cfg["model"]["message_type"] == "physick"
    assert cfg["model"]["mem_dim"] == 128
    assert cfg["train"]["epochs"] == 50
