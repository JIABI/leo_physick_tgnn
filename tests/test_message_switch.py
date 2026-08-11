import torch
from leo_pg.models.registry import build_model

def _make_cfg(message_type: str):
    return {
        "model": {
            "node_in_dim": 6, "edge_in_dim": 6,
            "mem_dim": 16, "msg_dim": 16, "emb_dim": 16,
            "message_type": message_type,
            "aggregator": "sum",
            "dropout": 0.0,
            "use_edge_type": True,
            "edge_type_vocab": 8,
        },
        "head": {"type": "forecast", "out_dim": 1}
    }

def test_switches_run_forward():
    device = torch.device("cpu")
    episode = {
        "steps": [{
            "node_x": torch.randn(10,6),
            "edge_index": torch.tensor([[0,1,2],[5,6,7]]),
            "edge_z": torch.randn(3,6).tanh(),
            "edge_type": torch.zeros(3, dtype=torch.long),
            "y": torch.randn(10,1),
            "t": 0,
        }]
    }
    for mt in ["mlp","kan","physick"]:
        cfg = _make_cfg(mt)
        model = build_model(cfg).to(device)
        out = model.forward_episode(episode, device=device)
        assert len(out["preds"]) == 1
        assert out["preds"][0].shape == (10,1)
