from __future__ import annotations
from typing import Dict, Any
from .tgn.tgn import TGN
from .heads.forecast import ForecastHead
from .heads.ranking import RankingHead

def build_model(cfg: Dict[str, Any]):
    model_cfg = cfg["model"]
    head_cfg = cfg.get("head", {"type": "forecast"})
    head_type = head_cfg.get("type", "forecast")

    if head_type == "forecast":
        head = ForecastHead(out_dim=int(head_cfg.get("out_dim", 1)))
    elif head_type == "ranking":
        head = RankingHead()
    else:
        raise ValueError(f"Unknown head.type={head_type}")

    model = TGN(
        node_in_dim=int(model_cfg["node_in_dim"]),
        edge_in_dim=int(model_cfg["edge_in_dim"]),
        mem_dim=int(model_cfg["mem_dim"]),
        msg_dim=int(model_cfg["msg_dim"]),
        emb_dim=int(model_cfg["emb_dim"]),
        message_type=str(model_cfg["message_type"]),
        aggregator=str(model_cfg.get("aggregator","sum")),
        dropout=float(model_cfg.get("dropout",0.0)),
        use_edge_type=bool(model_cfg.get("use_edge_type", True)),
        edge_type_vocab=int(model_cfg.get("edge_type_vocab", 8)),
        head=head,
    )
    return model
