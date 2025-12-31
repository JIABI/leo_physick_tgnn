from __future__ import annotations
from typing import Dict, Any

from .tgn.tgn import TGN
from .heads.forecast import ForecastHead
from .heads.ranking import RankingHead


def build_model(cfg: Dict[str, Any]):
    """Factory for building the TGN + head stack.

    IMPORTANT: We pass `cfg` into TGN so message functions (e.g., PhysiCK) can
    read optional knobs from:
      - cfg["physick"][...]
      - cfg["model"]["physick"][...]
      - cfg["train"]["edge_chunk"] (memory control)
    """
    model_cfg = cfg["model"]
    head_cfg = cfg.get("head", {"type": "forecast"})
    head_type = head_cfg.get("type", "forecast")

    if head_type == "forecast":
        head = ForecastHead(out_dim=int(head_cfg.get("out_dim", 1)),
                            in_dim=int(model_cfg.get("emb_dim", 64)))
    elif head_type == "ranking":
        head = RankingHead(in_dim=int(model_cfg.get("emb_dim", 64)))
    else:
        raise ValueError(f"Unknown head.type={head_type}")

    # edge_chunk can live in train config; keep a safe default
    edge_chunk = int(cfg.get("train", {}).get("edge_chunk", model_cfg.get("edge_chunk", 50000)))

    model = TGN(
        node_in_dim=int(model_cfg["node_in_dim"]),
        edge_in_dim=int(model_cfg["edge_in_dim"]),
        mem_dim=int(model_cfg["mem_dim"]),
        msg_dim=int(model_cfg["msg_dim"]),
        emb_dim=int(model_cfg["emb_dim"]),
        message_type=str(model_cfg["message_type"]),
        aggregator=str(model_cfg.get("aggregator", "sum")),
        dropout=float(model_cfg.get("dropout", 0.0)),
        use_edge_type=bool(model_cfg.get("use_edge_type", True)),
        edge_type_vocab=int(model_cfg.get("edge_type_vocab", 8)),
        head=head,
        edge_chunk=edge_chunk,
        cfg=cfg,  # <<< critical: allows TGN to read physick knobs safely
    )
    return model
