from __future__ import annotations
from typing import Dict, Any

from .tgn.tgn import TGN
from .heads.forecast import ForecastHead
from .heads.intensity_flow import IntensityFlowHead
from .heads.ranking import RankingHead
from ..sim.state import PAPER_EDGE_FEATURE_NAMES, PAPER_NODE_FEATURE_NAMES


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
    head_type = str(head_cfg.get("type", "forecast")).lower()

    if head_type == "forecast":
        head = ForecastHead(out_dim=int(head_cfg.get("out_dim", 1)),
                            in_dim=int(model_cfg.get("emb_dim", 64)),
                            output_activation=str(head_cfg.get("output_activation", "none")))
    elif head_type == "ranking":
        head = RankingHead(in_dim=int(model_cfg.get("emb_dim", 64)))
    elif head_type == "intensity_flow":
        if int(model_cfg.get("node_in_dim", 0)) != len(PAPER_NODE_FEATURE_NAMES):
            raise ValueError(
                "paper intensity_flow head requires node_in_dim="
                f"{len(PAPER_NODE_FEATURE_NAMES)}"
            )
        if int(model_cfg.get("edge_in_dim", 0)) != len(PAPER_EDGE_FEATURE_NAMES):
            raise ValueError(
                "paper intensity_flow head requires edge_in_dim="
                f"{len(PAPER_EDGE_FEATURE_NAMES)}"
            )
        head = IntensityFlowHead(
            in_dim=int(model_cfg.get("emb_dim", 64)),
            edge_in_dim=int(model_cfg.get("edge_in_dim", 0)),
            hidden_dim=int(head_cfg.get("hidden_dim", model_cfg.get("emb_dim", 64))),
            dropout=float(head_cfg.get("dropout", model_cfg.get("dropout", 0.0))),
        )
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
