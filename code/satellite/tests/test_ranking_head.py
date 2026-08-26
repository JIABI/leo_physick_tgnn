from __future__ import annotations

import torch

from leo_pg.models.heads.ranking import RankingHead


def test_ranking_head_scores_only_candidate_edges() -> None:
    head = RankingHead(in_dim=4, edge_in_dim=2, hidden_dim=8)
    embedding = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10.0
    scores = head(
        embedding,
        {
            "edge_index": torch.tensor([[0, 1, 2], [3, 4, 3]]),
            "edge_z": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "edge_type": torch.tensor([0, 1, 0]),
        },
    )
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_ranking_head_returns_empty_for_no_candidate_edges() -> None:
    head = RankingHead(in_dim=3, edge_in_dim=1)
    result = head(
        torch.zeros(2, 3),
        {
            "edge_index": torch.tensor([[0], [1]]),
            "edge_z": torch.zeros(1, 1),
            "edge_type": torch.ones(1, dtype=torch.long),
        },
    )
    assert result.shape == (0,)
