import torch

from leo_pg.graph.candidates import build_elevation_candidates, pairwise_elevation_deg


def test_elevation_mask_topk_and_satellite_id_tie_break():
    users = torch.tensor([[1.0, 0.0, 0.0]])
    satellites = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.1, 0.0],
            [2.0, -0.1, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    candidates = build_elevation_candidates(
        users,
        satellites,
        minimum_elevation_deg=10.0,
        topk=2,
    )
    # Satellite 0 is highest. Satellites 1 and 2 tie, so smaller id 1 wins.
    assert candidates.edge_ids.tolist() == [[0, 0], [0, 1]]
    assert candidates.edge_index.tolist() == [[0, 0], [1, 2]]
    assert torch.all(candidates.elevation_deg >= 10.0)


def test_pairwise_elevation_uses_local_horizon():
    users = torch.tensor([[1.0, 0.0, 0.0]])
    satellites = torch.tensor([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    elevation = pairwise_elevation_deg(users, satellites)
    assert torch.allclose(elevation[0, 0], torch.tensor(90.0))
    assert torch.allclose(elevation[0, 1], torch.tensor(0.0))


def test_candidate_builder_does_not_fallback_below_elevation_mask():
    users = torch.tensor([[1.0, 0.0, 0.0]])
    satellites = torch.tensor([[0.0, 2.0, 0.0]])
    candidates = build_elevation_candidates(
        users,
        satellites,
        minimum_elevation_deg=10.0,
        topk=6,
    )
    assert candidates.edge_ids.shape == (0, 2)
    assert candidates.edge_index.shape == (2, 0)
