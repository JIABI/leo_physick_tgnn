import torch
from leo_pg.graph.builder import build_user_sat_edges

def test_build_edges():
    user_pos = torch.tensor([[0.0,0.0,0.0],[10.0,0.0,0.0]])
    sat_pos = torch.tensor([[0.1,0.0,0.0],[9.9,0.0,0.0]])
    edge_index, ctx = build_user_sat_edges(user_pos, sat_pos, visibility_radius=0.5, user_offset=0, sat_offset=2)
    assert edge_index.shape[0] == 2
    # each user sees one sat
    assert edge_index.shape[1] == 2
    assert (edge_index[1] >= 2).all()
