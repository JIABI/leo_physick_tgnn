import torch
from leo_pg.physics.descriptors import compute_edge_descriptors

def test_descriptor_shapes_and_ranges():
    device = torch.device("cpu")
    edge_ctx = {"dist": torch.tensor([0.1, 0.2, 0.5]),
                "dst_s": torch.tensor([0, 1, 2])}
    rate = torch.tensor([1.0, 0.5, 0.1])
    ho = torch.tensor([0.2, 0.5, 1.2])
    sat_load = torch.tensor([0.0, 0.5, 1.0])
    cox_cfg = {"shells":[{"P":72,"Q":22,"theta":0.02,"alpha_deg":53.0,"p_feas":0.7}]}

    z = compute_edge_descriptors(edge_ctx, rate, ho, sat_load, cox_cfg, device)
    assert z.shape == (3,6)
    assert torch.isfinite(z).all()
    assert (z.abs() <= 1.0 + 1e-3).all()
