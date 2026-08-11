import torch
import pytest

from leo_pg.sim.environment import MultiUserLEOEnv
from leo_pg.train.metrics import ping_pong_rate
from leo_pg.train.system_metrics import (
    adjacent_aba_fraction,
    assignment_failure_fraction,
    greedy_assign_from_load,
    pingpong_rate,
)


def test_pingpong_uses_only_valid_triplets_in_denominator():
    serving = torch.tensor(
        [
            [1, -1],
            [2, 3],
            [1, 4],
        ]
    )
    assert pingpong_rate(serving) == 1.0
    assert adjacent_aba_fraction(serving) == 1.0
    assert ping_pong_rate(serving) == 1.0


def test_assignment_failure_fraction_is_not_attempt_normalized():
    failures = torch.tensor([[False, True], [False, False]])
    assert assignment_failure_fraction(failures) == 0.25


def test_greedy_assignment_is_deterministic_by_default():
    node_x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    edge_index = torch.tensor([[0, 0, 1, 1], [2, 3, 2, 3]])
    edge_type = torch.zeros(4, dtype=torch.long)
    kwargs = dict(
        node_x=node_x,
        edge_index=edge_index,
        edge_type=edge_type,
        sat_load=torch.zeros(2),
        K_users=2,
        sat_capacity_users=1,
        channel_cfg={"base_sinr": 10.0, "dist_scale": 5.0},
    )
    first = greedy_assign_from_load(**kwargs)[0]
    torch.manual_seed(999)
    second = greedy_assign_from_load(**kwargs)[0]
    assert torch.equal(first, second)


def test_allocator_reconstruction_is_oracle_consistent():
    cfg = {
        "seed": 19,
        "T": 2,
        "K_users": 8,
        "S_sats": 12,
        "graph": {"visibility_radius": 1.0, "topk": 5},
        "channel": {"base_sinr": 10.0, "dist_scale": 5.0, "sinr_min": -1.0},
        "load": {"sat_capacity_users": 4, "momentum": 0.8},
    }
    step = MultiUserLEOEnv(cfg, device=torch.device("cpu")).reset()
    meta = step["meta"]
    serving, _, failures = greedy_assign_from_load(
        node_x=step["node_x"],
        edge_index=step["edge_index"],
        edge_type=step["edge_type"],
        sat_load=meta["sat_load_pre"],
        K_users=meta["K_users"],
        sat_capacity_users=meta["sat_capacity_users"],
        channel_cfg=cfg["channel"],
        sinr_min=cfg["channel"]["sinr_min"],
        user_order=meta["allocation_user_order"],
    )
    assert torch.equal(serving, meta["serving_sat"])
    assert torch.equal(failures, meta["ho_fail"])
    incoming = torch.bincount(serving[serving >= 0], minlength=meta["S_sats"]).float()
    reconstructed_load = 0.8 * meta["sat_load_pre"] + 0.2 * (incoming / 4.0)
    assert torch.allclose(reconstructed_load, step["y"][meta["K_users"] :, 0])


def test_allocator_rejects_user_sat_edge_with_user_destination():
    with pytest.raises(ValueError, match="USER_SAT edges"):
        greedy_assign_from_load(
            node_x=torch.zeros(2, 6),
            edge_index=torch.tensor([[0], [0]]),
            edge_type=torch.tensor([0]),
            sat_load=torch.zeros(1),
            K_users=1,
            sat_capacity_users=1,
            channel_cfg={"base_sinr": 10.0, "dist_scale": 5.0},
        )


def test_allocator_rejects_nonfinite_channel_protocol():
    with pytest.raises(ValueError, match="must be finite"):
        greedy_assign_from_load(
            node_x=torch.zeros(2, 6),
            edge_index=torch.tensor([[0], [1]]),
            edge_type=torch.tensor([0]),
            sat_load=torch.zeros(1),
            K_users=1,
            sat_capacity_users=1,
            channel_cfg={"base_sinr": float("nan"), "dist_scale": 5.0},
        )
