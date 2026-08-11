import pytest
import torch

from dataset_verify import verify_bundle
from leo_pg.data.schema import DATASET_SCHEMA_VERSION


def _bundle(episodes):
    return {"dataset_schema_version": DATASET_SCHEMA_VERSION, "episodes": episodes}


def _episode(index: int, edge_index: torch.Tensor | None = None):
    if edge_index is None:
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return {
        "meta": {"episode": index},
        "steps": [
            {
                "t": 0,
                "node_x": torch.zeros(2, 6),
                "edge_index": edge_index,
                "edge_z": torch.zeros(edge_index.size(1), 6),
                "edge_type": torch.zeros(edge_index.size(1), dtype=torch.long),
                "y": torch.zeros(2, 1),
                "meta": {
                    "K_users": 1,
                    "S_sats": 1,
                    "sat_capacity_users": 1,
                    "sat_load_pre": torch.zeros(1),
                    "allocation_user_order": torch.tensor([0]),
                    "allocator_protocol": {
                        "name": "rate_greedy_v1",
                        "base_sinr": 10.0,
                        "dist_scale": 5.0,
                        "sinr_min": -1.0,
                        "load_momentum": 0.8,
                    },
                    "serving_sat": torch.tensor([0]),
                    "ho_fail": torch.tensor([False]),
                },
            }
        ],
    }


def test_verifier_rejects_split_leakage(tmp_path):
    first, second = _episode(0), _episode(1)
    path = tmp_path / "leaky.pt"
    torch.save(
        {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "episodes": [first, second],
            "train": [first],
            "val": [],
            "test": [first, second],
        },
        path,
    )
    with pytest.raises(ValueError, match="not mutually exclusive"):
        verify_bundle(str(path))


def test_verifier_rejects_out_of_range_edges(tmp_path):
    episode = _episode(0, edge_index=torch.tensor([[0], [2]], dtype=torch.long))
    path = tmp_path / "bad_edge.pt"
    torch.save(_bundle([episode]), path)
    with pytest.raises(ValueError, match="outside"):
        verify_bundle(str(path))


def test_verifier_rejects_user_sat_edge_with_user_destination(tmp_path):
    episode = _episode(0, edge_index=torch.tensor([[0], [0]], dtype=torch.long))
    path = tmp_path / "wrong_endpoint_type.pt"
    torch.save(_bundle([episode]), path)
    with pytest.raises(ValueError, match="USER_SAT edges"):
        verify_bundle(str(path))


def test_verifier_rejects_nonfinite_allocator_protocol(tmp_path):
    episode = _episode(0)
    episode["steps"][0]["meta"]["allocator_protocol"]["base_sinr"] = float("nan")
    path = tmp_path / "nan_protocol.pt"
    torch.save(_bundle([episode]), path)
    with pytest.raises(ValueError, match="must be finite"):
        verify_bundle(str(path))
