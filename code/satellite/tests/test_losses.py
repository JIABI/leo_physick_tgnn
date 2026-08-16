import pytest
import torch

from leo_pg.train.losses import satellite_target_mse


def test_satellite_target_mse_excludes_user_padding():
    pred = torch.tensor([[10.0], [10.0], [1.0], [3.0]])
    target = torch.zeros_like(pred)
    step = {"meta": {"K_users": 2}}
    assert satellite_target_mse(pred, target, step) == torch.tensor(5.0)


def test_satellite_target_mse_rejects_empty_target_slice():
    pred = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="no valid satellite"):
        satellite_target_mse(pred, pred, {"meta": {"K_users": 2}})


def test_satellite_target_mse_requires_user_boundary():
    pred = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="K_users is required"):
        satellite_target_mse(pred, pred, {"meta": {}})


def test_satellite_target_mse_rejects_nonfinite_prediction():
    pred = torch.tensor([[float("nan")]])
    with pytest.raises(FloatingPointError, match="prediction"):
        satellite_target_mse(pred, torch.zeros_like(pred), {"meta": {"K_users": 0}})
