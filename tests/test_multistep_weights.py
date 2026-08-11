import pytest
import torch

from leo_pg.train.trainer import _build_multistep_weights


@pytest.mark.parametrize(
    "cfg,match",
    [
        ({"scheme": "typo"}, "unknown"),
        ({"scheme": "exp", "exp_beta": 1.0e6}, "non-finite"),
        (
            {"scheme": "milestones", "milestones": [{"t": 1, "w": 0.0}, {"t": 3, "w": 0.0}]},
            "positive sum",
        ),
    ],
)
def test_invalid_multistep_weights_fail_fast(cfg, match):
    with pytest.raises(ValueError, match=match):
        _build_multistep_weights(3, cfg, device=torch.device("cpu"))
