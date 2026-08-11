import numpy as np

import pytest

from scripts.plot_system_metrics import _contract_id, _scale_factor
from leo_pg.train.frozen_diagnostics import EVALUATION_MODE, METRIC_CONTRACT


def test_plot_scaling_uses_one_factor_for_cross_model_values():
    factor, suffix = _scale_factor(np.asarray([1.0e-5, 1.0e-4]))
    assert factor == 1.0e4
    assert "-4" in suffix
    scaled = np.asarray([1.0e-5, 1.0e-4]) * factor
    assert np.allclose(scaled, [0.1, 1.0])


def test_plot_contract_rejects_legacy_by_default():
    with pytest.raises(ValueError, match="incompatible"):
        _contract_id({"results": []}, allow_legacy=False)
    assert (
        _contract_id(
            {"evaluation_mode": EVALUATION_MODE, "metric_contract": METRIC_CONTRACT},
            allow_legacy=False,
        )
        == "canonical-v1"
    )
