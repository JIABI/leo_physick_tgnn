from pathlib import Path

import pytest
import torch
import yaml

from leo_pg.paper.model_identity import (
    ReportedModelIdentityMismatch,
    audit_reported_models,
    build_exact_reported_model,
    reported_model_identities,
)
from leo_pg.paper.models import build_paper_model


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paper_protocol.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _step(epoch: int):
    user_count, satellite_count = 2, 3
    candidate_ids = torch.tensor(
        [[0, 0], [0, 1], [1, 1], [1, 2]], dtype=torch.long
    )
    return {
        "t": epoch,
        "node_x": torch.randn(user_count + satellite_count, 7),
        "edge_index": torch.stack(
            (candidate_ids[:, 0], candidate_ids[:, 1] + user_count)
        ),
        "edge_z": torch.randn(candidate_ids.size(0), 7),
        "edge_type": torch.zeros(candidate_ids.size(0), dtype=torch.long),
        "meta": {
            "K_users": user_count,
            "S_sats": satellite_count,
            "candidate_edge_ids": candidate_ids,
        },
    }


def test_reported_parameter_ledger_has_the_five_exact_table_s11_identities():
    identities = reported_model_identities(_config())
    assert {
        implementation_id: identity.reported_online_parameters
        for implementation_id, identity in identities.items()
    } == {
        "TGN-PHYSICK-SNAPSHOT-v1.0": 742_913,
        "TGN-PHYSICK-IF-v1.0": 758_401,
        "LTT-R-H10-v1.0": 12_041_857,
        "DA-GWM-SNAPSHOT-H10-v1.0": 8_527_361,
        "DA-GWM-IF-H10-v1.0": 8_684_929,
    }


def test_reference_constructors_report_real_mismatches_instead_of_padding():
    audits = {row.identity.implementation_id: row for row in audit_reported_models(_config())}
    assert {
        implementation_id: row.constructed_online_parameters
        for implementation_id, row in audits.items()
    } == {
        "TGN-PHYSICK-SNAPSHOT-v1.0": 847_107,
        "TGN-PHYSICK-IF-v1.0": 847_236,
        "LTT-R-H10-v1.0": 12_609_795,
        "DA-GWM-SNAPSHOT-H10-v1.0": 1_032_707,
        "DA-GWM-IF-H10-v1.0": 1_132_804,
    }
    assert not any(row.exact_match for row in audits.values())
    assert {
        row.status for row in audits.values()
    } == {"structural_reference_not_checkpoint_exact"}


def test_exact_identity_builder_fails_closed_on_architecture_level_gap():
    with pytest.raises(
        ReportedModelIdentityMismatch,
        match="unused parameter padding is forbidden",
    ):
        build_exact_reported_model(_config(), "LTT-R-H10-v1.0")


@pytest.mark.parametrize(
    "method",
    [
        "snapshot_physick",
        "tgn_physick",
        "snapshot_ltt_r",
        "snapshot_da_gwm",
        "da_gwm",
    ],
)
def test_every_reference_parameter_participates_in_a_two_epoch_forward(method):
    torch.manual_seed(20260826)
    model = build_paper_model(_config(), method)
    model.train()
    state = None
    objective = torch.zeros(())
    for epoch in range(2):
        prediction, state = model.predict_step(_step(epoch), state, "cpu")
        objective = objective + sum(value.sum() for value in prediction.as_dict().values())
    objective.backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    zero = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.any(parameter.grad))
    ]
    assert missing == []
    assert zero == []
