from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from uav_cfs.protocol import (
    EstimatedDynamicsParameters,
    ManuscriptExactParameters,
    UAVSharedProtocol,
)
from uav_cfs.config import resolve_uav_shared_protocol
from uav_cfs.randomness import keyed_order, keyed_uniform


def test_manuscript_values_and_estimates_have_separate_provenance() -> None:
    protocol = resolve_uav_shared_protocol(
        Path(__file__).parents[1] / "configs" / "model_mlp.yaml"
    )
    manifest = protocol.manifest()

    assert protocol.uav_count == 24
    assert protocol.station_count == 6
    assert protocol.horizon_steps == 120
    assert manifest["manuscript_exact"]["candidate_topk"]["value"] == 3
    assert manifest["manuscript_exact"]["active_slots_per_station"]["value"] == 2
    assert manifest["manuscript_exact"]["fifo_queue_capacity"]["value"] == 8
    assert (
        manifest["manuscript_exact"]["feasible_start_reserve_fraction"]["value"]
        == 0.15
    )
    assert manifest["manuscript_exact"]["intensity_lookahead_s"]["value"] == 12.0
    assert all(
        item["status"] == "manuscript_exact"
        for item in manifest["manuscript_exact"].values()
    )
    assert all(
        item["status"] == "runtime_required_not_manuscript_exact"
        for item in manifest["author_supplied_dynamics"].values()
    )
    assert "max_speed_m_s" in manifest["author_supplied_dynamics"]
    assert "nominal_service_duration_s" in manifest["author_supplied_dynamics"]


def test_manuscript_exact_fields_cannot_be_silently_changed() -> None:
    with pytest.raises(ValueError, match="manuscript-exact"):
        ManuscriptExactParameters(candidate_topk=4)

    with pytest.raises(ValueError, match="density_multiplier"):
        replace(_base_protocol(), density_multiplier=6)

    with pytest.raises(ValueError, match="capacity_compression"):
        replace(_base_protocol(), capacity_compression=0.9)


def test_estimated_parameter_change_changes_protocol_fingerprint() -> None:
    base = _base_protocol()
    changed = replace(
        base, estimated=replace(base.estimated, max_speed_m_s=30.0)
    )
    assert base.fingerprint != changed.fingerprint


def test_keyed_randomness_is_entity_and_iteration_order_independent() -> None:
    forward = keyed_order(range(24), 19, 7, "arrival")
    reverse = keyed_order(reversed(range(24)), 19, 7, "arrival")
    assert forward == reverse
    assert sorted(forward) == list(range(24))
    assert keyed_uniform(19, 7, "motion", 3) == keyed_uniform(
        19, 7, "motion", 3
    )
    assert keyed_uniform(19, 7, "motion", 3) != keyed_uniform(
        19, 7, "motion", 4
    )


def _base_protocol() -> UAVSharedProtocol:
    return resolve_uav_shared_protocol(
        Path(__file__).parents[1] / "configs" / "model_mlp.yaml"
    )
