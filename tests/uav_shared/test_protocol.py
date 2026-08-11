from __future__ import annotations

from dataclasses import replace

import pytest

from leo_pg.uav_shared.protocol import (
    EstimatedDynamicsParameters,
    ManuscriptExactParameters,
    UAVSharedProtocol,
)
from leo_pg.uav_shared.randomness import keyed_order, keyed_uniform


def test_manuscript_values_and_estimates_have_separate_provenance() -> None:
    protocol = UAVSharedProtocol()
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
        item["status"] == "explicit_estimate"
        for item in manifest["explicit_estimates"].values()
    )
    assert "max_speed_m_s" in manifest["explicit_estimates"]
    assert "nominal_service_duration_s" in manifest["explicit_estimates"]


def test_manuscript_exact_fields_cannot_be_silently_changed() -> None:
    with pytest.raises(ValueError, match="manuscript-exact"):
        ManuscriptExactParameters(candidate_topk=4)

    with pytest.raises(ValueError, match="density_multiplier"):
        UAVSharedProtocol(density_multiplier=6)

    with pytest.raises(ValueError, match="capacity_compression"):
        UAVSharedProtocol(capacity_compression=0.9)


def test_estimated_parameter_change_changes_protocol_fingerprint() -> None:
    base = UAVSharedProtocol()
    changed = UAVSharedProtocol(
        estimated=replace(
            EstimatedDynamicsParameters(), max_speed_m_s=30.0
        )
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
