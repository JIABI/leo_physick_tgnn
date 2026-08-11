from pathlib import Path

import pytest

from leo_pg.sim.paper_environment import protocol_fingerprint


def _closure(scale: float):
    def phi(load):
        return scale * load

    return phi


def _config_with_phi(phi):
    return {
        "ephemeris": {"mode": "debug"},
        "paper_protocol": {"flow": {"phi": phi}},
    }


def test_callable_requires_explicit_stable_protocol_fingerprint():
    with pytest.raises(
        ValueError,
        match=r"config\.paper_protocol\.flow\.phi.*__protocol_fingerprint__",
    ):
        protocol_fingerprint(_config_with_phi(_closure(1.0)))


def test_same_qualname_closures_do_not_collide_when_explicitly_versioned():
    first = _closure(1.0)
    second = _closure(9.0)
    first.__protocol_fingerprint__ = "linear-phi/scale=1/v1"
    second.__protocol_fingerprint__ = "linear-phi/scale=9/v1"

    assert protocol_fingerprint(_config_with_phi(first)) != protocol_fingerprint(
        _config_with_phi(second)
    )


def test_equivalent_callable_versions_have_stable_fingerprints():
    first = _closure(1.0)
    second = _closure(1.0)
    first.__protocol_fingerprint__ = "linear-phi/scale=1/v1"
    second.__protocol_fingerprint__ = "linear-phi/scale=1/v1"

    assert protocol_fingerprint(_config_with_phi(first)) == protocol_fingerprint(
        _config_with_phi(second)
    )


def _tle_config(path: Path):
    return {
        "ephemeris": {
            "mode": "skyfield_tle",
            "tle_path": str(path),
        }
    }


def test_tle_protocol_fingerprint_tracks_file_content(tmp_path: Path):
    tle_path = tmp_path / "constellation.tle"
    tle_path.write_bytes(b"tle-version-one\n")
    first = protocol_fingerprint(_tle_config(tle_path))

    tle_path.write_bytes(b"tle-version-two\n")
    second = protocol_fingerprint(_tle_config(tle_path))

    assert first != second


def test_tle_protocol_fingerprint_is_stable_for_unchanged_content(tmp_path: Path):
    tle_path = tmp_path / "constellation.tle"
    tle_path.write_bytes(b"same-tle-content\n")
    config = _tle_config(tle_path)

    assert protocol_fingerprint(config) == protocol_fingerprint(config)


def test_missing_tle_cannot_be_protocol_fingerprinted(tmp_path: Path):
    missing_path = tmp_path / "missing.tle"

    with pytest.raises(FileNotFoundError, match="ephemeris.tle_path does not exist"):
        protocol_fingerprint(_tle_config(missing_path))
