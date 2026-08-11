from datetime import datetime, timezone

import pytest

from scripts.make_synthetic_tle import generate
from leo_pg.sim.ephemeris import load_tle_file


def test_synthetic_tle_generation_is_reproducible(tmp_path):
    epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)
    first = tmp_path / "first.tle"
    second = tmp_path / "second.tle"
    kwargs = dict(N=8, alt_km=550.0, inc_deg=53.0, planes=2, ecc=0.0001, seed=17, epoch=epoch)
    generate(str(first), **kwargs)
    generate(str(second), **kwargs)
    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_text().splitlines()) == 24
    assert len(load_tle_file(str(first))) == 8


def test_tle_loader_rejects_truncated_and_bad_checksum(tmp_path):
    epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)
    path = tmp_path / "base.tle"
    generate(str(path), N=1, alt_km=550.0, inc_deg=53.0, planes=1, ecc=0.0001, seed=7, epoch=epoch)

    truncated = tmp_path / "truncated.tle"
    truncated.write_text(path.read_text() + "DANGLING-NAME\n")
    with pytest.raises(ValueError, match="Truncated"):
        load_tle_file(str(truncated))

    lines = path.read_text().splitlines()
    lines[1] = lines[1][:-1] + str((int(lines[1][-1]) + 1) % 10)
    bad_checksum = tmp_path / "bad_checksum.tle"
    bad_checksum.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_tle_file(str(bad_checksum))
