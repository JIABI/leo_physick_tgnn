import pytest
import torch

from leo_pg.sim.environment import MultiUserLEOEnv


def _cfg(**overrides):
    cfg = {"seed": 3, "T": 3, "dt": 0.25, "K_users": 2, "S_sats": 3}
    cfg.update(overrides)
    return cfg


def test_reset_is_t0_and_step_advances_exactly_once():
    env = MultiUserLEOEnv(_cfg(), device=torch.device("cpu"))
    initial_position = env.ephem.pos.clone()
    initial_velocity = env.ephem.vel.clone()
    first = env.reset()
    assert first["t"] == 0
    assert torch.allclose(first["node_x"][:, :3], initial_position)
    second = env.step()
    assert second["t"] == 1
    assert torch.allclose(second["node_x"][:, :3], initial_position + 0.25 * initial_velocity)
    env.step()
    with pytest.raises(RuntimeError, match="finished"):
        env.step()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"T": 0}, "T must"),
        ({"S_sats": 0}, "S_sats"),
        ({"load": {"momentum": 1.2}}, "momentum"),
        ({"channel": {"base_sinr": float("nan")}}, "must be finite"),
        ({"channel": {"dist_scale": -1.0}}, "dist_scale"),
    ],
)
def test_invalid_environment_protocol_fails_fast(overrides, match):
    with pytest.raises(ValueError, match=match):
        MultiUserLEOEnv(_cfg(**overrides), device=torch.device("cpu"))
