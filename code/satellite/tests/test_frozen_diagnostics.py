import torch

from leo_pg.sim.environment import MultiUserLEOEnv
from leo_pg.train.frozen_diagnostics import frozen_diagnostics_for_episode


class _Memory:
    @staticmethod
    def init(node_x):
        return torch.zeros(node_x.size(0), 1, device=node_x.device)


class _OracleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = _Memory()

    def forward_step(self, step, mem, device):
        target = step["y"].to(device).float()
        return target.clone(), target, mem


def test_oracle_predictions_reproduce_nondefault_allocator_protocol():
    cfg = {
        "seed": 5,
        "T": 4,
        "K_users": 10,
        "S_sats": 6,
        "graph": {"visibility_radius": 1.0, "topk": 4},
        "channel": {"base_sinr": 7.0, "dist_scale": 3.0, "sinr_min": 2.0},
        "load": {"sat_capacity_users": 3, "momentum": 0.2},
    }
    env = MultiUserLEOEnv(cfg, device=torch.device("cpu"))
    steps = [env.reset()]
    for _ in range(3):
        steps.append(env.step())
    result = frozen_diagnostics_for_episode(
        {},
        _OracleModel(),
        {"steps": steps},
        torch.device("cpu"),
        horizon=4,
    )
    assert result["adjacent_aba_pred"] == result["adjacent_aba_gt"]
    assert result["assignment_failure_pred"] == result["assignment_failure_gt"]
    assert abs(result["load_var_pred"] - result["load_var_gt"]) < 1e-7
    assert abs(result["load_peak_pred"] - result["load_peak_gt"]) < 1e-7
