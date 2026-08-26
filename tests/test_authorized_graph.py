from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from leo_pg.sim.paper_environment import PaperAlignedLEOEnv


ROOT = Path(__file__).resolve().parents[1]


def test_model_observation_materializes_only_authorized_edges() -> None:
    config = yaml.safe_load((ROOT / "code/satellite/configs/paper_protocol.yaml").read_text())
    config = deepcopy(config)
    config["K_users"] = 8
    config["S_sats"] = 24
    config["paper_protocol"]["horizon_steps"] = 2
    config["paper_protocol"]["visibility"]["minimum"] = 10
    config["paper_protocol"]["visibility"]["maximum"] = 20
    config["paper_protocol"]["visibility"]["mean"] = 15
    environment = PaperAlignedLEOEnv(config)
    observation = environment.reset_control()
    assert observation.meta["candidate_graph_domain"] == "authorized_post_mask"
    assert observation.meta["geometry_candidate_count"] >= observation.edge_count
    assert bool(observation.sim_descriptors.feasible_edge.all())
    model_step = observation.as_model_step()
    assert model_step["edge_index"].shape[1] == observation.edge_count
    assert model_step["edge_z"].shape[0] == observation.edge_count

