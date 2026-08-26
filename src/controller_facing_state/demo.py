"""Small deterministic CPU demonstration of the implemented semantic path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import ROOT, load_experiment_registry, load_manuscript_config


def _satellite_observation():
    from leo_pg.sim.state import (
        ControlObservation,
        PolicyDescriptors,
        SimulatorDescriptors,
    )

    candidate_ids = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 2]], dtype=torch.long)
    candidate_index = torch.stack((candidate_ids[:, 0], candidate_ids[:, 1] + 2))
    descriptors = PolicyDescriptors(
        gamma_edge=torch.tensor([0.8, 0.3, 0.1, 0.9]),
        intensity_edge=torch.tensor([0.4, 0.8, 1.0, 0.2]),
        flow_node=torch.tensor([0.25, 0.75, 0.15]),
    )
    observation = ControlObservation(
        observation_id=(17, 10),
        node_x=torch.zeros((5, 7)),
        candidate_edge_index=candidate_index,
        candidate_edge_ids=candidate_ids,
        edge_features=torch.zeros((4, 7)),
        elevation_deg=torch.tensor([40.0, 30.0, 32.0, 42.0]),
        sim_descriptors=SimulatorDescriptors(
            policy_fields=descriptors.clone(),
            feasible_edge=torch.ones(4, dtype=torch.bool),
        ),
        policy_descriptors=descriptors,
        current_serving=torch.tensor([0, 1], dtype=torch.long),
        hold_steps=torch.tensor([10, 10], dtype=torch.long),
        user_order=torch.tensor([0, 1], dtype=torch.long),
        meta={"candidate_graph_domain": "authorized_post_mask"},
    )
    observation.validate()
    return observation


def _message_switch(config: dict[str, Any]) -> dict[str, Any]:
    from leo_pg.kernels.kan import KANMessage
    from leo_pg.kernels.mlp import MLPMessage
    from leo_pg.kernels.physick.paper_physick_message import PaperPhysiCKMessage

    torch.manual_seed(20260825)
    edges, mem_dim, edge_dim, msg_dim = 5, 128, 7, 128
    src = torch.randn(edges, mem_dim)
    dst = torch.randn(edges, mem_dim)
    edge = torch.randn(edges, edge_dim)
    edge_type = torch.zeros(edges, dtype=torch.long)
    operators = {
        "mlp": MLPMessage(mem_dim, edge_dim, msg_dim, dropout=0.1),
        "kan": KANMessage(mem_dim, edge_dim, msg_dim, num_knots=16),
        "physick": PaperPhysiCKMessage(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            msg_dim=msg_dim,
            num_kernels=16,
            latent_dim=128,
            descriptor_dim=16,
            kernel_hidden_dim=128,
            coeff_hidden_dim=128,
            dropout=0.1,
            projection_radius=float(
                config["model"]["message_operators"]["physick"][
                    "deployed_projection_radius"
                ]
            ),
            operating_clip=5.0,
        ),
    }
    result: dict[str, Any] = {}
    for name, operator in operators.items():
        operator.eval()
        with torch.no_grad():
            message = operator(src, dst, edge, edge_type)
        result[name] = {
            "shape": list(message.shape),
            "finite": bool(torch.isfinite(message).all()),
            "parameter_count": sum(p.numel() for p in operator.parameters()),
        }
    return result


def _satellite_policy(config: dict[str, Any]) -> dict[str, Any]:
    from leo_pg.control.policy import FixedRankPolicy, FixedRankPolicyConfig

    local_weight, constraint_weight, pressure_weight = config["roles"]["weights"]
    policy = FixedRankPolicy(
        FixedRankPolicyConfig(
            gamma_weight=float(local_weight),
            load_weight=float(pressure_weight),
            intensity_weight=float(constraint_weight),
            hard_feasibility_mask=True,
            min_dwell_steps=10,
            hysteresis=1.0 / 6.0,
        )
    )
    observation = _satellite_observation()
    scores = policy.score(observation)
    action = policy(observation)
    return {
        "candidate_graph_domain": observation.meta["candidate_graph_domain"],
        "authorized_edge_count": observation.edge_count,
        "requested_serving": action.requested_serving.tolist(),
        "score_weights_gamma_flow_intensity": [1.0, 0.7, 0.5],
        "all_ranked_edges_eligible": bool(scores.eligible.all()),
    }


def _uav_step() -> dict[str, Any]:
    from uav_cfs.config import build_uav_environment
    from uav_cfs.policy import UAVFixedRankPolicy, UAVFixedRankPolicyConfig

    config_path = ROOT / "code" / "uav" / "configs" / "common.yaml"
    environment = build_uav_environment(config_path, device="cpu")
    observation = environment.reset_control()
    policy = UAVFixedRankPolicy(
        UAVFixedRankPolicyConfig(
            eta_weight=1.0,
            intensity_weight=0.5,
            flow_weight=0.7,
            hard_feasible_start_mask=True,
            hysteresis=1.0 / 3.0,
        )
    )
    action = policy(observation)
    _, execution, _ = environment.step_action(action)
    return {
        "uav_count": observation.user_count,
        "station_count": observation.station_count,
        "candidate_edge_count": observation.edge_count,
        "hard_mask": True,
        "executed_service_starts": int(execution.service_started.sum().item()),
    }


def _mrg_ppo_smoke() -> dict[str, Any]:
    import yaml

    from leo_pg.paper.mrg_ppo import (
        build_mrg_ppo_from_config,
        parameter_count,
        verify_reported_parameter_counts,
    )

    config_path = ROOT / "configs" / "models" / "mrg_ppo.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    actor, critic = build_mrg_ppo_from_config(config)
    verify_reported_parameter_counts(actor, critic)
    torch.manual_seed(20260825)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [3, 4, 4, 5, 3, 5]], dtype=torch.long
    )
    node_features = torch.randn(1, 6, 7)
    history = torch.randn(1, 6, 2)
    edge_features = torch.randn(1, 6, 7)
    association_ids = torch.tensor([[12, 44, -1, 12, 44, 71]])
    association_active = association_ids >= 0
    edge_mask = torch.ones(1, 6, dtype=torch.bool)
    actor.eval()
    critic.eval()
    with torch.no_grad():
        actor_output = actor(
            node_features,
            history,
            edge_index,
            edge_features,
            association_ids,
            association_active,
            edge_mask=edge_mask,
        )
        values = critic(
            node_features,
            history,
            edge_index,
            edge_features,
            torch.tensor([[0, 0, 0, 1, 1, 1]]),
            torch.arange(6).remainder(8).unsqueeze(0),
            torch.zeros(1, 160),
            association_ids,
            association_active,
            edge_mask=edge_mask,
        )
    return {
        "actor_parameters": parameter_count(actor),
        "critic_parameters": parameter_count(critic),
        "actor_logits_shape": list(actor_output.logits.shape),
        "finite": bool(
            torch.isfinite(actor_output.logits).all() and torch.isfinite(values).all()
        ),
        "training_performed": False,
    }


def run_demo(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_manuscript_config(config_path)
    registry = load_experiment_registry()
    experiments = registry["experiments"]
    report = {
        "kind": "deterministic_semantic_demo",
        "manuscript_result_reproduction": False,
        "training_performed": False,
        "seed": 20260825,
        "satellite_policy": _satellite_policy(config),
        "message_operators": _message_switch(config),
        "mrg_ppo": _mrg_ppo_smoke(),
        "uav_execution": _uav_step(),
        "experiment_registry": {
            name: {
                "components_implemented": name != "EXP4",
                "full_historical_rerun_from_release": bool(
                    value.get(
                        "historical_eight_cell_rerun_from_release",
                        value.get("reproducible_from_release", False),
                    )
                ),
            }
            for name, value in experiments.items()
        },
    }
    target = Path(output).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    with (target / "demo_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (target / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report
