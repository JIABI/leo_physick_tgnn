from pathlib import Path

import pytest
import torch
import yaml

from leo_pg.paper.mrg_ppo import (
    REPORTED_ACTOR_PARAMETERS,
    REPORTED_CRITIC_PARAMETERS,
    REPORTED_TOTAL_TRAINING_PARAMETERS,
    build_mrg_ppo_from_config,
    clipped_ppo_loss,
    fixed_executed_reward,
    generalized_advantage_estimate,
    masked_categorical,
    parameter_count,
    verify_reported_parameter_counts,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "models" / "mrg_ppo.yaml"


def _config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _graph_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(19)
    # Three user nodes followed by three satellite nodes.  Every stored edge is
    # an authorized user--satellite candidate before optional batch padding.
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [3, 4, 4, 5, 3, 5]],
        dtype=torch.long,
    )
    edge_count = edge_index.shape[1]
    node_types = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long).expand(
        batch_size, -1
    )
    return {
        "node_features": torch.randn(batch_size, 6, 7),
        "history_features": torch.randn(batch_size, 6, 2),
        "edge_index": edge_index,
        "edge_features": torch.randn(batch_size, edge_count, 7),
        "node_types": node_types,
        "edge_types": torch.arange(edge_count).remainder(8).unsqueeze(0).expand(
            batch_size, -1
        ),
        "load_summary": torch.rand(batch_size, 160),
        "association_ids": torch.tensor(
            [[12, 44, -1, 12, 44, 71]], dtype=torch.long
        ).expand(batch_size, -1),
        "association_active": torch.tensor(
            [[True, True, False, True, True, True]], dtype=torch.bool
        ).expand(batch_size, -1),
        "edge_mask": torch.tensor(
            [[True, True, False, True, True, False], [True, False, True, True, False, True]],
            dtype=torch.bool,
        ),
    }


def test_models_match_reported_parameter_ledger() -> None:
    actor, critic = build_mrg_ppo_from_config(_config())

    assert parameter_count(actor) == REPORTED_ACTOR_PARAMETERS == 731_905
    assert parameter_count(critic) == REPORTED_CRITIC_PARAMETERS == 687_233
    assert parameter_count(actor) + parameter_count(critic) == (
        REPORTED_TOTAL_TRAINING_PARAMETERS
    )
    verify_reported_parameter_counts(actor, critic)


def test_actor_critic_forward_masked_action_and_gradients() -> None:
    actor, critic = build_mrg_ppo_from_config(_config())
    batch = _graph_batch()

    actor_output = actor(
        batch["node_features"],
        batch["history_features"],
        batch["edge_index"],
        batch["edge_features"],
        batch["association_ids"],
        batch["association_active"],
        edge_mask=batch["edge_mask"],
    )
    assert actor_output.logits.shape == (2, 6)
    assert actor_output.hidden_state.shape == (2, 6, 128)

    distribution = actor.distribution(actor_output.logits, batch["edge_mask"])
    assert torch.equal(
        distribution.probs.masked_select(~batch["edge_mask"]),
        torch.zeros_like(distribution.probs.masked_select(~batch["edge_mask"])),
    )
    action, log_prob, entropy = actor.select_action(
        actor_output.logits,
        batch["edge_mask"],
        deterministic=True,
    )
    assert action.shape == log_prob.shape == entropy.shape == (2,)
    assert bool(batch["edge_mask"].gather(-1, action.unsqueeze(-1)).all())

    values = critic(
        batch["node_features"],
        batch["history_features"],
        batch["edge_index"],
        batch["edge_features"],
        batch["node_types"],
        batch["edge_types"],
        batch["load_summary"],
        batch["association_ids"],
        batch["association_active"],
        edge_mask=batch["edge_mask"],
    )
    assert values.shape == (2,)
    assert bool(torch.isfinite(values).all())

    objective = -log_prob.mean() + 0.01 * actor_output.hidden_state.square().mean()
    objective = objective + values.square().mean()
    objective.backward()
    assert actor.node_encoder.weight.grad is not None
    assert critic.node_encoder.weight.grad is not None
    assert bool(torch.isfinite(actor.node_encoder.weight.grad).all())
    assert bool(torch.isfinite(critic.node_encoder.weight.grad).all())


def test_masked_categorical_supports_one_distribution_per_user() -> None:
    logits = torch.tensor(
        [[[2.0, 100.0, 1.0], [0.0, -1.0, 3.0]]],
        requires_grad=True,
    )
    mask = torch.tensor([[[True, False, True], [False, True, True]]])
    distribution = masked_categorical(logits, mask)
    actions = torch.argmax(distribution.logits, dim=-1)

    assert actions.tolist() == [[0, 2]]
    assert distribution.probs[0, 0, 1].item() == 0.0
    assert distribution.probs[0, 1, 0].item() == 0.0
    (-distribution.log_prob(actions).mean()).backward()
    assert logits.grad is not None


def test_masked_categorical_rejects_empty_authorized_set() -> None:
    with pytest.raises(ValueError, match="authorized action"):
        masked_categorical(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))


def test_generalized_advantage_estimate_respects_terminal_boundary() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    dones = torch.tensor([False, False, True])

    advantages, returns = generalized_advantage_estimate(
        rewards,
        values,
        dones,
        torch.tensor(9.0),
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([3.0, 2.0, 1.0]))
    torch.testing.assert_close(returns, advantages)


def test_fixed_reward_uses_only_five_bounded_executed_terms() -> None:
    reward = fixed_executed_reward(
        delivered_service_p10_ratio=torch.tensor([0.8]),
        outage_fraction=torch.tensor([0.1]),
        rejected_handover_fraction=torch.tensor([0.2]),
        aba_return_fraction=torch.tensor([0.0]),
        active_load_cv=torch.tensor([0.25]),
    )
    torch.testing.assert_close(reward, torch.tensor([(0.8 + 0.9 + 0.8 + 1.0 + 0.8) / 5]))


def test_clipped_ppo_loss_is_finite_and_differentiable() -> None:
    new_log_prob = torch.tensor([0.25, -0.15, 0.03, -0.04], requires_grad=True)
    old_log_prob = torch.zeros(4)
    advantages = torch.tensor([1.2, -0.4, 0.7, -1.1])
    new_values = torch.tensor([0.3, 0.1, -0.2, 0.6], requires_grad=True)
    returns = torch.tensor([0.8, -0.1, 0.2, 0.4])
    entropy = torch.tensor([0.4, 0.3, 0.5, 0.2])

    loss = clipped_ppo_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        new_values=new_values,
        returns=returns,
        entropy=entropy,
        clip_range=0.20,
        value_coefficient=0.50,
        entropy_coefficient=0.005,
    )

    for value in (
        loss.total,
        loss.policy,
        loss.value,
        loss.entropy,
        loss.approx_kl,
        loss.clip_fraction,
    ):
        assert bool(torch.isfinite(value))
    assert 0.0 <= loss.clip_fraction.item() <= 1.0
    loss.total.backward()
    assert new_log_prob.grad is not None
    assert new_values.grad is not None


def test_config_records_the_paper_budget_and_optimization_contract() -> None:
    config = _config()
    reporting = config["training_budget"]["reporting"]
    tuning = config["training_budget"]["tuning"]
    optimization = config["optimization"]

    assert reporting["episodes_per_seed"] * reporting["steps_per_episode"] == 400_000
    assert reporting["environment_interactions_per_seed"] == 400_000
    assert reporting["reporting_seeds"] == 10
    assert tuning["total_environment_interactions"] == 4_800_000
    assert tuning["learning_rate_candidates"] == [0.0001, 0.0003]
    assert tuning["entropy_coefficient_candidates"] == [0.005, 0.010]
    assert optimization["discount_gamma"] == 0.99
    assert optimization["gae_lambda"] == 0.95
    assert optimization["policy_clip_range"] == 0.20
    assert optimization["ppo_updates_per_seed"] == 250
    assert optimization["ppo_epochs_per_update"] == 4
    assert optimization["minibatches_per_epoch"] == 16
    assert optimization["gradient_optimizer_steps_per_seed"] == 16_000
