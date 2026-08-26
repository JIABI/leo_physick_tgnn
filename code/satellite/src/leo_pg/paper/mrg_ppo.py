"""Masked recurrent graph PPO used as the model-free paper comparator.

The deployed actor consumes only the authorized Snapshot graph and causal
association/dwell history.  It has no transition-prediction head and never
receives Intensity or Flow.  The centralized critic is used during training
only.  This module contains the policy/value networks and the two numerical
building blocks needed by PPO; it does not contain or report trained results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


REPORTED_ACTOR_PARAMETERS = 731_905
REPORTED_CRITIC_PARAMETERS = 687_233
REPORTED_TOTAL_TRAINING_PARAMETERS = 1_419_138


def parameter_count(module: nn.Module) -> int:
    """Return the number of trainable parameters in ``module``."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _require_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")


def _edge_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    edge_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch_size, edge_count, device=device, dtype=dtype)
    if mask.ndim == 1:
        if mask.numel() != edge_count:
            raise ValueError("one-dimensional edge_mask must have one entry per edge")
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    if mask.shape != (batch_size, edge_count):
        raise ValueError(
            f"edge_mask must have shape {(batch_size, edge_count)}, got {tuple(mask.shape)}"
        )
    return mask.to(device=device, dtype=dtype)


def _association_encoding(
    association_ids: torch.Tensor,
    association_active: torch.Tensor,
    *,
    identity_embedding: nn.Embedding,
    active_embedding: nn.Embedding,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if association_ids.shape != association_active.shape:
        raise ValueError("association_ids and association_active must have identical shapes")
    ids = association_ids.to(device=device, dtype=torch.long)
    active = association_active.to(device=device, dtype=torch.bool)
    if bool((ids < -1).any()) or bool((ids >= identity_embedding.num_embeddings).any()):
        raise ValueError("association_ids must be -1 or a valid satellite identifier")
    identity = identity_embedding(ids.clamp_min(0))
    identity = identity * active.unsqueeze(-1).to(dtype=identity.dtype)
    if identity.shape[-1] > hidden_dim:
        raise ValueError("association identity embedding cannot exceed hidden_dim")
    identity = F.pad(identity, (0, hidden_dim - identity.shape[-1]))
    status = active_embedding(active.to(dtype=torch.long))
    return (identity + status).to(dtype=dtype)


class GraphMessageLayer(nn.Module):
    """One pure-PyTorch, bidirectional message-passing layer.

    A shared MLP forms messages from the source, destination and authorized
    edge state.  Incoming messages are degree-normalized before a second MLP
    updates each node.  Passing only authorized edges, or an equivalent
    ``edge_mask`` for padded batches, enforces the graph part of the action
    contract before ranking.
    """

    def __init__(self, hidden_dim: int = 128, message_hidden_dim: int = 320) -> None:
        super().__init__()
        if hidden_dim <= 0 or message_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        self.hidden_dim = int(hidden_dim)
        self.message_hidden_dim = int(message_hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.message_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.message_hidden_dim, self.hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.message_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.message_hidden_dim, self.hidden_dim),
        )

    def forward(
        self,
        node_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require_rank("node_state", node_state, 3)
        _require_rank("edge_index", edge_index, 2)
        _require_rank("edge_state", edge_state, 3)
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        batch_size, node_count, hidden_dim = node_state.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"node_state final dimension must be {self.hidden_dim}")
        if edge_state.shape != (batch_size, edge_index.shape[1], self.hidden_dim):
            raise ValueError("edge_state must have shape [B, E, hidden_dim]")

        edge_index = edge_index.to(device=node_state.device, dtype=torch.long)
        if edge_index.numel() and (
            int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= node_count
        ):
            raise ValueError("edge_index contains a node outside the current graph")
        source, destination = edge_index[0], edge_index[1]
        edge_count = int(edge_index.shape[1])
        weight = _edge_mask(
            edge_mask,
            batch_size=batch_size,
            edge_count=edge_count,
            device=node_state.device,
            dtype=node_state.dtype,
        ).unsqueeze(-1)

        source_state = node_state.index_select(1, source)
        destination_state = node_state.index_select(1, destination)
        forward_message = self.message_mlp(
            torch.cat((source_state, destination_state, edge_state), dim=-1)
        ) * weight
        reverse_message = self.message_mlp(
            torch.cat((destination_state, source_state, edge_state), dim=-1)
        ) * weight

        aggregate = node_state.new_zeros(batch_size, node_count, self.hidden_dim)
        degree = node_state.new_zeros(batch_size, node_count, 1)
        aggregate.index_add_(1, destination, forward_message)
        aggregate.index_add_(1, source, reverse_message)
        degree.index_add_(1, destination, weight)
        degree.index_add_(1, source, weight)
        aggregate = aggregate / degree.clamp_min(1.0)

        update = self.update_mlp(torch.cat((node_state, aggregate), dim=-1))
        return torch.tanh(node_state + update)


@dataclass(frozen=True)
class ActorOutput:
    """Edge logits and the next per-node recurrent state."""

    logits: torch.Tensor
    hidden_state: torch.Tensor


class MaskedRecurrentGraphActor(nn.Module):
    """Two-layer graph actor with a GRU-128 state at every graph node.

    The frozen paper implementation consumes the seven current node fields,
    two causal history scalars, the current association identity and its active
    flag.  Each authorized user--satellite edge has the seven fields listed in
    the protocol config.  The same implementation produces one scalar logit
    per authorized edge.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int = 7,
        history_feature_dim: int = 2,
        edge_feature_dim: int = 7,
        association_vocabulary_size: int = 160,
        association_embedding_dim: int = 32,
        hidden_dim: int = 128,
        message_hidden_dim: int = 320,
        readout_hidden_dim: int = 384,
        graph_layers: int = 2,
    ) -> None:
        super().__init__()
        dimensions = (
            node_feature_dim,
            history_feature_dim,
            edge_feature_dim,
            association_vocabulary_size,
            association_embedding_dim,
            hidden_dim,
            message_hidden_dim,
            readout_hidden_dim,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("all feature and hidden dimensions must be positive")
        if int(graph_layers) != 2:
            raise ValueError("the paper MRG-PPO actor uses exactly two graph layers")

        self.node_feature_dim = int(node_feature_dim)
        self.history_feature_dim = int(history_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.node_encoder = nn.Linear(
            self.node_feature_dim + self.history_feature_dim, self.hidden_dim
        )
        self.edge_encoder = nn.Linear(self.edge_feature_dim, self.hidden_dim)
        self.association_embedding = nn.Embedding(
            int(association_vocabulary_size), int(association_embedding_dim)
        )
        self.association_active_embedding = nn.Embedding(2, self.hidden_dim)
        self.graph_layers = nn.ModuleList(
            [
                GraphMessageLayer(self.hidden_dim, int(message_hidden_dim))
                for _ in range(2)
            ]
        )
        self.recurrent = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_dim, int(readout_hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(readout_hidden_dim), 1),
        )

    def initial_state(
        self,
        batch_size: int,
        node_count: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0 or node_count <= 0:
            raise ValueError("batch_size and node_count must be positive")
        reference = self.node_encoder.weight
        return torch.zeros(
            batch_size,
            node_count,
            self.hidden_dim,
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        history_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        association_ids: torch.Tensor,
        association_active: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> ActorOutput:
        _require_rank("node_features", node_features, 3)
        _require_rank("history_features", history_features, 3)
        _require_rank("edge_features", edge_features, 3)
        batch_size, node_count, node_dim = node_features.shape
        if node_dim != self.node_feature_dim:
            raise ValueError(f"node_features final dimension must be {self.node_feature_dim}")
        if history_features.shape != (
            batch_size,
            node_count,
            self.history_feature_dim,
        ):
            raise ValueError("history_features has the wrong shape")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        edge_count = int(edge_index.shape[1])
        if edge_features.shape != (batch_size, edge_count, self.edge_feature_dim):
            raise ValueError("edge_features has the wrong shape")

        node_state = torch.tanh(
            self.node_encoder(torch.cat((node_features, history_features), dim=-1))
            + _association_encoding(
                association_ids,
                association_active,
                identity_embedding=self.association_embedding,
                active_embedding=self.association_active_embedding,
                hidden_dim=self.hidden_dim,
                device=node_features.device,
                dtype=node_features.dtype,
            )
        )
        edge_state = torch.tanh(self.edge_encoder(edge_features))
        for layer in self.graph_layers:
            node_state = layer(node_state, edge_index, edge_state, edge_mask)

        if hidden_state is None:
            hidden_state = self.initial_state(
                batch_size,
                node_count,
                device=node_state.device,
                dtype=node_state.dtype,
            )
        if hidden_state.shape != (batch_size, node_count, self.hidden_dim):
            raise ValueError("hidden_state has the wrong shape")
        next_hidden = self.recurrent(
            node_state.reshape(batch_size * node_count, self.hidden_dim),
            hidden_state.reshape(batch_size * node_count, self.hidden_dim),
        ).reshape(batch_size, node_count, self.hidden_dim)

        index = edge_index.to(device=next_hidden.device, dtype=torch.long)
        source_state = next_hidden.index_select(1, index[0])
        destination_state = next_hidden.index_select(1, index[1])
        policy_edge_state = source_state + destination_state + edge_state
        logits = self.actor_head(policy_edge_state).squeeze(-1)
        return ActorOutput(logits=logits, hidden_state=next_hidden)

    @staticmethod
    def distribution(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
        """Create a categorical distribution with infeasible actions at zero mass.

        ``logits`` and ``action_mask`` may be ``[B, K]`` or ``[B, U, K]``.
        In the latter case PyTorch creates one categorical distribution per
        user.  Candidate-edge logits can therefore be grouped by their stable
        user ordering before this call.
        """

        return masked_categorical(logits, action_mask)

    @staticmethod
    def select_action(
        logits: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = masked_categorical(logits, action_mask)
        action = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy()


class CentralizedGraphCritic(nn.Module):
    """Training-only critic over the observed graph and admitted-load summary.

    User and satellite embeddings are pooled separately.  The observed
    160-satellite admitted-load vector is encoded and concatenated with those
    pools before a two-hidden-layer value MLP of width 128.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int = 7,
        history_feature_dim: int = 2,
        edge_feature_dim: int = 7,
        association_vocabulary_size: int = 160,
        association_embedding_dim: int = 32,
        load_summary_dim: int = 160,
        hidden_dim: int = 128,
        message_hidden_dim: int = 320,
        graph_layers: int = 2,
        node_type_count: int = 2,
        edge_type_count: int = 8,
    ) -> None:
        super().__init__()
        dimensions = (
            node_feature_dim,
            history_feature_dim,
            edge_feature_dim,
            association_vocabulary_size,
            association_embedding_dim,
            load_summary_dim,
            hidden_dim,
            message_hidden_dim,
            node_type_count,
            edge_type_count,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("all feature, vocabulary and hidden dimensions must be positive")
        if int(graph_layers) != 2:
            raise ValueError("the paper MRG-PPO critic uses exactly two graph layers")

        self.node_feature_dim = int(node_feature_dim)
        self.history_feature_dim = int(history_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.load_summary_dim = int(load_summary_dim)
        self.hidden_dim = int(hidden_dim)
        self.node_encoder = nn.Linear(
            self.node_feature_dim + self.history_feature_dim, self.hidden_dim
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.association_embedding = nn.Embedding(
            int(association_vocabulary_size), int(association_embedding_dim)
        )
        self.association_active_embedding = nn.Embedding(2, self.hidden_dim)
        self.node_type_embedding = nn.Embedding(int(node_type_count), self.hidden_dim)
        self.edge_type_embedding = nn.Embedding(int(edge_type_count), self.hidden_dim)
        self.graph_layers = nn.ModuleList(
            [
                GraphMessageLayer(self.hidden_dim, int(message_hidden_dim))
                for _ in range(2)
            ]
        )
        self.load_encoder = nn.Linear(self.load_summary_dim, self.hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _pool(
        state: torch.Tensor,
        selected: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        weights = selected.to(dtype=state.dtype).unsqueeze(-1)
        count = weights.sum(dim=1)
        if bool((count == 0).any()):
            raise ValueError(f"every graph must contain at least one unpadded {name} node")
        return (state * weights).sum(dim=1) / count

    def forward(
        self,
        node_features: torch.Tensor,
        history_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_types: torch.Tensor,
        edge_types: torch.Tensor,
        load_summary: torch.Tensor,
        association_ids: torch.Tensor,
        association_active: torch.Tensor,
        *,
        node_mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require_rank("node_features", node_features, 3)
        _require_rank("history_features", history_features, 3)
        _require_rank("edge_features", edge_features, 3)
        batch_size, node_count, node_dim = node_features.shape
        if node_dim != self.node_feature_dim:
            raise ValueError(f"node_features final dimension must be {self.node_feature_dim}")
        if history_features.shape != (
            batch_size,
            node_count,
            self.history_feature_dim,
        ):
            raise ValueError("history_features has the wrong shape")
        if node_types.shape != (batch_size, node_count):
            raise ValueError("node_types must have shape [B, N]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        edge_count = int(edge_index.shape[1])
        if edge_features.shape != (batch_size, edge_count, self.edge_feature_dim):
            raise ValueError("edge_features has the wrong shape")
        if edge_types.shape != (batch_size, edge_count):
            raise ValueError("edge_types must have shape [B, E]")
        if load_summary.shape != (batch_size, self.load_summary_dim):
            raise ValueError("load_summary has the wrong shape")

        node_types = node_types.to(device=node_features.device, dtype=torch.long)
        edge_types = edge_types.to(device=edge_features.device, dtype=torch.long)
        node_state = torch.tanh(
            self.node_encoder(torch.cat((node_features, history_features), dim=-1))
            + _association_encoding(
                association_ids,
                association_active,
                identity_embedding=self.association_embedding,
                active_embedding=self.association_active_embedding,
                hidden_dim=self.hidden_dim,
                device=node_features.device,
                dtype=node_features.dtype,
            )
            + self.node_type_embedding(node_types)
        )
        edge_state = torch.tanh(
            self.edge_encoder(edge_features) + self.edge_type_embedding(edge_types)
        )
        for layer in self.graph_layers:
            node_state = layer(node_state, edge_index, edge_state, edge_mask)

        if node_mask is None:
            valid_node = torch.ones(
                batch_size, node_count, dtype=torch.bool, device=node_state.device
            )
        else:
            if node_mask.shape != (batch_size, node_count):
                raise ValueError("node_mask must have shape [B, N]")
            valid_node = node_mask.to(device=node_state.device, dtype=torch.bool)
        user_pool = self._pool(
            node_state,
            valid_node & (node_types == 0),
            name="user",
        )
        satellite_pool = self._pool(
            node_state,
            valid_node & (node_types == 1),
            name="satellite",
        )
        load_state = torch.tanh(self.load_encoder(load_summary))
        return self.value_head(
            torch.cat((user_pool, satellite_pool, load_state), dim=-1)
        ).squeeze(-1)


def masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
    """Return a categorical distribution supported only on authorized actions."""

    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if action_mask.shape != logits.shape:
        raise ValueError("action_mask must have the same shape as logits")
    mask = action_mask.to(device=logits.device, dtype=torch.bool)
    if bool((~mask.any(dim=-1)).any()):
        raise ValueError("each categorical action set must contain an authorized action")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


def fixed_executed_reward(
    *,
    delivered_service_p10_ratio: torch.Tensor,
    outage_fraction: torch.Tensor,
    rejected_handover_fraction: torch.Tensor,
    aba_return_fraction: torch.Tensor,
    active_load_cv: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen equal-weight, five-term executed-control reward.

    The first four inputs are bounded fractions in ``[0,1]``.  Callers set the
    rejected-handover fraction to zero when no handover was attempted.  Load
    balance is transformed as ``1/(1+CV)`` and all five terms receive weight
    one fifth.  No oracle or held-out quantity enters this reward.
    """

    bounded = {
        "delivered_service_p10_ratio": delivered_service_p10_ratio,
        "outage_fraction": outage_fraction,
        "rejected_handover_fraction": rejected_handover_fraction,
        "aba_return_fraction": aba_return_fraction,
    }
    shape = delivered_service_p10_ratio.shape
    for name, value in bounded.items():
        if value.shape != shape:
            raise ValueError("all reward components must have the same shape")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite floating tensor")
        if bool(((value < 0) | (value > 1)).any()):
            raise ValueError(f"{name} must lie in [0,1]")
    if active_load_cv.shape != shape:
        raise ValueError("active_load_cv must match the reward component shape")
    if (
        not active_load_cv.is_floating_point()
        or not bool(torch.isfinite(active_load_cv).all())
        or bool((active_load_cv < 0).any())
    ):
        raise ValueError("active_load_cv must be finite and non-negative")
    balance = 1.0 / (1.0 + active_load_cv)
    return (
        delivered_service_p10_ratio
        + (1.0 - outage_fraction)
        + (1.0 - rejected_handover_fraction)
        + (1.0 - aba_return_fraction)
        + balance
    ) / 5.0


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and value targets along the leading time axis."""

    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values and dones must have identical shapes")
    if rewards.ndim < 1 or rewards.shape[0] == 0:
        raise ValueError("rollout tensors must have a non-empty time dimension")
    if bootstrap_value.shape != rewards.shape[1:]:
        raise ValueError("bootstrap_value must match one rollout time slice")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0,1]")
    if not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("gae_lambda must lie in [0,1]")

    rewards = rewards.detach()
    values = values.detach()
    bootstrap_value = bootstrap_value.detach().to(device=values.device, dtype=values.dtype)
    done = dones.to(device=values.device, dtype=values.dtype)
    advantages = torch.zeros_like(values)
    running_advantage = torch.zeros_like(bootstrap_value)
    next_value = bootstrap_value
    for time in range(rewards.shape[0] - 1, -1, -1):
        continuation = 1.0 - done[time]
        delta = rewards[time] + float(gamma) * next_value * continuation - values[time]
        running_advantage = (
            delta
            + float(gamma) * float(gae_lambda) * continuation * running_advantage
        )
        advantages[time] = running_advantage
        next_value = values[time]
    return advantages, advantages + values


@dataclass(frozen=True)
class PPOLoss:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def clipped_ppo_loss(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_range: float = 0.20,
    value_coefficient: float = 0.50,
    entropy_coefficient: float,
    normalize_advantages: bool = True,
    old_values: torch.Tensor | None = None,
    value_clip_range: float | None = None,
) -> PPOLoss:
    """Compute the clipped PPO objective and training diagnostics."""

    tensors = (old_log_prob, advantages, new_values, returns, entropy)
    if any(tensor.shape != new_log_prob.shape for tensor in tensors):
        raise ValueError("all PPO tensors must have identical shapes")
    if float(clip_range) < 0.0:
        raise ValueError("clip_range must be non-negative")
    if float(value_coefficient) < 0.0 or float(entropy_coefficient) < 0.0:
        raise ValueError("loss coefficients must be non-negative")
    if value_clip_range is not None and float(value_clip_range) < 0.0:
        raise ValueError("value_clip_range must be non-negative")

    advantage = advantages.detach()
    if normalize_advantages and advantage.numel() > 1:
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    log_ratio = new_log_prob - old_log_prob.detach()
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantage
    clipped = torch.clamp(
        ratio,
        1.0 - float(clip_range),
        1.0 + float(clip_range),
    ) * advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    value_error = (new_values - returns.detach()).square()
    if value_clip_range is not None:
        if old_values is None or old_values.shape != new_values.shape:
            raise ValueError("old_values matching new_values is required for value clipping")
        clipped_values = old_values.detach() + torch.clamp(
            new_values - old_values.detach(),
            -float(value_clip_range),
            float(value_clip_range),
        )
        value_error = torch.maximum(
            value_error,
            (clipped_values - returns.detach()).square(),
        )
    value_loss = 0.5 * value_error.mean()
    entropy_mean = entropy.mean()
    total = (
        policy_loss
        + float(value_coefficient) * value_loss
        - float(entropy_coefficient) * entropy_mean
    )
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > float(clip_range)).float().mean()
    return PPOLoss(
        total=total,
        policy=policy_loss,
        value=value_loss,
        entropy=entropy_mean,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )


def build_mrg_ppo_from_config(
    config: Mapping[str, Any],
) -> tuple[MaskedRecurrentGraphActor, CentralizedGraphCritic]:
    """Build the actor and critic from ``configs/models/mrg_ppo.yaml``."""

    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("model configuration is required")
    actor_cfg = model.get("actor")
    critic_cfg = model.get("critic")
    if not isinstance(actor_cfg, Mapping) or not isinstance(critic_cfg, Mapping):
        raise ValueError("model.actor and model.critic mappings are required")
    actor = MaskedRecurrentGraphActor(
        node_feature_dim=int(actor_cfg["node_feature_dim"]),
        history_feature_dim=int(actor_cfg["history_feature_dim"]),
        edge_feature_dim=int(actor_cfg["edge_feature_dim"]),
        association_vocabulary_size=int(actor_cfg["association_vocabulary_size"]),
        association_embedding_dim=int(actor_cfg["association_embedding_dim"]),
        hidden_dim=int(actor_cfg["hidden_dim"]),
        message_hidden_dim=int(actor_cfg["message_hidden_dim"]),
        readout_hidden_dim=int(actor_cfg["readout_hidden_dim"]),
        graph_layers=int(actor_cfg["graph_layers"]),
    )
    critic = CentralizedGraphCritic(
        node_feature_dim=int(critic_cfg["node_feature_dim"]),
        history_feature_dim=int(critic_cfg["history_feature_dim"]),
        edge_feature_dim=int(critic_cfg["edge_feature_dim"]),
        association_vocabulary_size=int(critic_cfg["association_vocabulary_size"]),
        association_embedding_dim=int(critic_cfg["association_embedding_dim"]),
        load_summary_dim=int(critic_cfg["load_summary_dim"]),
        hidden_dim=int(critic_cfg["hidden_dim"]),
        message_hidden_dim=int(critic_cfg["message_hidden_dim"]),
        graph_layers=int(critic_cfg["graph_layers"]),
        node_type_count=int(critic_cfg["node_type_count"]),
        edge_type_count=int(critic_cfg["edge_type_count"]),
    )
    return actor, critic


def verify_reported_parameter_counts(
    actor: nn.Module,
    critic: nn.Module,
) -> None:
    """Fail if a code/config change drifts from the manuscript ledger."""

    actor_count = parameter_count(actor)
    critic_count = parameter_count(critic)
    if actor_count != REPORTED_ACTOR_PARAMETERS:
        raise ValueError(
            f"MRG-PPO actor has {actor_count:,} parameters; "
            f"expected {REPORTED_ACTOR_PARAMETERS:,}"
        )
    if critic_count != REPORTED_CRITIC_PARAMETERS:
        raise ValueError(
            f"MRG-PPO critic has {critic_count:,} parameters; "
            f"expected {REPORTED_CRITIC_PARAMETERS:,}"
        )


__all__ = [
    "ActorOutput",
    "CentralizedGraphCritic",
    "GraphMessageLayer",
    "MaskedRecurrentGraphActor",
    "PPOLoss",
    "REPORTED_ACTOR_PARAMETERS",
    "REPORTED_CRITIC_PARAMETERS",
    "REPORTED_TOTAL_TRAINING_PARAMETERS",
    "build_mrg_ppo_from_config",
    "clipped_ppo_loss",
    "fixed_executed_reward",
    "generalized_advantage_estimate",
    "masked_categorical",
    "parameter_count",
    "verify_reported_parameter_counts",
]
