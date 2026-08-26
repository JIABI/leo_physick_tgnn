"""Executable protocol objects for the manuscript's extended experiments.

The functions in this module implement analysis and model-construction rules;
they do not contain observed result values and do not launch training.  In
particular, checkpoint selection accepts predictive losses only, so a caller
cannot accidentally use a controller outcome to choose a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from leo_pg.kernels.interface import MessageFunction
from leo_pg.kernels.mlp import MLPMessage
from leo_pg.kernels.physick.paper_kernel_bank import PaperVectorKernelBank
from leo_pg.kernels.physick.projection import project_onto_l1_ball


# ---------------------------------------------------------------------------
# EXT--EXP1: outcome-blind checkpoint selection and predictive equivalence


@dataclass(frozen=True)
class CheckpointLoss:
    """Predictive validation record allowed into checkpoint selection."""

    checkpoint_id: str
    epoch: int
    aggregate_loss: float
    field_losses: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")
        if (
            isinstance(self.epoch, bool)
            or int(self.epoch) != self.epoch
            or self.epoch < 0
        ):
            raise ValueError("epoch must be a non-negative integer")
        _require_positive_finite(self.aggregate_loss, "aggregate_loss")
        if not self.field_losses:
            raise ValueError("field_losses must be non-empty")
        for name, value in self.field_losses.items():
            if not name:
                raise ValueError("field-loss names must be non-empty")
            _require_positive_finite(value, f"field_losses[{name!r}]")


@dataclass(frozen=True)
class CheckpointSelectionRule:
    """Frozen EXP1 selection rule parsed from its registered identifier."""

    rule_id: str
    relative_margin: float | None

    @classmethod
    def parse(cls, value: str) -> "CheckpointSelectionRule":
        normalized = str(value).strip().lower()
        if normalized == "minimum_validation_loss":
            return cls(normalized, None)
        match = re.fullmatch(r"loss_matched_(0(?:\.\d+)?)", normalized)
        if match is None:
            raise ValueError(
                "selection rule must be minimum_validation_loss or "
                "loss_matched_<relative-margin>"
            )
        margin = float(match.group(1))
        if not 0.0 < margin < 1.0:
            raise ValueError("loss-matched relative margin must lie in (0,1)")
        return cls(normalized, margin)


@dataclass(frozen=True)
class SelectedCheckpointPair:
    reference: CheckpointLoss
    comparison: CheckpointLoss
    rule: CheckpointSelectionRule
    aggregate_loss_ratio: float


def select_checkpoint_pair(
    reference: Sequence[CheckpointLoss],
    comparison: Sequence[CheckpointLoss],
    rule: str | CheckpointSelectionRule,
) -> SelectedCheckpointPair:
    """Select a model pair using predictive validation losses only.

    The minimum-loss rule selects each model independently.  A loss-matched
    rule first retains checkpoints within the stated relative margin of that
    model's own validation minimum, then chooses the cross-model pair closest
    on the log-loss scale.  Remaining ties are resolved deterministically.
    """

    parsed = CheckpointSelectionRule.parse(rule) if isinstance(rule, str) else rule
    ref = _validated_checkpoint_pool(reference, "reference")
    cmp = _validated_checkpoint_pool(comparison, "comparison")
    ref_min = min(ref, key=_checkpoint_sort_key)
    cmp_min = min(cmp, key=_checkpoint_sort_key)

    if parsed.relative_margin is None:
        selected_ref, selected_cmp = ref_min, cmp_min
    else:
        ref_limit = ref_min.aggregate_loss * (1.0 + parsed.relative_margin)
        cmp_limit = cmp_min.aggregate_loss * (1.0 + parsed.relative_margin)
        ref_eligible = [item for item in ref if item.aggregate_loss <= ref_limit]
        cmp_eligible = [item for item in cmp if item.aggregate_loss <= cmp_limit]

        def pair_key(pair: tuple[CheckpointLoss, CheckpointLoss]) -> tuple[object, ...]:
            left, right = pair
            mismatch = abs(math.log(right.aggregate_loss / left.aggregate_loss))
            excess = (
                left.aggregate_loss / ref_min.aggregate_loss
                + right.aggregate_loss / cmp_min.aggregate_loss
            )
            return mismatch, excess, left.epoch, right.epoch, left.checkpoint_id, right.checkpoint_id

        selected_ref, selected_cmp = min(
            ((left, right) for left in ref_eligible for right in cmp_eligible),
            key=pair_key,
        )
    return SelectedCheckpointPair(
        reference=selected_ref,
        comparison=selected_cmp,
        rule=parsed,
        aggregate_loss_ratio=selected_cmp.aggregate_loss / selected_ref.aggregate_loss,
    )


def _checkpoint_sort_key(record: CheckpointLoss) -> tuple[float, int, str]:
    return record.aggregate_loss, record.epoch, record.checkpoint_id


def _validated_checkpoint_pool(
    records: Sequence[CheckpointLoss], name: str
) -> tuple[CheckpointLoss, ...]:
    pool = tuple(records)
    if not pool:
        raise ValueError(f"{name} checkpoint pool must be non-empty")
    identifiers = [record.checkpoint_id for record in pool]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} checkpoint identifiers must be unique")
    return pool


@dataclass(frozen=True)
class RatioEquivalenceResult:
    n_pairs: int
    geometric_mean_ratio: float
    ci_low: float
    ci_high: float
    lower_tost_p: float
    upper_tost_p: float
    band_low: float
    band_high: float
    alpha: float
    equivalent: bool

    @property
    def tost_p(self) -> float:
        return max(self.lower_tost_p, self.upper_tost_p)


def paired_ratio_equivalence(
    reference: Sequence[float],
    comparison: Sequence[float],
    *,
    band: tuple[float, float],
    confidence: float = 0.90,
) -> RatioEquivalenceResult:
    """Paired log-ratio TOST with its matching two-sided confidence interval."""

    if len(reference) != len(comparison) or len(reference) < 2:
        raise ValueError("equivalence requires equally sized vectors with at least two pairs")
    low, high = (float(band[0]), float(band[1]))
    if not 0.0 < low < 1.0 < high:
        raise ValueError("equivalence band must satisfy 0 < low < 1 < high")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    logs: list[float] = []
    for index, (left, right) in enumerate(zip(reference, comparison)):
        _require_positive_finite(left, f"reference[{index}]")
        _require_positive_finite(right, f"comparison[{index}]")
        logs.append(math.log(float(right) / float(left)))

    n_pairs = len(logs)
    mean = sum(logs) / n_pairs
    variance = sum((value - mean) ** 2 for value in logs) / (n_pairs - 1)
    standard_error = math.sqrt(variance / n_pairs)
    alpha = (1.0 - confidence) / 2.0
    log_low, log_high = math.log(low), math.log(high)
    if standard_error == 0.0:
        ci_log_low = ci_log_high = mean
        lower_p = 0.0 if mean > log_low else 1.0
        upper_p = 0.0 if mean < log_high else 1.0
    else:
        degrees_freedom = n_pairs - 1
        critical = _student_t_ppf(1.0 - alpha, degrees_freedom)
        ci_log_low = mean - critical * standard_error
        ci_log_high = mean + critical * standard_error
        lower_statistic = (mean - log_low) / standard_error
        upper_statistic = (mean - log_high) / standard_error
        lower_p = 1.0 - _student_t_cdf(lower_statistic, degrees_freedom)
        upper_p = _student_t_cdf(upper_statistic, degrees_freedom)
    ci_low, ci_high = math.exp(ci_log_low), math.exp(ci_log_high)
    equivalent = (
        lower_p < alpha
        and upper_p < alpha
        and ci_low > low
        and ci_high < high
    )
    return RatioEquivalenceResult(
        n_pairs=n_pairs,
        geometric_mean_ratio=math.exp(mean),
        ci_low=ci_low,
        ci_high=ci_high,
        lower_tost_p=lower_p,
        upper_tost_p=upper_p,
        band_low=low,
        band_high=high,
        alpha=alpha,
        equivalent=equivalent,
    )


@dataclass(frozen=True)
class EquivalenceGateResult:
    aggregate: RatioEquivalenceResult
    fields: Mapping[str, RatioEquivalenceResult]
    passed: bool


def evaluate_equivalence_gate(
    aggregate_reference: Sequence[float],
    aggregate_comparison: Sequence[float],
    field_reference: Mapping[str, Sequence[float]],
    field_comparison: Mapping[str, Sequence[float]],
    *,
    aggregate_band: tuple[float, float] = (0.95, 1.05),
    field_band: tuple[float, float] = (0.90, 1.10),
) -> EquivalenceGateResult:
    """Apply the registered aggregate and per-field EXP1 guardrails."""

    if not field_reference or set(field_reference) != set(field_comparison):
        raise ValueError("reference and comparison must contain the same non-empty fields")
    aggregate = paired_ratio_equivalence(
        aggregate_reference, aggregate_comparison, band=aggregate_band
    )
    fields = {
        name: paired_ratio_equivalence(
            field_reference[name], field_comparison[name], band=field_band
        )
        for name in sorted(field_reference)
    }
    return EquivalenceGateResult(
        aggregate=aggregate,
        fields=fields,
        passed=aggregate.equivalent and all(item.equivalent for item in fields.values()),
    )


@dataclass(frozen=True)
class NearTieCascade:
    supported_count: int
    ranking_disagreement_count: int
    proposal_disagreement_count: int
    executed_disagreement_count: int
    ranking_disagreement_rate: float
    proposal_disagreement_rate: float
    executed_disagreement_rate: float


def near_tie_cascade(
    supported: Sequence[bool],
    ranking_disagreement: Sequence[bool],
    proposal_disagreement: Sequence[bool],
    executed_disagreement: Sequence[bool],
) -> NearTieCascade:
    """Count the supported -> rank -> proposal -> execution near-tie chain.

    All rates use the supported near-tie set as denominator.  The function
    rejects non-nested flags rather than silently repairing a broken ledger.
    """

    vectors = tuple(map(tuple, (supported, ranking_disagreement, proposal_disagreement, executed_disagreement)))
    if len({len(item) for item in vectors}) != 1 or not vectors[0]:
        raise ValueError("near-tie ledgers must be non-empty and equally sized")
    supported_v, ranking_v, proposal_v, executed_v = vectors
    for index, flags in enumerate(zip(supported_v, ranking_v, proposal_v, executed_v)):
        support, ranking, proposal, executed = map(bool, flags)
        if ranking and not support:
            raise ValueError(f"ranking disagreement outside support at row {index}")
        if proposal and not ranking:
            raise ValueError(f"proposal disagreement without ranking disagreement at row {index}")
        if executed and not proposal:
            raise ValueError(f"executed disagreement without proposal disagreement at row {index}")
    counts = [sum(map(bool, item)) for item in vectors]
    denominator = counts[0]
    if denominator == 0:
        raise ValueError("near-tie support set is empty")
    return NearTieCascade(
        supported_count=denominator,
        ranking_disagreement_count=counts[1],
        proposal_disagreement_count=counts[2],
        executed_disagreement_count=counts[3],
        ranking_disagreement_rate=counts[1] / denominator,
        proposal_disagreement_rate=counts[2] / denominator,
        executed_disagreement_rate=counts[3] / denominator,
    )


# ---------------------------------------------------------------------------
# EXT--EXP3: objective x operator factorial


class EXP3Operator(str, Enum):
    MLP = "mlp"
    PHYSICK = "physick"


class EXP3Objective(str, Enum):
    PREDICTIVE = "predictive"
    DECISION_AWARE = "decision_aware"


@dataclass(frozen=True)
class EXP3Arm:
    arm_id: str
    operator: EXP3Operator
    objective: EXP3Objective


EXP3_ARMS: tuple[EXP3Arm, ...] = (
    EXP3Arm("A1", EXP3Operator.MLP, EXP3Objective.PREDICTIVE),
    EXP3Arm("A2", EXP3Operator.PHYSICK, EXP3Objective.PREDICTIVE),
    EXP3Arm("B1", EXP3Operator.MLP, EXP3Objective.DECISION_AWARE),
    EXP3Arm("B2", EXP3Operator.PHYSICK, EXP3Objective.DECISION_AWARE),
)


@dataclass(frozen=True)
class DecisionAwareLossWeights:
    field: float = 1.0
    pairwise_logistic: float = 0.20
    listmle_top5: float = 0.10
    margin: float = 0.05
    margin_size: float = 0.05

    def __post_init__(self) -> None:
        for name in ("field", "pairwise_logistic", "listmle_top5", "margin"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} weight must be non-negative")
        if self.margin_size <= 0.0:
            raise ValueError("margin_size must be positive")


@dataclass(frozen=True)
class DecisionAwareLossBreakdown:
    total: torch.Tensor
    field: torch.Tensor
    pairwise_logistic: torch.Tensor
    listmle_top5: torch.Tensor
    margin: torch.Tensor
    ordered_pair_count: int
    group_count: int


def decision_aware_combined_loss(
    field_loss: torch.Tensor,
    predicted_scores: torch.Tensor,
    oracle_scores: torch.Tensor,
    *,
    group_ids: torch.Tensor | None = None,
    persistent_mask: torch.Tensor | None = None,
    weights: DecisionAwareLossWeights = DecisionAwareLossWeights(),
) -> DecisionAwareLossBreakdown:
    """Compute the registered field + pairwise + top-5 ListMLE + margin loss.

    Ranking terms are formed only within a decision group and only for
    persistent candidates.  Oracle ties do not create pairwise or margin
    targets.  An empty ranking ledger contributes exact differentiable zeros.
    """

    predicted_scores = predicted_scores.reshape(-1)
    oracle_scores = oracle_scores.reshape(-1).to(
        device=predicted_scores.device, dtype=predicted_scores.dtype
    )
    if predicted_scores.numel() != oracle_scores.numel() or predicted_scores.numel() == 0:
        raise ValueError("predicted_scores and oracle_scores must have equal non-zero size")
    if not torch.isfinite(predicted_scores).all() or not torch.isfinite(oracle_scores).all():
        raise ValueError("score tensors contain NaN or Inf")
    field = field_loss.mean()
    if not torch.isfinite(field):
        raise ValueError("field_loss contains NaN or Inf")
    count = predicted_scores.numel()
    if group_ids is None:
        group_ids = torch.zeros(count, dtype=torch.long, device=predicted_scores.device)
    else:
        group_ids = group_ids.reshape(-1).to(device=predicted_scores.device)
    if persistent_mask is None:
        persistent_mask = torch.ones(count, dtype=torch.bool, device=predicted_scores.device)
    else:
        persistent_mask = persistent_mask.reshape(-1).to(
            device=predicted_scores.device, dtype=torch.bool
        )
    if group_ids.numel() != count or persistent_mask.numel() != count:
        raise ValueError("group_ids and persistent_mask must match the score vectors")

    pair_terms: list[torch.Tensor] = []
    margin_terms: list[torch.Tensor] = []
    list_terms: list[torch.Tensor] = []
    group_count = 0
    for group in torch.unique(group_ids, sorted=True):
        indices = torch.nonzero((group_ids == group) & persistent_mask, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        group_count += 1
        pred = predicted_scores[indices]
        oracle = oracle_scores[indices]
        differences = oracle[:, None] - oracle[None, :]
        ordered = torch.nonzero(differences > 0, as_tuple=False)
        if ordered.numel():
            predicted_differences = pred[ordered[:, 0]] - pred[ordered[:, 1]]
            pair_terms.extend(F.softplus(-predicted_differences).unbind())
            margin_terms.extend(
                F.relu(weights.margin_size - predicted_differences).unbind()
            )

        # Stable sorting makes tied oracle scores reproducible without treating
        # the arbitrary tie order as an additional pairwise target.
        order = torch.argsort(oracle, descending=True, stable=True)
        ranked_pred = pred[order]
        for position in range(min(5, ranked_pred.numel())):
            list_terms.append(
                torch.logsumexp(ranked_pred[position:], dim=0) - ranked_pred[position]
            )

    zero = predicted_scores.sum() * 0.0
    pairwise = torch.stack(pair_terms).mean() if pair_terms else zero
    margin = torch.stack(margin_terms).mean() if margin_terms else zero
    listmle = torch.stack(list_terms).mean() if list_terms else zero
    total = (
        weights.field * field
        + weights.pairwise_logistic * pairwise
        + weights.listmle_top5 * listmle
        + weights.margin * margin
    )
    return DecisionAwareLossBreakdown(
        total=total,
        field=field,
        pairwise_logistic=pairwise,
        listmle_top5=listmle,
        margin=margin,
        ordered_pair_count=len(pair_terms),
        group_count=group_count,
    )


def aba_ratio_of_rate_ratios(
    a1_events: int,
    a2_events: int,
    b1_events: int,
    b2_events: int,
    *,
    correction: float = 0.5,
) -> float:
    """EXP3 secondary ratio-of-rate-ratios with the frozen 0.5 correction."""

    events = (a1_events, a2_events, b1_events, b2_events)
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in events):
        raise ValueError("ABA event counts must be non-negative integers")
    if correction <= 0.0 or not math.isfinite(correction):
        raise ValueError("correction must be positive and finite")
    predictive_ratio = (a2_events + correction) / (a1_events + correction)
    decision_aware_ratio = (b2_events + correction) / (b1_events + correction)
    return decision_aware_ratio / predictive_ratio


# ---------------------------------------------------------------------------
# EXT--EXP5: ordered operator factory and frozen diagnostics


class EXP5Variant(str, Enum):
    MLP = "mlp"
    GENERIC_DYNAMIC_MIXTURE = "generic_dynamic_mixture"
    UNPROJECTED_PHYSICK = "unprojected_physick"
    PROJECTED_PHYSICK = "projected_physick"


class GenericDynamicMixtureMessage(MessageFunction):
    """Input-conditioned mixture of generic learned message experts."""

    def __init__(
        self,
        *,
        mem_dim: int,
        edge_dim: int,
        msg_dim: int,
        num_experts: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        edge_type_vocab: int = 8,
        use_edge_type: bool = True,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.mem_dim, self.edge_dim, self.msg_dim = mem_dim, edge_dim, msg_dim
        self.num_experts = int(num_experts)
        in_dim = 2 * mem_dim + edge_dim
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, msg_dim),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_experts),
        )
        self.output_norm = nn.LayerNorm(msg_dim)
        self.type_embedding = (
            nn.Embedding(edge_type_vocab, msg_dim) if use_edge_type else None
        )

    def coefficient_tensors(
        self, mem_src: torch.Tensor, mem_dst: torch.Tensor, z_ij: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = _message_inputs(mem_src, mem_dst, z_ij, self.mem_dim, self.edge_dim)
        logits = self.gate(inputs)
        return logits, torch.softmax(logits, dim=-1)

    def forward(
        self,
        mem_src: torch.Tensor,
        mem_dst: torch.Tensor,
        z_ij: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = _message_inputs(mem_src, mem_dst, z_ij, self.mem_dim, self.edge_dim)
        _, coefficients = self.coefficient_tensors(mem_src, mem_dst, z_ij)
        experts = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        message = (coefficients.unsqueeze(-1) * experts).sum(dim=1)
        return self.output_norm(_apply_edge_type(message, edge_type, self.type_embedding))


class _PhysiCKMechanismMessage(MessageFunction):
    """Shared PhysiCK mechanism with projection toggled for EXP5."""

    def __init__(
        self,
        *,
        mem_dim: int,
        edge_dim: int,
        msg_dim: int,
        projected: bool,
        projection_radius: float,
        num_kernels: int = 16,
        latent_dim: int = 128,
        descriptor_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        edge_type_vocab: int = 8,
        use_edge_type: bool = True,
        operating_clip: float = 5.0,
    ) -> None:
        super().__init__()
        if projected and projection_radius <= 0.0:
            raise ValueError("projection_radius must be positive")
        self.mem_dim, self.edge_dim, self.msg_dim = mem_dim, edge_dim, msg_dim
        self.projected = bool(projected)
        self.projection_radius = float(projection_radius)
        self.phi_in = nn.Sequential(
            nn.Linear(2 * mem_dim + edge_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(latent_dim),
        )
        self.bank = PaperVectorKernelBank(
            latent_dim=latent_dim,
            descriptor_dim=descriptor_dim,
            msg_dim=msg_dim,
            num_kernels=num_kernels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            descriptor_clip=operating_clip,
        )
        self.coefficient_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_kernels),
        )
        self.output_norm = nn.LayerNorm(msg_dim)
        self.type_embedding = (
            nn.Embedding(edge_type_vocab, msg_dim) if use_edge_type else None
        )

    def coefficient_tensors(
        self, mem_src: torch.Tensor, mem_dst: torch.Tensor, z_ij: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = _message_inputs(mem_src, mem_dst, z_ij, self.mem_dim, self.edge_dim)
        latent = self.phi_in(inputs)
        raw = self.coefficient_head(latent)
        effective = (
            project_onto_l1_ball(raw.float(), self.projection_radius).to(raw.dtype)
            if self.projected
            else raw
        )
        return raw, effective

    def forward(
        self,
        mem_src: torch.Tensor,
        mem_dst: torch.Tensor,
        z_ij: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = _message_inputs(mem_src, mem_dst, z_ij, self.mem_dim, self.edge_dim)
        latent = self.phi_in(inputs)
        kernels, _ = self.bank(latent)
        raw = self.coefficient_head(latent)
        coefficients = (
            project_onto_l1_ball(raw.float(), self.projection_radius).to(kernels.dtype)
            if self.projected
            else raw.to(kernels.dtype)
        )
        message = (coefficients.unsqueeze(-1) * kernels).sum(dim=1)
        return self.output_norm(_apply_edge_type(message, edge_type, self.type_embedding))


def build_exp5_operator(
    variant: EXP5Variant | str,
    *,
    mem_dim: int,
    edge_dim: int,
    msg_dim: int,
    num_kernels: int = 16,
    hidden_dim: int = 128,
    descriptor_dim: int = 16,
    dropout: float = 0.1,
    projection_radius: float = 1.25,
    edge_type_vocab: int = 8,
    use_edge_type: bool = True,
) -> MessageFunction:
    """Construct one of the four registered EXP5 operators."""

    selected = EXP5Variant(variant)
    if selected is EXP5Variant.MLP:
        return MLPMessage(mem_dim, edge_dim, msg_dim, dropout=dropout)
    if selected is EXP5Variant.GENERIC_DYNAMIC_MIXTURE:
        return GenericDynamicMixtureMessage(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            msg_dim=msg_dim,
            num_experts=num_kernels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            edge_type_vocab=edge_type_vocab,
            use_edge_type=use_edge_type,
        )
    return _PhysiCKMechanismMessage(
        mem_dim=mem_dim,
        edge_dim=edge_dim,
        msg_dim=msg_dim,
        projected=selected is EXP5Variant.PROJECTED_PHYSICK,
        projection_radius=projection_radius,
        num_kernels=num_kernels,
        descriptor_dim=descriptor_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        edge_type_vocab=edge_type_vocab,
        use_edge_type=use_edge_type,
    )


@dataclass(frozen=True)
class CoefficientDiagnostics:
    available: bool
    observations: int
    active_projection_count: int | None
    active_projection_fraction: float | None
    raw_l1_median: float | None
    effective_l1_median: float | None


@dataclass(frozen=True)
class OperatorDiagnostics:
    coefficient: CoefficientDiagnostics
    operator_jump_p95: float
    expansive_jacobian_fraction: float


def coefficient_diagnostics(
    operator: MessageFunction,
    mem_src: torch.Tensor,
    mem_dst: torch.Tensor,
    z_ij: torch.Tensor,
    *,
    tolerance: float = 1e-7,
) -> CoefficientDiagnostics:
    """Summarize coefficient exposure for one diagnostic minibatch.

    For projected PhysiCK, an observation is active when raw L1 mass exceeds
    the registered radius.  This batch-level count can be accumulated by a
    training loop to obtain projection-active updates and update exposure.
    """

    method = getattr(operator, "coefficient_tensors", None)
    if method is None:
        return CoefficientDiagnostics(False, int(z_ij.size(0)), None, None, None, None)
    with torch.no_grad():
        raw, effective = method(mem_src, mem_dst, z_ij)
        raw_l1 = raw.abs().sum(dim=-1)
        effective_l1 = effective.abs().sum(dim=-1)
        if getattr(operator, "projected", False):
            radius = float(getattr(operator, "projection_radius"))
            active_count: int | None = int((raw_l1 > radius + tolerance).sum().item())
            active_fraction: float | None = active_count / max(1, raw_l1.numel())
        else:
            active_count, active_fraction = None, None
    return CoefficientDiagnostics(
        available=True,
        observations=int(raw_l1.numel()),
        active_projection_count=active_count,
        active_projection_fraction=active_fraction,
        raw_l1_median=float(raw_l1.median().item()),
        effective_l1_median=float(effective_l1.median().item()),
    )


def operator_jump_p95(
    operator: MessageFunction,
    base_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    perturbed_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    edge_type: torch.Tensor | None = None,
) -> float:
    """95th percentile of per-edge message jumps for a paired perturbation."""

    if any(left.shape != right.shape for left, right in zip(base_inputs, perturbed_inputs)):
        raise ValueError("base and perturbed message inputs must have matching shapes")
    prior_mode = operator.training
    operator.eval()
    try:
        with torch.no_grad():
            before = operator(*base_inputs, edge_type=edge_type)
            after = operator(*perturbed_inputs, edge_type=edge_type)
            jumps = torch.linalg.vector_norm(after - before, dim=-1)
            if jumps.numel() == 0:
                raise ValueError("operator jump diagnostic requires at least one edge")
            return float(torch.quantile(jumps.float(), 0.95).item())
    finally:
        operator.train(prior_mode)


def expansive_jacobian_fraction(
    operator: MessageFunction,
    mem_src: torch.Tensor,
    mem_dst: torch.Tensor,
    z_ij: torch.Tensor,
    *,
    edge_type: torch.Tensor | None = None,
    threshold: float = 1.0,
) -> float:
    """Fraction of edge-local message Jacobians with spectral norm > threshold."""

    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("Jacobian threshold must be non-negative and finite")
    if z_ij.size(0) == 0:
        raise ValueError("Jacobian diagnostic requires at least one edge")
    prior_mode = operator.training
    operator.eval()
    expansive = 0
    try:
        for index in range(z_ij.size(0)):
            combined = torch.cat(
                (mem_src[index], mem_dst[index], z_ij[index]), dim=0
            ).detach().requires_grad_(True)
            mem_dim = mem_src.size(1)
            edge_dim = z_ij.size(1)
            local_type = None if edge_type is None else edge_type[index : index + 1]

            def local_message(value: torch.Tensor) -> torch.Tensor:
                left = value[:mem_dim].unsqueeze(0)
                right = value[mem_dim : 2 * mem_dim].unsqueeze(0)
                edge = value[2 * mem_dim : 2 * mem_dim + edge_dim].unsqueeze(0)
                return operator(left, right, edge, edge_type=local_type).squeeze(0)

            jacobian = torch.autograd.functional.jacobian(local_message, combined)
            spectral_norm = torch.linalg.matrix_norm(jacobian.float(), ord=2)
            expansive += int(float(spectral_norm.item()) > threshold)
        return expansive / z_ij.size(0)
    finally:
        operator.train(prior_mode)


def exp5_operator_diagnostics(
    operator: MessageFunction,
    base_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    perturbed_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    edge_type: torch.Tensor | None = None,
    jacobian_threshold: float = 1.0,
) -> OperatorDiagnostics:
    """Run the registered coefficient, jump and expansivity diagnostics."""

    return OperatorDiagnostics(
        coefficient=coefficient_diagnostics(operator, *base_inputs),
        operator_jump_p95=operator_jump_p95(
            operator, base_inputs, perturbed_inputs, edge_type=edge_type
        ),
        expansive_jacobian_fraction=expansive_jacobian_fraction(
            operator,
            *base_inputs,
            edge_type=edge_type,
            threshold=jacobian_threshold,
        ),
    )


def _message_inputs(
    mem_src: torch.Tensor,
    mem_dst: torch.Tensor,
    z_ij: torch.Tensor,
    mem_dim: int,
    edge_dim: int,
) -> torch.Tensor:
    if mem_src.shape != mem_dst.shape or mem_src.ndim != 2:
        raise ValueError("source and destination memories must have matching rank-2 shapes")
    if mem_src.size(1) != mem_dim or z_ij.shape != (mem_src.size(0), edge_dim):
        raise ValueError("message input dimensions do not match the configured operator")
    if not all(torch.isfinite(item).all() for item in (mem_src, mem_dst, z_ij)):
        raise ValueError("message input contains NaN or Inf")
    return torch.cat((mem_src, mem_dst, z_ij), dim=-1)


def _apply_edge_type(
    message: torch.Tensor,
    edge_type: torch.Tensor | None,
    embedding: nn.Embedding | None,
) -> torch.Tensor:
    if embedding is None:
        return message
    if edge_type is None:
        edge_type = torch.zeros(message.size(0), dtype=torch.long, device=message.device)
    if edge_type.shape != (message.size(0),):
        raise ValueError("edge_type must have shape [E]")
    return message * torch.tanh(embedding(edge_type.long()))


def _require_positive_finite(value: float, name: str) -> None:
    if float(value) <= 0.0 or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be positive and finite")


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta using a stable continued fraction."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_beta_term)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations, epsilon, floor = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > floor else floor)
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        numerator = iteration * (b - iteration) * x / (
            (qam + even) * (a + even)
        )
        d = 1.0 + numerator * d
        d = d if abs(d) > floor else floor
        c = 1.0 + numerator / c
        c = c if abs(c) > floor else floor
        d = 1.0 / d
        result *= d * c
        numerator = -(a + iteration) * (qab + iteration) * x / (
            (a + even) * (qap + even)
        )
        d = 1.0 + numerator * d
        d = d if abs(d) > floor else floor
        c = 1.0 + numerator / c
        c = c if abs(c) > floor else floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _student_t_cdf(value: float, degrees_freedom: int) -> float:
    if degrees_freedom <= 0:
        raise ValueError("degrees_freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees_freedom / (degrees_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        x, degrees_freedom / 2.0, 0.5
    )
    return 1.0 - tail if value > 0.0 else tail


def _student_t_ppf(probability: float, degrees_freedom: int) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0,1)")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -_student_t_ppf(1.0 - probability, degrees_freedom)
    low, high = 0.0, 1.0
    while _student_t_cdf(high, degrees_freedom) < probability:
        high *= 2.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _student_t_cdf(midpoint, degrees_freedom) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


__all__ = [
    "CheckpointLoss",
    "CheckpointSelectionRule",
    "CoefficientDiagnostics",
    "DecisionAwareLossBreakdown",
    "DecisionAwareLossWeights",
    "EXP3Arm",
    "EXP3Objective",
    "EXP3Operator",
    "EXP3_ARMS",
    "EXP5Variant",
    "EquivalenceGateResult",
    "GenericDynamicMixtureMessage",
    "NearTieCascade",
    "OperatorDiagnostics",
    "RatioEquivalenceResult",
    "SelectedCheckpointPair",
    "aba_ratio_of_rate_ratios",
    "build_exp5_operator",
    "coefficient_diagnostics",
    "decision_aware_combined_loss",
    "evaluate_equivalence_gate",
    "exp5_operator_diagnostics",
    "expansive_jacobian_fraction",
    "near_tie_cascade",
    "operator_jump_p95",
    "paired_ratio_equivalence",
    "select_checkpoint_pair",
]
