"""Typed registry for the EXT--EXP2 controller-interface crossover.

The crossover treats descriptor domain, staging semantics and score map as
separate code factors.  This module deliberately contains no result values and
does not infer factors from a cell label: every cell is registered explicitly
and validated against the complete 2 x 2 x 2 Cartesian design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, Callable, Mapping, Protocol

import torch

from leo_pg.control.policy import normalized_ordinal_rank
from leo_pg.paper.snapshot import SnapshotOutput
from leo_pg.sim.state import ControlObservation, PolicyDescriptors, ServingAction


class DescriptorCode(str, Enum):
    """Controller-facing descriptor domain used by one crossover cell."""

    D0 = "D0"
    D1 = "D1"

    @property
    def level(self) -> str:
        return {
            DescriptorCode.D0: "snapshot_descriptor",
            DescriptorCode.D1: "intensity_flow_descriptor",
        }[self]


class StagingCode(str, Enum):
    """Timing/identity semantics used to populate the policy copy."""

    STAGE0 = "Stage0"
    STAGE1 = "Stage1"

    @property
    def level(self) -> str:
        return {
            StagingCode.STAGE0: "current_epoch_fields",
            StagingCode.STAGE1: "causal_staged_fields",
        }[self]


class ScoreMapCode(str, Enum):
    """Score-map factor used after the current hard mask."""

    S0 = "S0"
    S1 = "S1"

    @property
    def level(self) -> str:
        return {
            ScoreMapCode.S0: "static_weighted_score",
            ScoreMapCode.S1: "hazard_aware_score_map",
        }[self]


class CrossoverPhase(str, Enum):
    ORACLE_DRIVEN = "oracle_driven"
    FROZEN_PREDICTOR = "frozen_predictor"


@dataclass(frozen=True)
class CrossoverCell:
    """One of the eight code-level cells in EXT--EXP2."""

    cell_id: str
    descriptor: DescriptorCode
    staging: StagingCode
    score_map: ScoreMapCode
    selector_weights_id: str = "W_BALANCED_V1"
    selector_rule: str = "current_hard_mask_then_argmax_min_dwell_1s"

    def as_manifest_row(self) -> dict[str, str]:
        return {
            "cell_id": self.cell_id,
            "descriptor_code": self.descriptor.value,
            "descriptor_level": self.descriptor.level,
            "staging_code": self.staging.value,
            "staging_level": self.staging.level,
            "score_map_code": self.score_map.value,
            "score_map_level": self.score_map.level,
            "selector_weights_id": self.selector_weights_id,
            "selector_rule": self.selector_rule,
        }


EXP2_CELLS: tuple[CrossoverCell, ...] = (
    CrossoverCell("C1", DescriptorCode.D0, StagingCode.STAGE0, ScoreMapCode.S0),
    CrossoverCell("C2", DescriptorCode.D0, StagingCode.STAGE0, ScoreMapCode.S1),
    CrossoverCell("C3", DescriptorCode.D0, StagingCode.STAGE1, ScoreMapCode.S0),
    CrossoverCell("C4", DescriptorCode.D0, StagingCode.STAGE1, ScoreMapCode.S1),
    CrossoverCell("C5", DescriptorCode.D1, StagingCode.STAGE0, ScoreMapCode.S0),
    CrossoverCell("C6", DescriptorCode.D1, StagingCode.STAGE0, ScoreMapCode.S1),
    CrossoverCell("C7", DescriptorCode.D1, StagingCode.STAGE1, ScoreMapCode.S0),
    CrossoverCell("C8", DescriptorCode.D1, StagingCode.STAGE1, ScoreMapCode.S1),
)


def validate_exp2_cells(cells: tuple[CrossoverCell, ...] = EXP2_CELLS) -> None:
    """Reject missing, duplicated or semantically duplicated EXP2 cells."""

    if len(cells) != 8:
        raise ValueError("EXT--EXP2 requires exactly eight crossover cells")
    identifiers = [cell.cell_id for cell in cells]
    if identifiers != [f"C{index}" for index in range(1, 9)]:
        raise ValueError("EXT--EXP2 cell identifiers must be ordered C1 through C8")
    combinations = {
        (cell.descriptor, cell.staging, cell.score_map) for cell in cells
    }
    expected = set(product(DescriptorCode, StagingCode, ScoreMapCode))
    if combinations != expected:
        raise ValueError("EXT--EXP2 cells must cover the complete D x Stage x S design")


validate_exp2_cells()


def crossover_cell(cell_id: str) -> CrossoverCell:
    """Return an explicitly registered cell by its stable identifier."""

    normalized = str(cell_id).strip().upper()
    for cell in EXP2_CELLS:
        if cell.cell_id == normalized:
            return cell
    raise KeyError(f"unknown EXT--EXP2 cell {cell_id!r}; expected C1 through C8")


@dataclass(frozen=True)
class CrossoverRunSpec:
    """One phase x cell execution contract (seed blocks remain replication)."""

    phase: CrossoverPhase
    cell: CrossoverCell
    matched_seed_blocks: int = 10
    held_out_episodes_per_block: int = 30

    def __post_init__(self) -> None:
        if self.matched_seed_blocks != 10:
            raise ValueError("EXT--EXP2 uses ten matched seed blocks")
        if self.held_out_episodes_per_block != 30:
            raise ValueError("EXT--EXP2 uses 30 held-out episodes per block")

    @property
    def run_spec_id(self) -> str:
        return f"{self.phase.value}:{self.cell.cell_id}"

    def as_manifest_row(self) -> dict[str, str | int]:
        return {
            "experiment_id": "EXP2_CONTRACT_CROSSOVER",
            "experiment_configuration_version": "EXP2-v1",
            "phase": self.phase.value,
            **self.cell.as_manifest_row(),
            "matched_seed_blocks": self.matched_seed_blocks,
            "held_out_episodes_per_block": self.held_out_episodes_per_block,
        }


def register_exp2_runs() -> tuple[CrossoverRunSpec, ...]:
    """Return the frozen 2 phases x 8 cells execution registry."""

    return tuple(
        CrossoverRunSpec(phase=phase, cell=cell)
        for phase in CrossoverPhase
        for cell in EXP2_CELLS
    )


@dataclass(frozen=True)
class CrossoverRuntime:
    """Dependency-injected runtime for executing a registered crossover cell.

    The three callable tables make the manipulation auditable.  A descriptor
    builder cannot silently choose staging or score semantics, and a score map
    cannot change the hard-mask selector owned by the surrounding executor.
    """

    descriptor_builders: Mapping[DescriptorCode, Callable[[Any], Any]]
    staging_functions: Mapping[StagingCode, Callable[[Any, Any], Any]]
    score_functions: Mapping[ScoreMapCode, Callable[[Any], Any]]

    def __post_init__(self) -> None:
        required = (
            ("descriptor_builders", set(DescriptorCode), set(self.descriptor_builders)),
            ("staging_functions", set(StagingCode), set(self.staging_functions)),
            ("score_functions", set(ScoreMapCode), set(self.score_functions)),
        )
        for name, expected, observed in required:
            if observed != expected:
                raise ValueError(f"{name} must register exactly {sorted(x.value for x in expected)}")

    def evaluate(self, cell: CrossoverCell | str, context: Any, history: Any) -> Any:
        selected = crossover_cell(cell) if isinstance(cell, str) else cell
        descriptor_values = self.descriptor_builders[selected.descriptor](context)
        staged = self.staging_functions[selected.staging](descriptor_values, history)
        return self.score_functions[selected.score_map](staged)


# ---------------------------------------------------------------------------
# Executable, typed EXP2 epoch contract
# ---------------------------------------------------------------------------


class DescriptorFrameOrigin(str, Enum):
    """Provenance of a descriptor frame before it enters the policy copy."""

    CURRENT_EPOCH = "current_epoch"
    CAUSAL_PREDICTION = "causal_prediction"
    FIXED_INITIALIZER = "fixed_initializer"


def _validate_observation_id(name: str, value: tuple[int, int]) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or int(value[0]) != value[0]
        or int(value[1]) != value[1]
        or int(value[0]) < 0
        or int(value[1]) < 0
    ):
        raise ValueError(f"{name} must be a non-negative (episode_seed, epoch) tuple")


def _validate_candidate_ids(
    candidate_edge_ids: torch.Tensor,
    *,
    user_count: int,
    satellite_count: int,
    name: str,
) -> None:
    if candidate_edge_ids.ndim != 2 or candidate_edge_ids.size(1) != 2:
        raise ValueError(f"{name} must have shape [E,2]")
    if candidate_edge_ids.dtype != torch.long:
        raise ValueError(f"{name} must have dtype long")
    if candidate_edge_ids.numel() == 0:
        return
    users, satellites = candidate_edge_ids.unbind(dim=1)
    if int(users.min()) < 0 or int(users.max()) >= user_count:
        raise ValueError(f"{name} contains an invalid local user id")
    if int(satellites.min()) < 0 or int(satellites.max()) >= satellite_count:
        raise ValueError(f"{name} contains an invalid local satellite id")
    if torch.unique(candidate_edge_ids, dim=0).size(0) != candidate_edge_ids.size(0):
        raise ValueError(f"{name} contains duplicate (user, satellite) identities")


@dataclass(frozen=True)
class CrossoverDecisionContext:
    """The executor-owned graph/history copied into one EXP2 decision.

    The context intentionally contains no descriptor values.  This prevents a
    score adapter from silently reading simulator state outside the selected
    descriptor domain.
    """

    observation_id: tuple[int, int]
    candidate_edge_ids: torch.Tensor
    feasible_edge: torch.Tensor
    current_serving: torch.Tensor
    hold_steps: torch.Tensor
    user_order: torch.Tensor
    satellite_count: int

    @property
    def edge_count(self) -> int:
        return int(self.candidate_edge_ids.size(0))

    @property
    def user_count(self) -> int:
        return int(self.current_serving.numel())

    def validate(self) -> None:
        _validate_observation_id("observation_id", self.observation_id)
        if isinstance(self.satellite_count, bool) or int(self.satellite_count) != self.satellite_count:
            raise TypeError("satellite_count must be an integer")
        if self.satellite_count <= 0:
            raise ValueError("satellite_count must be positive")
        if self.current_serving.ndim != 1 or self.current_serving.numel() == 0:
            raise ValueError("current_serving must be a non-empty vector")
        if self.current_serving.dtype != torch.long:
            raise ValueError("current_serving must have dtype long")
        expected_users = self.user_count
        for name, value in (
            ("hold_steps", self.hold_steps),
            ("user_order", self.user_order),
        ):
            if value.shape != (expected_users,) or value.dtype != torch.long:
                raise ValueError(f"{name} must be a long vector of length {expected_users}")
        if torch.any(self.current_serving < -1) or torch.any(
            self.current_serving >= self.satellite_count
        ):
            raise ValueError("current_serving contains an invalid local satellite id")
        if torch.any(self.hold_steps < 0):
            raise ValueError("hold_steps must be non-negative")
        expected_order = torch.arange(expected_users, device=self.user_order.device)
        if not torch.equal(torch.sort(self.user_order).values, expected_order):
            raise ValueError("user_order must be a permutation of [0,user_count)")
        _validate_candidate_ids(
            self.candidate_edge_ids,
            user_count=expected_users,
            satellite_count=self.satellite_count,
            name="candidate_edge_ids",
        )
        if self.feasible_edge.shape != (self.edge_count,) or self.feasible_edge.dtype != torch.bool:
            raise ValueError("feasible_edge must be a bool vector in candidate-edge order")
        expected_device = self.candidate_edge_ids.device
        mismatched = [
            name
            for name, value in (
                ("feasible_edge", self.feasible_edge),
                ("current_serving", self.current_serving),
                ("hold_steps", self.hold_steps),
                ("user_order", self.user_order),
            )
            if value.device != expected_device
        ]
        if mismatched:
            raise ValueError("crossover context tensors must share a device: " + ", ".join(mismatched))

    @classmethod
    def from_observation(cls, observation: ControlObservation) -> "CrossoverDecisionContext":
        observation.validate()
        result = cls(
            observation_id=observation.observation_id,
            candidate_edge_ids=observation.candidate_edge_ids.clone(),
            feasible_edge=observation.sim_descriptors.feasible_edge.clone(),
            current_serving=observation.current_serving.clone(),
            hold_steps=observation.hold_steps.clone(),
            user_order=observation.user_order.clone(),
            satellite_count=observation.satellite_count,
        )
        result.validate()
        return result


@dataclass(frozen=True)
class SnapshotDescriptorFrame:
    """D0 values plus the candidate identities on which edge fields are stored."""

    candidate_edge_ids: torch.Tensor
    descriptors: SnapshotOutput
    produced_at: tuple[int, int]
    intended_for: tuple[int, int]
    origin: DescriptorFrameOrigin
    source_id: str

    @property
    def descriptor_code(self) -> DescriptorCode:
        return DescriptorCode.D0

    @property
    def edge_count(self) -> int:
        return int(self.candidate_edge_ids.size(0))

    def validate(self, *, user_count: int, satellite_count: int) -> None:
        _validate_observation_id("produced_at", self.produced_at)
        _validate_observation_id("intended_for", self.intended_for)
        if not isinstance(self.origin, DescriptorFrameOrigin):
            raise TypeError("origin must be a DescriptorFrameOrigin")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty provenance identifier")
        _validate_candidate_ids(
            self.candidate_edge_ids,
            user_count=user_count,
            satellite_count=satellite_count,
            name="Snapshot candidate_edge_ids",
        )
        self.descriptors.validate(self.edge_count, satellite_count)
        if self.descriptors.gamma_edge.device != self.candidate_edge_ids.device:
            raise ValueError("Snapshot frame tensors must share a device")


@dataclass(frozen=True)
class IntensityFlowDescriptorFrame:
    """D1 values plus the candidate identities on which edge fields are stored."""

    candidate_edge_ids: torch.Tensor
    descriptors: PolicyDescriptors
    produced_at: tuple[int, int]
    intended_for: tuple[int, int]
    origin: DescriptorFrameOrigin
    source_id: str

    @property
    def descriptor_code(self) -> DescriptorCode:
        return DescriptorCode.D1

    @property
    def edge_count(self) -> int:
        return int(self.candidate_edge_ids.size(0))

    def validate(self, *, user_count: int, satellite_count: int) -> None:
        _validate_observation_id("produced_at", self.produced_at)
        _validate_observation_id("intended_for", self.intended_for)
        if not isinstance(self.origin, DescriptorFrameOrigin):
            raise TypeError("origin must be a DescriptorFrameOrigin")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty provenance identifier")
        _validate_candidate_ids(
            self.candidate_edge_ids,
            user_count=user_count,
            satellite_count=satellite_count,
            name="Intensity--Flow candidate_edge_ids",
        )
        self.descriptors.validate(self.edge_count, satellite_count)
        if any(
            value.device != self.candidate_edge_ids.device
            for value in (
                self.descriptors.gamma_edge,
                self.descriptors.intensity_edge,
                self.descriptors.flow_node,
            )
        ):
            raise ValueError("Intensity--Flow frame tensors must share a device")


DescriptorFrame = SnapshotDescriptorFrame | IntensityFlowDescriptorFrame
DescriptorValues = SnapshotOutput | PolicyDescriptors


class SnapshotCrossoverInitializer(Protocol):
    initializer_id: str

    def __call__(
        self,
        context: CrossoverDecisionContext,
        missing_edge: torch.Tensor,
    ) -> SnapshotOutput: ...


class IntensityFlowCrossoverInitializer(Protocol):
    initializer_id: str

    def __call__(
        self,
        context: CrossoverDecisionContext,
        missing_edge: torch.Tensor,
    ) -> PolicyDescriptors: ...


def _finite_scalar(name: str, value: float, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class ConstantSnapshotCrossoverInitializer:
    """Explicit D0 initializer; constants are supplied by the frozen config."""

    gamma: float
    feasibility_margin: float
    admitted_load: float
    initializer_id: str = "constant_snapshot_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma", _finite_scalar("gamma", self.gamma))
        object.__setattr__(
            self,
            "feasibility_margin",
            _finite_scalar("feasibility_margin", self.feasibility_margin),
        )
        object.__setattr__(
            self,
            "admitted_load",
            _finite_scalar("admitted_load", self.admitted_load, nonnegative=True),
        )
        if not self.initializer_id.strip():
            raise ValueError("initializer_id must be non-empty")

    def __call__(
        self,
        context: CrossoverDecisionContext,
        missing_edge: torch.Tensor,
    ) -> SnapshotOutput:
        context.validate()
        if missing_edge.shape != (context.edge_count,) or missing_edge.dtype != torch.bool:
            raise ValueError("missing_edge must be a bool vector in current candidate order")
        device = context.candidate_edge_ids.device
        if missing_edge.device != device:
            raise ValueError("initializer mask and context must share a device")
        result = SnapshotOutput(
            gamma_edge=torch.full((context.edge_count,), self.gamma, device=device),
            feasibility_margin_edge=torch.full(
                (context.edge_count,), self.feasibility_margin, device=device
            ),
            admitted_load_node=torch.full(
                (context.satellite_count,), self.admitted_load, device=device
            ),
        )
        result.validate(context.edge_count, context.satellite_count)
        return result


@dataclass(frozen=True)
class ConstantIntensityFlowCrossoverInitializer:
    """Explicit D1 initializer; constants are supplied by the frozen config."""

    gamma: float
    intensity: float
    flow: float
    initializer_id: str = "configured_prior"

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma", _finite_scalar("gamma", self.gamma))
        object.__setattr__(
            self,
            "intensity",
            _finite_scalar("intensity", self.intensity, nonnegative=True),
        )
        object.__setattr__(self, "flow", _finite_scalar("flow", self.flow))
        if not self.initializer_id.strip():
            raise ValueError("initializer_id must be non-empty")

    def __call__(
        self,
        context: CrossoverDecisionContext,
        missing_edge: torch.Tensor,
    ) -> PolicyDescriptors:
        context.validate()
        if missing_edge.shape != (context.edge_count,) or missing_edge.dtype != torch.bool:
            raise ValueError("missing_edge must be a bool vector in current candidate order")
        device = context.candidate_edge_ids.device
        if missing_edge.device != device:
            raise ValueError("initializer mask and context must share a device")
        result = PolicyDescriptors(
            gamma_edge=torch.full((context.edge_count,), self.gamma, device=device),
            intensity_edge=torch.full((context.edge_count,), self.intensity, device=device),
            flow_node=torch.full((context.satellite_count,), self.flow, device=device),
        )
        result.validate(context.edge_count, context.satellite_count)
        return result


@dataclass(frozen=True)
class CrossoverEpochInput:
    """All typed descriptor sources available at one EXP2 decision epoch."""

    phase: CrossoverPhase
    context: CrossoverDecisionContext
    current_snapshot: SnapshotDescriptorFrame | None = None
    current_intensity_flow: IntensityFlowDescriptorFrame | None = None
    causal_snapshot: SnapshotDescriptorFrame | None = None
    causal_intensity_flow: IntensityFlowDescriptorFrame | None = None
    snapshot_initializer: SnapshotCrossoverInitializer | None = None
    intensity_flow_initializer: IntensityFlowCrossoverInitializer | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, CrossoverPhase):
            raise TypeError("phase must be a CrossoverPhase")
        self.context.validate()


@dataclass(frozen=True)
class StagedCrossoverDescriptors:
    """A selected descriptor frame aligned to the current candidate order."""

    descriptor_code: DescriptorCode
    values: DescriptorValues
    candidate_edge_ids: torch.Tensor
    initialized_edge: torch.Tensor
    staging_code: StagingCode
    source_id: str

    def validate(self, context: CrossoverDecisionContext) -> None:
        context.validate()
        if not isinstance(self.descriptor_code, DescriptorCode):
            raise TypeError("descriptor_code must be a DescriptorCode")
        if not isinstance(self.staging_code, StagingCode):
            raise TypeError("staging_code must be a StagingCode")
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not torch.equal(self.candidate_edge_ids, context.candidate_edge_ids):
            raise ValueError("staged candidate ids must equal the current candidate order")
        if self.initialized_edge.shape != (context.edge_count,) or self.initialized_edge.dtype != torch.bool:
            raise ValueError("initialized_edge must be a bool vector in current order")
        if self.initialized_edge.device != context.candidate_edge_ids.device:
            raise ValueError("staged tensors must share the context device")
        if self.descriptor_code is DescriptorCode.D0:
            if not isinstance(self.values, SnapshotOutput):
                raise TypeError("D0 must carry SnapshotOutput values")
        elif not isinstance(self.values, PolicyDescriptors):
            raise TypeError("D1 must carry PolicyDescriptors values")
        self.values.validate(context.edge_count, context.satellite_count)


def _selected_frame(epoch: CrossoverEpochInput, cell: CrossoverCell) -> DescriptorFrame:
    if cell.descriptor is DescriptorCode.D0:
        frame = (
            epoch.current_snapshot
            if cell.staging is StagingCode.STAGE0
            else epoch.causal_snapshot
        )
    else:
        frame = (
            epoch.current_intensity_flow
            if cell.staging is StagingCode.STAGE0
            else epoch.causal_intensity_flow
        )
    if frame is None:
        raise ValueError(
            f"{cell.cell_id} requires {cell.descriptor.value}/{cell.staging.value} "
            "descriptor values"
        )
    frame.validate(
        user_count=epoch.context.user_count,
        satellite_count=epoch.context.satellite_count,
    )
    if frame.candidate_edge_ids.device != epoch.context.candidate_edge_ids.device:
        raise ValueError("descriptor frame and current decision must share a device")
    return frame


def _validate_frame_timing(
    frame: DescriptorFrame,
    context: CrossoverDecisionContext,
    staging: StagingCode,
) -> None:
    current = context.observation_id
    if frame.intended_for != current:
        raise ValueError("descriptor frame intended_for does not match the current decision")
    if staging is StagingCode.STAGE0:
        if frame.origin is not DescriptorFrameOrigin.CURRENT_EPOCH:
            raise ValueError("Stage0 requires a current-epoch descriptor frame")
        if frame.produced_at != current:
            raise ValueError("Stage0 values must be produced at the current epoch")
        if not torch.equal(frame.candidate_edge_ids, context.candidate_edge_ids):
            raise ValueError("Stage0 descriptor values must use the current candidate order")
        return

    if frame.origin is DescriptorFrameOrigin.FIXED_INITIALIZER:
        if frame.produced_at != current:
            raise ValueError("a Stage1 fixed initializer must be installed at the current epoch")
        if not torch.equal(frame.candidate_edge_ids, context.candidate_edge_ids):
            raise ValueError("fixed initializer values must use the current candidate order")
        return
    if frame.origin is not DescriptorFrameOrigin.CAUSAL_PREDICTION:
        raise ValueError("Stage1 requires a causal prediction or explicit fixed initializer")
    seed, epoch = current
    if frame.produced_at != (seed, epoch - 1) or epoch <= 0:
        raise ValueError("Stage1 causal values must be produced exactly one epoch earlier")


def _persistent_join(
    source_ids: torch.Tensor,
    current_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source index per current edge and a persistent-edge mask."""

    lookup = {
        (int(user), int(satellite)): index
        for index, (user, satellite) in enumerate(source_ids.tolist())
    }
    source_index = torch.full(
        (current_ids.size(0),),
        -1,
        dtype=torch.long,
        device=current_ids.device,
    )
    for current_index, (user, satellite) in enumerate(current_ids.tolist()):
        previous_index = lookup.get((int(user), int(satellite)))
        if previous_index is not None:
            source_index[current_index] = previous_index
    return source_index, source_index >= 0


def stage_crossover_descriptors(
    epoch: CrossoverEpochInput,
    cell: CrossoverCell | str,
) -> StagedCrossoverDescriptors:
    """Apply Stage0/Stage1 without changing descriptor or score semantics."""

    selected = crossover_cell(cell) if isinstance(cell, str) else cell
    context = epoch.context
    frame = _selected_frame(epoch, selected)
    _validate_frame_timing(frame, context, selected.staging)

    if selected.staging is StagingCode.STAGE0 or frame.origin is DescriptorFrameOrigin.FIXED_INITIALIZER:
        initialized = torch.full(
            (context.edge_count,),
            frame.origin is DescriptorFrameOrigin.FIXED_INITIALIZER,
            dtype=torch.bool,
            device=context.candidate_edge_ids.device,
        )
        values = frame.descriptors.clone()
    else:
        source_index, persistent = _persistent_join(
            frame.candidate_edge_ids,
            context.candidate_edge_ids,
        )
        missing = ~persistent
        if isinstance(frame, SnapshotDescriptorFrame):
            initializer = epoch.snapshot_initializer
            if bool(missing.any()) and initializer is None:
                raise ValueError("new Stage1 Snapshot candidates require an explicit initializer")
            initialized_values = None if initializer is None else initializer(context, missing)
            gamma = torch.empty(
                context.edge_count,
                dtype=frame.descriptors.gamma_edge.dtype,
                device=frame.descriptors.gamma_edge.device,
            )
            margin = torch.empty_like(gamma)
            if bool(persistent.any()):
                source = source_index[persistent]
                gamma[persistent] = frame.descriptors.gamma_edge.index_select(0, source)
                margin[persistent] = frame.descriptors.feasibility_margin_edge.index_select(0, source)
            if bool(missing.any()):
                assert initialized_values is not None
                initialized_values.validate(context.edge_count, context.satellite_count)
                gamma[missing] = initialized_values.gamma_edge[missing]
                margin[missing] = initialized_values.feasibility_margin_edge[missing]
            values = SnapshotOutput(
                gamma_edge=gamma,
                feasibility_margin_edge=margin,
                admitted_load_node=frame.descriptors.admitted_load_node.clone(),
            )
        else:
            initializer = epoch.intensity_flow_initializer
            if bool(missing.any()) and initializer is None:
                raise ValueError("new Stage1 Intensity--Flow candidates require an explicit initializer")
            initialized_values = None if initializer is None else initializer(context, missing)
            gamma = torch.empty(
                context.edge_count,
                dtype=frame.descriptors.gamma_edge.dtype,
                device=frame.descriptors.gamma_edge.device,
            )
            intensity = torch.empty_like(gamma)
            if bool(persistent.any()):
                source = source_index[persistent]
                gamma[persistent] = frame.descriptors.gamma_edge.index_select(0, source)
                intensity[persistent] = frame.descriptors.intensity_edge.index_select(0, source)
            if bool(missing.any()):
                assert initialized_values is not None
                initialized_values.validate(context.edge_count, context.satellite_count)
                gamma[missing] = initialized_values.gamma_edge[missing]
                intensity[missing] = initialized_values.intensity_edge[missing]
            values = PolicyDescriptors(
                gamma_edge=gamma,
                intensity_edge=intensity,
                flow_node=frame.descriptors.flow_node.clone(),
            )
        initialized = missing

    result = StagedCrossoverDescriptors(
        descriptor_code=selected.descriptor,
        values=values,
        candidate_edge_ids=context.candidate_edge_ids.clone(),
        initialized_edge=initialized,
        staging_code=selected.staging,
        source_id=frame.source_id,
    )
    result.validate(context)
    return result


@dataclass(frozen=True)
class CrossoverCandidateScores:
    """Masked per-edge scores returned by one explicit D x S binding."""

    component_ranks: Mapping[str, torch.Tensor]
    total: torch.Tensor
    eligible: torch.Tensor
    binding_id: str
    evidence_basis: str

    def validate(self, context: CrossoverDecisionContext) -> None:
        context.validate()
        if self.total.shape != (context.edge_count,) or not self.total.is_floating_point():
            raise ValueError("total score must be a floating vector in candidate order")
        if self.eligible.shape != (context.edge_count,) or self.eligible.dtype != torch.bool:
            raise ValueError("eligible must be a bool vector in candidate order")
        if self.total.device != context.candidate_edge_ids.device or self.eligible.device != self.total.device:
            raise ValueError("score tensors and context must share a device")
        if not torch.equal(self.eligible, context.feasible_edge):
            raise ValueError("EXP2 bindings must apply the simulator hard mask before ranking")
        if bool(self.eligible.any()) and not torch.isfinite(self.total[self.eligible]).all():
            raise ValueError("eligible candidate scores must be finite")
        if bool((~self.eligible).any()) and not torch.isneginf(self.total[~self.eligible]).all():
            raise ValueError("ineligible candidate scores must be -inf")
        for name, value in self.component_ranks.items():
            if not isinstance(name, str) or not name:
                raise ValueError("component rank names must be non-empty strings")
            if value.shape != (context.edge_count,) or value.device != self.total.device:
                raise ValueError(f"component rank {name!r} has the wrong shape/device")
            if not torch.isfinite(value).all():
                raise ValueError(f"component rank {name!r} contains NaN or Inf")
        if not self.binding_id.strip() or not self.evidence_basis.strip():
            raise ValueError("score binding id and evidence basis must be non-empty")


class CrossoverScoreBinding(Protocol):
    descriptor_code: DescriptorCode
    score_map_code: ScoreMapCode
    binding_id: str
    evidence_basis: str

    def score(
        self,
        context: CrossoverDecisionContext,
        staged: StagedCrossoverDescriptors,
    ) -> CrossoverCandidateScores: ...


def _ordinal_reference_score(
    context: CrossoverDecisionContext,
    components: tuple[tuple[str, torch.Tensor, bool, bool, float], ...],
    *,
    binding_id: str,
    evidence_basis: str,
) -> CrossoverCandidateScores:
    """Score documented fields by normalized ordinal roles after hard masking.

    Each component tuple is ``(name, values, node_indexed, higher_is_better,
    weight)``.  Node-indexed values follow local satellite ids; other values
    follow candidate-edge order.
    """

    context.validate()
    edge_count = context.edge_count
    device = context.candidate_edge_ids.device
    dtype = next((value.dtype for _, value, _, _, _ in components if value.is_floating_point()), torch.float32)
    ranks = {
        name: torch.zeros(edge_count, dtype=dtype, device=device)
        for name, _, _, _, _ in components
    }
    total = torch.full((edge_count,), -torch.inf, dtype=dtype, device=device)
    edge_users = context.candidate_edge_ids[:, 0]
    edge_satellites = context.candidate_edge_ids[:, 1]

    for user in context.user_order.tolist():
        user_edges = torch.nonzero(edge_users == int(user), as_tuple=False).flatten()
        ranked_edges = user_edges[context.feasible_edge.index_select(0, user_edges)]
        if ranked_edges.numel() == 0:
            continue
        satellites = edge_satellites.index_select(0, ranked_edges)
        user_total = torch.zeros(ranked_edges.numel(), dtype=dtype, device=device)
        for name, values, node_indexed, higher_is_better, weight in components:
            selected_values = values.index_select(
                0, satellites if node_indexed else ranked_edges
            ).to(dtype=dtype)
            component_rank = normalized_ordinal_rank(
                selected_values,
                satellites,
                higher_is_better=higher_is_better,
            ).to(dtype=dtype)
            ranks[name][ranked_edges] = component_rank
            user_total = user_total + float(weight) * component_rank
        total[ranked_edges] = user_total

    result = CrossoverCandidateScores(
        component_ranks=ranks,
        total=total,
        eligible=context.feasible_edge.clone(),
        binding_id=binding_id,
        evidence_basis=evidence_basis,
    )
    result.validate(context)
    return result


@dataclass(frozen=True)
class SnapshotStaticReferenceBinding:
    """Documented native D0 x S0 Snapshot score equation."""

    descriptor_code: DescriptorCode = DescriptorCode.D0
    score_map_code: ScoreMapCode = ScoreMapCode.S0
    binding_id: str = "D0_S0_snapshot_static_equation_v1"
    evidence_basis: str = (
        "main Methods 'Platform contracts and score maps': Snapshot score uses "
        "gamma, signed current margin and admitted occupancy with W_BALANCED_V1"
    )
    gamma_weight: float = 1.0
    feasibility_weight: float = 0.5
    load_weight: float = 0.7

    def __post_init__(self) -> None:
        for name in ("gamma_weight", "feasibility_weight", "load_weight"):
            object.__setattr__(
                self,
                name,
                _finite_scalar(name, getattr(self, name), nonnegative=True),
            )
        if self.gamma_weight + self.feasibility_weight + self.load_weight <= 0:
            raise ValueError("at least one Snapshot score weight must be positive")

    def score(
        self,
        context: CrossoverDecisionContext,
        staged: StagedCrossoverDescriptors,
    ) -> CrossoverCandidateScores:
        staged.validate(context)
        if staged.descriptor_code is not self.descriptor_code or not isinstance(staged.values, SnapshotOutput):
            raise TypeError("Snapshot static binding requires staged D0 values")
        return _ordinal_reference_score(
            context,
            (
                ("gamma", staged.values.gamma_edge, False, True, self.gamma_weight),
                (
                    "feasibility_margin",
                    staged.values.feasibility_margin_edge,
                    False,
                    True,
                    self.feasibility_weight,
                ),
                (
                    "admitted_load",
                    staged.values.admitted_load_node,
                    True,
                    False,
                    self.load_weight,
                ),
            ),
            binding_id=self.binding_id,
            evidence_basis=self.evidence_basis,
        )


@dataclass(frozen=True)
class IntensityFlowHazardReferenceBinding:
    """Documented native D1 x S1 Intensity--Flow score equation."""

    descriptor_code: DescriptorCode = DescriptorCode.D1
    score_map_code: ScoreMapCode = ScoreMapCode.S1
    binding_id: str = "D1_S1_intensity_flow_hazard_equation_v1"
    evidence_basis: str = (
        "main Methods 'Platform contracts and score maps': Intensity--Flow "
        "score uses gamma, Flow and integrated violation Intensity with W_BALANCED_V1"
    )
    gamma_weight: float = 1.0
    flow_weight: float = 0.7
    intensity_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in ("gamma_weight", "flow_weight", "intensity_weight"):
            object.__setattr__(
                self,
                name,
                _finite_scalar(name, getattr(self, name), nonnegative=True),
            )
        if self.gamma_weight + self.flow_weight + self.intensity_weight <= 0:
            raise ValueError("at least one Intensity--Flow score weight must be positive")

    def score(
        self,
        context: CrossoverDecisionContext,
        staged: StagedCrossoverDescriptors,
    ) -> CrossoverCandidateScores:
        staged.validate(context)
        if staged.descriptor_code is not self.descriptor_code or not isinstance(staged.values, PolicyDescriptors):
            raise TypeError("Intensity--Flow hazard binding requires staged D1 values")
        return _ordinal_reference_score(
            context,
            (
                ("gamma", staged.values.gamma_edge, False, True, self.gamma_weight),
                ("flow", staged.values.flow_node, True, False, self.flow_weight),
                (
                    "intensity",
                    staged.values.intensity_edge,
                    False,
                    False,
                    self.intensity_weight,
                ),
            ),
            binding_id=self.binding_id,
            evidence_basis=self.evidence_basis,
        )


RawScoreFunction = Callable[
    [CrossoverDecisionContext, StagedCrossoverDescriptors], torch.Tensor
]


@dataclass(frozen=True)
class ExplicitCrossoverScoreAdapter:
    """Author-supplied D x S mapping for a pairing absent from the archive.

    The callable returns one finite raw score per current candidate.  This
    wrapper applies the simulator-owned hard mask and records the mapping's
    provenance.  Merely labelling a callable ``S1`` is therefore insufficient:
    both a stable mapping id and its evidence basis are mandatory.
    """

    descriptor_code: DescriptorCode
    score_map_code: ScoreMapCode
    binding_id: str
    evidence_basis: str
    score_function: RawScoreFunction

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor_code, DescriptorCode):
            raise TypeError("descriptor_code must be a DescriptorCode")
        if not isinstance(self.score_map_code, ScoreMapCode):
            raise TypeError("score_map_code must be a ScoreMapCode")
        if not self.binding_id.strip() or not self.evidence_basis.strip():
            raise ValueError("explicit score adapter requires id and evidence basis")
        if not callable(self.score_function):
            raise TypeError("score_function must be callable")

    def score(
        self,
        context: CrossoverDecisionContext,
        staged: StagedCrossoverDescriptors,
    ) -> CrossoverCandidateScores:
        staged.validate(context)
        if staged.descriptor_code is not self.descriptor_code:
            raise TypeError("explicit score adapter received the wrong descriptor domain")
        raw = self.score_function(context, staged)
        if not isinstance(raw, torch.Tensor):
            raise TypeError("explicit score_function must return a torch.Tensor")
        if raw.shape != (context.edge_count,) or not raw.is_floating_point():
            raise ValueError("explicit score_function must return a floating [E] vector")
        if raw.device != context.candidate_edge_ids.device or not torch.isfinite(raw).all():
            raise ValueError("explicit raw scores must be finite and share the context device")
        total = raw.clone()
        total[~context.feasible_edge] = -torch.inf
        result = CrossoverCandidateScores(
            component_ranks={},
            total=total,
            eligible=context.feasible_edge.clone(),
            binding_id=self.binding_id,
            evidence_basis=self.evidence_basis,
        )
        result.validate(context)
        return result


class MissingScoreBindingError(RuntimeError):
    """Raised instead of synthesizing an unrecorded EXP2 field mapping."""


@dataclass(frozen=True)
class CrossoverBindingRegistry:
    """Exact lookup from descriptor/score codes to executable bindings."""

    bindings: Mapping[tuple[DescriptorCode, ScoreMapCode], CrossoverScoreBinding]

    def __post_init__(self) -> None:
        copied = dict(self.bindings)
        for key, binding in copied.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], DescriptorCode)
                or not isinstance(key[1], ScoreMapCode)
            ):
                raise TypeError("score binding keys must be (DescriptorCode, ScoreMapCode)")
            if key != (binding.descriptor_code, binding.score_map_code):
                raise ValueError("score binding key does not match binding metadata")
        object.__setattr__(self, "bindings", copied)

    def resolve(
        self,
        descriptor: DescriptorCode,
        score_map: ScoreMapCode,
    ) -> CrossoverScoreBinding:
        binding = self.bindings.get((descriptor, score_map))
        if binding is None:
            detail = ""
            if descriptor is DescriptorCode.D0 and score_map is ScoreMapCode.S1:
                detail = (
                    " The v4 EXP2 source records this S1 x D0 cell but not the "
                    "hazard field read by S1."
                )
            raise MissingScoreBindingError(
                f"no documented {descriptor.value} x {score_map.value} score binding."
                f"{detail} Supply ExplicitCrossoverScoreAdapter with author provenance."
            )
        return binding

    @property
    def missing_pairs(self) -> tuple[tuple[DescriptorCode, ScoreMapCode], ...]:
        return tuple(
            pair
            for pair in product(DescriptorCode, ScoreMapCode)
            if pair not in self.bindings
        )


def documented_reference_bindings(
    *explicit_adapters: ExplicitCrossoverScoreAdapter,
) -> CrossoverBindingRegistry:
    """Return native equations plus any explicitly evidenced crossover maps."""

    native: tuple[CrossoverScoreBinding, ...] = (
        SnapshotStaticReferenceBinding(),
        IntensityFlowHazardReferenceBinding(),
    )
    bindings: dict[tuple[DescriptorCode, ScoreMapCode], CrossoverScoreBinding] = {
        (binding.descriptor_code, binding.score_map_code): binding
        for binding in native
    }
    for adapter in explicit_adapters:
        key = (adapter.descriptor_code, adapter.score_map_code)
        if key in bindings:
            raise ValueError(f"duplicate score binding for {key[0].value} x {key[1].value}")
        bindings[key] = adapter
    return CrossoverBindingRegistry(bindings)


def _best_scored_edge(
    edge_indices: torch.Tensor,
    total: torch.Tensor,
    satellite_ids: torch.Tensor,
) -> int:
    satellite_order = torch.argsort(
        satellite_ids.index_select(0, edge_indices), stable=True
    )
    by_satellite = edge_indices.index_select(0, satellite_order)
    score_order = torch.argsort(
        total.index_select(0, by_satellite), descending=True, stable=True
    )
    return int(by_satellite[score_order[0]].item())


def select_crossover_action(
    context: CrossoverDecisionContext,
    scores: CrossoverCandidateScores,
    *,
    min_dwell_steps: int = 10,
    hysteresis: float = 1.0 / 6.0,
) -> ServingAction:
    """Apply the shared dwell, hysteresis and identifier tie convention."""

    context.validate()
    scores.validate(context)
    if isinstance(min_dwell_steps, bool) or int(min_dwell_steps) != min_dwell_steps or min_dwell_steps < 0:
        raise ValueError("min_dwell_steps must be a non-negative integer")
    hysteresis = _finite_scalar("hysteresis", hysteresis, nonnegative=True)
    requested = context.current_serving.clone()
    edge_users = context.candidate_edge_ids[:, 0]
    edge_satellites = context.candidate_edge_ids[:, 1]

    for user in context.user_order.tolist():
        user_edges = torch.nonzero(edge_users == int(user), as_tuple=False).flatten()
        ranked_edges = user_edges[scores.eligible.index_select(0, user_edges)]
        if ranked_edges.numel() == 0:
            # Match the released fixed-controller/executor contract: preserve
            # the current request and let the authoritative transition gate it.
            continue
        best_edge = _best_scored_edge(ranked_edges, scores.total, edge_satellites)
        best_satellite = int(edge_satellites[best_edge].item())
        current = int(context.current_serving[user].item())
        if current < 0:
            requested[user] = best_satellite
            continue
        current_edges = user_edges[
            edge_satellites.index_select(0, user_edges) == current
        ]
        if current_edges.numel() != 1 or not bool(scores.eligible[current_edges[0]]):
            continue
        if best_satellite == current or int(context.hold_steps[user]) < min_dwell_steps:
            continue
        current_edge = int(current_edges[0].item())
        best_score = float(scores.total[best_edge])
        required = float(scores.total[current_edge]) + hysteresis
        if best_score >= required or math.isclose(
            best_score, required, rel_tol=1e-7, abs_tol=1e-8
        ):
            requested[user] = best_satellite

    result = ServingAction(
        observation_id=context.observation_id,
        requested_serving=requested,
    )
    result.validate(context.user_count, context.satellite_count)
    return result


@dataclass(frozen=True)
class CrossoverEpochResult:
    phase: CrossoverPhase
    cell: CrossoverCell
    staged: StagedCrossoverDescriptors
    scores: CrossoverCandidateScores
    action: ServingAction


class CrossoverRunner:
    """Execute one registered EXP2 cell against typed epoch inputs.

    This runner is intentionally outcome-free.  A surrounding environment owns
    transitions and metrics; this class owns only the registered D/Stage/S
    manipulation and the common proposal rule.
    """

    def __init__(
        self,
        bindings: CrossoverBindingRegistry | None = None,
        *,
        min_dwell_steps: int = 10,
        hysteresis: float = 1.0 / 6.0,
    ) -> None:
        self.bindings = bindings or documented_reference_bindings()
        self.min_dwell_steps = min_dwell_steps
        self.hysteresis = hysteresis

    def run_epoch(
        self,
        cell: CrossoverCell | str,
        epoch: CrossoverEpochInput,
    ) -> CrossoverEpochResult:
        selected = crossover_cell(cell) if isinstance(cell, str) else cell
        staged = stage_crossover_descriptors(epoch, selected)
        binding = self.bindings.resolve(selected.descriptor, selected.score_map)
        scores = binding.score(epoch.context, staged)
        action = select_crossover_action(
            epoch.context,
            scores,
            min_dwell_steps=self.min_dwell_steps,
            hysteresis=self.hysteresis,
        )
        return CrossoverEpochResult(
            phase=epoch.phase,
            cell=selected,
            staged=staged,
            scores=scores,
            action=action,
        )


__all__ = [
    "ConstantIntensityFlowCrossoverInitializer",
    "ConstantSnapshotCrossoverInitializer",
    "CrossoverBindingRegistry",
    "CrossoverCandidateScores",
    "CrossoverCell",
    "CrossoverDecisionContext",
    "CrossoverEpochInput",
    "CrossoverEpochResult",
    "CrossoverPhase",
    "CrossoverRunSpec",
    "CrossoverRunner",
    "CrossoverRuntime",
    "DescriptorFrameOrigin",
    "DescriptorCode",
    "EXP2_CELLS",
    "ExplicitCrossoverScoreAdapter",
    "IntensityFlowDescriptorFrame",
    "IntensityFlowHazardReferenceBinding",
    "MissingScoreBindingError",
    "ScoreMapCode",
    "SnapshotDescriptorFrame",
    "SnapshotStaticReferenceBinding",
    "StagedCrossoverDescriptors",
    "StagingCode",
    "crossover_cell",
    "documented_reference_bindings",
    "register_exp2_runs",
    "select_crossover_action",
    "stage_crossover_descriptors",
    "validate_exp2_cells",
]
