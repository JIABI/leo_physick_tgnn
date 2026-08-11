"""Classical reassociation controllers and auditable parameter sweeps.

The controllers in this module share the paper environment's candidate graph,
stable local satellite identifiers, dwell accounting, and simulator-authority
transition.  They only select requested associations; feasibility and capacity
remain the responsibility of :class:`~leo_pg.sim.PaperAlignedLEOEnv`.

The manuscript fixes the A3 offset/TTT grid but does not uniquely specify every
CHO or load-aware hyperparameter.  Those values are therefore carried with an
explicit provenance label in every sweep specification instead of being hidden
inside the implementation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Protocol

import torch

from leo_pg.sim.state import ControlObservation, ServingAction


class ReassociationController(Protocol):
    """Minimal controller contract consumed by the paper evaluation runner."""

    def reset(self) -> None: ...

    def select_action(self, observation: ControlObservation) -> ServingAction: ...


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _candidate_rows(observation: ControlObservation, user: int) -> torch.Tensor:
    return torch.nonzero(
        observation.candidate_edge_ids[:, 0] == user,
        as_tuple=False,
    ).flatten()


def _edge_for_satellite(
    observation: ControlObservation,
    rows: torch.Tensor,
    satellite: int,
) -> int | None:
    if rows.numel() == 0:
        return None
    matches = rows[
        observation.candidate_edge_ids.index_select(0, rows)[:, 1] == int(satellite)
    ]
    if matches.numel() == 0:
        return None
    if matches.numel() != 1:
        raise ValueError("candidate graph contains a duplicate user-satellite edge")
    return int(matches[0].item())


def _best_by_value(
    observation: ControlObservation,
    rows: torch.Tensor,
    values: torch.Tensor,
) -> tuple[int, int, float] | None:
    """Return ``(edge, satellite, value)`` with deterministic sat-id ties."""

    if rows.numel() == 0:
        return None
    satellites = observation.candidate_edge_ids.index_select(0, rows)[:, 1]
    satellite_order = torch.argsort(satellites, stable=True)
    ordered_rows = rows.index_select(0, satellite_order)
    value_order = torch.argsort(
        values.index_select(0, ordered_rows),
        descending=True,
        stable=True,
    )
    edge = int(ordered_rows[value_order[0]].item())
    satellite = int(observation.candidate_edge_ids[edge, 1].item())
    return edge, satellite, float(values[edge].item())


def _action(observation: ControlObservation, requested: torch.Tensor) -> ServingAction:
    result = ServingAction(
        observation_id=observation.observation_id,
        requested_serving=requested,
    )
    result.validate(observation.user_count, observation.satellite_count)
    return result


@dataclass(frozen=True)
class A3Config:
    """Stateful 3GPP-style event-A3 comparator.

    ``ttt_steps`` is expressed in decision epochs.  The manuscript sweep uses
    ``{1, 3, 5, 8}`` without a recoverable seconds/epoch declaration, so the
    publication runner records it as a step-domain parameter.
    """

    offset_db: float = 3.0
    ttt_steps: int | None = 3
    ttt_seconds: float | None = None
    dt_ctrl: float = 0.1
    min_dwell_steps: int = 10

    def __post_init__(self) -> None:
        _finite_nonnegative("offset_db", self.offset_db)
        if (self.ttt_steps is None) == (self.ttt_seconds is None):
            raise ValueError("specify exactly one of ttt_steps or ttt_seconds")
        if self.ttt_steps is not None:
            _positive_int("ttt_steps", self.ttt_steps)
        if self.ttt_seconds is not None:
            if not math.isfinite(float(self.ttt_seconds)) or float(self.ttt_seconds) <= 0:
                raise ValueError("ttt_seconds must be finite and positive")
        if not math.isfinite(float(self.dt_ctrl)) or float(self.dt_ctrl) <= 0:
            raise ValueError("dt_ctrl must be finite and positive")
        if isinstance(self.min_dwell_steps, bool) or int(self.min_dwell_steps) != self.min_dwell_steps:
            raise ValueError("min_dwell_steps must be a non-negative integer")
        if int(self.min_dwell_steps) < 0:
            raise ValueError("min_dwell_steps must be a non-negative integer")

    @property
    def resolved_ttt_steps(self) -> int:
        if self.ttt_steps is not None:
            return int(self.ttt_steps)
        assert self.ttt_seconds is not None
        return int(math.ceil(float(self.ttt_seconds) / float(self.dt_ctrl)))


class A3Controller:
    """A3 offset plus time-to-trigger controller on simulator SINR."""

    def __init__(self, config: A3Config | None = None) -> None:
        self.config = config or A3Config()
        self._candidate: torch.Tensor | None = None
        self._streak: torch.Tensor | None = None

    def reset(self) -> None:
        self._candidate = None
        self._streak = None

    def _state(self, observation: ControlObservation) -> tuple[torch.Tensor, torch.Tensor]:
        if self._candidate is None or self._candidate.numel() != observation.user_count:
            device = observation.current_serving.device
            self._candidate = torch.full(
                (observation.user_count,), -1, dtype=torch.long, device=device
            )
            self._streak = torch.zeros(
                observation.user_count, dtype=torch.long, device=device
            )
        assert self._streak is not None
        return self._candidate, self._streak

    def select_action(self, observation: ControlObservation) -> ServingAction:
        observation.validate()
        candidate, streak = self._state(observation)
        requested = observation.current_serving.clone()
        gamma = observation.sim_descriptors.policy_fields.gamma_edge

        for user in observation.user_order.tolist():
            rows = _candidate_rows(observation, user)
            best = _best_by_value(observation, rows, gamma)
            current = int(observation.current_serving[user].item())
            if best is None:
                candidate[user] = -1
                streak[user] = 0
                continue
            _, best_satellite, best_gamma = best
            if current < 0:
                requested[user] = best_satellite
                candidate[user] = -1
                streak[user] = 0
                continue
            current_edge = _edge_for_satellite(observation, rows, current)
            if current_edge is None or best_satellite == current:
                candidate[user] = -1
                streak[user] = 0
                continue
            condition = best_gamma >= float(gamma[current_edge].item()) + self.config.offset_db
            if not condition or int(observation.hold_steps[user].item()) < self.config.min_dwell_steps:
                candidate[user] = -1
                streak[user] = 0
                continue
            if int(candidate[user].item()) == best_satellite:
                streak[user] += 1
            else:
                candidate[user] = best_satellite
                streak[user] = 1
            if int(streak[user].item()) >= self.config.resolved_ttt_steps:
                requested[user] = best_satellite
                candidate[user] = -1
                streak[user] = 0
        return _action(observation, requested)

    def __call__(self, observation: ControlObservation) -> ServingAction:
        return self.select_action(observation)


@dataclass(frozen=True)
class CHOConfig:
    """Conditional-handover preparation and execution parameters.

    The manuscript names CHO but does not expose a complete preparation and
    validity contract.  Defaults are consequently labelled ``estimated`` by
    :func:`default_controller_sweeps` and remain user-overridable.
    """

    preparation_offset_db: float = 3.0
    preparation_steps: int = 2
    execution_offset_db: float = 0.0
    validity_steps: int = 10
    min_dwell_steps: int = 10

    def __post_init__(self) -> None:
        _finite_nonnegative("preparation_offset_db", self.preparation_offset_db)
        _positive_int("preparation_steps", self.preparation_steps)
        _finite_nonnegative("execution_offset_db", self.execution_offset_db)
        _positive_int("validity_steps", self.validity_steps)
        if isinstance(self.min_dwell_steps, bool) or int(self.min_dwell_steps) != self.min_dwell_steps:
            raise ValueError("min_dwell_steps must be a non-negative integer")
        if int(self.min_dwell_steps) < 0:
            raise ValueError("min_dwell_steps must be a non-negative integer")


class CHOController:
    """Stateful conditional-handover controller with an expiring target."""

    def __init__(self, config: CHOConfig | None = None) -> None:
        self.config = config or CHOConfig()
        self._pre_candidate: torch.Tensor | None = None
        self._pre_streak: torch.Tensor | None = None
        self._prepared: torch.Tensor | None = None
        self._expiry: torch.Tensor | None = None

    def reset(self) -> None:
        self._pre_candidate = None
        self._pre_streak = None
        self._prepared = None
        self._expiry = None

    def _state(self, observation: ControlObservation) -> tuple[torch.Tensor, ...]:
        if self._prepared is None or self._prepared.numel() != observation.user_count:
            device = observation.current_serving.device
            shape = (observation.user_count,)
            self._pre_candidate = torch.full(shape, -1, dtype=torch.long, device=device)
            self._pre_streak = torch.zeros(shape, dtype=torch.long, device=device)
            self._prepared = torch.full(shape, -1, dtype=torch.long, device=device)
            self._expiry = torch.full(shape, -1, dtype=torch.long, device=device)
        assert self._pre_candidate is not None
        assert self._pre_streak is not None
        assert self._prepared is not None
        assert self._expiry is not None
        return self._pre_candidate, self._pre_streak, self._prepared, self._expiry

    def select_action(self, observation: ControlObservation) -> ServingAction:
        observation.validate()
        pre_candidate, pre_streak, prepared, expiry = self._state(observation)
        requested = observation.current_serving.clone()
        gamma = observation.sim_descriptors.policy_fields.gamma_edge

        for user in observation.user_order.tolist():
            rows = _candidate_rows(observation, user)
            current = int(observation.current_serving[user].item())
            best = _best_by_value(observation, rows, gamma)
            if best is None:
                pre_candidate[user] = -1
                pre_streak[user] = 0
                prepared[user] = -1
                expiry[user] = -1
                continue
            _, best_satellite, best_gamma = best
            if current < 0:
                requested[user] = best_satellite
                pre_candidate[user] = -1
                pre_streak[user] = 0
                prepared[user] = -1
                expiry[user] = -1
                continue
            current_edge = _edge_for_satellite(observation, rows, current)
            if current_edge is None:
                continue
            current_gamma = float(gamma[current_edge].item())

            prepared_satellite = int(prepared[user].item())
            if prepared_satellite >= 0 and observation.epoch <= int(expiry[user].item()):
                prepared_edge = _edge_for_satellite(observation, rows, prepared_satellite)
                if (
                    prepared_edge is not None
                    and prepared_satellite != current
                    and int(observation.hold_steps[user].item()) >= self.config.min_dwell_steps
                    and float(gamma[prepared_edge].item())
                    >= current_gamma + self.config.execution_offset_db
                ):
                    requested[user] = prepared_satellite
                    prepared[user] = -1
                    expiry[user] = -1
                    pre_candidate[user] = -1
                    pre_streak[user] = 0
                    continue
            elif prepared_satellite >= 0:
                prepared[user] = -1
                expiry[user] = -1

            prepare = (
                best_satellite != current
                and best_gamma >= current_gamma + self.config.preparation_offset_db
            )
            if not prepare:
                pre_candidate[user] = -1
                pre_streak[user] = 0
                continue
            if int(pre_candidate[user].item()) == best_satellite:
                pre_streak[user] += 1
            else:
                pre_candidate[user] = best_satellite
                pre_streak[user] = 1
            if int(pre_streak[user].item()) >= self.config.preparation_steps:
                prepared[user] = best_satellite
                expiry[user] = observation.epoch + self.config.validity_steps
                pre_candidate[user] = -1
                pre_streak[user] = 0
        return _action(observation, requested)

    def __call__(self, observation: ControlObservation) -> ServingAction:
        return self.select_action(observation)


@dataclass(frozen=True)
class LoadAwareGreedyConfig:
    """SINR minus destination-load penalty greedy controller."""

    load_penalty_db: float = 6.0
    hysteresis_db: float = 3.0
    min_dwell_steps: int = 10

    def __post_init__(self) -> None:
        _finite_nonnegative("load_penalty_db", self.load_penalty_db)
        _finite_nonnegative("hysteresis_db", self.hysteresis_db)
        if isinstance(self.min_dwell_steps, bool) or int(self.min_dwell_steps) != self.min_dwell_steps:
            raise ValueError("min_dwell_steps must be a non-negative integer")
        if int(self.min_dwell_steps) < 0:
            raise ValueError("min_dwell_steps must be a non-negative integer")


class LoadAwareGreedyController:
    """Greedy score ``gamma - load_penalty_db * destination_flow``."""

    def __init__(self, config: LoadAwareGreedyConfig | None = None) -> None:
        self.config = config or LoadAwareGreedyConfig()

    def reset(self) -> None:
        return None

    def select_action(self, observation: ControlObservation) -> ServingAction:
        observation.validate()
        requested = observation.current_serving.clone()
        fields = observation.sim_descriptors.policy_fields
        satellites = observation.candidate_edge_ids[:, 1]
        utility = fields.gamma_edge - self.config.load_penalty_db * fields.flow_node[satellites]

        for user in observation.user_order.tolist():
            rows = _candidate_rows(observation, user)
            best = _best_by_value(observation, rows, utility)
            current = int(observation.current_serving[user].item())
            if best is None:
                continue
            _, best_satellite, best_utility = best
            if current < 0:
                requested[user] = best_satellite
                continue
            current_edge = _edge_for_satellite(observation, rows, current)
            if current_edge is None or best_satellite == current:
                continue
            if int(observation.hold_steps[user].item()) < self.config.min_dwell_steps:
                continue
            if best_utility >= float(utility[current_edge].item()) + self.config.hysteresis_db:
                requested[user] = best_satellite
        return _action(observation, requested)

    def __call__(self, observation: ControlObservation) -> ServingAction:
        return self.select_action(observation)


@dataclass(frozen=True)
class SweepPoint:
    controller: str
    parameters: Mapping[str, Any]
    provenance: str
    rationale: str
    selected: bool = False
    grid: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "controller": self.controller,
            "parameters": dict(self.parameters),
            "provenance": self.provenance,
            "rationale": self.rationale,
            "selected": bool(self.selected),
        }
        if self.grid is not None:
            result["grid"] = dict(self.grid)
        return result


def default_controller_sweeps() -> tuple[SweepPoint, ...]:
    """Return the frozen sweep registry without executing any experiment."""

    points: list[SweepPoint] = []
    for offset in (1.0, 3.0, 5.0, 7.0):
        for ttt in (1, 3, 5, 8):
            cfg = A3Config(
                offset_db=offset,
                ttt_steps=ttt,
                ttt_seconds=None,
                dt_ctrl=0.1,
                min_dwell_steps=10,
            )
            points.append(
                SweepPoint(
                    controller="a3",
                    parameters=asdict(cfg),
                    provenance="manuscript_grid",
                    rationale=(
                        "A3 offset/TTT grid reported in the manuscript table. The "
                        "table is unitless while prose defines seconds; this primary "
                        "manifest uses control steps and the API also supports an "
                        "explicit ttt_seconds conversion."
                    ),
                    selected=(offset == 3.0 and ttt == 3),
                )
            )
    for preparation_offset in (1.0, 3.0, 5.0):
        for preparation in (1, 2, 3):
            for validity in (5, 10, 20):
                cfg = CHOConfig(
                    preparation_offset_db=preparation_offset,
                    preparation_steps=preparation,
                    execution_offset_db=0.0,
                    validity_steps=validity,
                    min_dwell_steps=10,
                )
                points.append(
                    SweepPoint(
                        controller="cho",
                        parameters=asdict(cfg),
                        provenance="estimated_grid",
                        rationale=(
                            "The manuscript names CHO but does not uniquely report its "
                            "preparation offset/timing; this grid spans all frozen "
                            "offset, preparation, and validity values."
                        ),
                    )
                )
    for penalty in (2.0, 4.0, 6.0, 8.0):
        for hysteresis in (1.0, 3.0, 5.0):
            cfg = LoadAwareGreedyConfig(
                load_penalty_db=penalty,
                hysteresis_db=hysteresis,
                min_dwell_steps=10,
            )
            points.append(
                SweepPoint(
                    controller="load_aware_greedy",
                    parameters=asdict(cfg),
                    provenance="estimated_grid",
                    rationale=(
                        "The manuscript does not expose the load-to-dB conversion; "
                        "2-8 dB per unit flow brackets a moderate penalty while the "
                        "1-5 dB hysteresis grid matches the A3 scale."
                    ),
                )
            )
    return tuple(points)


def _grid_mapping(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"controller_sweeps.{name} must be a mapping")
    return dict(value)


def _grid_values(config: Mapping[str, Any], name: str) -> tuple[Any, ...]:
    value = config.get(name)
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"controller sweep field {name!r} must be a non-empty list")
    return tuple(value)


def expand_controller_sweep_config(
    config: Mapping[str, Any],
) -> tuple[SweepPoint, ...]:
    """Expand the top-level frozen YAML grids into auditable sweep points.

    The expansion is deliberately explicit: A3 is ``4 x 4``, CHO is
    ``3 x 3 x 3``, and load-aware greedy is ``4 x 3`` for the released config.
    Alternative non-empty lists remain supported and are recorded verbatim in
    every point's grid metadata.
    """

    if not isinstance(config, Mapping):
        raise TypeError("controller_sweeps must be a mapping")
    points: list[SweepPoint] = []

    a3 = _grid_mapping(config, "a3")
    a3_offsets = _grid_values(a3, "offsets_db")
    a3_ttt = _grid_values(a3, "ttt_steps")
    a3_selected = a3.get("selected", {})
    if not isinstance(a3_selected, Mapping):
        raise TypeError("controller_sweeps.a3.selected must be a mapping")
    for offset in a3_offsets:
        for ttt in a3_ttt:
            parameters: dict[str, Any] = {
                "offset_db": offset,
                "ttt_steps": ttt,
            }
            for optional in ("dt_ctrl", "min_dwell_steps"):
                if optional in a3:
                    parameters[optional] = a3[optional]
            selected = bool(a3_selected) and all(
                parameters.get(key) == value for key, value in a3_selected.items()
            )
            points.append(
                SweepPoint(
                    controller="a3",
                    parameters=parameters,
                    provenance=str(a3.get("provenance", "unspecified")),
                    rationale=(
                        "Cartesian expansion of controller_sweeps.a3 offsets_db "
                        "and ttt_steps."
                    ),
                    selected=selected,
                    grid={
                        "offsets_db": list(a3_offsets),
                        "ttt_steps": list(a3_ttt),
                        "declared_selected": dict(a3_selected),
                    },
                )
            )

    cho = _grid_mapping(config, "cho")
    cho_offsets = _grid_values(cho, "preparation_offset_db")
    cho_preparation = _grid_values(cho, "preparation_steps")
    cho_validity = _grid_values(cho, "validity_steps")
    for offset in cho_offsets:
        for preparation in cho_preparation:
            for validity in cho_validity:
                parameters = {
                    "preparation_offset_db": offset,
                    "preparation_steps": preparation,
                    "execution_offset_db": cho.get("execution_offset_db", 0.0),
                    "validity_steps": validity,
                }
                if "min_dwell_steps" in cho:
                    parameters["min_dwell_steps"] = cho["min_dwell_steps"]
                points.append(
                    SweepPoint(
                        controller="cho",
                        parameters=parameters,
                        provenance=str(cho.get("provenance", "unspecified")),
                        rationale=(
                            "Cartesian expansion of controller_sweeps.cho "
                            "preparation offset, preparation steps, and validity steps."
                        ),
                        grid={
                            "preparation_offset_db": list(cho_offsets),
                            "preparation_steps": list(cho_preparation),
                            "validity_steps": list(cho_validity),
                            "execution_offset_db": cho.get("execution_offset_db", 0.0),
                        },
                    )
                )

    load = _grid_mapping(config, "load_aware_greedy")
    load_penalties = _grid_values(load, "load_penalty_db")
    load_hysteresis = _grid_values(load, "hysteresis_db")
    for penalty in load_penalties:
        for hysteresis in load_hysteresis:
            parameters = {
                "load_penalty_db": penalty,
                "hysteresis_db": hysteresis,
            }
            if "min_dwell_steps" in load:
                parameters["min_dwell_steps"] = load["min_dwell_steps"]
            points.append(
                SweepPoint(
                    controller="load_aware_greedy",
                    parameters=parameters,
                    provenance=str(load.get("provenance", "unspecified")),
                    rationale=(
                        "Cartesian expansion of controller_sweeps.load_aware_greedy "
                        "load penalty and hysteresis."
                    ),
                    grid={
                        "load_penalty_db": list(load_penalties),
                        "hysteresis_db": list(load_hysteresis),
                    },
                )
            )
    return tuple(points)


def build_controller(name: str, parameters: Mapping[str, Any]) -> ReassociationController:
    """Construct a controller from a manifest/sweep entry."""

    normalized = str(name).strip().lower().replace("-", "_")
    if normalized == "a3":
        return A3Controller(A3Config(**dict(parameters)))
    if normalized == "cho":
        return CHOController(CHOConfig(**dict(parameters)))
    if normalized in {"load_aware", "load_aware_greedy"}:
        return LoadAwareGreedyController(LoadAwareGreedyConfig(**dict(parameters)))
    raise ValueError(f"unknown classical controller {name!r}")


def sweep_manifest(points: Iterable[SweepPoint] | None = None) -> list[dict[str, Any]]:
    return [point.as_dict() for point in (points or default_controller_sweeps())]


__all__ = [
    "A3Config",
    "A3Controller",
    "CHOConfig",
    "CHOController",
    "LoadAwareGreedyConfig",
    "LoadAwareGreedyController",
    "ReassociationController",
    "SweepPoint",
    "build_controller",
    "default_controller_sweeps",
    "expand_controller_sweep_config",
    "sweep_manifest",
]
