"""Machine-readable contracts for experiments and the Zenodo record."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar, Mapping


def _required_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class RunManifest:
    """One independently initialized training run.

    A row is a training run only when it has its own model, optimizer, data-order
    seed, and checkpoint identity. Evaluation seeds never create new run rows.
    """

    platform: str
    condition_id: str
    cell_id: str
    method: str
    run_id: str
    training_seed: int
    model_init_seed: int
    optimizer_seed: int
    data_order_seed: int
    checkpoint_id: str
    checkpoint_path: str
    config_path: str
    config_sha256: str
    code_commit: str = ""
    training_status: str = "planned"
    notes: str = ""

    CSV_FIELDS: ClassVar[tuple[str, ...]] = (
        "platform",
        "condition_id",
        "cell_id",
        "method",
        "run_id",
        "training_seed",
        "model_init_seed",
        "optimizer_seed",
        "data_order_seed",
        "checkpoint_id",
        "checkpoint_path",
        "config_path",
        "config_sha256",
        "code_commit",
        "training_status",
        "notes",
    )

    def __post_init__(self) -> None:
        for name in ("platform", "condition_id", "cell_id", "method", "run_id"):
            _required_text(name, getattr(self, name))
        if self.training_status == "complete":
            _required_text("checkpoint_id", self.checkpoint_id)
            _required_text("config_sha256", self.config_sha256)
            if len(self.config_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.config_sha256.lower()
            ):
                raise ValueError("config_sha256 must contain 64 hexadecimal characters")
            _required_text("code_commit", self.code_commit)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "RunManifest":
        values = dict(row)
        for key in ("training_seed", "model_init_seed", "optimizer_seed", "data_order_seed"):
            values[key] = int(values[key])
        return cls(**{key: values.get(key, "") for key in cls.CSV_FIELDS})


@dataclass(frozen=True)
class EpisodeManifest:
    """Identity of one held-out episode and its exogenous random stream."""

    platform: str
    run_id: str
    split: str
    episode_id: str
    episode_seed: int
    exogenous_sequence_id: str
    exogenous_seed: int
    panel_id: str
    initial_state_id: str = ""
    notes: str = ""

    CSV_FIELDS: ClassVar[tuple[str, ...]] = (
        "platform",
        "run_id",
        "split",
        "episode_id",
        "episode_seed",
        "exogenous_sequence_id",
        "exogenous_seed",
        "panel_id",
        "initial_state_id",
        "notes",
    )

    def __post_init__(self) -> None:
        for name in ("platform", "run_id", "split", "episode_id", "exogenous_sequence_id", "panel_id"):
            _required_text(name, getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "EpisodeManifest":
        values = dict(row)
        values["episode_seed"] = int(values["episode_seed"])
        values["exogenous_seed"] = int(values["exogenous_seed"])
        return cls(**{key: values.get(key, "") for key in cls.CSV_FIELDS})


@dataclass(frozen=True)
class MetricObservation:
    """Metric value with an explicit support/denominator contract.

    Undefined ratios are represented by ``defined=False`` and ``value=None``.
    They must never be coerced to zero. ``support_n`` counts the primitive units
    entering a directly aggregated metric, while ``denominator`` is used for a
    rate or ratio.
    """

    metric: str
    value: float | None
    defined: bool
    numerator: float | None = None
    denominator: float | None = None
    support_n: int | None = None
    unit: str = "dimensionless"
    missing_reason: str = ""

    def __post_init__(self) -> None:
        _required_text("metric", self.metric)
        if self.defined:
            if self.value is None or not isfinite(float(self.value)):
                raise ValueError("a defined metric must have a finite value")
            if self.denominator is not None and self.denominator <= 0:
                raise ValueError("a defined rate must have a positive denominator")
        else:
            if self.value is not None:
                raise ValueError("an undefined metric must store value=None")
            _required_text("missing_reason", self.missing_reason)


@dataclass(frozen=True)
class ResultRecord:
    """One condition-by-run-by-episode metric record."""

    platform: str
    condition_id: str
    cell_id: str
    method: str
    interface: str
    operator: str
    run_id: str
    checkpoint_id: str
    episode_id: str
    exogenous_sequence_id: str
    metric: str
    value: float | None
    defined: bool
    provenance: str
    numerator: float | None = None
    denominator: float | None = None
    support_n: int | None = None
    unit: str = "dimensionless"
    stress_id: str = "nominal"
    split: str = "test"
    missing_reason: str = ""

    CSV_FIELDS: ClassVar[tuple[str, ...]] = (
        "platform",
        "condition_id",
        "cell_id",
        "method",
        "interface",
        "operator",
        "run_id",
        "checkpoint_id",
        "episode_id",
        "exogenous_sequence_id",
        "metric",
        "value",
        "defined",
        "provenance",
        "numerator",
        "denominator",
        "support_n",
        "unit",
        "stress_id",
        "split",
        "missing_reason",
    )

    def __post_init__(self) -> None:
        for name in (
            "platform",
            "condition_id",
            "cell_id",
            "method",
            "interface",
            "operator",
            "run_id",
            "checkpoint_id",
            "episode_id",
            "exogenous_sequence_id",
            "metric",
            "provenance",
            "unit",
            "stress_id",
            "split",
        ):
            _required_text(name, getattr(self, name))
        if self.provenance not in {
            "original_experiment_output",
            "derived_from_original_experiment_output",
        }:
            raise ValueError(
                "ResultRecord accepts only original or explicitly derived original "
                "run-level output; reported/display-reconstructed aggregates belong "
                "in the separate reported-data schemas"
            )
        MetricObservation(
            metric=self.metric,
            value=self.value,
            defined=self.defined,
            numerator=self.numerator,
            denominator=self.denominator,
            support_n=self.support_n,
            unit=self.unit,
            missing_reason=self.missing_reason,
        )

    @property
    def key(self) -> tuple[str, ...]:
        return (
            self.platform,
            self.condition_id,
            self.cell_id,
            self.method,
            self.interface,
            self.operator,
            self.run_id,
            self.checkpoint_id,
            self.episode_id,
            self.exogenous_sequence_id,
            self.stress_id,
            self.split,
            self.metric,
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["defined"] = "true" if self.defined else "false"
        for key in ("value", "numerator", "denominator", "support_n"):
            if row[key] is None:
                row[key] = ""
        return row

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ResultRecord":
        values = dict(row)
        defined_value = str(values.get("defined", "")).strip().lower()
        if defined_value not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"invalid defined value: {values.get('defined')!r}")
        values["defined"] = defined_value in {"true", "1", "yes"}
        for key in ("value", "numerator", "denominator"):
            raw = values.get(key, "")
            values[key] = None if raw in (None, "") else float(raw)
        raw_support = values.get("support_n", "")
        values["support_n"] = None if raw_support in (None, "") else int(raw_support)
        return cls(**{key: values.get(key, "") for key in cls.CSV_FIELDS})


def require_relative_path(path: str) -> None:
    """Reject absolute or parent-traversing paths in public manifests."""

    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"manifest path must be relative and contained: {path}")
