"""Risk calibration and an optional policy-side Cox risk shield.

No fitter in this module touches simulator feasibility or capacity.  It only
calibrates the policy-facing integrated violation intensity and may veto a
requested *handover* before the simulator applies its authoritative gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import torch

from leo_pg.control.policy import FixedRankPolicyConfig, score_candidates
from leo_pg.sim.state import ControlObservation, ServingAction


CALIBRATOR_MANIFEST_SCHEMA_VERSION = 1
CALIBRATOR_KINDS = ("cox_ratio", "covariate_error", "split_conformal")


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sample_count(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("sample_count must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("sample_count must be an integer") from exc
    if result != value or result <= 0:
        raise ValueError("sample_count must be a positive integer")
    return result


def _manifest(kind: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    if kind not in CALIBRATOR_KINDS:
        raise ValueError(f"unknown calibrator kind: {kind!r}")
    return {
        "schema_version": CALIBRATOR_MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "parameters": dict(parameters),
    }


def _manifest_parameters(
    payload: Mapping[str, Any],
    *,
    expected_kind: str,
    required: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("calibrator manifest must be a mapping")
    allowed_top_level = {"schema_version", "kind", "parameters", "fit"}
    unknown_top_level = sorted(set(payload).difference(allowed_top_level))
    if unknown_top_level:
        raise ValueError(
            "calibrator manifest has unknown fields: "
            + ", ".join(unknown_top_level)
        )
    version = payload.get("schema_version")
    if version != CALIBRATOR_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported calibrator manifest schema_version: "
            f"{version!r}; expected {CALIBRATOR_MANIFEST_SCHEMA_VERSION}"
        )
    kind = str(payload.get("kind", "")).strip().lower()
    if kind != expected_kind:
        raise ValueError(
            f"calibrator kind mismatch: expected {expected_kind!r}, got {kind!r}"
        )
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("calibrator manifest requires a parameters mapping")
    missing = [name for name in required if name not in parameters]
    unknown = sorted(set(parameters).difference(required))
    if missing:
        raise ValueError("calibrator parameters are missing: " + ", ".join(missing))
    if unknown:
        raise ValueError(
            "calibrator parameters have unknown fields: " + ", ".join(unknown)
        )
    fit = payload.get("fit")
    if fit is not None and not isinstance(fit, Mapping):
        raise ValueError("calibrator manifest fit metadata must be a mapping")
    return {name: parameters[name] for name in required}


def _save_json_payload(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    output.write_text(serialized + "\n", encoding="utf-8")
    return output


def _load_json_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"calibrator manifest does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid calibrator JSON manifest {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("calibrator JSON root must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not value.is_floating_point():
        value = value.to(torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def nearest_rank_quantile(values: torch.Tensor, probability: float) -> torch.Tensor:
    """Nearest-rank quantile with the manuscript's finite-sample convention."""

    values = _finite_vector("values", values)
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
        raise ValueError("probability must lie in (0, 1]")
    rank = max(1, math.ceil(probability * values.numel()))
    return torch.sort(values).values[rank - 1]


@dataclass(frozen=True)
class CoxInflationCalibrator:
    """Multiplicative upper calibration for integrated Cox intensities."""

    inflation: float
    quantile: float
    sample_count: int
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        inflation = _finite_number("inflation", self.inflation)
        quantile = _finite_number("quantile", self.quantile)
        epsilon = _finite_number("epsilon", self.epsilon)
        sample_count = _sample_count(self.sample_count)
        if inflation < 1.0:
            raise ValueError("Cox inflation must be at least 1")
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must lie in (0, 1]")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "inflation", inflation)
        object.__setattr__(self, "quantile", quantile)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "epsilon", epsilon)

    @classmethod
    def fit(
        cls,
        predicted_intensity: torch.Tensor,
        observed_integrated_hazard: torch.Tensor,
        *,
        quantile: float = 0.95,
        epsilon: float = 1e-8,
    ) -> "CoxInflationCalibrator":
        predicted = _finite_vector("predicted_intensity", predicted_intensity)
        observed = _finite_vector(
            "observed_integrated_hazard", observed_integrated_hazard
        ).to(device=predicted.device, dtype=predicted.dtype)
        if predicted.shape != observed.shape:
            raise ValueError("predicted and observed vectors must have the same shape")
        if torch.any(predicted < 0) or torch.any(observed < 0):
            raise ValueError("integrated hazards must be non-negative")
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        ratio = (observed + epsilon) / (predicted + epsilon)
        factor = max(1.0, float(nearest_rank_quantile(ratio, quantile).item()))
        return cls(
            inflation=factor,
            quantile=float(quantile),
            sample_count=int(predicted.numel()),
            epsilon=epsilon,
        )

    def transform(self, predicted_intensity: torch.Tensor) -> torch.Tensor:
        predicted = torch.as_tensor(predicted_intensity)
        if not torch.isfinite(predicted).all() or torch.any(predicted < 0):
            raise ValueError("predicted intensity must be finite and non-negative")
        return predicted * self.inflation

    def violation_probability(self, predicted_intensity: torch.Tensor) -> torch.Tensor:
        return -torch.expm1(-self.transform(predicted_intensity))

    def to_dict(self) -> dict[str, Any]:
        return _manifest(
            "cox_ratio",
            {
                "inflation": self.inflation,
                "quantile": self.quantile,
                "sample_count": self.sample_count,
                "epsilon": self.epsilon,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CoxInflationCalibrator":
        parameters = _manifest_parameters(
            payload,
            expected_kind="cox_ratio",
            required=("inflation", "quantile", "sample_count", "epsilon"),
        )
        return cls(**parameters)

    def save_json(
        self,
        path: str | Path,
        *,
        fit: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_calibrator_json(self, path, fit=fit)

    @classmethod
    def load_json(cls, path: str | Path) -> "CoxInflationCalibrator":
        calibrator = load_calibrator_json(path, expected_kind="cox_ratio")
        if not isinstance(calibrator, cls):
            raise TypeError("loaded calibrator is not CoxInflationCalibrator")
        return calibrator


@dataclass(frozen=True)
class CovariateErrorCoxCalibrator:
    """Manuscript-form Cox envelope ``Lambda*exp(||beta||_2*r_q)``."""

    beta_l2_norm: float
    error_radius: float
    quantile: float
    sample_count: int

    def __post_init__(self) -> None:
        beta_l2_norm = _finite_number("beta_l2_norm", self.beta_l2_norm)
        error_radius = _finite_number("error_radius", self.error_radius)
        quantile = _finite_number("quantile", self.quantile)
        sample_count = _sample_count(self.sample_count)
        if beta_l2_norm < 0.0 or error_radius < 0.0:
            raise ValueError("beta_l2_norm and error_radius must be non-negative")
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must lie in (0, 1]")
        try:
            inflation = math.exp(beta_l2_norm * error_radius)
        except OverflowError as exc:
            raise ValueError("Cox covariate-error inflation overflows float64") from exc
        if not math.isfinite(inflation):
            raise ValueError("Cox covariate-error inflation must be finite")
        object.__setattr__(self, "beta_l2_norm", beta_l2_norm)
        object.__setattr__(self, "error_radius", error_radius)
        object.__setattr__(self, "quantile", quantile)
        object.__setattr__(self, "sample_count", sample_count)

    @classmethod
    def fit(
        cls,
        covariate_error_l2: torch.Tensor,
        *,
        beta_l2_norm: float,
        quantile: float = 0.95,
    ) -> "CovariateErrorCoxCalibrator":
        errors = _finite_vector("covariate_error_l2", covariate_error_l2)
        if torch.any(errors < 0):
            raise ValueError("covariate error magnitudes must be non-negative")
        beta_l2_norm = float(beta_l2_norm)
        if not math.isfinite(beta_l2_norm) or beta_l2_norm < 0:
            raise ValueError("beta_l2_norm must be finite and non-negative")
        radius = float(nearest_rank_quantile(errors, quantile).item())
        return cls(
            beta_l2_norm=beta_l2_norm,
            error_radius=radius,
            quantile=float(quantile),
            sample_count=int(errors.numel()),
        )

    @property
    def inflation(self) -> float:
        return math.exp(self.beta_l2_norm * self.error_radius)

    def transform(self, predicted_intensity: torch.Tensor) -> torch.Tensor:
        predicted = torch.as_tensor(predicted_intensity)
        if not torch.isfinite(predicted).all() or torch.any(predicted < 0):
            raise ValueError("predicted intensity must be finite and non-negative")
        return predicted * self.inflation

    def violation_probability(self, predicted_intensity: torch.Tensor) -> torch.Tensor:
        return -torch.expm1(-self.transform(predicted_intensity))

    def to_dict(self) -> dict[str, Any]:
        return _manifest(
            "covariate_error",
            {
                "beta_l2_norm": self.beta_l2_norm,
                "error_radius": self.error_radius,
                "quantile": self.quantile,
                "sample_count": self.sample_count,
            },
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "CovariateErrorCoxCalibrator":
        parameters = _manifest_parameters(
            payload,
            expected_kind="covariate_error",
            required=(
                "beta_l2_norm",
                "error_radius",
                "quantile",
                "sample_count",
            ),
        )
        return cls(**parameters)

    def save_json(
        self,
        path: str | Path,
        *,
        fit: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_calibrator_json(self, path, fit=fit)

    @classmethod
    def load_json(cls, path: str | Path) -> "CovariateErrorCoxCalibrator":
        calibrator = load_calibrator_json(path, expected_kind="covariate_error")
        if not isinstance(calibrator, cls):
            raise TypeError("loaded calibrator is not CovariateErrorCoxCalibrator")
        return calibrator


@dataclass(frozen=True)
class SplitConformalUpperCalibrator:
    """Finite-sample split-conformal upper correction for non-negative targets."""

    residual_quantile: float
    alpha: float
    sample_count: int

    def __post_init__(self) -> None:
        residual_quantile = _finite_number(
            "residual_quantile", self.residual_quantile
        )
        alpha = _finite_number("alpha", self.alpha)
        sample_count = _sample_count(self.sample_count)
        if residual_quantile < 0.0:
            raise ValueError("residual_quantile must be non-negative")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        object.__setattr__(self, "residual_quantile", residual_quantile)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "sample_count", sample_count)

    @classmethod
    def fit(
        cls,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        alpha: float = 0.05,
    ) -> "SplitConformalUpperCalibrator":
        prediction = _finite_vector("prediction", prediction)
        target = _finite_vector("target", target).to(
            device=prediction.device, dtype=prediction.dtype
        )
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        n = prediction.numel()
        finite_sample_probability = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
        residual = target - prediction
        correction = max(
            0.0,
            float(nearest_rank_quantile(residual, finite_sample_probability).item()),
        )
        return cls(
            residual_quantile=correction,
            alpha=alpha,
            sample_count=int(n),
        )

    def transform(self, prediction: torch.Tensor) -> torch.Tensor:
        prediction = torch.as_tensor(prediction)
        if not torch.isfinite(prediction).all():
            raise ValueError("prediction contains NaN or Inf")
        return (prediction + self.residual_quantile).clamp_min(0.0)

    def to_dict(self) -> dict[str, Any]:
        return _manifest(
            "split_conformal",
            {
                "residual_quantile": self.residual_quantile,
                "alpha": self.alpha,
                "sample_count": self.sample_count,
            },
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "SplitConformalUpperCalibrator":
        parameters = _manifest_parameters(
            payload,
            expected_kind="split_conformal",
            required=("residual_quantile", "alpha", "sample_count"),
        )
        return cls(**parameters)

    def save_json(
        self,
        path: str | Path,
        *,
        fit: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_calibrator_json(self, path, fit=fit)

    @classmethod
    def load_json(cls, path: str | Path) -> "SplitConformalUpperCalibrator":
        calibrator = load_calibrator_json(path, expected_kind="split_conformal")
        if not isinstance(calibrator, cls):
            raise TypeError("loaded calibrator is not SplitConformalUpperCalibrator")
        return calibrator


class IntensityCalibrator(Protocol):
    def transform(self, predicted_intensity: torch.Tensor) -> torch.Tensor: ...


FittedCalibrator: TypeAlias = (
    CoxInflationCalibrator
    | CovariateErrorCoxCalibrator
    | SplitConformalUpperCalibrator
)


def calibrator_from_dict(payload: Mapping[str, Any]) -> FittedCalibrator:
    """Reconstruct one fitted calibrator from a versioned JSON-safe mapping."""

    if not isinstance(payload, Mapping):
        raise TypeError("calibrator manifest must be a mapping")
    kind = str(payload.get("kind", "")).strip().lower()
    classes = {
        "cox_ratio": CoxInflationCalibrator,
        "covariate_error": CovariateErrorCoxCalibrator,
        "split_conformal": SplitConformalUpperCalibrator,
    }
    calibrator_class = classes.get(kind)
    if calibrator_class is None:
        raise ValueError(
            f"unknown calibrator kind {kind!r}; expected "
            + ", ".join(CALIBRATOR_KINDS)
        )
    return calibrator_class.from_dict(payload)


def save_calibrator_json(
    calibrator: FittedCalibrator,
    path: str | Path,
    *,
    fit: Mapping[str, Any] | None = None,
) -> Path:
    """Save a fitted scalar calibrator as a portable versioned JSON manifest."""

    if not isinstance(
        calibrator,
        (
            CoxInflationCalibrator,
            CovariateErrorCoxCalibrator,
            SplitConformalUpperCalibrator,
        ),
    ):
        raise TypeError("calibrator must be one of the fitted calibration dataclasses")
    payload = calibrator.to_dict()
    if fit is not None:
        if not isinstance(fit, Mapping):
            raise TypeError("fit metadata must be a mapping")
        payload["fit"] = dict(fit)
    return _save_json_payload(path, payload)


def load_calibrator_json(
    path: str | Path,
    *,
    expected_kind: str | None = None,
) -> FittedCalibrator:
    """Load and validate a fitted calibrator JSON manifest."""

    payload = _load_json_payload(path)
    if expected_kind is not None:
        normalized = str(expected_kind).strip().lower()
        if normalized not in CALIBRATOR_KINDS:
            raise ValueError(
                f"unknown expected calibrator kind {expected_kind!r}; expected "
                + ", ".join(CALIBRATOR_KINDS)
            )
        actual = str(payload.get("kind", "")).strip().lower()
        if actual != normalized:
            raise ValueError(
                f"calibrator kind mismatch: expected {normalized!r}, got {actual!r}"
            )
    return calibrator_from_dict(payload)


def load_calibrator_factory(
    cfg: Mapping[str, Any],
    spec: Mapping[str, Any],
    device: str | torch.device,
) -> FittedCalibrator:
    """Evaluator-compatible factory loading ``spec.options.path``.

    Fitted calibrators contain only scalar parameters, so ``device`` is
    validated for extension compatibility but does not alter the loaded
    object.  Optional ``options.kind`` and ``options.sha256`` assertions bind
    an evaluation spec to the intended manifest.
    """

    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")
    if not isinstance(spec, Mapping):
        raise TypeError("calibrator spec must be a mapping")
    torch.device(device)
    options = spec.get("options")
    if not isinstance(options, Mapping):
        raise ValueError("calibrator spec requires an options mapping")
    allowed = {"path", "kind", "sha256"}
    unknown = sorted(set(options).difference(allowed))
    if unknown:
        raise ValueError(
            "calibrator options have unknown fields: " + ", ".join(unknown)
        )
    raw_path = options.get("path")
    if raw_path is None or not str(raw_path).strip():
        raise ValueError("calibrator spec.options.path is required")
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"calibrator manifest does not exist: {path}")
    expected_digest = options.get("sha256")
    if expected_digest is not None:
        normalized_digest = str(expected_digest).strip().lower()
        if (
            len(normalized_digest) != 64
            or any(character not in "0123456789abcdef" for character in normalized_digest)
        ):
            raise ValueError("calibrator options.sha256 must be 64 hexadecimal digits")
        actual_digest = _sha256(path)
        if actual_digest != normalized_digest:
            raise ValueError(
                "calibrator manifest SHA-256 mismatch: "
                f"expected {normalized_digest}, got {actual_digest}"
            )
    expected_kind = options.get("kind")
    return load_calibrator_json(
        path,
        expected_kind=None if expected_kind is None else str(expected_kind),
    )


@dataclass(frozen=True)
class RiskShieldConfig:
    """Optional handover veto using a calibrated first-violation probability.

    ``delta=1`` is deliberately inert and matches the manuscript's unshielded
    main protocol.  Smaller values define explicitly labelled shield analyses.
    """

    delta: float = 1.0
    protect_existing_association: bool = True
    mode: str = "veto"
    downweight_strength: float = 1.0

    def __post_init__(self) -> None:
        delta = float(self.delta)
        if not math.isfinite(delta) or not 0.0 <= delta <= 1.0:
            raise ValueError("delta must lie in [0, 1]")
        if str(self.mode).strip().lower() not in {"veto", "downweight"}:
            raise ValueError("mode must be 'veto' or 'downweight'")
        strength = float(self.downweight_strength)
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("downweight_strength must be finite and non-negative")


class RiskShield:
    def __init__(
        self,
        config: RiskShieldConfig | None = None,
        calibrator: IntensityCalibrator | None = None,
    ) -> None:
        self.config = config or RiskShieldConfig()
        self.calibrator = calibrator

    def calibrated_intensity(self, observation: ControlObservation) -> torch.Tensor:
        intensity = observation.policy_descriptors.intensity_edge
        if self.calibrator is not None:
            intensity = self.calibrator.transform(intensity)
        if not torch.isfinite(intensity).all() or torch.any(intensity < 0):
            raise ValueError("calibrated intensity must be finite and non-negative")
        return intensity

    def apply(
        self,
        observation: ControlObservation,
        proposed: ServingAction,
    ) -> ServingAction:
        proposed.validate(observation.user_count, observation.satellite_count)
        if proposed.observation_id != observation.observation_id:
            raise ValueError("shield received a stale action")
        if self.config.mode.lower() != "veto":
            raise ValueError(
                "RiskShield.apply implements veto semantics; use "
                "RiskAdjustedFixedRankPolicy for soft down-weighting"
            )
        if self.config.delta >= 1.0:
            return ServingAction(
                observation_id=proposed.observation_id,
                requested_serving=proposed.requested_serving.clone(),
            )

        requested = proposed.requested_serving.clone()
        intensity = self.calibrated_intensity(observation)
        probability = -torch.expm1(-intensity)
        edge_lookup = {
            (int(user), int(satellite)): edge
            for edge, (user, satellite) in enumerate(
                observation.candidate_edge_ids.tolist()
            )
        }
        for user in range(observation.user_count):
            target = int(requested[user].item())
            current = int(observation.current_serving[user].item())
            if target < 0 or target == current:
                continue
            edge = edge_lookup.get((user, target))
            should_veto = edge is None or float(probability[edge].item()) > self.config.delta
            if should_veto:
                requested[user] = current if self.config.protect_existing_association else -1
        result = ServingAction(
            observation_id=observation.observation_id,
            requested_serving=requested,
        )
        result.validate(observation.user_count, observation.satellite_count)
        return result


class ShieldedController:
    """Compose any reassociation controller with the optional risk shield."""

    def __init__(self, controller: object, shield: RiskShield) -> None:
        if not hasattr(controller, "select_action"):
            raise TypeError("controller must implement select_action")
        self.controller = controller
        self.shield = shield

    def reset(self) -> None:
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()

    def select_action(self, observation: ControlObservation) -> ServingAction:
        proposed = self.controller.select_action(observation)
        return self.shield.apply(observation, proposed)

    def __call__(self, observation: ControlObservation) -> ServingAction:
        return self.select_action(observation)


class RiskAdjustedFixedRankPolicy:
    """Fixed rank policy with explicitly labelled veto or soft-risk coupling.

    The main protocol is recovered exactly with ``delta=1`` and
    ``downweight_strength=0``.  Veto and down-weight modes are shield
    ablations, not replacements for the simulator feasibility gate.
    """

    def __init__(
        self,
        policy_config: FixedRankPolicyConfig,
        shield_config: RiskShieldConfig,
        calibrator: IntensityCalibrator | None = None,
    ) -> None:
        self.policy_config = policy_config
        self.shield_config = shield_config
        self.calibrator = calibrator

    def reset(self) -> None:
        return None

    def _probability(self, observation: ControlObservation) -> torch.Tensor:
        intensity = observation.policy_descriptors.intensity_edge
        if self.calibrator is not None:
            intensity = self.calibrator.transform(intensity)
        if not torch.isfinite(intensity).all() or torch.any(intensity < 0):
            raise ValueError("calibrated intensity must be finite and non-negative")
        return -torch.expm1(-intensity)

    def select_action(self, observation: ControlObservation) -> ServingAction:
        observation.validate()
        candidate_scores = score_candidates(observation, self.policy_config)
        probability = self._probability(observation)
        adjusted = candidate_scores.total.clone()
        mode = self.shield_config.mode.lower()
        if mode == "downweight":
            adjusted = adjusted - self.shield_config.downweight_strength * probability
        requested = observation.current_serving.clone()
        edge_users = observation.candidate_edge_ids[:, 0]
        edge_satellites = observation.candidate_edge_ids[:, 1]

        for user in observation.user_order.tolist():
            rows = torch.nonzero(edge_users == user, as_tuple=False).flatten()
            rows = rows[candidate_scores.eligible.index_select(0, rows)]
            if mode == "veto" and rows.numel():
                rows = rows[
                    probability.index_select(0, rows) <= self.shield_config.delta
                ]
            if rows.numel() == 0:
                continue
            satellites = edge_satellites.index_select(0, rows)
            satellite_order = torch.argsort(satellites, stable=True)
            ordered = rows.index_select(0, satellite_order)
            score_order = torch.argsort(
                adjusted.index_select(0, ordered), descending=True, stable=True
            )
            best_edge = int(ordered[score_order[0]].item())
            best_satellite = int(edge_satellites[best_edge].item())
            current = int(observation.current_serving[user].item())
            if current < 0:
                requested[user] = best_satellite
                continue
            current_rows = rows[edge_satellites.index_select(0, rows) == current]
            if current_rows.numel() != 1 or best_satellite == current:
                continue
            if int(observation.hold_steps[user].item()) < self.policy_config.min_dwell_steps:
                continue
            current_edge = int(current_rows[0].item())
            margin = self.policy_config.hysteresis
            if margin is None:
                margin = 1.0 / float(rows.numel())
            best_score = float(adjusted[best_edge].item())
            current_score = float(adjusted[current_edge].item())
            if best_score >= current_score + float(margin) or math.isclose(
                best_score,
                current_score + float(margin),
                rel_tol=1e-7,
                abs_tol=1e-8,
            ):
                requested[user] = best_satellite
        result = ServingAction(
            observation_id=observation.observation_id,
            requested_serving=requested,
        )
        result.validate(observation.user_count, observation.satellite_count)
        return result

    def __call__(self, observation: ControlObservation) -> ServingAction:
        return self.select_action(observation)


__all__ = [
    "CALIBRATOR_KINDS",
    "CALIBRATOR_MANIFEST_SCHEMA_VERSION",
    "CoxInflationCalibrator",
    "CovariateErrorCoxCalibrator",
    "FittedCalibrator",
    "IntensityCalibrator",
    "RiskShield",
    "RiskShieldConfig",
    "RiskAdjustedFixedRankPolicy",
    "ShieldedController",
    "SplitConformalUpperCalibrator",
    "calibrator_from_dict",
    "load_calibrator_factory",
    "load_calibrator_json",
    "nearest_rank_quantile",
    "save_calibrator_json",
]
