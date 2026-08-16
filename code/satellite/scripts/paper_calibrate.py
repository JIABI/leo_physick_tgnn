"""Fit and serialize one paper risk calibrator from a safe tensor bundle.

The input must be a ``.pt`` mapping readable by
``torch.load(..., weights_only=True)``.  Tensor keys may be dotted paths into
nested mappings.  Samples are flattened in stored order and the original shape,
key, source path, and source SHA-256 are recorded in the output JSON manifest.

Default input keys by calibration kind are:

* ``cox_ratio``: ``predicted_intensity`` and
  ``observed_integrated_hazard``;
* ``covariate_error``: ``covariate_error_l2`` and scalar ``beta_l2_norm``;
* ``split_conformal``: ``prediction`` and ``target``.

Keys can be overridden explicitly on the command line.  The generated JSON is
loadable by ``leo_pg.paper.calibration:load_calibrator_factory`` when its path
is provided as ``risk_shield.calibrator.options.path``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from leo_pg.paper.calibration import (
    CALIBRATOR_KINDS,
    CoxInflationCalibrator,
    CovariateErrorCoxCalibrator,
    FittedCalibrator,
    SplitConformalUpperCalibrator,
    save_calibrator_json,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_weights_only_bundle(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"calibration input does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("calibration .pt input must contain a mapping")
    return payload


def _lookup(payload: Mapping[str, Any], dotted_key: str) -> Any:
    key = str(dotted_key).strip()
    if not key:
        raise ValueError("calibration tensor key must be non-empty")
    current: Any = payload
    traversed: list[str] = []
    for component in key.split("."):
        traversed.append(component)
        if not isinstance(current, Mapping):
            raise KeyError(
                f"calibration key {key!r} traverses a non-mapping at "
                f"{'.'.join(traversed[:-1])!r}"
            )
        if component not in current:
            raise KeyError(f"calibration input is missing key {key!r}")
        current = current[component]
    return current


def _sample_vector(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[torch.Tensor, list[int]]:
    value = _lookup(payload, key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"calibration input {key!r} must be a torch.Tensor")
    if value.numel() == 0:
        raise ValueError(f"calibration input {key!r} must be non-empty")
    if value.layout != torch.strided:
        raise TypeError(f"calibration input {key!r} must be a dense tensor")
    if value.is_complex():
        raise TypeError(f"calibration input {key!r} must be real-valued")
    original_shape = [int(size) for size in value.shape]
    vector = value.detach().cpu().reshape(-1)
    if not vector.is_floating_point():
        vector = vector.to(torch.float64)
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"calibration input {key!r} contains NaN or Inf")
    return vector, original_shape


def _scalar(
    payload: Mapping[str, Any],
    *,
    override: float | None,
    key: str,
) -> float:
    raw = override if override is not None else _lookup(payload, key)
    if isinstance(raw, torch.Tensor):
        if raw.numel() != 1:
            raise ValueError(f"calibration scalar {key!r} must contain one value")
        raw = raw.detach().cpu().item()
    if isinstance(raw, bool):
        raise TypeError(f"calibration scalar {key!r} must be real-valued")
    try:
        result = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"calibration scalar {key!r} must be real-valued") from exc
    if not math.isfinite(result):
        raise ValueError(f"calibration scalar {key!r} must be finite")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Cox-ratio, Cox covariate-error, or split-conformal risk "
            "calibrator from a weights_only PyTorch tensor bundle"
        )
    )
    parser.add_argument("--input", required=True, help="Input tensor bundle .pt")
    parser.add_argument("--out", required=True, help="Output calibrator manifest .json")
    parser.add_argument(
        "--kind",
        required=True,
        choices=CALIBRATOR_KINDS,
    )
    parser.add_argument(
        "--prediction-key",
        default=None,
        help="Prediction tensor key (cox_ratio or split_conformal)",
    )
    parser.add_argument(
        "--target-key",
        default=None,
        help="Observed target tensor key (cox_ratio or split_conformal)",
    )
    parser.add_argument(
        "--error-key",
        default="covariate_error_l2",
        help="Covariate L2-error tensor key",
    )
    parser.add_argument(
        "--beta-l2-norm",
        type=float,
        default=None,
        help="Cox beta L2 norm; otherwise read from --beta-key",
    )
    parser.add_argument(
        "--beta-key",
        default="beta_l2_norm",
        help="Scalar Cox beta L2-norm key",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def _fit_calibrator(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
) -> tuple[FittedCalibrator, dict[str, Any]]:
    if args.kind == "cox_ratio":
        prediction_key = args.prediction_key or "predicted_intensity"
        target_key = args.target_key or "observed_integrated_hazard"
        prediction, prediction_shape = _sample_vector(payload, prediction_key)
        target, target_shape = _sample_vector(payload, target_key)
        if prediction.shape != target.shape:
            raise ValueError(
                "cox_ratio prediction and target have different flattened sizes"
            )
        calibrator = CoxInflationCalibrator.fit(
            prediction,
            target,
            quantile=args.quantile,
            epsilon=args.epsilon,
        )
        fit = {
            "inputs": {
                "prediction": {
                    "key": prediction_key,
                    "shape": prediction_shape,
                },
                "target": {"key": target_key, "shape": target_shape},
            },
            "hyperparameters": {
                "quantile": float(args.quantile),
                "epsilon": float(args.epsilon),
            },
        }
        return calibrator, fit

    if args.kind == "covariate_error":
        errors, error_shape = _sample_vector(payload, args.error_key)
        beta_l2_norm = _scalar(
            payload,
            override=args.beta_l2_norm,
            key=args.beta_key,
        )
        calibrator = CovariateErrorCoxCalibrator.fit(
            errors,
            beta_l2_norm=beta_l2_norm,
            quantile=args.quantile,
        )
        fit = {
            "inputs": {
                "covariate_error_l2": {
                    "key": args.error_key,
                    "shape": error_shape,
                },
                "beta_l2_norm": {
                    "source": "command_line"
                    if args.beta_l2_norm is not None
                    else "input_bundle",
                    "key": None if args.beta_l2_norm is not None else args.beta_key,
                    "value": beta_l2_norm,
                },
            },
            "hyperparameters": {"quantile": float(args.quantile)},
        }
        return calibrator, fit

    prediction_key = args.prediction_key or "prediction"
    target_key = args.target_key or "target"
    prediction, prediction_shape = _sample_vector(payload, prediction_key)
    target, target_shape = _sample_vector(payload, target_key)
    if prediction.shape != target.shape:
        raise ValueError(
            "split_conformal prediction and target have different flattened sizes"
        )
    calibrator = SplitConformalUpperCalibrator.fit(
        prediction,
        target,
        alpha=args.alpha,
    )
    fit = {
        "inputs": {
            "prediction": {"key": prediction_key, "shape": prediction_shape},
            "target": {"key": target_key, "shape": target_shape},
        },
        "hyperparameters": {"alpha": float(args.alpha)},
    }
    return calibrator, fit


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    if input_path.suffix.lower() not in {".pt", ".pth"}:
        parser.error("--input must be a .pt or .pth weights_only tensor bundle")
    if output_path.suffix.lower() != ".json":
        parser.error("--out must have a .json suffix")
    if input_path == output_path:
        parser.error("--input and --out must be different files")

    payload = _load_weights_only_bundle(input_path)
    calibrator, fit = _fit_calibrator(args, payload)
    fit_manifest = {
        "source": {
            "filename": input_path.name,
            "sha256": _sha256(input_path),
            "loader": "torch.load(weights_only=True,map_location='cpu')",
        },
        "kind": args.kind,
        "sample_count": calibrator.sample_count,
        **fit,
    }
    saved = save_calibrator_json(calibrator, output_path, fit=fit_manifest)
    print(
        f"[OK] kind={args.kind} samples={calibrator.sample_count} "
        f"manifest={saved} sha256={_sha256(saved)}"
    )


if __name__ == "__main__":
    main()
