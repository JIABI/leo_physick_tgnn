"""Run the paper's paired, action-coupled closed-loop evaluation protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from leo_pg.paper.evaluation import (
    DEFAULT_PAIRED_MODES,
    load_frozen_model,
    make_initializer_factory,
    run_paper_evaluation,
    save_evaluation_bundle,
)
from leo_pg.paper.models import normalize_paper_method, resolved_model_config
from leo_pg.utils.config import load_cfg


def _csv_strings(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return parsed


def _csv_ints(value: str) -> list[int]:
    try:
        parsed = [int(item) for item in _csv_strings(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if any(seed < 0 for seed in parsed):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return parsed


def _json_mapping(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON mapping: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must decode to a JSON object")
    return parsed


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return copy.deepcopy(dict(value))


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_classical_conditions(
    evaluation_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw = evaluation_cfg.get("classical_conditions", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("paper_evaluation.classical_conditions must be a list")
    conditions: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            conditions.append({"name": item, "parameters": {}})
        elif isinstance(item, Mapping):
            condition = copy.deepcopy(dict(item))
            if "name" not in condition and "factory" not in condition:
                raise ValueError(
                    f"classical condition {index} needs a name or factory"
                )
            conditions.append(condition)
        else:
            raise TypeError(
                f"classical condition {index} must be a name or mapping"
            )
    return conditions


def _merge_classical_conditions(
    configured: list[dict[str, Any]],
    names: list[str] | None,
    *,
    factory: str | None,
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = copy.deepcopy(configured)
    labels = {
        str(item.get("label") or item.get("name") or item.get("factory"))
        for item in result
    }
    for name in names or []:
        label = name
        if label in labels:
            continue
        condition: dict[str, Any] = {
            "name": name,
            "parameters": copy.deepcopy(dict(parameters)),
        }
        if factory:
            condition["factory"] = factory
        result.append(condition)
        labels.add(label)
    return result


def _controller_sweep_conditions(
    root_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand the frozen classical grid with stable labels and provenance."""

    controllers = importlib.import_module("leo_pg.paper.controllers")
    configured = root_cfg.get("controller_sweeps")
    if configured is None:
        entries = getattr(controllers, "sweep_manifest")()
        registry_source = "leo_pg.paper.controllers.default_controller_sweeps"
    else:
        if isinstance(configured, Mapping):
            expanded = getattr(controllers, "expand_controller_sweep_config")(
                copy.deepcopy(dict(configured))
            )
            entries = getattr(controllers, "sweep_manifest")(expanded)
            registry_source = "controller_sweeps"
        elif isinstance(configured, list):
            entries = copy.deepcopy(configured)
            registry_source = "controller_sweeps (legacy flat list)"
        else:
            raise TypeError("controller_sweeps must be a mapping or flat list")

    conditions: list[dict[str, Any]] = []
    labels: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise TypeError(f"controller sweep point {index} must be a mapping")
        controller = str(raw.get("controller", "")).strip()
        parameters = raw.get("parameters", {})
        if not controller:
            raise ValueError(f"controller sweep point {index} has no controller")
        if not isinstance(parameters, Mapping):
            raise TypeError(
                f"controller sweep point {index} parameters must be a mapping"
            )
        canonical = json.dumps(
            {"controller": controller, "parameters": dict(parameters)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
        label = f"sweep_{controller}_{suffix}"
        if label in labels:
            raise ValueError(f"duplicate controller sweep point: {label}")
        labels.add(label)
        conditions.append(
            {
                "name": controller,
                "label": label,
                "parameters": copy.deepcopy(dict(parameters)),
                "sweep_provenance": {
                    "registry": registry_source,
                    "index": int(index),
                    "provenance": raw.get("provenance", "unspecified"),
                    "rationale": raw.get("rationale", ""),
                    "declared_selected": bool(raw.get("selected", False)),
                    "grid": copy.deepcopy(dict(raw.get("grid", {}))),
                    "parameter_sha256": hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest(),
                },
            }
        )
    return conditions


def _append_unique_conditions(
    conditions: list[dict[str, Any]],
    additions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = copy.deepcopy(conditions)
    labels = {
        str(item.get("label") or item.get("name") or item.get("factory"))
        for item in result
    }
    for addition in additions:
        item = copy.deepcopy(dict(addition))
        label = str(item.get("label") or item.get("name") or item.get("factory"))
        if not label:
            raise ValueError("classical condition must have a stable label")
        if label in labels:
            raise ValueError(f"duplicate classical condition label: {label!r}")
        result.append(item)
        labels.add(label)
    return result


def _initializer_spec(
    evaluation_cfg: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    cfg: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    configured = evaluation_cfg.get("initializer", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise TypeError("paper_evaluation.initializer must be a mapping")
    configured = copy.deepcopy(dict(configured))
    reference = args.initializer_factory or configured.get("factory")
    checkpoint = args.initializer_checkpoint or configured.get("checkpoint")
    kind = args.initializer_kind or configured.get("kind")
    fingerprint = args.initializer_fingerprint or configured.get("fingerprint")
    options = copy.deepcopy(dict(configured.get("options", {})))
    if args.initializer_options:
        options.update(args.initializer_options)
    factory = make_initializer_factory(
        cfg,
        device=args.device,
        factory_reference=None if reference is None else str(reference),
        checkpoint_path=None if checkpoint is None else str(checkpoint),
        kind=None if kind is None else str(kind),
        fingerprint=None if fingerprint is None else str(fingerprint),
        options=options,
    )
    manifest: dict[str, Any] = {
        "factory": "configured_prior" if reference is None else str(reference),
        "kind_override": kind,
        "fingerprint_override": fingerprint,
        "options": options,
    }
    if checkpoint is not None:
        checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
        manifest["checkpoint"] = {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        }
    return factory, manifest


def _shield_spec(
    evaluation_cfg: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    configured = evaluation_cfg.get("risk_shield", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise TypeError("paper_evaluation.risk_shield must be a mapping")
    spec = copy.deepcopy(dict(configured))
    if args.risk_shield is not None:
        spec["enabled"] = bool(args.risk_shield)
    if args.shield_factory:
        spec["factory"] = args.shield_factory
    if args.shield_delta is not None:
        spec["delta"] = float(args.shield_delta)
    if args.shield_mode:
        spec["mode"] = args.shield_mode
    if args.shield_options:
        spec.update(args.shield_options)
    if args.calibrator_factory:
        spec["calibrator"] = {
            "factory": args.calibrator_factory,
            "options": copy.deepcopy(args.calibrator_options or {}),
        }
    if not bool(spec.get("enabled", False)):
        return None
    return spec


def _hook_spec(
    *,
    cli_enabled: bool | None,
    cli_factory: str | None,
    configured: Any,
    built_in_options: Mapping[str, Any] | None = None,
) -> str | dict[str, Any] | None:
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise TypeError("hook configuration must be a mapping")
    value = copy.deepcopy(dict(configured))
    if cli_enabled is not None:
        value["enabled"] = bool(cli_enabled)
    if cli_factory:
        value["factory"] = cli_factory
        value["enabled"] = True
    if built_in_options:
        options = copy.deepcopy(dict(value.get("options", {})))
        options.update(copy.deepcopy(dict(built_in_options)))
        value["options"] = options
    if not bool(value.get("enabled", False)):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one frozen checkpoint under paired action-coupled "
            "model/oracle/descriptor-substitution conditions"
        )
    )
    parser.add_argument("--cfg", required=True, help="Paper protocol YAML")
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Frozen model checkpoint (or paper_evaluation.checkpoint)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Full trace .pt output (or paper_evaluation.output)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON manifest path; defaults beside --out",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Paper method name (tgn_mlp, tgn_kan, tgn_physick, or baseline)",
    )
    parser.add_argument(
        "--modes",
        type=_csv_strings,
        default=None,
        help="Comma-separated paired modes",
    )
    parser.add_argument(
        "--seeds",
        type=_csv_ints,
        default=None,
        help="Comma-separated base simulator seeds",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Episodes per base seed",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override paper_protocol.horizon_steps",
    )
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Explicit migration override for semantic checkpoint mismatch",
    )
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="Explicitly allow a checkpoint with no semantic signature",
    )
    parser.add_argument(
        "--allow-oracle-warm-start",
        action="store_true",
        help="Labelled diagnostic only; disabled in the main paper protocol",
    )

    parser.add_argument("--initializer-factory", default=None)
    parser.add_argument("--initializer-checkpoint", default=None)
    parser.add_argument("--initializer-kind", default=None)
    parser.add_argument("--initializer-fingerprint", default=None)
    parser.add_argument(
        "--initializer-options",
        type=_json_mapping,
        default=None,
        metavar="JSON",
    )

    parser.add_argument(
        "--classical",
        type=_csv_strings,
        default=None,
        help="Comma-separated built-in classical controller names",
    )
    parser.add_argument(
        "--controller-factory",
        default=None,
        help="Optional module:callable factory for --classical entries",
    )
    parser.add_argument(
        "--controller-parameters",
        type=_json_mapping,
        default={},
        metavar="JSON",
    )
    parser.add_argument(
        "--controller-sweep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Expand the complete frozen A3/CHO/load-aware controller grid",
    )

    parser.add_argument(
        "--risk-shield",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--shield-factory", default=None)
    parser.add_argument("--shield-delta", type=float, default=None)
    parser.add_argument("--shield-mode", choices=("veto", "downweight"), default=None)
    parser.add_argument("--shield-options", type=_json_mapping, default=None)
    parser.add_argument("--calibrator-factory", default=None)
    parser.add_argument("--calibrator-options", type=_json_mapping, default=None)

    parser.add_argument(
        "--metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--metrics-hook", default=None)
    parser.add_argument(
        "--shrink-jump-audit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--audit-hook", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg_path = Path(args.cfg).expanduser().resolve()
    cfg = load_cfg(str(cfg_path))
    evaluation_cfg = _mapping(cfg, "paper_evaluation")

    checkpoint_value = args.ckpt or evaluation_cfg.get("checkpoint")
    if checkpoint_value is None:
        parser.error("provide --ckpt or paper_evaluation.checkpoint")
    output_value = args.out or evaluation_cfg.get("output")
    if output_value is None:
        output_dir = evaluation_cfg.get("output_dir")
        if output_dir:
            fallback_message = _mapping(cfg, "model").get("message_type", "physick")
            method_name = args.method or cfg.get(
                "paper_method", f"tgn_{fallback_message}"
            )
            output_value = str(Path(str(output_dir)) / f"{method_name}_paired_trace.pt")
        else:
            parser.error("provide --out, paper_evaluation.output, or output_dir")

    fallback_message = _mapping(cfg, "model").get("message_type", "physick")
    method = normalize_paper_method(
        str(
            args.method
            or evaluation_cfg.get("method")
            or cfg.get("paper_method")
            or f"tgn_{fallback_message}"
        )
    )
    if method.startswith("snapshot_"):
        parser.error(
            "Snapshot methods use the isolated Snapshot feature/evaluation contract; "
            "call scripts/paper_snapshot_evaluate.py or scripts/paper_protocol.sh"
        )
    cfg = resolved_model_config(cfg, method)

    configured_seeds = cfg.get("run_seeds", [cfg.get("seed", 7)])
    if args.seeds is not None:
        seeds = args.seeds
    elif isinstance(configured_seeds, (list, tuple)):
        seeds = [int(value) for value in configured_seeds]
    else:
        seeds = [int(configured_seeds)]
    if not seeds or any(seed < 0 for seed in seeds):
        parser.error("evaluation seeds must be a non-empty list of non-negative integers")

    episodes = _positive_int(
        "episodes",
        int(
            args.episodes
            if args.episodes is not None
            else evaluation_cfg.get("episodes_per_seed", 1)
        ),
    )
    horizon_value = (
        args.horizon
        if args.horizon is not None
        else evaluation_cfg.get("horizon_steps")
    )
    horizon = None if horizon_value is None else _positive_int("horizon", int(horizon_value))
    modes = args.modes or evaluation_cfg.get(
        "modes", [mode.value for mode in DEFAULT_PAIRED_MODES]
    )
    if not isinstance(modes, (list, tuple)):
        parser.error("paper_evaluation.modes must be a list")

    classical = _merge_classical_conditions(
        _config_classical_conditions(evaluation_cfg),
        args.classical,
        factory=args.controller_factory,
        parameters=args.controller_parameters,
    )
    sweep_enabled = (
        bool(args.controller_sweep)
        if args.controller_sweep is not None
        else bool(evaluation_cfg.get("controller_sweep", False))
    )
    if sweep_enabled:
        classical = _append_unique_conditions(
            classical,
            _controller_sweep_conditions(cfg),
        )
    shield = _shield_spec(evaluation_cfg, args)
    if (
        classical
        and shield
        and str(shield.get("mode", "veto")).lower() == "downweight"
        and not shield.get("factory")
    ):
        parser.error(
            "built-in downweight reranks FixedRankPolicy and cannot post-process "
            "classical controller actions; evaluate those conditions separately or "
            "provide a custom shield factory"
        )
    initializer_factory, initializer_manifest = _initializer_spec(
        evaluation_cfg,
        args,
        cfg=cfg,
    )

    metric_options = {
        "dt_ctrl": float(
            _mapping(cfg, "paper_protocol").get("dt_ctrl", 0.1)
        ),
        "outage_gamma_db": float(evaluation_cfg.get("outage_gamma_db", -3.0)),
        "pingpong_window_seconds": float(
            evaluation_cfg.get("pingpong_window_seconds", 2.0)
        ),
        "bandwidth_hz": float(evaluation_cfg.get("bandwidth_hz", 1.0)),
        "bootstrap_resamples": int(
            evaluation_cfg.get("bootstrap_resamples", 10_000)
        ),
        "bootstrap_seed": int(evaluation_cfg.get("bootstrap_seed", 17)),
        "bootstrap_confidence": float(
            evaluation_cfg.get("bootstrap_confidence", 0.95)
        ),
        "tail_risk_probability": float(
            evaluation_cfg.get("tail_risk_probability", 0.10)
        ),
    }
    metrics = _hook_spec(
        cli_enabled=args.metrics,
        cli_factory=args.metrics_hook,
        configured=evaluation_cfg.get("metrics", {}),
        built_in_options=metric_options,
    )
    shrink_cfg = evaluation_cfg.get("shrink_jump", {})
    shrink_options = (
        copy.deepcopy(dict(shrink_cfg))
        if isinstance(shrink_cfg, Mapping)
        else {}
    )
    audit = _hook_spec(
        cli_enabled=args.shrink_jump_audit,
        cli_factory=args.audit_hook,
        configured=evaluation_cfg.get("audit", {}),
        built_in_options=shrink_options,
    )

    model, checkpoint_manifest = load_frozen_model(
        cfg,
        checkpoint_value,
        method=method,
        device=args.device,
        allow_config_mismatch=args.allow_config_mismatch,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
    payload = run_paper_evaluation(
        cfg,
        model,
        checkpoint_manifest=checkpoint_manifest,
        seeds=seeds,
        episodes_per_seed=episodes,
        horizon=horizon,
        modes=[str(mode) for mode in modes],
        device=args.device,
        initializer_factory=initializer_factory,
        initializer_manifest=initializer_manifest,
        classical_conditions=classical,
        shield_spec=shield,
        metrics_spec=metrics,
        audit_spec=audit,
        allow_oracle_warm_start=args.allow_oracle_warm_start,
    )
    payload["manifest"]["config"] = {
        "path": str(cfg_path),
        "sha256": _sha256(cfg_path),
        "paper_method": method,
        "message_type": _mapping(cfg, "model").get("message_type"),
    }
    output, manifest = save_evaluation_bundle(
        payload,
        output_value,
        manifest_path=args.manifest,
    )
    print(
        f"[OK] wrote {output} and {manifest} | "
        f"episodes={len(payload['units'])} modes={','.join(str(mode) for mode in modes)}"
    )


if __name__ == "__main__":
    main()
