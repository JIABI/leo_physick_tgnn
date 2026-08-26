"""Command-line entry point.  No command trains a model implicitly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .config import (
    DEFAULT_EXPERIMENT_REGISTRY,
    DEFAULT_MANUSCRIPT_CONFIG,
    load_experiment_registry,
    load_manuscript_config,
)
from .demo import run_demo
from .source_data import verify_source_data


DEFAULT_REPORTED_MODEL_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "code"
    / "satellite"
    / "configs"
    / "paper_protocol.yaml"
)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="controller-facing-state",
        description="Inspect and exercise the manuscript reference implementation (CPU, no training).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the deterministic semantic demonstration")
    demo.add_argument("--config", default=str(DEFAULT_MANUSCRIPT_CONFIG))
    demo.add_argument("--output", default="outputs/demo")

    experiments = sub.add_parser("experiments", help="list or inspect registered experiments")
    experiments.add_argument("--registry", default=str(DEFAULT_EXPERIMENT_REGISTRY))
    experiments.add_argument("--id", dest="experiment_id")

    verify_config = sub.add_parser("verify-config", help="validate frozen manuscript parameters")
    verify_config.add_argument("--config", default=str(DEFAULT_MANUSCRIPT_CONFIG))

    audit_models = sub.add_parser(
        "audit-model-identities",
        help="compare executable structural references with the Table S11 ledger",
    )
    audit_models.add_argument("--config", default=str(DEFAULT_REPORTED_MODEL_CONFIG))
    audit_models.add_argument(
        "--require-exact",
        action="store_true",
        help="return a non-zero status unless every constructor exactly matches its frozen ID",
    )

    verify_data = sub.add_parser("verify-source-data", help="verify a Zenodo v4 source-data directory")
    verify_data.add_argument("--source-data", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "demo":
        report = run_demo(args.config, args.output)
        _print(report)
        return 0
    if args.command == "experiments":
        registry = load_experiment_registry(args.registry)
        experiments = registry["experiments"]
        if args.experiment_id:
            if args.experiment_id not in experiments:
                raise SystemExit(f"unknown experiment id: {args.experiment_id}")
            _print({args.experiment_id: experiments[args.experiment_id]})
        else:
            _print(
                {
                    name: value.get("description", value.get("name", ""))
                    for name, value in experiments.items()
                }
            )
        return 0
    if args.command == "verify-config":
        config = load_manuscript_config(args.config)
        _print({"passed": True, "version": config["software_version"], "config": str(Path(args.config).resolve())})
        return 0
    if args.command == "audit-model-identities":
        from leo_pg.paper.model_identity import audit_reported_models

        config_path = Path(args.config).expanduser().resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise TypeError(f"{config_path} must contain a YAML mapping")
        audits = audit_reported_models(config)
        all_exact = all(row.exact_match for row in audits)
        _print(
            {
                "audit_complete": True,
                "all_frozen_identities_exact": all_exact,
                "config": str(config_path),
                "models": [row.as_dict() for row in audits],
                "training_performed": False,
                "unused_parameter_padding_allowed": False,
            }
        )
        return 0 if all_exact or not args.require_exact else 2
    if args.command == "verify-source-data":
        _print(verify_source_data(args.source_data).as_dict())
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
