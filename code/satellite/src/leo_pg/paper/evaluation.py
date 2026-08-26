"""Action-coupled paper evaluation and provenance-preserving serialization.

The stable path in this module depends only on the simulator, the frozen model,
and the closed-loop evaluator. Paper extensions (classical controllers,
calibration/risk shields, metrics, and shrink-jump audits) are imported lazily
when their corresponding option is requested.

Extension contracts
-------------------
An initializer factory is a dotted callable (package.module:callable) accepting
cfg, device, checkpoint_path, and options keyword arguments. It must return an
initializer with non-empty kind and fingerprint attributes. Controller, shield,
metrics, and audit factories follow the keyword contracts documented by their
builder functions below.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

from leo_pg.control.policy import FixedRankPolicy
from leo_pg.eval.closed_loop import (
    ClosedLoopResult,
    ConstantPolicyStreamInitializer,
    PolicyInitializerKind,
    TGNDescriptorProvider,
    run_closed_loop,
    run_paired_conditions,
)
from leo_pg.eval.substitution import SubstitutionMode
from leo_pg.paper.dataset import (
    serialize_action,
    serialize_execution,
    serialize_observation,
    serialize_policy_descriptors,
)
from leo_pg.paper.models import build_paper_model, normalize_paper_method
from leo_pg.sim.paper_environment import PAPER_PROTOCOL_VERSION, PaperAlignedLEOEnv
from leo_pg.train.checkpoint import load_ckpt


EVALUATION_SCHEMA_VERSION = 2
DEFAULT_PAIRED_MODES: tuple[SubstitutionMode, ...] = (
    SubstitutionMode.MODEL,
    SubstitutionMode.ORACLE,
    SubstitutionMode.GAMMA,
    SubstitutionMode.INTENSITY,
    SubstitutionMode.FLOW,
    SubstitutionMode.INTENSITY_FLOW,
)


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().to(device="cpu").contiguous()
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise ValueError("evaluation trace contains a NaN or Inf tensor")
    return result


def _pure(value: Any, *, path: str = "root") -> Any:
    """Return a weights-only-safe tree, rejecting ambiguous live objects."""

    if isinstance(value, torch.Tensor):
        return _cpu_tensor(value)
    if isinstance(value, Enum):
        return _pure(value.value, path=path)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite scalar at {path}: {value!r}")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"mapping key at {path} must be str, got {type(key)!r}")
            out[key] = _pure(item, path=f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_pure(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"unsupported serialization value at {path}: {type(value)!r}")


def _json_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_tree(item) for item in value]
    return value


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_symbol(reference: str) -> Any:
    """Load module:attribute without importing optional modules eagerly."""

    if ":" not in reference:
        raise ValueError(
            f"extension reference must use 'module:attribute' syntax, got {reference!r}"
        )
    module_name, attribute_path = reference.split(":", 1)
    if not module_name or not attribute_path:
        raise ValueError(f"invalid extension reference: {reference!r}")
    value: Any = importlib.import_module(module_name)
    for component in attribute_path.split("."):
        value = getattr(value, component)
    return value


def _invoke_extension(factory: Callable[..., Any], **kwargs: Any) -> Any:
    """Call an extension using only keywords declared by its signature."""

    signature = inspect.signature(factory)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    call_kwargs = {
        key: value
        for key, value in kwargs.items()
        if accepts_kwargs or key in signature.parameters
    }
    return factory(**call_kwargs)


def parse_modes(values: Iterable[str | SubstitutionMode]) -> tuple[SubstitutionMode, ...]:
    modes: list[SubstitutionMode] = []
    for value in values:
        mode = (
            value
            if isinstance(value, SubstitutionMode)
            else SubstitutionMode.parse(str(value))
        )
        if mode in modes:
            raise ValueError(f"duplicate substitution mode: {mode.value}")
        modes.append(mode)
    if not modes:
        raise ValueError("at least one substitution mode is required")
    return tuple(modes)


def load_frozen_model(
    cfg: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    method: str | None = None,
    device: str | torch.device = "cpu",
    allow_config_mismatch: bool = False,
    allow_legacy_checkpoint: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build and strictly load the paper model plus a checkpoint manifest."""

    cfg_copy = copy.deepcopy(dict(cfg))
    selected_method = normalize_paper_method(
        method
        or str(cfg_copy.get("paper_method", ""))
        or f"tgn_{dict(cfg_copy.get('model', {})).get('message_type', 'physick')}"
    )
    target_device = torch.device(device)
    model = build_paper_model(cfg_copy, selected_method)
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_ckpt(
        str(resolved_checkpoint),
        model,
        map_location=target_device,
        strict=True,
        allow_config_mismatch=allow_config_mismatch,
        allow_legacy_checkpoint=allow_legacy_checkpoint,
    )
    model.to(target_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_method = str(checkpoint.get("paper_method", "")).strip().lower()
    if checkpoint_method and checkpoint_method != selected_method:
        raise RuntimeError(
            f"checkpoint paper_method={checkpoint_method!r} does not match "
            f"requested method={selected_method!r}"
        )

    selected: dict[str, Any] = {}
    for key in (
        "checkpoint_schema_version",
        "epoch",
        "global_step",
        "run_name",
        "model_signature",
        "config_signature",
        "paper_method",
    ):
        if key in checkpoint:
            selected[key] = checkpoint[key]
    trainer_state = checkpoint.get("trainer_state")
    if isinstance(trainer_state, Mapping):
        training: dict[str, Any] = {}
        for key in (
            "trainer_schema_version",
            "best_validation",
            "best_epoch",
            "latest_validation",
            "loss_weights",
            "batch_semantics",
            "batch_size_control_graphs",
            "gradient_accumulation_episodes",
            "dataset",
            "action_train_seeds",
            "action_validation_seeds",
            "model_class",
            "paper_method",
            "trainable_parameters",
        ):
            if key in trainer_state:
                training[key] = trainer_state[key]
        selected["training"] = training
    manifest = {
        "path": str(resolved_checkpoint),
        "sha256": _file_sha256(resolved_checkpoint),
        "paper_method": selected_method,
        "metadata": _pure(selected, path="checkpoint.metadata"),
        "allow_config_mismatch": bool(allow_config_mismatch),
        "allow_legacy_checkpoint": bool(allow_legacy_checkpoint),
    }
    return model, manifest


class _InitializerMetadataOverride:
    """Delegate an initializer while supplying explicit provenance metadata."""

    def __init__(self, initializer: Any, *, kind: str | None, fingerprint: str | None) -> None:
        self._initializer = initializer
        observed_kind = getattr(initializer, "kind", None)
        if observed_kind is not None and kind is not None:
            if PolicyInitializerKind.parse(observed_kind) is not PolicyInitializerKind.parse(
                kind
            ):
                raise ValueError("initializer kind override disagrees with the initializer")
        source_kind = kind if kind is not None else observed_kind
        self.kind = PolicyInitializerKind.parse(source_kind)
        observed_fingerprint = str(
            getattr(initializer, "fingerprint", "")
        ).strip()
        declared_fingerprint = str(fingerprint or "").strip()
        if (
            observed_fingerprint
            and declared_fingerprint
            and observed_fingerprint != declared_fingerprint
        ):
            raise ValueError(
                "initializer fingerprint override disagrees with the initializer"
            )
        self.fingerprint = observed_fingerprint or declared_fingerprint
        if not self.fingerprint:
            raise ValueError("initializer must expose a non-empty fingerprint")

    def initialize(self, observation: Any, missing_edge: torch.Tensor) -> Any:
        return self._initializer.initialize(observation, missing_edge)


def make_initializer_factory(
    cfg: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
    factory_reference: str | None = None,
    checkpoint_path: str | Path | None = None,
    kind: str | None = None,
    fingerprint: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> Callable[[], Any]:
    """Create a fresh, fingerprinted initializer for every paired condition."""

    cfg_copy = copy.deepcopy(dict(cfg))
    option_copy = copy.deepcopy(dict(options or {}))
    if factory_reference is None:
        if checkpoint_path is not None:
            raise ValueError(
                "initializer_checkpoint requires an initializer_factory; configured_prior "
                "does not load learned weights"
            )

        def configured_factory() -> Any:
            initializer = ConstantPolicyStreamInitializer.from_config(cfg_copy)
            return _InitializerMetadataOverride(
                initializer,
                kind=kind,
                fingerprint=fingerprint,
            )

        return configured_factory

    extension = _load_symbol(factory_reference)
    if not callable(extension):
        raise TypeError(f"initializer factory is not callable: {factory_reference!r}")

    def extension_factory() -> Any:
        initializer = _invoke_extension(
            extension,
            cfg=copy.deepcopy(cfg_copy),
            device=torch.device(device),
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
            options=copy.deepcopy(option_copy),
        )
        return _InitializerMetadataOverride(
            initializer,
            kind=kind,
            fingerprint=fingerprint,
        )

    return extension_factory


class _ShieldedPolicyAdapter:
    """Preserve the fixed-policy config contract while applying a risk shield."""

    def __init__(self, policy: FixedRankPolicy, shield: Any) -> None:
        self._policy = policy
        self._shield = shield
        self.config = policy.config

    def select_action(self, observation: Any) -> Any:
        proposed = self._policy.select_action(observation)
        return self._shield.apply(observation, proposed)


class _ConfiguredPolicyAdapter:
    """Expose the environment's fixed config for a compatible policy variant."""

    def __init__(self, policy: Any, config: Any) -> None:
        self._policy = policy
        self.config = config

    def select_action(self, observation: Any) -> Any:
        return self._policy.select_action(observation)


class _BuiltInRiskAdjustment:
    """Select the correct built-in policy composition for veto/downweight."""

    def __init__(
        self,
        *,
        calibration_module: Any,
        shield_config: Any,
        calibrator: Any,
    ) -> None:
        self._calibration = calibration_module
        self.config = shield_config
        self.calibrator = calibrator
        self._veto = calibration_module.RiskShield(
            shield_config,
            calibrator=calibrator,
        )

    def wrap_fixed_policy(self, policy: FixedRankPolicy) -> Any:
        if str(self.config.mode).lower() == "downweight":
            adjusted = self._calibration.RiskAdjustedFixedRankPolicy(
                policy.config,
                self.config,
                calibrator=self.calibrator,
            )
            return _ConfiguredPolicyAdapter(adjusted, policy.config)
        return _ShieldedPolicyAdapter(policy, self._veto)

    def apply(self, observation: Any, proposed: Any) -> Any:
        if str(self.config.mode).lower() == "downweight":
            raise ValueError(
                "downweight risk adjustment must wrap FixedRankPolicy before ranking; "
                "it cannot be applied after an arbitrary classical controller proposal"
            )
        return self._veto.apply(observation, proposed)


def make_shield_factory(
    cfg: Mapping[str, Any],
    spec: Mapping[str, Any] | None,
    *,
    device: str | torch.device = "cpu",
) -> Callable[[], Any] | None:
    """Build a lazy shield factory, optionally with a calibrator factory."""

    if not spec or not bool(spec.get("enabled", False)):
        return None
    spec_copy = copy.deepcopy(dict(spec))
    custom_reference = spec_copy.get("factory")
    if custom_reference:
        custom = _load_symbol(str(custom_reference))
        if not callable(custom):
            raise TypeError(f"shield factory is not callable: {custom_reference!r}")

        def custom_factory() -> Any:
            shield = _invoke_extension(
                custom,
                cfg=copy.deepcopy(dict(cfg)),
                spec=copy.deepcopy(spec_copy),
                device=torch.device(device),
            )
            if not hasattr(shield, "apply"):
                raise TypeError("shield factory must return an object with apply()")
            return shield

        return custom_factory

    calibration = importlib.import_module("leo_pg.paper.calibration")
    calibrator_spec = spec_copy.get("calibrator")

    def built_in_factory() -> Any:
        calibrator = None
        if calibrator_spec:
            if not isinstance(calibrator_spec, Mapping):
                raise TypeError("risk_shield.calibrator must be a mapping")
            calibrator_reference = calibrator_spec.get("factory")
            if not calibrator_reference:
                raise ValueError(
                    "a calibrator specification must provide factory='module:callable'; "
                    "fitted calibration parameters are never inferred"
                )
            calibrator_factory = _load_symbol(str(calibrator_reference))
            calibrator = _invoke_extension(
                calibrator_factory,
                cfg=copy.deepcopy(dict(cfg)),
                spec=copy.deepcopy(dict(calibrator_spec)),
                device=torch.device(device),
            )

        config_class = getattr(calibration, "RiskShieldConfig")
        shield_config = config_class(
            delta=float(spec_copy.get("delta", 1.0)),
            protect_existing_association=bool(
                spec_copy.get("protect_existing_association", True)
            ),
            mode=str(spec_copy.get("mode", "veto")),
            downweight_strength=float(spec_copy.get("downweight_strength", 1.0)),
        )
        return _BuiltInRiskAdjustment(
            calibration_module=calibration,
            shield_config=shield_config,
            calibrator=calibrator,
        )

    return built_in_factory


def _run_paired_with_shield(
    *,
    environment_factory: Callable[[], PaperAlignedLEOEnv],
    modes: Sequence[SubstitutionMode],
    descriptor_provider_factory: Callable[[], Any],
    policy_initializer_factory: Callable[[], Any],
    shield_factory: Callable[[], Any],
    allow_oracle_warm_start: bool,
) -> dict[SubstitutionMode, ClosedLoopResult]:
    """Shielded paired evaluation with fresh state for every condition."""

    results: dict[SubstitutionMode, ClosedLoopResult] = {}
    for mode in modes:
        env = environment_factory()
        provider = None if mode is SubstitutionMode.ORACLE else descriptor_provider_factory()
        initializer = (
            None if mode is SubstitutionMode.ORACLE else policy_initializer_factory()
        )
        base_policy = FixedRankPolicy(env.fixed_policy_config())
        shield = shield_factory()
        wrap_fixed = getattr(shield, "wrap_fixed_policy", None)
        policy = (
            wrap_fixed(base_policy)
            if callable(wrap_fixed)
            else _ShieldedPolicyAdapter(base_policy, shield)
        )
        results[mode] = run_closed_loop(
            env,
            mode=mode,
            descriptor_provider=provider,
            policy_initializer=initializer,
            policy=policy,
            allow_oracle_warm_start=allow_oracle_warm_start,
        )

    protocols = {result.protocol_fingerprint for result in results.values()}
    if len(protocols) != 1:
        raise RuntimeError("paired shielded conditions produced different protocols")
    if len({result.action_count for result in results.values()}) != 1:
        raise RuntimeError("paired shielded conditions used different horizons")
    initial_states = {
        (
            tuple(
                tuple(int(value) for value in edge)
                for edge in result.records[0].candidate_edge_ids.tolist()
            ),
            result.records[0]
            .sim_descriptors.policy_fields.gamma_edge.detach()
            .cpu()
            .numpy()
            .tobytes(),
            result.records[0]
            .sim_descriptors.policy_fields.intensity_edge.detach()
            .cpu()
            .numpy()
            .tobytes(),
            result.records[0]
            .sim_descriptors.policy_fields.flow_node.detach()
            .cpu()
            .numpy()
            .tobytes(),
            result.records[0]
            .sim_descriptors.feasible_edge.detach()
            .cpu()
            .numpy()
            .tobytes(),
        )
        for result in results.values()
        if result.records
    }
    if len(initial_states) > 1:
        raise RuntimeError("paired shielded conditions did not start from identical state")
    return results


def serialize_closed_loop_result(result: ClosedLoopResult) -> dict[str, Any]:
    """Serialize every field in the evaluator's per-decision audit trace."""

    records: list[dict[str, Any]] = []
    for record in result.records:
        records.append(
            {
                "observation_id": [
                    int(record.observation_id[0]),
                    int(record.observation_id[1]),
                ],
                "mode": record.mode.value,
                "candidate_edge_ids": _cpu_tensor(record.candidate_edge_ids),
                "sim_descriptors": {
                    **serialize_policy_descriptors(
                        record.sim_descriptors.policy_fields
                    ),
                    "feasible_edge": _cpu_tensor(
                        record.sim_descriptors.feasible_edge
                    ),
                },
                "model_input_descriptors": serialize_policy_descriptors(
                    record.model_input_descriptors
                ),
                "model_descriptors": (
                    None
                    if record.model_descriptors is None
                    else serialize_policy_descriptors(record.model_descriptors)
                ),
                "next_model_prediction": (
                    None
                    if record.next_model_prediction is None
                    else serialize_policy_descriptors(record.next_model_prediction)
                ),
                "policy_descriptors": serialize_policy_descriptors(
                    record.policy_descriptors
                ),
                "initialized_edge": _cpu_tensor(record.initialized_edge),
                "feasible_edge": _cpu_tensor(record.feasible_edge),
                "action": serialize_action(record.action),
                "execution": serialize_execution(record.execution),
            }
        )
    return {
        "mode": result.mode.value,
        "episode_seed": int(result.episode_seed),
        "protocol_version": int(result.protocol_version),
        "protocol_fingerprint": str(result.protocol_fingerprint),
        "provider_fingerprint": result.provider_fingerprint,
        "initializer_kind": result.initializer_kind,
        "initializer_fingerprint": result.initializer_fingerprint,
        "policy_config": _pure(asdict(result.policy_config), path="policy_config"),
        "hard_feasibility_mask": bool(result.hard_feasibility_mask),
        "initial_policy_strategy": str(result.initial_policy_strategy),
        "new_edge_strategy": str(result.new_edge_strategy),
        "records": records,
        "final_serving": _cpu_tensor(result.final_serving),
        "final_flow": _cpu_tensor(result.final_flow),
    }


def _resolve_hook(
    spec: str | Mapping[str, Any] | None,
    *,
    default_module: str,
    default_names: Sequence[str],
) -> tuple[Callable[..., Any] | None, dict[str, Any]]:
    if spec is None:
        return None, {"enabled": False}
    if isinstance(spec, str):
        reference = spec
        options: dict[str, Any] = {}
    elif isinstance(spec, Mapping):
        if not bool(spec.get("enabled", True)):
            return None, {"enabled": False}
        reference = str(spec.get("factory", "")).strip()
        options = copy.deepcopy(dict(spec.get("options", {})))
    else:
        raise TypeError("hook spec must be a string, mapping, or None")

    if reference:
        hook = _load_symbol(reference)
    else:
        module = importlib.import_module(default_module)
        hook = None
        reference = default_module
        for name in default_names:
            candidate = getattr(module, name, None)
            if callable(candidate):
                hook = candidate
                reference = f"{default_module}:{name}"
                break
        if hook is None:
            raise AttributeError(
                f"{default_module!r} exposes none of the supported hooks: "
                + ", ".join(default_names)
            )
    if not callable(hook):
        raise TypeError(f"hook is not callable: {reference!r}")
    return hook, {"enabled": True, "factory": reference, "options": options}


def _apply_result_hook(
    hook: Callable[..., Any] | None,
    *,
    result: Any,
    context: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Any:
    if hook is None:
        return None
    value = _invoke_extension(
        hook,
        result=result,
        context=dict(context),
        options=copy.deepcopy(dict(options)),
    )
    return _pure(value, path="hook_result")


def _is_builtin_metrics_adapter(hook: Callable[..., Any] | None) -> bool:
    return bool(
        hook is not None
        and getattr(hook, "__module__", "") == "leo_pg.paper.metrics"
        and getattr(hook, "__name__", "") == "evaluate_closed_loop_result"
    )


def _compact_metric_aggregation(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Keep estimates/CIs in JSON while leaving unit vectors in the trace bundle."""

    compact_effects: dict[str, Any] = {}
    raw_effects = aggregate.get("paired_effects_vs_oracle", {})
    if not isinstance(raw_effects, Mapping):
        raise TypeError("paired_effects_vs_oracle must be a mapping")
    for condition, raw in raw_effects.items():
        if not isinstance(raw, Mapping):
            raise TypeError("paired effect rows must be mappings")
        metrics = raw.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise TypeError("paired effect metrics must be a mapping")
        compact_effects[str(condition)] = {
            key: _pure(value, path=f"metric_aggregation.effect.{condition}.{key}")
            for key, value in raw.items()
            if key != "metrics"
        }
        compact_effects[str(condition)]["metrics"] = {
            str(metric): {
                key: _pure(
                    value,
                    path=f"metric_aggregation.effect.{condition}.{metric}.{key}",
                )
                for key, value in details.items()
                if key != "unit_differences"
            }
            for metric, details in metrics.items()
            if isinstance(details, Mapping)
        }

    compact_tail: dict[str, Any] = {}
    raw_tail = aggregate.get("tail_risk_p10", {})
    if not isinstance(raw_tail, Mapping):
        raise TypeError("tail_risk_p10 must be a mapping")
    for condition, row in raw_tail.items():
        if not isinstance(row, Mapping):
            raise TypeError("tail-risk rows must be mappings")
        compact_tail[str(condition)] = {
            key: _pure(value, path=f"metric_aggregation.tail.{condition}.{key}")
            for key, value in row.items()
            if key != "unit_ratios"
        }

    return {
        "enabled": True,
        "schema_version": aggregate["schema_version"],
        "metric_contract_version": aggregate["metric_contract_version"],
        "unit_contract": _pure(aggregate["unit_contract"]),
        "bootstrap": _pure(aggregate["bootstrap"]),
        "oracle_condition": aggregate["oracle_condition"],
        "condition_summaries": _pure(aggregate["condition_summaries"]),
        "paired_effects_vs_oracle": compact_effects,
        "tail_risk_p10": compact_tail,
        "controller_selection": _pure(aggregate["controller_selection"]),
        "unit_vectors_location": "payload.aggregates.metrics",
    }


def _built_in_shrink_jump_audit(
    result: ClosedLoopResult,
    *,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the shrink-jump audit lazily from a closed-loop result."""

    if result.mode is SubstitutionMode.ORACLE:
        return {"status": "not_applicable", "reason": "oracle has no model error"}
    audit = importlib.import_module("leo_pg.paper.audit")
    records = [
        record for record in result.records if record.model_descriptors is not None
    ]
    if len(records) < 2:
        return {"status": "insufficient_samples", "sample_count": len(records)}

    scales_config = dict(options.get("scales", {}))
    scales = getattr(audit, "DescriptorScales")(
        gamma=float(scales_config.get("gamma", 10.0)),
        intensity=float(scales_config.get("intensity", 1.0)),
        flow=float(scales_config.get("flow", 1.0)),
    )
    error_norms = [
        getattr(audit, "descriptor_error_norm")(
            record.model_descriptors,
            record.sim_descriptors.policy_fields,
            scales=scales,
            norm=str(options.get("norm", "l2")),
        )
        for record in records
    ]
    switch_between = [
        bool(records[index].execution.handover_executed.any().item())
        for index in range(len(records) - 1)
    ]
    samples = getattr(audit, "build_shrink_jump_samples")(
        error_norms,
        switch_between,
        epsilon=float(options.get("epsilon", 1e-8)),
        seed=int(result.episode_seed),
    )
    raw_dwell = options.get("dwell_steps", result.policy_config.min_dwell_steps)
    dwell_values = (
        [int(value) for value in raw_dwell]
        if isinstance(raw_dwell, (list, tuple))
        else [int(raw_dwell)]
    )
    raw_quantiles = options.get("quantiles")
    probability_pairs = (
        [
            (float(probability), float(probability))
            for probability in raw_quantiles
        ]
        if isinstance(raw_quantiles, (list, tuple))
        else [
            (
                float(options.get("alpha_probability", 0.95)),
                float(options.get("beta_probability", 0.95)),
            )
        ]
    )
    if not dwell_values:
        raise ValueError("shrink-jump dwell_steps must be non-empty")
    if not probability_pairs:
        raise ValueError("shrink-jump quantiles must be non-empty")
    summaries: list[dict[str, Any]] = []
    try:
        for dwell_steps in dwell_values:
            for alpha_probability, beta_probability in probability_pairs:
                summary = getattr(audit, "summarize_shrink_jump")(
                    samples,
                    dwell_steps=dwell_steps,
                    alpha_probability=alpha_probability,
                    beta_probability=beta_probability,
                    epsilon=float(options.get("epsilon", 1e-8)),
                )
                summaries.append(summary.as_dict())
    except ValueError as exc:
        return {
            "status": "insufficient_samples",
            "sample_count": len(samples),
            "reason": str(exc),
            "samples": _pure(
                [asdict(sample) for sample in samples],
                path="audit.samples",
            ),
        }
    return {
        "status": "ok",
        "sample_count": len(samples),
        "summaries": _pure(summaries, path="audit.summaries"),
        "samples": _pure(
            [asdict(sample) for sample in samples],
            path="audit.samples",
        ),
    }


def _controller_from_condition(
    cfg: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    device: str | torch.device,
) -> Any:
    reference = condition.get("factory")
    if reference:
        factory = _load_symbol(str(reference))
        controller = _invoke_extension(
            factory,
            cfg=copy.deepcopy(dict(cfg)),
            condition=copy.deepcopy(dict(condition)),
            device=torch.device(device),
        )
    else:
        controllers = importlib.import_module("leo_pg.paper.controllers")
        controller = getattr(controllers, "build_controller")(
            str(condition["name"]),
            condition.get("parameters") or {},
        )
    if not hasattr(controller, "select_action"):
        raise TypeError("controller must expose select_action(observation)")
    return controller


def _run_classical_condition(
    cfg: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    device: str | torch.device,
    shield_factory: Callable[[], Any] | None,
) -> dict[str, Any]:
    """Run a simulator-oracle classical controller with a full observation trace."""

    env = PaperAlignedLEOEnv(copy.deepcopy(dict(cfg)), device=device)
    controller = _controller_from_condition(cfg, condition, device=device)
    reset = getattr(controller, "reset", None)
    if callable(reset):
        reset()
    shield = None if shield_factory is None else shield_factory()

    observation = env.reset_control()
    records: list[dict[str, Any]] = []
    while observation is not None:
        proposed = controller.select_action(observation)
        action = proposed if shield is None else shield.apply(observation, proposed)
        next_observation, execution, _done = env.step_action(action)
        records.append(
            {
                "observation": serialize_observation(observation),
                "proposed_action": serialize_action(proposed),
                "action": serialize_action(action),
                "execution": serialize_execution(execution),
            }
        )
        observation = next_observation

    label = str(condition.get("label") or condition.get("name") or "controller")
    return {
        "condition": label,
        "controller_name": str(condition.get("name", label)),
        "controller_factory": condition.get("factory"),
        "controller_parameters": _pure(
            dict(condition.get("parameters") or {}),
            path="controller.parameters",
        ),
        "sweep_provenance": _pure(
            dict(condition.get("sweep_provenance") or {}),
            path="controller.sweep_provenance",
        ),
        "episode_seed": int(env.seed),
        "protocol_version": int(PAPER_PROTOCOL_VERSION),
        "protocol_fingerprint": str(env.protocol_fingerprint),
        "hard_feasibility_mask": bool(env.hard_feasibility_mask),
        "records": records,
        "final_serving": _cpu_tensor(env.current_serving),
        "final_flow": _cpu_tensor(env.flow),
    }


def _episode_seed(base_seed: int, episode_index: int) -> int:
    if int(episode_index) < 0 or int(episode_index) >= 1_000_000:
        raise ValueError("episode_index must lie in [0, 1000000)")
    seed = int(base_seed) * 1_000_000 + int(episode_index)
    if seed < 0:
        raise ValueError("derived episode seeds must be non-negative")
    return seed


def run_paper_evaluation(
    cfg: Mapping[str, Any],
    model: nn.Module,
    *,
    checkpoint_manifest: Mapping[str, Any],
    seeds: Sequence[int],
    episodes_per_seed: int = 1,
    horizon: int | None = None,
    modes: Sequence[str | SubstitutionMode] = DEFAULT_PAIRED_MODES,
    device: str | torch.device = "cpu",
    initializer_factory: Callable[[], Any] | None = None,
    initializer_manifest: Mapping[str, Any] | None = None,
    classical_conditions: Sequence[Mapping[str, Any]] = (),
    shield_spec: Mapping[str, Any] | None = None,
    metrics_spec: str | Mapping[str, Any] | None = None,
    audit_spec: str | Mapping[str, Any] | None = None,
    allow_oracle_warm_start: bool = False,
) -> dict[str, Any]:
    """Run paired action-coupled paper evaluation across seeds and episodes."""

    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    if episodes_per_seed <= 0:
        raise ValueError("episodes_per_seed must be positive")
    if horizon is not None and (
        isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) <= 0
    ):
        raise ValueError("horizon must be a positive integer")
    parsed_modes = parse_modes(modes)
    base_cfg = copy.deepcopy(dict(cfg))
    model.eval()
    classical_labels: set[str] = set()
    for condition in classical_conditions:
        label = str(condition.get("label") or condition.get("name") or "controller")
        if label in classical_labels:
            raise ValueError(f"duplicate classical condition label: {label!r}")
        classical_labels.add(label)
    provider_template = (
        None
        if all(mode is SubstitutionMode.ORACLE for mode in parsed_modes)
        else TGNDescriptorProvider(model, device=device)
    )

    def fresh_provider() -> TGNDescriptorProvider:
        if provider_template is None:
            raise RuntimeError("oracle-only evaluation does not create a model provider")
        provider = copy.copy(provider_template)
        provider.reset()
        return provider

    if initializer_factory is None:
        initializer_factory = make_initializer_factory(base_cfg, device=device)
        initializer_manifest = {"kind": "configured_prior"}

    metrics_hook, metrics_manifest = _resolve_hook(
        metrics_spec,
        default_module="leo_pg.paper.metrics",
        default_names=(
            "evaluate_closed_loop_result",
            "summarize_closed_loop_result",
            "compute_closed_loop_metrics",
        ),
    )
    if _is_builtin_metrics_adapter(metrics_hook) and SubstitutionMode.ORACLE not in parsed_modes:
        raise ValueError(
            "standard paper metrics require the oracle mode so every seed-by-episode "
            "unit has a matched normalization and paired-effect reference"
        )
    audit_hook: Callable[..., Any] | None
    audit_manifest: dict[str, Any]
    built_in_audit = False
    if audit_spec is None:
        audit_hook = None
        audit_manifest = {"enabled": False}
    elif isinstance(audit_spec, Mapping) and not audit_spec.get("factory"):
        if not bool(audit_spec.get("enabled", True)):
            audit_hook = None
            audit_manifest = {"enabled": False}
        else:
            audit_hook = None
            built_in_audit = True
            audit_manifest = {
                "enabled": True,
                "factory": "leo_pg.paper.audit:built_in_shrink_jump_adapter",
                "options": copy.deepcopy(dict(audit_spec.get("options", {}))),
            }
    else:
        audit_hook, audit_manifest = _resolve_hook(
            audit_spec,
            default_module="leo_pg.paper.audit",
            default_names=("evaluate_shrink_jump", "audit_closed_loop_result"),
        )

    shield_factory = make_shield_factory(base_cfg, shield_spec, device=device)
    shield_manifest = (
        {"enabled": False}
        if shield_factory is None
        else {
            "enabled": True,
            "spec": _pure(dict(shield_spec or {}), path="shield.spec"),
        }
    )

    derived_seeds: list[int] = []
    for base_seed in seeds:
        for episode_index in range(episodes_per_seed):
            derived_seeds.append(_episode_seed(int(base_seed), episode_index))
    if len(set(derived_seeds)) != len(derived_seeds):
        raise ValueError(
            "base seeds and episodes_per_seed derive duplicate simulator seeds; "
            "base seeds must be unique and each run may contain fewer than 1000000 episodes"
        )

    units: list[dict[str, Any]] = []
    protocol_fingerprints: set[str] = set()
    provider_fingerprints: set[str] = set()
    initializer_fingerprints: set[str] = set()
    for base_seed in seeds:
        for episode_index in range(episodes_per_seed):
            episode_cfg = copy.deepcopy(base_cfg)
            episode_cfg["seed"] = _episode_seed(int(base_seed), episode_index)
            if horizon is not None:
                protocol = episode_cfg.get("paper_protocol")
                if not isinstance(protocol, Mapping):
                    raise TypeError("paper_protocol must be a mapping")
                protocol_copy = dict(protocol)
                protocol_copy["horizon_steps"] = int(horizon)
                episode_cfg["paper_protocol"] = protocol_copy

            def environment_factory(c=episode_cfg):
                return PaperAlignedLEOEnv(
                    copy.deepcopy(c),
                    device=device,
                )
            provider_factory = fresh_provider
            if shield_factory is None:
                paired = run_paired_conditions(
                    environment_factory=environment_factory,
                    modes=parsed_modes,
                    descriptor_provider_factory=provider_factory,
                    policy_initializer_factory=initializer_factory,
                    allow_oracle_warm_start=allow_oracle_warm_start,
                )
            else:
                paired = _run_paired_with_shield(
                    environment_factory=environment_factory,
                    modes=parsed_modes,
                    descriptor_provider_factory=provider_factory,
                    policy_initializer_factory=initializer_factory,
                    shield_factory=shield_factory,
                    allow_oracle_warm_start=allow_oracle_warm_start,
                )

            serialized_paired: dict[str, Any] = {}
            metrics_by_mode: dict[str, Any] = {}
            audit_by_mode: dict[str, Any] = {}
            for mode, result in paired.items():
                serialized_paired[mode.value] = serialize_closed_loop_result(result)
                context = {
                    "condition_type": "paired_substitution",
                    "mode": mode.value,
                    "base_seed": int(base_seed),
                    "episode_index": int(episode_index),
                    "episode_seed": int(result.episode_seed),
                    "matched_oracle_result": (
                        None
                        if mode is SubstitutionMode.ORACLE
                        else paired.get(SubstitutionMode.ORACLE)
                    ),
                }
                if metrics_hook is not None:
                    metrics_by_mode[mode.value] = _apply_result_hook(
                        metrics_hook,
                        result=result,
                        context=context,
                        options=metrics_manifest.get("options", {}),
                    )
                if built_in_audit:
                    audit_by_mode[mode.value] = _built_in_shrink_jump_audit(
                        result,
                        options=audit_manifest.get("options", {}),
                    )
                elif audit_hook is not None:
                    audit_by_mode[mode.value] = _apply_result_hook(
                        audit_hook,
                        result=result,
                        context=context,
                        options=audit_manifest.get("options", {}),
                    )
                protocol_fingerprints.add(result.protocol_fingerprint)
                if result.mode is not SubstitutionMode.ORACLE:
                    provider_fingerprints.add(result.provider_fingerprint)
                if result.mode is not SubstitutionMode.ORACLE:
                    initializer_fingerprints.add(result.initializer_fingerprint)

            classical: dict[str, Any] = {}
            classical_metrics: dict[str, Any] = {}
            matched_oracle_result = paired.get(SubstitutionMode.ORACLE)
            for condition in classical_conditions:
                classical_result = _run_classical_condition(
                    episode_cfg,
                    condition,
                    device=device,
                    shield_factory=shield_factory,
                )
                label = str(classical_result["condition"])
                if label in classical:
                    raise ValueError(f"duplicate classical condition label: {label!r}")
                classical[label] = classical_result
                protocol_fingerprints.add(str(classical_result["protocol_fingerprint"]))
                if metrics_hook is not None:
                    classical_metrics[label] = _apply_result_hook(
                        metrics_hook,
                        result=classical_result,
                        context={
                            "condition_type": "classical_controller",
                            "condition": label,
                            "base_seed": int(base_seed),
                            "episode_index": int(episode_index),
                            "episode_seed": int(classical_result["episode_seed"]),
                            "matched_oracle_result": matched_oracle_result,
                        },
                        options=metrics_manifest.get("options", {}),
                    )

            unit: dict[str, Any] = {
                "base_seed": int(base_seed),
                "episode_index": int(episode_index),
                "episode_seed": int(episode_cfg["seed"]),
                "paired": serialized_paired,
                "classical": classical,
            }
            if metrics_by_mode or classical_metrics:
                unit["metrics"] = {
                    "paired": metrics_by_mode,
                    "classical": classical_metrics,
                }
            if audit_by_mode:
                unit["shrink_jump_audit"] = audit_by_mode
            units.append(unit)

    if len(provider_fingerprints) > 1:
        raise RuntimeError("evaluation used more than one model/provider fingerprint")
    if len(initializer_fingerprints) > 1:
        raise RuntimeError("evaluation used more than one initializer fingerprint")

    metric_aggregates: dict[str, Any] | None = None
    if _is_builtin_metrics_adapter(metrics_hook):
        metrics_module = importlib.import_module("leo_pg.paper.metrics")
        aggregate_hook = getattr(metrics_module, "aggregate_evaluation_units")
        metric_aggregates = _pure(
            _invoke_extension(
                aggregate_hook,
                units=units,
                options=copy.deepcopy(dict(metrics_manifest.get("options", {}))),
                context={
                    "base_seeds": [int(seed) for seed in seeds],
                    "episodes_per_seed": int(episodes_per_seed),
                    "paired_modes": [mode.value for mode in parsed_modes],
                },
            ),
            path="metric_aggregates",
        )

    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "leo_pg_action_coupled_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(torch.device(device)),
        "checkpoint": _pure(dict(checkpoint_manifest), path="checkpoint"),
        "model_provider_fingerprints": sorted(provider_fingerprints),
        "initializer": {
            **_pure(
                dict(initializer_manifest or {}),
                path="initializer.manifest",
            ),
            "observed_fingerprints": sorted(initializer_fingerprints),
        },
        "protocol_fingerprints": sorted(protocol_fingerprints),
        "paired_modes": [mode.value for mode in parsed_modes],
        "base_seeds": [int(seed) for seed in seeds],
        "episodes_per_seed": int(episodes_per_seed),
        "episode_seed_derivation": "base_seed * 1000000 + episode_index",
        "episode_count": len(units),
        "horizon_override": None if horizon is None else int(horizon),
        "allow_oracle_warm_start": bool(allow_oracle_warm_start),
        "action_coupled": True,
        "paired_fresh_environment_per_mode": True,
        "shield": shield_manifest,
        "classical_conditions": _pure(
            list(classical_conditions),
            path="classical_conditions",
        ),
        "metrics_hook": _pure(metrics_manifest, path="metrics_hook"),
        "metric_aggregation": (
            {"enabled": False}
            if metric_aggregates is None
            else _compact_metric_aggregation(metric_aggregates)
        ),
        "audit_hook": _pure(audit_manifest, path="audit_hook"),
        "trace_contract": {
            "paired": "complete ClosedLoopResult and ClosedLoopStep fields",
            "classical": "complete observation/proposal/action/execution at every step",
        },
    }
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "manifest": manifest,
        "units": units,
    }
    if metric_aggregates is not None:
        payload["aggregates"] = {"metrics": metric_aggregates}
    return _pure(payload)


def save_evaluation_bundle(
    payload: Mapping[str, Any],
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Save the full weights-only trace bundle and a readable manifest."""

    pure_payload = _pure(dict(payload))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pure_payload, output)

    manifest_output = (
        Path(manifest_path)
        if manifest_path is not None
        else output.with_suffix(".manifest.json")
    )
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(
            _json_tree(pure_payload["manifest"]),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output, manifest_output


__all__ = [
    "DEFAULT_PAIRED_MODES",
    "EVALUATION_SCHEMA_VERSION",
    "load_frozen_model",
    "make_initializer_factory",
    "make_shield_factory",
    "parse_modes",
    "run_paper_evaluation",
    "save_evaluation_bundle",
    "serialize_closed_loop_result",
]
