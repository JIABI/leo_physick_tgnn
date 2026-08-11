"""Reproducible runtime and peak-memory profiling primitives.

The module is deliberately platform-agnostic.  Paper NTN, Snapshot, and UAV
adapters live in ``scripts/profile_runtime.py`` and supply zero-argument
operations with an explicit timing contract.  No optimizer is constructed and
no target is accepted by this API.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn


RUNTIME_PROFILE_SCHEMA_VERSION = 1
RUNTIME_PROFILE_CSV_VERSION = 1
RUNTIME_PROFILE_KIND = "leo_pg.runtime_profile"


@dataclass(frozen=True)
class RuntimeProfileConfig:
    """Timing controls shared by every profiled stage."""

    warmup_steps: int = 20
    repeats: int = 600

    def __post_init__(self) -> None:
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if type(self.repeats) is not int or self.repeats <= 0:
            raise ValueError("repeats must be a positive integer")


def file_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    """Return release-safe checkpoint identity without deserializing it."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    return {
        "file_name": source.name,
        "size_bytes": int(source.stat().st_size),
        "sha256": file_sha256(source),
        "load_contract": "torch_weights_only_strict_signature_checked",
    }


def model_inventory(
    model: nn.Module,
    *,
    declared_trainable_parameters: int | None = None,
) -> dict[str, Any]:
    """Describe model size and the actual parameter dtype/device layout."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    requiring_grad = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if declared_trainable_parameters is None:
        # Frozen evaluation loaders intentionally clear requires_grad.  In that
        # case the architecture parameter count is the only honest fallback;
        # the provenance field prevents it being confused with live grad state.
        trainable = requiring_grad if requiring_grad > 0 else total
        provenance = (
            "model_requires_grad"
            if requiring_grad > 0
            else "architecture_total_after_evaluation_freeze"
        )
    else:
        if (
            type(declared_trainable_parameters) is not int
            or declared_trainable_parameters < 0
        ):
            raise ValueError("declared_trainable_parameters must be non-negative")
        trainable = declared_trainable_parameters
        provenance = "checkpoint_training_metadata"
    floating_dtypes = sorted(
        {str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters() if parameter.is_floating_point()}
    )
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameters_total": total,
        "parameters_total_millions": total / 1_000_000.0,
        "trainable_parameters": trainable,
        "trainable_parameters_millions": trainable / 1_000_000.0,
        "trainable_parameter_provenance": provenance,
        "parameters_requiring_grad_at_profile_time": requiring_grad,
        "floating_parameter_dtypes": floating_dtypes,
        "parameter_devices": devices,
    }


def hardware_inventory(device: torch.device | str) -> dict[str, Any]:
    target = torch.device(device)
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "requested_device": str(target),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_cuda_device_count": int(torch.cuda.device_count()),
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
        index = target.index if target.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result["cuda"] = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(index)),
            "cuda_runtime_version": torch.version.cuda,
        }
    result["hardware_sha256"] = hashlib.sha256(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ru_maxrss_bytes() -> tuple[int | None, str]:
    """Return process-lifetime peak RSS and its explicit scope."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None, "unavailable"
    # macOS reports bytes; Linux and the BSDs exposed by CPython report KiB.
    if sys.platform == "darwin":
        return value, "process_lifetime_ru_maxrss_bytes"
    return value * 1024, "process_lifetime_ru_maxrss_kib"


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def latency_summary(samples_ms: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples_ms]
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("latency samples must be a non-empty finite non-negative list")
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "stdev_population": statistics.pstdev(values),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
        "samples": values,
    }


def profile_operation(
    name: str,
    operation: Callable[[], Any],
    *,
    device: torch.device | str,
    config: RuntimeProfileConfig,
    timing_contract: str,
    reset_after_warmup: Callable[[], None] | None = None,
    prepare_each: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Profile one operation with CUDA synchronization around every sample.

    ``reset_after_warmup`` is intentionally outside the timed region.  Stateful
    full-decision adapters use it to restart their descriptor-autoregressive
    stream before reported samples are collected.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("profile name must be non-empty")
    if not callable(operation):
        raise TypeError("operation must be callable")
    if not isinstance(timing_contract, str) or not timing_contract.strip():
        raise ValueError("timing_contract must be non-empty")
    target = torch.device(device)
    with torch.inference_mode():
        for _ in range(config.warmup_steps):
            if prepare_each is not None:
                prepare_each()
            operation()
            _synchronize(target)
        if reset_after_warmup is not None:
            reset_after_warmup()
        _synchronize(target)

        if target.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target)
        rss_before, rss_scope = _ru_maxrss_bytes()
        samples_ms: list[float] = []
        for _ in range(config.repeats):
            if prepare_each is not None:
                prepare_each()
            _synchronize(target)
            started = time.perf_counter_ns()
            operation()
            _synchronize(target)
            elapsed_ns = time.perf_counter_ns() - started
            samples_ms.append(elapsed_ns / 1_000_000.0)
        rss_peak, _ = _ru_maxrss_bytes()

    cuda_memory: dict[str, int] | None = None
    if target.type == "cuda":
        cuda_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(target)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target)),
            "end_allocated_bytes": int(torch.cuda.memory_allocated(target)),
            "end_reserved_bytes": int(torch.cuda.memory_reserved(target)),
        }
    cpu_delta = (
        None
        if rss_before is None or rss_peak is None
        else max(0, int(rss_peak) - int(rss_before))
    )
    cpu_memory = {
        "ru_maxrss_before_bytes": rss_before,
        "ru_maxrss_peak_bytes": rss_peak,
        "ru_maxrss_observed_delta_bytes": cpu_delta,
        "scope": rss_scope,
        "limitation": (
            "ru_maxrss is a process-lifetime high-water mark; its delta can be "
            "zero when model loading or an earlier stage already set a higher peak"
        ),
    }
    selected_peak = (
        cuda_memory["peak_allocated_bytes"]
        if cuda_memory is not None
        else rss_peak
    )
    return {
        "name": name.strip(),
        "timing_contract": timing_contract.strip(),
        "device": str(target),
        "warmup_steps": config.warmup_steps,
        "repeats": config.repeats,
        "cuda_synchronized_each_sample": target.type == "cuda",
        "latency_ms": latency_summary(samples_ms),
        "memory": {
            "cpu_rss": cpu_memory,
            "cuda": cuda_memory,
            "reported_peak_kind": (
                "cuda_peak_allocated" if cuda_memory is not None else "cpu_ru_maxrss"
            ),
            "reported_peak_bytes": selected_peak,
            "reported_peak_gib": (
                None if selected_peak is None else selected_peak / float(1024**3)
            ),
        },
    }


def build_runtime_report(
    *,
    platform_name: str,
    method: str,
    model: nn.Module,
    checkpoint: Mapping[str, Any],
    input_provenance: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    device: torch.device | str,
    configuration: Mapping[str, Any],
    evaluation_horizon_steps: int,
    batch_size: int = 1,
    declared_trainable_parameters: int | None = None,
) -> dict[str, Any]:
    if not profiles:
        raise ValueError("at least one profile is required")
    if type(evaluation_horizon_steps) is not int or evaluation_horizon_steps <= 1:
        raise ValueError("evaluation_horizon_steps must be an integer greater than one")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    hardware = hardware_inventory(device)
    full_profile = next(
        (profile for profile in profiles if profile.get("name") == "full_decision_epoch"),
        None,
    )
    comparable_reasons: list[str] = []
    if str(platform_name) not in {"ntn", "snapshot"}:
        comparable_reasons.append(
            "platform is outside the SI 60-s/600-step NTN-Snapshot runtime table"
        )
    if torch.device(device).type != "cuda":
        comparable_reasons.append("SI runtime table requires CUDA; CPU is diagnostic only")
    cuda_hardware = hardware.get("cuda")
    cuda_name = "" if not isinstance(cuda_hardware, Mapping) else str(cuda_hardware.get("name", ""))
    if "A100" not in cuda_name.upper():
        comparable_reasons.append("profiled CUDA device is not an NVIDIA A100")
    if batch_size != 1:
        comparable_reasons.append("SI runtime table requires batch_size=1")
    if evaluation_horizon_steps != 600:
        comparable_reasons.append("SI runtime table requires a 600-step horizon")
    if full_profile is None:
        comparable_reasons.append("full_decision_epoch profile is missing")
    else:
        if int(full_profile.get("repeats", -1)) != evaluation_horizon_steps:
            comparable_reasons.append(
                "measured full-decision samples do not equal the formal horizon"
            )
        if int(full_profile.get("first_measured_epoch", -1)) != 0:
            comparable_reasons.append("full-decision measurement does not start at t=0")
        if not bool(full_profile.get("complete_evaluation_sequence", False)):
            comparable_reasons.append(
                "timing samples do not form one complete evaluation sequence"
            )
        if not bool(full_profile.get("uniform_model_prediction_each_sample", False)):
            comparable_reasons.append(
                "at least one decision sample lacks a model prediction"
            )
    report = {
        "schema_version": RUNTIME_PROFILE_SCHEMA_VERSION,
        "kind": RUNTIME_PROFILE_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": str(platform_name),
        "method": str(method),
        "model": model_inventory(
            model,
            declared_trainable_parameters=declared_trainable_parameters,
        ),
        "checkpoint": dict(checkpoint),
        "configuration": dict(configuration),
        "input": dict(input_provenance),
        "hardware": hardware,
        "profiles": [dict(profile) for profile in profiles],
        "si_runtime_table_comparable": not comparable_reasons,
        "si_runtime_table_comparability_reasons": comparable_reasons,
        "scientific_contract": {
            "training_performed": False,
            "optimizer_constructed": False,
            "target_visible_to_predict_step": False,
            "formal_input_required": True,
            "synthetic_input_allowed": False,
            "batch_size": batch_size,
            "evaluation_horizon_steps": evaluation_horizon_steps,
            "cpu_rss_role": "diagnostic_process_high_water_mark_only",
            "si_runtime_table_comparable": not comparable_reasons,
            "si_runtime_table_comparability_reasons": comparable_reasons,
            "si_runtime_table_reference": (
                "single NVIDIA A100; batch_size=1; complete 60-s/600-step "
                "descriptor-autoregressive decision rollout from t=0"
            ),
            "si_peak_memory_definition": "CUDA peak allocated bytes",
        },
    }
    fingerprint_payload = dict(report)
    fingerprint_payload.pop("created_utc")
    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return report


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    model = report["model"]
    checkpoint = report["checkpoint"]
    input_provenance = report["input"]
    rows: list[dict[str, Any]] = []
    for profile_result in report["profiles"]:
        latency = profile_result["latency_ms"]
        memory = profile_result["memory"]
        cuda = memory.get("cuda") or {}
        cpu = memory["cpu_rss"]
        rows.append(
            {
                "csv_version": RUNTIME_PROFILE_CSV_VERSION,
                "report_schema_version": report["schema_version"],
                "report_sha256": report["report_sha256"],
                "platform": report["platform"],
                "method": report["method"],
                "profile": profile_result["name"],
                "timing_contract": profile_result["timing_contract"],
                "device": profile_result["device"],
                "batch_size": report["scientific_contract"]["batch_size"],
                "evaluation_horizon_steps": report["scientific_contract"]["evaluation_horizon_steps"],
                "si_runtime_table_comparable": report["scientific_contract"]["si_runtime_table_comparable"],
                "si_runtime_table_comparability_reasons": "; ".join(
                    report["scientific_contract"]["si_runtime_table_comparability_reasons"]
                ),
                "config_sha256": report["configuration"].get("sha256"),
                "hardware_sha256": report["hardware"].get("hardware_sha256"),
                "dtype": ";".join(model["floating_parameter_dtypes"]),
                "trainable_params": model["trainable_parameters"],
                "params_m": model["trainable_parameters_millions"],
                "checkpoint_file": checkpoint["file_name"],
                "checkpoint_sha256": checkpoint["sha256"],
                "input_split": input_provenance.get("split"),
                "input_episode_index": input_provenance.get("episode_index"),
                "input_episode_seed": input_provenance.get("episode_seed"),
                "warmup_steps": profile_result["warmup_steps"],
                "repeats": profile_result["repeats"],
                "latency_mean_ms": latency["mean"],
                "latency_stdev_ms": latency["stdev_population"],
                "latency_min_ms": latency["min"],
                "latency_p50_ms": latency["p50"],
                "latency_p90_ms": latency["p90"],
                "latency_p95_ms": latency["p95"],
                "latency_p99_ms": latency["p99"],
                "latency_max_ms": latency["max"],
                "peak_memory_gib": memory["reported_peak_gib"],
                "peak_memory_kind": memory["reported_peak_kind"],
                "cpu_ru_maxrss_peak_bytes": cpu["ru_maxrss_peak_bytes"],
                "cpu_ru_maxrss_delta_bytes": cpu["ru_maxrss_observed_delta_bytes"],
                "cuda_peak_allocated_bytes": cuda.get("peak_allocated_bytes"),
                "cuda_peak_reserved_bytes": cuda.get("peak_reserved_bytes"),
            }
        )
    return rows


def write_runtime_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """Atomically write the versioned rich JSON and flat SI-table CSV."""

    json_output = Path(json_path).expanduser()
    csv_output = Path(csv_path).expanduser()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = json_output.with_suffix(json_output.suffix + ".tmp")
    csv_tmp = csv_output.with_suffix(csv_output.suffix + ".tmp")
    with json_tmp.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    rows = _csv_rows(report)
    if not rows:
        raise ValueError("runtime report contains no CSV rows")
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_tmp.replace(json_output)
    csv_tmp.replace(csv_output)
    return json_output, csv_output


__all__ = [
    "RUNTIME_PROFILE_CSV_VERSION",
    "RUNTIME_PROFILE_KIND",
    "RUNTIME_PROFILE_SCHEMA_VERSION",
    "RuntimeProfileConfig",
    "build_runtime_report",
    "checkpoint_identity",
    "file_sha256",
    "hardware_inventory",
    "latency_summary",
    "model_inventory",
    "profile_operation",
    "write_runtime_report",
]
