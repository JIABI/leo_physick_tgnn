from __future__ import annotations

from typing import Any, Dict


def resolve_run_name(
    cfg: Dict[str, Any],
    message_type: str | None = None,
    mode: str | None = None,
) -> str:
    """Resolve one consistent run directory name across train/eval/rollout."""
    train_cfg = cfg.get("train", {})
    message = str(message_type or cfg.get("model", {}).get("message_type", "mlp"))
    mode_value = str(mode if mode is not None else train_cfg.get("mode", cfg.get("mode", ""))).strip()
    raw = str(train_cfg.get("run_name", "run_$message_type$"))
    name = raw
    for token in ("message_type", "message"):
        name = name.replace(f"${token}$", message).replace(f"{{{token}}}", message)
    name = name.replace("$mode$", mode_value).replace("{mode}", mode_value)

    has_message_placeholder = any(token in raw for token in ("$message_type$", "{message_type}", "$message$", "{message}"))
    if not has_message_placeholder and message not in name.split("_"):
        name = f"{name}_{message}"

    has_mode_placeholder = "$mode$" in raw or "{mode}" in raw
    if mode_value and not has_mode_placeholder and mode_value not in name.split("_"):
        name = f"{name}_{mode_value}"
    return name.strip("_")

