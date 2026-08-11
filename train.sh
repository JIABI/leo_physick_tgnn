#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CFG="${CFG:-configs/smoke.yaml}"
MESSAGE="${MESSAGE:-physick}"
DATA_PATH="${DATA_PATH:-data/synthetic_debug.pt}"
MODE="${MODE:-}"
DEVICE="${DEVICE:-cpu}"

args=(scripts/train.py --cfg "$CFG" --message "$MESSAGE" --data "$DATA_PATH" --device "$DEVICE")
if [[ -n "$MODE" ]]; then
  args+=(--mode "$MODE")
fi

"$PYTHON" "${args[@]}"
