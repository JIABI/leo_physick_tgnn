#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CFG="${CFG:-configs/smoke.yaml}"
MESSAGE="${MESSAGE:-physick}"
DATA_PATH="${DATA_PATH:-data/synthetic_debug.pt}"
SPLIT="${SPLIT:-test}"
HS="${HS:-5,10}"
MAX_EPS="${MAX_EPS:-2}"
WHICH="${WHICH:-last}"
MODE="${MODE:-}"
DEVICE="${DEVICE:-cpu}"
CKPT="${CKPT:-}"
OUT="${OUT:-}"

args=(
  scripts/rollout.py
  --cfg "$CFG"
  --message "$MESSAGE"
  --data "$DATA_PATH"
  --split "$SPLIT"
  --which "$WHICH"
  --Hs "$HS"
  --max_eps "$MAX_EPS"
  --device "$DEVICE"
)
if [[ -n "$MODE" ]]; then
  args+=(--mode "$MODE")
fi
if [[ -n "$CKPT" ]]; then
  args+=(--ckpt "$CKPT")
fi
if [[ -n "$OUT" ]]; then
  args+=(--out "$OUT")
fi

"$PYTHON" "${args[@]}"
