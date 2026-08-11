#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CFG="${CFG:-configs/data/synthetic_debug.yaml}"
OUT="${OUT:-data/synthetic_debug.pt}"
EPISODES="${EPISODES:-9}"
SPLIT_RATIOS="${SPLIT_RATIOS:-0.67,0.11,0.22}"
DEVICE="${DEVICE:-cpu}"

"$PYTHON" scripts/gen_data.py \
  --cfg "$CFG" \
  --out "$OUT" \
  --episodes "$EPISODES" \
  --split-ratios "$SPLIT_RATIOS" \
  --device "$DEVICE"
