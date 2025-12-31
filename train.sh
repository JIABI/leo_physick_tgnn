#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
# Usage:
#   MODE=single MESSAGE=mlp  ./scripts/train.sh
#   MODE=multi  MESSAGE=physick DATA_MULTI=data/starlink_like_2000.pt ./scripts/train.sh
#
# Notes:
#   - MODE only affects which dataset path is used and run naming.
#   - MESSAGE selects the message kernel (mlp|kan|physick).

MODE="${MODE:-multi}"          # single|multi
MESSAGE="${MESSAGE:-mlp}"      # mlp|kan|physick
CFG="${CFG:-configs/default.yaml}"

DATA_SINGLE="${DATA_SINGLE:-data/starlink_like_2000_single.pt}"
DATA_MULTI="${DATA_MULTI:-data/starlink_like_1000.pt}"

if [[ "$MODE" == "single" ]]; then
  DATA_PATH="${DATA_PATH:-$DATA_SINGLE}"
elif [[ "$MODE" == "multi" ]]; then
  DATA_PATH="${DATA_PATH:-$DATA_MULTI}"
else
  echo "[ERR] MODE must be single or multi, got: $MODE" >&2
  exit 1
fi

#python scripts/train.py --cfg "$CFG" --message mlp --mode "$MODE" --data "$DATA_PATH"
#python scripts/train.py --cfg "$CFG" --message kan --mode "$MODE" --data "$DATA_PATH"
python scripts/train.py --cfg "$CFG" --message physick --mode "$MODE" --data "$DATA_PATH"
