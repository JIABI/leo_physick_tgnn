#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED="0"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m controller_facing_state.cli verify-config
"${PYTHON_BIN}" -m controller_facing_state.cli audit-model-identities

if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [UNPACKED_ZENODO_V4_ROOT]" >&2
  exit 2
fi

if [[ "$#" -eq 1 ]]; then
  "${PYTHON_BIN}" -m controller_facing_state.cli verify-source-data --source-data "$1"
fi
