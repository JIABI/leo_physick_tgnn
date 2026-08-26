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
exec "${PYTHON_BIN}" -m controller_facing_state.cli demo "$@"
