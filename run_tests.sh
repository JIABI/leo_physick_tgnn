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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

cd "${ROOT_DIR}"
if [[ "$#" -gt 0 ]]; then
  PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/code:${ROOT_DIR}/code/satellite/src:${ROOT_DIR}/code/satellite:${ROOT_DIR}/code/uav" \
    exec "${PYTHON_BIN}" -m pytest "$@"
fi
"${PYTHON_BIN}" -m pytest -q tests
(
  cd "${ROOT_DIR}/code/satellite"
  PYTHONPATH="src:." "${PYTHON_BIN}" -m pytest -q tests
)
(
  cd "${ROOT_DIR}/code/uav"
  PYTHONPATH="." "${PYTHON_BIN}" -m pytest -q tests
)
