#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  echo "Missing pyproject.toml in ${ROOT_DIR}" >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  if [[ "${CFS_USE_SYSTEM_SITE_PACKAGES:-0}" == "1" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  else
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
fi

if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "The virtual environment has no working pip: ${VENV_DIR}" >&2
  echo "Move that generated directory aside or choose a new VENV_DIR, then rerun setup.sh." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" -m pip install "setuptools>=69" wheel
"${VENV_DIR}/bin/python" -m pip install --no-build-isolation \
  -e "${ROOT_DIR}[test]" \
  -e "${ROOT_DIR}/code/satellite[test]" \
  -e "${ROOT_DIR}/code/uav"

echo "Environment ready: ${VENV_DIR}"
echo "Run ${ROOT_DIR}/run_demo.sh to execute the CPU semantic demonstration."
