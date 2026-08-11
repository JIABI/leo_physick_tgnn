#!/usr/bin/env bash
# Reproducible paper pipeline:
# dependencies -> typed data -> method-matrix training -> paired evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN=("${PYTHON_BIN}" -m pip)

CONFIG="${CONFIG:-${REPO_ROOT}/configs/paper_protocol.yaml}"
DATASET="${DATASET:-${REPO_ROOT}/artifacts/data/ntn_paper_schema_v1.pt}"
SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT:-${REPO_ROOT}/artifacts/data/snapshot}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/artifacts/paper_runs}"
# s4/mamba2/conformer are PaperPhysiCK temporal-mixer swaps, not independent
# graph baselines; their resolved checkpoint configs carry that architecture.
METHODS="${METHODS:-tgn_mlp,tgn_kan,tgn_physick,snapshot_mlp,snapshot_physick,snapshot_ltt_r,snapshot_da_gwm,da_gwm,s4,mamba2,conformer,big_mlp,edge_attn,temp_trans}"

SETUP_DEPENDENCIES="${SETUP_DEPENDENCIES:-1}"
INSTALL_OPTIONAL_DEPENDENCIES="${INSTALL_OPTIONAL_DEPENDENCIES:-1}"
REUSE_DATASET="${REUSE_DATASET:-0}"
REUSE_CHECKPOINTS="${REUSE_CHECKPOINTS:-0}"
CONTROLLER_SWEEP="${CONTROLLER_SWEEP:-1}"
EVAL_METRICS="${EVAL_METRICS:-1}"
EVAL_SHRINK_JUMP_AUDIT="${EVAL_SHRINK_JUMP_AUDIT:-1}"
EVAL_RISK_SHIELD="${EVAL_RISK_SHIELD:-0}"
RUN_SNAPSHOT_SCHEDULED_SAMPLING="${RUN_SNAPSHOT_SCHEDULED_SAMPLING:-1}"
RUN_SNAPSHOT_ROLLOUT_LADDER="${RUN_SNAPSHOT_ROLLOUT_LADDER:-1}"
SNAPSHOT_ROLLOUT_HORIZONS="${SNAPSHOT_ROLLOUT_HORIZONS:-5,10,20}"
# Optional SI runtime table stage. It is disabled by default so the historical
# train/evaluate protocol is unchanged unless explicitly requested.
RUN_RUNTIME_PROFILE="${RUN_RUNTIME_PROFILE:-0}"
PROFILE_DEVICE="${PROFILE_DEVICE:-${EVAL_DEVICE:-cuda}}"
PROFILE_WARMUP="${PROFILE_WARMUP:-20}"
PROFILE_REPEATS="${PROFILE_REPEATS:-}"
PROFILE_MODEL_DIAGNOSTIC="${PROFILE_MODEL_DIAGNOSTIC:-0}"
RUN_UAV_RUNTIME_PROFILE="${RUN_UAV_RUNTIME_PROFILE:-0}"
UAV_PROFILE_DATA="${UAV_PROFILE_DATA:-${REPO_ROOT}/artifacts/uav/uav_dataset.pt}"
UAV_PROFILE_CKPT="${UAV_PROFILE_CKPT:-}"
UAV_PROFILE_RUN_SEED="${UAV_PROFILE_RUN_SEED:-}"

S4_GIT_URL="${S4_GIT_URL:-https://github.com/state-spaces/s4.git}"
S4_GIT_REF="${S4_GIT_REF:-e757cef57d89e448c413de7325ed5601aceaac13}"
OPTIONAL_SOURCE_ROOT="${OPTIONAL_SOURCE_ROOT:-${RUN_ROOT}/dependencies}"
MAMBA2_PACKAGE="${MAMBA2_PACKAGE:-mamba-ssm==2.3.2.post1}"
TORCHAUDIO_PACKAGE="${TORCHAUDIO_PACKAGE:-auto}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "paper protocol config does not exist: ${CONFIG}" >&2
  exit 2
fi

IFS=',' read -r -a RAW_METHODS <<< "${METHODS}"
METHOD_MATRIX=()
for raw_method in "${RAW_METHODS[@]}"; do
  method="$(printf '%s' "${raw_method}" | tr '[:upper:]-' '[:lower:]_')"
  case "${method}" in
    mlp) method="tgn_mlp" ;;
    kan) method="tgn_kan" ;;
    physick) method="tgn_physick" ;;
    lttr) method="snapshot_ltt_r" ;;
    dagwm) method="da_gwm" ;;
    bigmlp) method="big_mlp" ;;
    edgeattn) method="edge_attn" ;;
    temptrans) method="temp_trans" ;;
    snapshot_tgn_mlp) method="snapshot_mlp" ;;
    snapshot_tgn_physick) method="snapshot_physick" ;;
  esac
  case "${method}" in
    tgn_mlp|tgn_kan|tgn_physick|snapshot_mlp|snapshot_physick|snapshot_ltt_r|snapshot_da_gwm|ltt_r|da_gwm|s4|mamba2|conformer|big_mlp|edge_attn|temp_trans)
      METHOD_MATRIX+=("${method}")
      ;;
    *)
      echo "unknown paper method: ${raw_method}" >&2
      exit 2
      ;;
  esac
done
if [[ ${#METHOD_MATRIX[@]} -eq 0 ]]; then
  echo "METHODS must select at least one method" >&2
  exit 2
fi

has_method() {
  local wanted="$1"
  local candidate
  for candidate in "${METHOD_MATRIX[@]}"; do
    if [[ "${candidate}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

is_snapshot_method() {
  [[ "$1" == snapshot_* ]]
}

NEEDS_INTENSITY_FLOW_DATASET=0
NEEDS_SNAPSHOT_PIPELINE=0
for method in "${METHOD_MATRIX[@]}"; do
  if is_snapshot_method "${method}"; then
    NEEDS_SNAPSHOT_PIPELINE=1
  else
    NEEDS_INTENSITY_FLOW_DATASET=1
  fi
done

install_s4_from_official_source() {
  local source_dir="${OPTIONAL_SOURCE_ROOT}/s4"
  mkdir -p "${OPTIONAL_SOURCE_ROOT}"
  if [[ -e "${source_dir}" ]]; then
    if [[ ! -d "${source_dir}/.git" ]]; then
      echo "S4 source target exists but is not a Git checkout: ${source_dir}" >&2
      exit 2
    fi
    local origin
    origin="$(git -C "${source_dir}" remote get-url origin)"
    if [[ "${origin}" != "${S4_GIT_URL}" ]]; then
      echo "existing S4 checkout has unexpected origin: ${origin}" >&2
      exit 2
    fi
  else
    if [[ -n "${S4_GIT_REF}" ]]; then
      git init "${source_dir}"
      git -C "${source_dir}" remote add origin "${S4_GIT_URL}"
      git -C "${source_dir}" fetch --depth 1 origin "${S4_GIT_REF}"
      git -C "${source_dir}" checkout --detach FETCH_HEAD
    else
      git clone --depth 1 "${S4_GIT_URL}" "${source_dir}"
    fi
  fi
  if [[ -n "${S4_GIT_REF}" ]]; then
    local expected_commit current_commit
    expected_commit="$(git -C "${source_dir}" rev-parse "${S4_GIT_REF}^{commit}" 2>/dev/null || true)"
    current_commit="$(git -C "${source_dir}" rev-parse HEAD)"
    if [[ -z "${expected_commit}" || "${current_commit}" != "${expected_commit}" ]]; then
      echo "S4 checkout is not at requested ref ${S4_GIT_REF}: ${current_commit}" >&2
      exit 2
    fi
  fi
  # The official repository's full experiment requirements include old
  # dataset/trainer packages that can replace the caller's PyTorch stack.  The
  # external S4Block adapter needs only this runtime subset.
  "${PIP_BIN[@]}" install numpy scipy "pytorch-lightning==2.0.4" einops
  export PYTHONPATH="${source_dir}${PYTHONPATH:+:${PYTHONPATH}}"
}

if [[ "${SETUP_DEPENDENCIES}" == "1" ]]; then
  "${PIP_BIN[@]}" install -e "${REPO_ROOT}[paper]"
  if [[ "${INSTALL_OPTIONAL_DEPENDENCIES}" == "1" ]]; then
    if has_method s4; then
      # S4 has no official PyPI distribution. Install the official source repo.
      install_s4_from_official_source
    fi
    if has_method mamba2; then
      # Mamba-2 is provided by the official mamba-ssm package.
      if [[ "$(uname -s)" != "Linux" ]]; then
        echo "Mamba2's official runtime requires Linux; select another METHODS set on this host." >&2
        exit 2
      fi
      "${PIP_BIN[@]}" install --no-build-isolation "${MAMBA2_PACKAGE}"
    fi
    if has_method conformer; then
      # The Conformer baseline uses the official torchaudio package.
      if [[ "${TORCHAUDIO_PACKAGE}" == "auto" ]]; then
        TORCH_BASE_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.__version__.split("+")[0])')"
        RESOLVED_TORCHAUDIO_PACKAGE="torchaudio==${TORCH_BASE_VERSION}"
      else
        RESOLVED_TORCHAUDIO_PACKAGE="${TORCHAUDIO_PACKAGE}"
      fi
      echo "Installing ${RESOLVED_TORCHAUDIO_PACKAGE}; it must match the active torch build."
      "${PIP_BIN[@]}" install "${RESOLVED_TORCHAUDIO_PACKAGE}"
    fi
    # LTT-R and DA-GWM are repository implementations. There is deliberately
    # no guessed or fabricated third-party pip package for either method.
  fi
fi

if [[ ! -f "${SCRIPT_DIR}/paper_generate.py" ]]; then
  echo "missing data generator: ${SCRIPT_DIR}/paper_generate.py" >&2
  exit 2
fi
if [[ ! -f "${SCRIPT_DIR}/paper_train.py" ]]; then
  echo "missing typed trainer: ${SCRIPT_DIR}/paper_train.py" >&2
  exit 2
fi
if [[ ! -f "${SCRIPT_DIR}/paper_evaluate.py" ]]; then
  echo "missing paired evaluator: ${SCRIPT_DIR}/paper_evaluate.py" >&2
  exit 2
fi
if [[ "${NEEDS_SNAPSHOT_PIPELINE}" == "1" ]]; then
  for snapshot_script in paper_snapshot_generate.py paper_snapshot_train.py paper_snapshot_evaluate.py; do
    if [[ ! -f "${SCRIPT_DIR}/${snapshot_script}" ]]; then
      echo "missing Snapshot pipeline entry point: ${SCRIPT_DIR}/${snapshot_script}" >&2
      exit 2
    fi
  done
fi
if [[ ( "${RUN_RUNTIME_PROFILE}" == "1" || "${RUN_UAV_RUNTIME_PROFILE}" == "1" ) && ! -f "${SCRIPT_DIR}/profile_runtime.py" ]]; then
  echo "missing runtime profiler: ${SCRIPT_DIR}/profile_runtime.py" >&2
  exit 2
fi

mkdir -p "$(dirname -- "${DATASET}")" "${SNAPSHOT_DATA_ROOT}" "${RUN_ROOT}"

GENERATE_ARGS=(--cfg "${CONFIG}" --out "${DATASET}")
if [[ -n "${GENERATE_EPISODES:-}" ]]; then
  GENERATE_ARGS+=(--episodes "${GENERATE_EPISODES}")
fi
if [[ -n "${GENERATE_HORIZON:-}" ]]; then
  GENERATE_ARGS+=(--horizon "${GENERATE_HORIZON}")
fi
if [[ -n "${GENERATE_SPLIT_COUNTS:-}" ]]; then
  GENERATE_ARGS+=(--split-counts "${GENERATE_SPLIT_COUNTS}")
fi
if [[ -n "${GENERATE_SPLIT_RATIOS:-}" ]]; then
  GENERATE_ARGS+=(--split-ratios "${GENERATE_SPLIT_RATIOS}")
fi
if [[ -n "${GENERATE_BASE_SEED:-}" ]]; then
  GENERATE_ARGS+=(--base-seed "${GENERATE_BASE_SEED}")
fi
if [[ -n "${GENERATE_SPLIT_SEED:-}" ]]; then
  GENERATE_ARGS+=(--split-seed "${GENERATE_SPLIT_SEED}")
fi
if [[ -n "${GENERATE_DEVICE:-}" ]]; then
  GENERATE_ARGS+=(--device "${GENERATE_DEVICE}")
fi
if [[ -n "${GENERATE_EXTRA_ARGS:-}" ]]; then
  read -r -a GENERATED_EXTRA <<< "${GENERATE_EXTRA_ARGS}"
  GENERATE_ARGS+=("${GENERATED_EXTRA[@]}")
fi

if [[ "${NEEDS_INTENSITY_FLOW_DATASET}" == "1" ]] && \
   [[ "${REUSE_DATASET}" != "1" || ! -f "${DATASET}" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/paper_generate.py" "${GENERATE_ARGS[@]}"
fi

run_snapshot_training_variant() {
  local label="$1"
  local dataset="$2"
  shift 2
  local variant_dir="${RUN_ROOT}/${label}"
  local checkpoint="${variant_dir}/checkpoint.pt"
  local trace="${variant_dir}/paired_trace.pt"
  mkdir -p "${variant_dir}"
  local train_args=(
    --cfg "${CONFIG}"
    --data "${dataset}"
    --method snapshot_physick
    --out "${checkpoint}"
    "$@"
  )
  if [[ -n "${TRAIN_DEVICE:-}" ]]; then
    train_args+=(--device "${TRAIN_DEVICE}")
  fi
  if [[ -n "${SNAPSHOT_TRAIN_EXTRA_ARGS:-}" ]]; then
    read -r -a snapshot_train_extra <<< "${SNAPSHOT_TRAIN_EXTRA_ARGS}"
    train_args+=("${snapshot_train_extra[@]}")
  fi
  if [[ "${REUSE_CHECKPOINTS}" != "1" || ! -f "${checkpoint}" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/paper_snapshot_train.py" "${train_args[@]}"
  fi
  local eval_args=(
    --cfg "${CONFIG}"
    --ckpt "${checkpoint}"
    --data "${dataset}"
    --method snapshot_physick
    --out "${trace}"
  )
  if [[ -n "${EVAL_SEEDS:-}" ]]; then
    eval_args+=(--seeds "${EVAL_SEEDS}")
  fi
  if [[ -n "${EVAL_EPISODES:-}" ]]; then
    eval_args+=(--episodes "${EVAL_EPISODES}")
  fi
  if [[ -n "${EVAL_HORIZON:-}" ]]; then
    eval_args+=(--horizon "${EVAL_HORIZON}")
  fi
  if [[ -n "${EVAL_DEVICE:-}" ]]; then
    eval_args+=(--device "${EVAL_DEVICE}")
  fi
  if [[ -n "${SNAPSHOT_EVAL_EXTRA_ARGS:-}" ]]; then
    read -r -a snapshot_eval_extra <<< "${SNAPSHOT_EVAL_EXTRA_ARGS}"
    eval_args+=("${snapshot_eval_extra[@]}")
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/paper_snapshot_evaluate.py" "${eval_args[@]}"
}

METHOD_ORDINAL=0
CONTROLLER_SWEEP_DONE=0
for method in "${METHOD_MATRIX[@]}"; do
  METHOD_DIR="${RUN_ROOT}/${method}"
  CHECKPOINT="${METHOD_DIR}/checkpoint.pt"
  TRACE="${METHOD_DIR}/paired_trace.pt"
  MANIFEST="${METHOD_DIR}/paired_trace.manifest.json"
  mkdir -p "${METHOD_DIR}"

  METHOD_DATASET="${DATASET}"
  TRAIN_ENTRY="${SCRIPT_DIR}/paper_train.py"
  EVALUATE_ENTRY="${SCRIPT_DIR}/paper_evaluate.py"
  IS_SNAPSHOT=0
  if is_snapshot_method "${method}"; then
    IS_SNAPSHOT=1
    METHOD_DATASET="${SNAPSHOT_DATA_ROOT}/${method}_snapshot_schema_v1.pt"
    TRAIN_ENTRY="${SCRIPT_DIR}/paper_snapshot_train.py"
    EVALUATE_ENTRY="${SCRIPT_DIR}/paper_snapshot_evaluate.py"
    SNAPSHOT_GENERATE_ARGS=(
      --cfg "${CONFIG}"
      --method "${method}"
      --out "${METHOD_DATASET}"
    )
    if [[ -n "${GENERATE_EPISODES:-}" ]]; then
      SNAPSHOT_GENERATE_ARGS+=(--episodes "${GENERATE_EPISODES}")
    fi
    if [[ -n "${GENERATE_HORIZON:-}" ]]; then
      SNAPSHOT_GENERATE_ARGS+=(--horizon "${GENERATE_HORIZON}")
    fi
    if [[ -n "${GENERATE_SPLIT_COUNTS:-}" ]]; then
      SNAPSHOT_GENERATE_ARGS+=(--split-counts "${GENERATE_SPLIT_COUNTS}")
    fi
    if [[ -n "${GENERATE_SPLIT_RATIOS:-}" ]]; then
      SNAPSHOT_GENERATE_ARGS+=(--split-ratios "${GENERATE_SPLIT_RATIOS}")
    fi
    if [[ -n "${GENERATE_BASE_SEED:-}" ]]; then
      SNAPSHOT_GENERATE_ARGS+=(--base-seed "${GENERATE_BASE_SEED}")
    fi
    if [[ -n "${GENERATE_SPLIT_SEED:-}" ]]; then
      SNAPSHOT_GENERATE_ARGS+=(--split-seed "${GENERATE_SPLIT_SEED}")
    fi
    if [[ -n "${GENERATE_DEVICE:-}" ]]; then
      SNAPSHOT_GENERATE_ARGS+=(--device "${GENERATE_DEVICE}")
    fi
    if [[ -n "${GENERATE_EXTRA_ARGS:-}" ]]; then
      read -r -a SNAPSHOT_GENERATED_EXTRA <<< "${GENERATE_EXTRA_ARGS}"
      SNAPSHOT_GENERATE_ARGS+=("${SNAPSHOT_GENERATED_EXTRA[@]}")
    fi
    if [[ "${REUSE_DATASET}" != "1" || ! -f "${METHOD_DATASET}" ]]; then
      "${PYTHON_BIN}" "${SCRIPT_DIR}/paper_snapshot_generate.py" \
        "${SNAPSHOT_GENERATE_ARGS[@]}"
    fi
  fi

  TRAIN_ARGS=(
    --cfg "${CONFIG}"
    --data "${METHOD_DATASET}"
    --method "${method}"
    --out "${CHECKPOINT}"
  )
  if [[ -n "${TRAIN_DEVICE:-}" ]]; then
    TRAIN_ARGS+=(--device "${TRAIN_DEVICE}")
  fi
  if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
    read -r -a TRAINED_EXTRA <<< "${TRAIN_EXTRA_ARGS}"
    TRAIN_ARGS+=("${TRAINED_EXTRA[@]}")
  fi
  if [[ "${REUSE_CHECKPOINTS}" != "1" || ! -f "${CHECKPOINT}" ]]; then
    "${PYTHON_BIN}" "${TRAIN_ENTRY}" "${TRAIN_ARGS[@]}"
  fi

  EVALUATE_ARGS=(
    --cfg "${CONFIG}"
    --ckpt "${CHECKPOINT}"
    --method "${method}"
    --out "${TRACE}"
  )
  if [[ "${IS_SNAPSHOT}" == "1" ]]; then
    EVALUATE_ARGS+=(--data "${METHOD_DATASET}")
  fi
  if [[ "${IS_SNAPSHOT}" == "0" ]]; then
    EVALUATE_ARGS+=(--manifest "${MANIFEST}")
  fi
  if [[ -n "${EVAL_SEEDS:-}" ]]; then
    EVALUATE_ARGS+=(--seeds "${EVAL_SEEDS}")
  fi
  if [[ -n "${EVAL_EPISODES:-}" ]]; then
    EVALUATE_ARGS+=(--episodes "${EVAL_EPISODES}")
  fi
  if [[ -n "${EVAL_HORIZON:-}" ]]; then
    EVALUATE_ARGS+=(--horizon "${EVAL_HORIZON}")
  fi
  if [[ "${IS_SNAPSHOT}" == "0" && -n "${EVAL_MODES:-}" ]]; then
    EVALUATE_ARGS+=(--modes "${EVAL_MODES}")
  fi
  if [[ -n "${EVAL_DEVICE:-}" ]]; then
    EVALUATE_ARGS+=(--device "${EVAL_DEVICE}")
  fi
  if [[ "${IS_SNAPSHOT}" == "0" && -n "${EVAL_CLASSICAL:-}" ]]; then
    EVALUATE_ARGS+=(--classical "${EVAL_CLASSICAL}")
  fi
  if [[ "${IS_SNAPSHOT}" == "0" && "${CONTROLLER_SWEEP}" == "1" && "${CONTROLLER_SWEEP_DONE}" == "0" ]]; then
    # Classical sweeps are method-independent, so execute the grid only once.
    EVALUATE_ARGS+=(--controller-sweep)
    CONTROLLER_SWEEP_DONE=1
  fi
  if [[ "${IS_SNAPSHOT}" == "0" && "${EVAL_METRICS}" == "1" ]]; then
    EVALUATE_ARGS+=(--metrics)
  fi
  if [[ "${IS_SNAPSHOT}" == "0" && "${EVAL_SHRINK_JUMP_AUDIT}" == "1" ]]; then
    EVALUATE_ARGS+=(--shrink-jump-audit)
  fi
  if [[ "${IS_SNAPSHOT}" == "0" && "${EVAL_RISK_SHIELD}" == "1" ]]; then
    EVALUATE_ARGS+=(--risk-shield)
  fi
  if [[ -n "${EVAL_EXTRA_ARGS:-}" ]]; then
    read -r -a EVALUATED_EXTRA <<< "${EVAL_EXTRA_ARGS}"
    EVALUATE_ARGS+=("${EVALUATED_EXTRA[@]}")
  fi
  "${PYTHON_BIN}" "${EVALUATE_ENTRY}" "${EVALUATE_ARGS[@]}"

  if [[ "${RUN_RUNTIME_PROFILE}" == "1" ]]; then
    PROFILE_PLATFORM="ntn"
    if [[ "${IS_SNAPSHOT}" == "1" ]]; then
      PROFILE_PLATFORM="snapshot"
    fi
    PROFILE_ARGS=(
      --platform "${PROFILE_PLATFORM}"
      --cfg "${CONFIG}"
      --method "${method}"
      --ckpt "${CHECKPOINT}"
      --data "${METHOD_DATASET}"
      --device "${PROFILE_DEVICE}"
      --warmup "${PROFILE_WARMUP}"
      --json-out "${METHOD_DIR}/runtime_profile.json"
      --csv-out "${METHOD_DIR}/runtime_profile.csv"
    )
    if [[ -n "${PROFILE_REPEATS}" ]]; then
      PROFILE_ARGS+=(--repeats "${PROFILE_REPEATS}")
    fi
    if [[ "${PROFILE_MODEL_DIAGNOSTIC}" == "1" ]]; then
      PROFILE_ARGS+=(--include-model-predict-step)
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_runtime.py" "${PROFILE_ARGS[@]}"
  fi

  if [[ "${method}" == "snapshot_physick" ]]; then
    if [[ "${RUN_SNAPSHOT_SCHEDULED_SAMPLING}" == "1" ]]; then
      run_snapshot_training_variant \
        snapshot_physick_scheduled_sampling \
        "${METHOD_DATASET}" \
        --training-objective scheduled_sampling
    fi
    if [[ "${RUN_SNAPSHOT_ROLLOUT_LADDER}" == "1" ]]; then
      IFS=',' read -r -a rollout_horizons <<< "${SNAPSHOT_ROLLOUT_HORIZONS}"
      for rollout_horizon in "${rollout_horizons[@]}"; do
        if [[ ! "${rollout_horizon}" =~ ^[0-9]+$ ]] || (( rollout_horizon < 2 )); then
          echo "invalid Snapshot rollout horizon: ${rollout_horizon}" >&2
          exit 2
        fi
        run_snapshot_training_variant \
          "snapshot_physick_rollout_h${rollout_horizon}" \
          "${METHOD_DATASET}" \
          --training-objective action_coupled \
          --rollout-horizon "${rollout_horizon}" \
          --budget-multiplier 1.0
      done
    fi
  fi
  METHOD_ORDINAL=$((METHOD_ORDINAL + 1))
done

if [[ "${RUN_UAV_RUNTIME_PROFILE}" == "1" ]]; then
  if [[ -z "${UAV_PROFILE_CKPT}" || ! -f "${UAV_PROFILE_CKPT}" ]]; then
    echo "RUN_UAV_RUNTIME_PROFILE=1 requires UAV_PROFILE_CKPT" >&2
    exit 2
  fi
  if [[ -z "${UAV_PROFILE_RUN_SEED}" ]]; then
    echo "RUN_UAV_RUNTIME_PROFILE=1 requires UAV_PROFILE_RUN_SEED" >&2
    exit 2
  fi
  if [[ ! -f "${UAV_PROFILE_DATA}" ]]; then
    echo "UAV formal dataset does not exist: ${UAV_PROFILE_DATA}" >&2
    exit 2
  fi
  UAV_PROFILE_DIR="${RUN_ROOT}/uav_runtime_run_${UAV_PROFILE_RUN_SEED}"
  mkdir -p "${UAV_PROFILE_DIR}"
  UAV_PROFILE_ARGS=(
    --platform uav
    --cfg "${CONFIG}"
    --ckpt "${UAV_PROFILE_CKPT}"
    --data "${UAV_PROFILE_DATA}"
    --run-seed "${UAV_PROFILE_RUN_SEED}"
    --device "${PROFILE_DEVICE}"
    --warmup "${PROFILE_WARMUP}"
    --json-out "${UAV_PROFILE_DIR}/runtime_profile.json"
    --csv-out "${UAV_PROFILE_DIR}/runtime_profile.csv"
  )
  if [[ -n "${PROFILE_REPEATS}" ]]; then
    UAV_PROFILE_ARGS+=(--repeats "${PROFILE_REPEATS}")
  fi
  if [[ "${PROFILE_MODEL_DIAGNOSTIC}" == "1" ]]; then
    UAV_PROFILE_ARGS+=(--include-model-predict-step)
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_runtime.py" "${UAV_PROFILE_ARGS[@]}"
fi

echo "[OK] paper protocol completed: ${RUN_ROOT}"
