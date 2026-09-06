#!/usr/bin/env bash
set -euo pipefail

# nohup bash server10_gpu0.sh >> server10_gpu0.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
SERVER=10
TIME_INTERVAL="${TIME_INTERVAL:-3m}"

TARGET_SCRIPTS=("eval.sh") # run_coex_0 eval eval_sampling
LOG_DIR="${LOG_DIR:-${WORK_DIR}/log}"

mkdir -p "${LOG_DIR}"
DATE_TAG="$(date +%m%d_%H%M%S)"
TARGET_TAG="${TARGET_SCRIPTS[*]}"
TARGET_TAG="${TARGET_TAG//.sh/}"
TARGET_TAG="${TARGET_TAG// /_}"
RUN_NAME="server${SERVER}_gpu${GPU_ID}_${TARGET_TAG}_${DATE_TAG}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
NTFY_TOPIC="${NTFY_TOPIC:-soeon_server${SERVER}}"

notify() {
  local msg="$1"
  curl -fsS -d "$msg" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
}

echo "=============================="
echo "SERVER: ${SERVER}"
echo "GPU_ID: ${GPU_ID}"
echo "WORK_DIR: ${WORK_DIR}"
echo "TARGET_SCRIPTS: ${TARGET_SCRIPTS[*]}"
echo "LOG_FILE: ${LOG_FILE}"
echo "TIME_INTERVAL: ${TIME_INTERVAL}"
echo "=============================="

cd "${WORK_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found"
  notify "[ERROR] server${SERVER} gpu${GPU_ID} | nvidia-smi not found"
  exit 1
fi

for script in "${TARGET_SCRIPTS[@]}"; do
  if [[ ! -f "${script}" ]]; then
    echo "[ERROR] ${WORK_DIR}/${script} not found"
    notify "[ERROR] server${SERVER} gpu${GPU_ID} | ${script} not found"
    exit 1
  fi
  if ! bash -n "${script}"; then
    echo "[ERROR] ${WORK_DIR}/${script} has invalid bash syntax"
    notify "[ERROR] server${SERVER} gpu${GPU_ID} | ${script} syntax error"
    exit 1
  fi
done

if [[ "${SERVER_CHECK_ONLY:-0}" == "1" ]]; then
  echo "SERVER_CHECK_ONLY_OK"
  exit 0
fi

while true; do
  if ! GPU_PIDS="$(
    nvidia-smi -i "${GPU_ID}" \
      --query-compute-apps=pid \
      --format=csv,noheader 2>/dev/null
  )"; then
    echo "[ERROR] failed to query GPU ${GPU_ID}; retrying in ${TIME_INTERVAL}"
    sleep "${TIME_INTERVAL}"
    continue
  fi
  PYTHON_PROCESS_COUNT="$(awk 'NF{c++} END{print c+0}' <<< "${GPU_PIDS}")"

  if [[ "${PYTHON_PROCESS_COUNT}" -eq 0 ]]; then
    echo ">>> GPU ${GPU_ID} free. START ${TARGET_SCRIPTS[*]}"
    echo ">>> LOG: ${LOG_FILE}"
    notify "[START] server${SERVER} gpu${GPU_ID} | ${TARGET_SCRIPTS[*]} | ${RUN_NAME}"

    if (
      set +e
      echo "========== RUN START =========="
      echo "date: $(date)"
      echo "host: $(hostname)"
      echo "work_dir: ${WORK_DIR}"
      echo "target_scripts: ${TARGET_SCRIPTS[*]}"
      echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
      echo "log_file: ${LOG_FILE}"
      echo "==============================="
      for script in "${TARGET_SCRIPTS[@]}"; do
        echo ""
        echo "========== SCRIPT START: ${script} =========="
        echo "date: $(date)"
        bash "${script}"
        SCRIPT_EXIT_CODE=$?
        echo "========== SCRIPT END: ${script} =========="
        echo "date: $(date)"
        echo "exit_code: ${SCRIPT_EXIT_CODE}"
        if [[ "${SCRIPT_EXIT_CODE}" -ne 0 ]]; then
          exit "${SCRIPT_EXIT_CODE}"
        fi
      done
      echo "========== RUN END =========="
      echo "date: $(date)"
      echo "exit_code: 0"
      exit 0
    ) > "${LOG_FILE}" 2>&1; then
      EXIT_CODE=0
    else
      EXIT_CODE=$?
    fi

    if [[ "${EXIT_CODE}" -eq 0 ]]; then
      echo ">>> DONE ${TARGET_SCRIPTS[*]}"
      notify "[DONE] server${SERVER} gpu${GPU_ID} | ${TARGET_SCRIPTS[*]} | ${RUN_NAME}"
    else
      echo ">>> ERROR ${TARGET_SCRIPTS[*]} exit_code=${EXIT_CODE}"
      notify "[ERROR] server${SERVER} gpu${GPU_ID} | ${TARGET_SCRIPTS[*]} | exit=${EXIT_CODE} | ${RUN_NAME}"
    fi
    exit "${EXIT_CODE}"
  else
    echo ">>> GPU ${GPU_ID} busy (${PYTHON_PROCESS_COUNT} compute process). sleep ${TIME_INTERVAL}"
  fi
  sleep "${TIME_INTERVAL}"
done
