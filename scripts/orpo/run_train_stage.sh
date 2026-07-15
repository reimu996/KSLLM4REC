#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 STAGE [OUTPUT_DIR]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
STAGE="$1"
RUN_ID="${STAGE}_$(date '+%Y%m%d_%H%M%S_%N')"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"
if [[ "${STAGE}" == "full_orpo_epoch_002" ]]; then
    OUTPUT_DIR="${2:-${FULL_OUTPUT}}"
else
    OUTPUT_DIR="${2:-${PROJECT_ROOT}/artifacts/orpo/runs/gates/${RUN_ID}}"
fi
MANIFEST="${RUN_DIR}/manifest.json"

mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1
trap 'status=$?; printf "exit_code=%s\nfinished_at=%s\n" "${status}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/shell-status.txt"' EXIT

COMMAND=(
    "${PYTHON}" -m torch.distributed.run
    --standalone --nnodes=1 --nproc-per-node=1
    --module ksllm4rec_orpo.cli train
    --config "${CONFIG}"
    --artifact-lock "${ARTIFACT_LOCK}"
    --environment-lock "${ENVIRONMENT_LOCK}"
    --output-dir "${OUTPUT_DIR}"
    --manifest "${MANIFEST}"
    --stage "${STAGE}"
    --log-root "${LOG_ROOT}"
    --gates-report "${GATES_REPORT}"
    --min-free-gib 20.5
    --max-reserved-gib 20.0
)
echo "run_id=${RUN_ID}"
echo "started_at=$(date --iso-8601=seconds)"
echo "output_dir=${OUTPUT_DIR}"
printf 'command='
printf '%q ' "${COMMAND[@]}"
printf '\n'
nvidia-smi
"${COMMAND[@]}"
