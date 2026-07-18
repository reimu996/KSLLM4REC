#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 STAGE CUTOFF_LEN MAX_STEPS [OUTPUT_DIR]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

STAGE="$1"
CUTOFF_LEN="$2"
MAX_STEPS="$3"
RUN_ID="${STAGE}_$(date '+%Y%m%d_%H%M%S_%N')"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"
OUTPUT_DIR="${4:-${GATE_OUTPUT_ROOT}/${RUN_ID}}"
MANIFEST="${RUN_DIR}/manifest.json"

require_python
if [[ ! "${CUTOFF_LEN}" =~ ^[0-9]+$ ]] || (( CUTOFF_LEN < 2 )); then
    echo "CUTOFF_LEN must be an integer >= 2." >&2
    exit 2
fi
if [[ "${MAX_STEPS}" != "full" ]] && { [[ ! "${MAX_STEPS}" =~ ^[0-9]+$ ]] || (( MAX_STEPS < 1 )); }; then
    echo "MAX_STEPS must be a positive integer or 'full'." >&2
    exit 2
fi

mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}" \
    "${PROJECT_ROOT}/artifacts/sft/cache/huggingface" \
    "${PROJECT_ROOT}/artifacts/sft/cache/triton"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1
trap 'status=$?; printf "exit_code=%s\nfinished_at=%s\n" "${status}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/shell-status.txt"' EXIT

export_offline_runtime

COMMAND=(
    "${PYTHON}" -m torch.distributed.run
    --standalone --nnodes=1 --nproc-per-node=1
    --module ksllm4rec_sft.cli train
    --profile "${PROFILE}"
    --config "${CONFIG}"
    --artifact-lock "${ARTIFACT_LOCK}"
    --environment-lock "${ENVIRONMENT_LOCK}"
    --output-dir "${OUTPUT_DIR}"
    --manifest "${MANIFEST}"
    --stage "${STAGE}"
    --cutoff-len "${CUTOFF_LEN}"
    --min-free-gib 20.5
    --max-reserved-gib 20.0
)
if [[ "${MAX_STEPS}" != "full" ]]; then
    COMMAND+=(--max-steps "${MAX_STEPS}")
fi

echo "profile=${PROFILE}"
echo "run_id=${RUN_ID}"
echo "started_at=$(date --iso-8601=seconds)"
echo "output_dir=${OUTPUT_DIR}"
printf 'command='
printf '%q ' "${COMMAND[@]}"
printf '\n'
nvidia-smi
"${COMMAND[@]}"
