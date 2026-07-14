#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 STAGE CUTOFF_LEN MAX_STEPS [OUTPUT_DIR]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LLAMAFACTORY_ROOT="/home/lyc/REC_PROJECTS/LlamaFactory"
ENV_NAME="onereason_lora_sft"
ENV_PREFIX="/home/lyc/miniconda3/envs/${ENV_NAME}"
PYTHON="${ENV_PREFIX}/bin/python"
STAGE="$1"
CUTOFF_LEN="$2"
MAX_STEPS="$3"
RUN_ID="${STAGE}_$(date '+%Y%m%d_%H%M%S_%N')"
RUN_DIR="${PROJECT_ROOT}/operation_logs/sft/${RUN_ID}"
OUTPUT_DIR="${4:-${PROJECT_ROOT}/artifacts/sft/runs/gates/${RUN_ID}}"
MANIFEST="${RUN_DIR}/manifest.json"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing environment Python: ${PYTHON}" >&2
    exit 2
fi
if [[ ! "${CUTOFF_LEN}" =~ ^[0-9]+$ ]] || (( CUTOFF_LEN < 2 )); then
    echo "CUTOFF_LEN must be an integer >= 2." >&2
    exit 2
fi
if [[ "${MAX_STEPS}" != "full" ]] && { [[ ! "${MAX_STEPS}" =~ ^[0-9]+$ ]] || (( MAX_STEPS < 1 )); }; then
    echo "MAX_STEPS must be a positive integer or 'full'." >&2
    exit 2
fi

mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}" "${PROJECT_ROOT}/artifacts/sft/cache/huggingface" \
    "${PROJECT_ROOT}/artifacts/sft/cache/triton"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1
trap 'status=$?; printf "exit_code=%s\nfinished_at=%s\n" "${status}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/shell-status.txt"' EXIT

export CONDA_PREFIX="${ENV_PREFIX}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/artifacts/sft/cache/huggingface/datasets"
export HF_HOME="${PROJECT_ROOT}/artifacts/sft/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="${PROJECT_ROOT}/artifacts/sft/cache/triton"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${LLAMAFACTORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

COMMAND=(
    "${PYTHON}" -m torch.distributed.run
    --standalone --nnodes=1 --nproc-per-node=1
    --module ksllm4rec_sft.cli train
    --config "${PROJECT_ROOT}/configs/sft/onereason_lora_focal_item.yaml"
    --artifact-lock "${PROJECT_ROOT}/configs/sft/artifacts.lock.json"
    --environment-lock "${PROJECT_ROOT}/configs/sft/environment.lock.txt"
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

echo "run_id=${RUN_ID}"
echo "started_at=$(date --iso-8601=seconds)"
echo "output_dir=${OUTPUT_DIR}"
printf 'command='
printf '%q ' "${COMMAND[@]}"
printf '\n'
nvidia-smi
"${COMMAND[@]}"
