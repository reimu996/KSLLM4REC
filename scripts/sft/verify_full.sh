#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LLAMAFACTORY_ROOT="/home/lyc/REC_PROJECTS/LlamaFactory"
ENV_NAME="onereason_lora_sft"
ENV_PREFIX="/home/lyc/miniconda3/envs/${ENV_NAME}"
PYTHON="${ENV_PREFIX}/bin/python"
RUN_ID="verify_full_$(date '+%Y%m%d_%H%M%S_%N')"
RUN_DIR="${PROJECT_ROOT}/operation_logs/sft/${RUN_ID}"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1
export CONDA_PREFIX="${ENV_PREFIX}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${LLAMAFACTORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m ksllm4rec_sft.cli verify \
    --model /home/lyc/models/OneReason-0.8B-pretrain-competition \
    --output-dir "${PROJECT_ROOT}/artifacts/sft/runs/full_epoch_001" \
    --log-root "${PROJECT_ROOT}/operation_logs/sft" \
    --report "${RUN_DIR}/verification.json" \
    --artifact-lock "${PROJECT_ROOT}/configs/sft/artifacts.lock.json" \
    --environment-lock "${PROJECT_ROOT}/configs/sft/environment.lock.txt" \
    --min-free-gib 22.0 \
    --max-reserved-gib 21.5
