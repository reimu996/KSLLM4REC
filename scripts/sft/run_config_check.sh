#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LLAMAFACTORY_ROOT="/home/lyc/REC_PROJECTS/LlamaFactory"
ENV_NAME="onereason_lora_sft"
PYTHON="/home/lyc/miniconda3/envs/${ENV_NAME}/bin/python"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${LLAMAFACTORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=1 \
    --module ksllm4rec_sft.cli config-check \
    --config "${PROJECT_ROOT}/configs/sft/onereason_lora_focal_item.yaml" \
    --artifact-lock "${PROJECT_ROOT}/configs/sft/artifacts.lock.json" \
    --environment-lock "${PROJECT_ROOT}/configs/sft/environment.lock.txt" \
    --report "${PROJECT_ROOT}/artifacts/sft/config_check.json"
