#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LLAMAFACTORY_ROOT="/home/lyc/REC_PROJECTS/LlamaFactory"
ENV_NAME="onereason_lora_sft"
PYTHON="/home/lyc/miniconda3/envs/${ENV_NAME}/bin/python"
DATA="${PROJECT_ROOT}/artifacts/sft/data/hf_baseline_091/train_alpaca.jsonl"
REPORT="${PROJECT_ROOT}/artifacts/sft/data/hf_baseline_091/tokenizer_report.json"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}/src:${LLAMAFACTORY_ROOT}/src" \
    "${PYTHON}" -m ksllm4rec_sft.cli preflight \
    --model /home/lyc/models/OneReason-0.8B-pretrain-competition \
    --data "${DATA}" \
    --report "${REPORT}" \
    --artifact-lock "${PROJECT_ROOT}/configs/sft/artifacts.lock.json" \
    --cutoff-len 16384
