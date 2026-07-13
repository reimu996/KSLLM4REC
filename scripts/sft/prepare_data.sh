#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SOURCE="/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/artifacts/sft/data/hf_baseline_091"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}/src" \
    /home/lyc/miniconda3/envs/onereason_lora_sft/bin/python -m ksllm4rec_sft.cli prepare-data \
    --source "${SOURCE}" \
    --output-dir "${OUTPUT_DIR}"
