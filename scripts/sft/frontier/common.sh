#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
readonly PROFILE="frontier_feedbackcore_listwise_invariant_v1"
readonly LLAMAFACTORY_ROOT="/home/lyc/REC_PROJECTS/LlamaFactory"
readonly ENV_NAME="onereason_lora_sft"
readonly ENV_PREFIX="/home/lyc/miniconda3/envs/${ENV_NAME}"
readonly PYTHON="${ENV_PREFIX}/bin/python"
readonly MODEL="/home/lyc/models/OneReason-0.8B-pretrain-competition"
readonly SOURCE="/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl"
readonly DATA_DIR="${PROJECT_ROOT}/artifacts/sft/data/${PROFILE}"
readonly DATA="${DATA_DIR}/train_alpaca.jsonl"
readonly CONFIG="${PROJECT_ROOT}/configs/sft/frontier_feedbackcore_listwise_invariant_v1.yaml"
readonly BASE_ARTIFACT_LOCK="${PROJECT_ROOT}/configs/sft/artifacts.lock.json"
readonly ARTIFACT_LOCK="${PROJECT_ROOT}/configs/sft/frontier_feedbackcore_listwise_invariant_v1.artifacts.lock.json"
readonly ENVIRONMENT_LOCK="${PROJECT_ROOT}/configs/sft/environment.lock.txt"
readonly LOG_ROOT="${PROJECT_ROOT}/operation_logs/sft/${PROFILE}"
readonly GATE_OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/sft/runs/${PROFILE}_gates"
readonly GATES_REPORT="${PROJECT_ROOT}/artifacts/sft/${PROFILE}_gates_report.json"
readonly CONFIG_REPORT="${PROJECT_ROOT}/artifacts/sft/${PROFILE}_config_check.json"
readonly FULL_OUTPUT="${PROJECT_ROOT}/artifacts/sft/runs/${PROFILE}_epoch_001"

require_python() {
    if [[ ! -x "${PYTHON}" ]]; then
        echo "Missing environment Python: ${PYTHON}" >&2
        exit 2
    fi
}

export_offline_runtime() {
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
}
