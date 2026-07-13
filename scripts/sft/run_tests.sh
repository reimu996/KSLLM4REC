#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LLAMAFACTORY_ROOT="/home/lyc/REC_PROJECTS/LlamaFactory"
ENV_NAME="onereason_lora_sft"
PYTHON="/home/lyc/miniconda3/envs/${ENV_NAME}/bin/python"

cd "${PROJECT_ROOT}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}/src:${LLAMAFACTORY_ROOT}/src" \
    "${PYTHON}" -m unittest discover --start-directory tests/sft --verbose
