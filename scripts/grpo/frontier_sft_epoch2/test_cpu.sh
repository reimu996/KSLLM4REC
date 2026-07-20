#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -m unittest discover -s tests/grpo -p 'test_*.py' -v \
  2>&1 | tee "${LOG_ROOT}/cpu_tests.log"
"${PYTHON}" -m py_compile src/ksllm4rec_grpo/*.py scripts/grpo/prepare_frontier_v2.py \
  scripts/grpo/prepare_frontier_hf_upload.py
git diff --check | tee "${LOG_ROOT}/git_diff_check.log"
