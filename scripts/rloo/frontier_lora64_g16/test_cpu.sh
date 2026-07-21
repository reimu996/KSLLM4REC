#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

for suite in sft orpo grpo rloo; do
  "${PYTHON}" -m unittest discover -s "tests/${suite}" -p 'test_*.py' -v
done 2>&1 | tee "${LOG_ROOT}/cpu_regression_tests.log"
"${PYTHON}" -m py_compile src/ksllm4rec_rloo/*.py
git diff --check | tee "${LOG_ROOT}/git_diff_check.log"
