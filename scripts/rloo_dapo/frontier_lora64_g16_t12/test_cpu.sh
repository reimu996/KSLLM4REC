#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

"${PYTHON}" -m unittest discover -s "${ROOT}/tests/rloo_dapo" -v
"${PYTHON}" -m unittest discover -s "${ROOT}/tests/rloo" -v
