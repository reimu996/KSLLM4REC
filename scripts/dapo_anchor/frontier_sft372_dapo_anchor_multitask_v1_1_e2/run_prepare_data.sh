#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

"${PYTHON}" "${SCRIPT_DIR}/prepare_data.py" \
  --config "${CONFIG}" \
  --output-dir "${TEXT_GROUPS_DIR}"

