#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/prepare_data.sh"
"${SCRIPT_DIR}/run_preflight.sh"
"${SCRIPT_DIR}/run_config_check.sh"
"${SCRIPT_DIR}/run_gates.sh"
"${SCRIPT_DIR}/run_full.sh"
"${SCRIPT_DIR}/verify_full.sh"
