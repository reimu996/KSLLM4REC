#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"${SCRIPT_DIR}/prepare_v3_1.sh"
"${SCRIPT_DIR}/test_cpu_v3_1.sh"
"${SCRIPT_DIR}/run_gates_v3_1.sh"
"${SCRIPT_DIR}/run_full_v3_1.sh"
"${SCRIPT_DIR}/run_probes_v3_1.sh"
"${SCRIPT_DIR}/verify_v3_1.sh"
