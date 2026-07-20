#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"${SCRIPT_DIR}/prepare.sh"
"${SCRIPT_DIR}/test_cpu.sh"
"${SCRIPT_DIR}/run_gates.sh"
"${SCRIPT_DIR}/run_full.sh"
"${SCRIPT_DIR}/run_probes.sh"
"${SCRIPT_DIR}/verify.sh"
