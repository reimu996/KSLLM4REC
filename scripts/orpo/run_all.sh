#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/freeze_sources.sh"
"${SCRIPT_DIR}/prepare_pairs.sh"
"${SCRIPT_DIR}/run_gates.sh"
"${SCRIPT_DIR}/run_full.sh"
"${SCRIPT_DIR}/run_probes.sh"
