#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
export_offline_runtime

"${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=1 \
    --module ksllm4rec_sft.cli config-check \
    --profile "${PROFILE}" \
    --config "${CONFIG}" \
    --artifact-lock "${ARTIFACT_LOCK}" \
    --environment-lock "${ENVIRONMENT_LOCK}" \
    --report "${CONFIG_REPORT}"
