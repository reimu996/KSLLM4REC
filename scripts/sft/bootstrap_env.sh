#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LLAMAFACTORY_ROOT="/home/lyc/REC_PROJECTS/LlamaFactory"
CONDA_BIN="/home/lyc/miniconda3/bin/conda"
ENV_NAME="onereason_lora_sft"
ENV_PREFIX="/home/lyc/miniconda3/envs/${ENV_NAME}"
RUN_ID="bootstrap_$(date '+%Y%m%d_%H%M%S_%N')"
RUN_DIR="${PROJECT_ROOT}/operation_logs/sft/${RUN_ID}"
CONSTRAINTS="${PROJECT_ROOT}/configs/sft/constraints.txt"
ENVIRONMENT_LOCK="${PROJECT_ROOT}/configs/sft/environment.lock.txt"
PIP_COMMON=(--disable-pip-version-check --retries 20 --timeout 120 --progress-bar off)
WHEELHOUSE="${PROJECT_ROOT}/artifacts/sft/wheelhouse"
PIP_CACHE_ROOT="${PIP_CACHE_DIR:-${HOME}/.cache/pip}"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/bootstrap.log") 2>&1
trap 'status=$?; echo "finished_at=$(date --iso-8601=seconds)"; echo "exit_code=${status}"; if [[ ${status} -ne 0 ]]; then echo "status=failed"; fi' EXIT

echo "run_id=${RUN_ID}"
echo "project_root=${PROJECT_ROOT}"
echo "llamafactory_root=${LLAMAFACTORY_ROOT}"
echo "environment=${ENV_NAME}"
echo "started_at=$(date --iso-8601=seconds)"

if [[ "$(git -C "${LLAMAFACTORY_ROOT}" rev-parse HEAD)" != "0b7aaf8f6a624bd89a01a155d4265ec861cbdf38" ]]; then
    echo "Pinned LLaMA-Factory commit is not checked out." >&2
    exit 2
fi
if [[ "$(git -C "${LLAMAFACTORY_ROOT}" rev-parse 'HEAD^{tree}')" != "02158cfe15d2019e3353901d61440b5a320993ab" ]]; then
    echo "Pinned LLaMA-Factory Git tree does not match." >&2
    exit 2
fi
if [[ -n "$(git -C "${LLAMAFACTORY_ROOT}" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "Pinned LLaMA-Factory working tree is not clean." >&2
    exit 2
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    "${CONDA_BIN}" create --name "${ENV_NAME}" --yes python=3.11.15 pip
fi

PYTHON="${ENV_PREFIX}/bin/python"
if [[ "$("${PYTHON}" -c 'import platform; print(platform.python_version())')" != "3.11.15" ]]; then
    echo "Environment ${ENV_NAME} must use Python 3.11.15." >&2
    exit 2
fi
mkdir -p "${WHEELHOUSE}"

"${PYTHON}" -m pip install "${PIP_COMMON[@]}" \
    "pip==26.1.2" "setuptools==82.0.1" "wheel==0.47.0"

download_resolved_wheel() {
    local report_name="$1"
    shift
    local wheel_url wheel_name wheel_sha wheel_size part_path final_path cache_candidate
    IFS=$'\t' read -r wheel_url wheel_name wheel_sha wheel_size < <("${PYTHON}" "${SCRIPT_DIR}/resolve_wheel.py" "$@")
    final_path="${WHEELHOUSE}/${wheel_name}"
    part_path="${final_path}.part"
    if [[ -f "${final_path}" ]] && ! printf '%s  %s\n' "${wheel_sha}" "${final_path}" | sha256sum --check --status; then
        rm -f "${final_path}"
    fi
    if [[ ! -f "${final_path}" && -d "${PIP_CACHE_ROOT}/http-v2" ]]; then
        while IFS= read -r cache_candidate; do
            if printf '%s  %s\n' "${wheel_sha}" "${cache_candidate}" | sha256sum --check --status; then
                echo "Seeding ${wheel_name} from verified pip cache: ${cache_candidate}"
                cp --reflink=auto "${cache_candidate}" "${part_path}"
                printf '%s  %s\n' "${wheel_sha}" "${part_path}" | sha256sum --check
                mv "${part_path}" "${final_path}"
                break
            fi
        done < <(find "${PIP_CACHE_ROOT}/http-v2" -type f -name '*.body' -size "${wheel_size}c" -print)
    fi
    if [[ ! -f "${final_path}" ]]; then
        wget --continue --tries=20 --timeout=120 --read-timeout=120 --progress=dot:giga \
            --output-document="${part_path}" "${wheel_url}"
        printf '%s  %s\n' "${wheel_sha}" "${part_path}" | sha256sum --check
        mv "${part_path}" "${final_path}"
    fi
    "${PYTHON}" -m pip install "${PIP_COMMON[@]}" --no-deps \
        --report "${RUN_DIR}/${report_name}-install-report.json" "${final_path}"
}
"${PYTHON}" -m pip install "${PIP_COMMON[@]}" \
    --report "${RUN_DIR}/torch-python-deps-install-report.json" \
    "filelock==3.18.0" \
    "fsspec==2025.3.0" \
    "jinja2==3.1.6" \
    "networkx==3.5" \
    "numpy==1.26.4" \
    "pillow==11.3.0" \
    "sympy==1.14.0" \
    "typing-extensions==4.15.0"

CUDA_PACKAGES=(
    "nvidia-cuda-nvrtc-cu12|12.6.77"
    "nvidia-cuda-runtime-cu12|12.6.77"
    "nvidia-cuda-cupti-cu12|12.6.80"
    "nvidia-cudnn-cu12|9.5.1.17"
    "nvidia-cublas-cu12|12.6.4.1"
    "nvidia-cufft-cu12|11.3.0.4"
    "nvidia-curand-cu12|10.3.7.77"
    "nvidia-cusolver-cu12|11.7.1.2"
    "nvidia-cusparse-cu12|12.5.4.2"
    "nvidia-cusparselt-cu12|0.6.3"
    "nvidia-nccl-cu12|2.26.2"
    "nvidia-nvtx-cu12|12.6.77"
    "nvidia-nvjitlink-cu12|12.6.85"
    "nvidia-cufile-cu12|1.11.1.6"
    "triton|3.3.1"
)
for package_spec in "${CUDA_PACKAGES[@]}"; do
    package="${package_spec%%|*}"
    version="${package_spec##*|}"
    report_name="$(printf '%s_%s' "${package}" "${version}" | tr '=.' '__')"
    download_resolved_wheel "${report_name}" --pypi-package "${package}" --version "${version}"
done

download_resolved_wheel "torch" \
    --simple-index https://download.pytorch.org/whl/cu126/torch/ \
    --filename 'torch-2.7.1+cu126-cp311-cp311-manylinux_2_28_x86_64.whl' \
    --size 822139676
download_resolved_wheel "torchvision" \
    --simple-index https://download.pytorch.org/whl/cu126/torchvision/ \
    --filename 'torchvision-0.22.1+cu126-cp311-cp311-manylinux_2_28_x86_64.whl' \
    --size 7487330
download_resolved_wheel "torchaudio" \
    --simple-index https://download.pytorch.org/whl/cu126/torchaudio/ \
    --filename 'torchaudio-2.7.1+cu126-cp311-cp311-manylinux_2_28_x86_64.whl' \
    --size 3456073

FLASH_WHEEL="${WHEELHOUSE}/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
FLASH_SIZE=414500830
FLASH_SHA256="22013b8c74a63fc70e69be1e10ff02e4ad8fec84a43600bdca67b434ed417113"
if [[ -f "${FLASH_WHEEL}" ]] && { [[ "$(stat -c '%s' "${FLASH_WHEEL}")" != "${FLASH_SIZE}" ]] || \
    ! printf '%s  %s\n' "${FLASH_SHA256}" "${FLASH_WHEEL}" | sha256sum --check --status; }; then
    rm -f "${FLASH_WHEEL}"
fi
if [[ ! -f "${FLASH_WHEEL}" ]]; then
    wget --continue --tries=20 --timeout=120 --read-timeout=120 --progress=dot:giga \
        --header='Accept: application/octet-stream' \
        --header='X-GitHub-Api-Version: 2022-11-28' \
        --header='User-Agent: KSLLM4REC-bootstrap' \
        --output-document="${FLASH_WHEEL}.part" \
        "https://api.github.com/repos/Dao-AILab/flash-attention/releases/assets/287924223"
    [[ "$(stat -c '%s' "${FLASH_WHEEL}.part")" == "${FLASH_SIZE}" ]]
    printf '%s  %s\n' "${FLASH_SHA256}" "${FLASH_WHEEL}.part" | sha256sum --check
    mv "${FLASH_WHEEL}.part" "${FLASH_WHEEL}"
fi
printf '%s  %s\n' "${FLASH_SHA256}" "${FLASH_WHEEL}" | sha256sum --check
sha256sum "${FLASH_WHEEL}" > "${RUN_DIR}/flash-attn-wheel.sha256"
"${PYTHON}" -m pip install "${PIP_COMMON[@]}" --no-deps \
    --report "${RUN_DIR}/flash-attn-install-report.json" "${FLASH_WHEEL}"

"${PYTHON}" -m pip install \
    "${PIP_COMMON[@]}" \
    --report "${RUN_DIR}/environment-lock-install-report.json" \
    --constraint "${CONSTRAINTS}" \
    --requirement "${ENVIRONMENT_LOCK}"

"${PYTHON}" -m pip check
"${PYTHON}" -m pip freeze > "${RUN_DIR}/pip-freeze.txt"
if ! cmp --silent "${ENVIRONMENT_LOCK}" "${RUN_DIR}/pip-freeze.txt"; then
    diff --unified "${ENVIRONMENT_LOCK}" "${RUN_DIR}/pip-freeze.txt" || true
    echo "Installed environment does not match environment.lock.txt." >&2
    exit 3
fi
"${PYTHON}" -m pip inspect > "${RUN_DIR}/pip-inspect.json"
"${CONDA_BIN}" list --name "${ENV_NAME}" --explicit > "${RUN_DIR}/conda-explicit.txt"
git -C "${LLAMAFACTORY_ROOT}" rev-parse HEAD > "${RUN_DIR}/llamafactory-commit.txt"
sha256sum "${CONSTRAINTS}" > "${RUN_DIR}/constraints.sha256"
sha256sum "${ENVIRONMENT_LOCK}" > "${RUN_DIR}/environment-lock.sha256"
find "${WHEELHOUSE}" -maxdepth 1 -type f -name '*.whl' -print0 | sort -z | xargs -0 sha256sum > "${RUN_DIR}/wheelhouse.sha256"

"${PYTHON}" - <<'PY'
import accelerate
import datasets
import flash_attn
import peft
import torch
import transformers
import trl

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
print("peft", peft.__version__)
print("trl", trl.__version__)
print("flash_attn", flash_attn.__version__)
PY

echo "finished_at=$(date --iso-8601=seconds)"
echo "status=passed"
