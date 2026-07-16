#!/usr/bin/env bash
# scripts/init_h20_env.sh
#
# Initialize a portable mini-vllm environment for a locked-down GPU box:
#   - CUDA 12.1 only (cu121), no driver upgrade allowed
#   - Python 3.9.x runtime
#
# What this script does NOT touch:
#   - NVIDIA driver / kernel module (never calls apt/dpkg/nvidia-installer)
#   - System CUDA toolkit (we only pull CUDA-12.1 *user-space* wheels via pip)
#
# What it does:
#   1. Verify the interpreter is Python 3.9.x (matches the target box).
#   2. Verify the installed NVIDIA driver's max-supported CUDA >= 12.1
#      (forward compatibility check, read-only via `nvidia-smi`).
#   3. Create an isolated venv and install torch==2.5.1+cu121 — the last
#      official PyTorch release that ships a CUDA-12.1 + cp39 wheel.
#   4. Install the rest of mini-vllm's runtime deps at versions that are
#      known to work on Python 3.9 (transformers pinned to 4.52.4, which
#      is <=4.x and predates PEP604-only typing in HF's own codebase).
#   5. Install mini-vllm itself in editable mode.
#   6. Run a verification pass (torch/cuda/gpu/bf16/fp8/CUDA-graph) and dump
#      an environment snapshot next to the venv for later benchmark records.
#
# Usage:
#   bash scripts/init_h20_env.sh [options]
#
# Options (all optional, can also be set as environment variables):
#   --python PATH          Python 3.9 interpreter to use (default: auto-detect)
#   --venv-dir PATH         Where to create the venv (default: .venv-h20)
#   --torch-version VER     Torch version to install (default: 2.5.1)
#   --pip-index URL         Override PyPI index for non-torch deps
#                           (e.g. https://pypi.tuna.tsinghua.edu.cn/simple)
#   --hf-endpoint URL       Set HF_ENDPOINT for later model downloads
#                           (e.g. https://hf-mirror.com)
#   --force-recreate        Delete and recreate the venv if it exists
#   --skip-driver-check     Skip the nvidia-smi driver/CUDA compatibility check
#   -h, --help              Show this help and exit
#
# Examples:
#   bash scripts/init_h20_env.sh
#   bash scripts/init_h20_env.sh --python /usr/bin/python3.9 --pip-index https://pypi.tuna.tsinghua.edu.cn/simple
#   PYTHON_BIN=/opt/python3.9/bin/python3.9 bash scripts/init_h20_env.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults (overridable via flags or env vars)
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-h20}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
CUDA_TAG="cu121"                    # fixed: target box only has CUDA 12.1
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.52.4}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
HF_ENDPOINT_OVERRIDE="${HF_ENDPOINT_OVERRIDE:-}"
FORCE_RECREATE=0
SKIP_DRIVER_CHECK=0

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
print_help() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --venv-dir) VENV_DIR="$2"; shift 2 ;;
        --torch-version) TORCH_VERSION="$2"; shift 2 ;;
        --pip-index) PIP_INDEX_URL="$2"; shift 2 ;;
        --hf-endpoint) HF_ENDPOINT_OVERRIDE="$2"; shift 2 ;;
        --force-recreate) FORCE_RECREATE=1; shift ;;
        --skip-driver-check) SKIP_DRIVER_CHECK=1; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown option: $1" >&2; print_help; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
c_reset="\033[0m"; c_red="\033[31m"; c_yellow="\033[33m"; c_green="\033[32m"; c_blue="\033[34m"
info()  { echo -e "${c_blue}[info]${c_reset} $*"; }
warn()  { echo -e "${c_yellow}[warn]${c_reset} $*"; }
error() { echo -e "${c_red}[error]${c_reset} $*" >&2; }
ok()    { echo -e "${c_green}[ok]${c_reset} $*"; }

trap 'error "Setup failed at line $LINENO. See message above."' ERR

cd "${PROJECT_ROOT}"
info "Project root: ${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# 1. Locate and verify Python 3.9
# ---------------------------------------------------------------------------
if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in python3.9 python3.9.6 python3; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            ver="$("${candidate}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
            if [[ "${ver}" == "3.9" ]]; then
                PYTHON_BIN="$(command -v "${candidate}")"
                break
            fi
        fi
    done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    error "Could not auto-detect a Python 3.9 interpreter."
    error "Pass one explicitly: --python /path/to/python3.9"
    exit 1
fi

py_full_ver="$("${PYTHON_BIN}" --version 2>&1)"
py_major_minor="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "${py_major_minor}" != "3.9" ]]; then
    error "Selected interpreter is ${py_full_ver}, but this box requires Python 3.9.x."
    error "mini-vllm's code uses 'from __future__ import annotations' to stay 3.9-compatible;"
    error "running it under a different minor version defeats the purpose of this script."
    exit 1
fi
ok "Using interpreter: ${PYTHON_BIN} (${py_full_ver})"

# ---------------------------------------------------------------------------
# 2. Driver / CUDA forward-compatibility check (read-only, never modifies driver)
# ---------------------------------------------------------------------------
if [[ "${SKIP_DRIVER_CHECK}" -eq 0 ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi not found — skipping driver/CUDA compatibility check."
        warn "If this box has no GPU visible right now, that's fine for setup, but re-run"
        warn "with GPU access before benchmarking."
    else
        smi_output="$(nvidia-smi 2>&1 || true)"
        # nvidia-smi's header prints e.g. "CUDA Version: 12.2" — that number is the
        # *maximum* CUDA version the installed driver supports, not the installed
        # CUDA toolkit. As long as it's >= 12.1, cu121 wheels run via driver forward
        # compatibility, with no need to touch the driver itself.
        driver_cuda="$(echo "${smi_output}" | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+' || true)"
        driver_ver="$(echo "${smi_output}" | grep -oE 'Driver Version: [0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 | awk '{print $3}' || true)"

        if [[ -z "${driver_cuda}" ]]; then
            warn "Could not parse driver's max CUDA version from nvidia-smi output."
            warn "Proceeding anyway — verify manually if install fails with a CUDA init error."
        else
            info "Driver version: ${driver_ver:-unknown}, driver max CUDA: ${driver_cuda}"
            # Compare driver_cuda >= 12.1 using python (avoids bc/awk float quirks)
            cmp_result="$("${PYTHON_BIN}" -c "print(1 if tuple(map(int, '${driver_cuda}'.split('.'))) >= (12, 1) else 0)")"
            if [[ "${cmp_result}" != "1" ]]; then
                error "Driver's max supported CUDA (${driver_cuda}) is below 12.1."
                error "cu121 wheels will fail to initialize CUDA on this box, and per your"
                error "constraint the driver cannot be upgraded. Stop here rather than install."
                exit 1
            fi
            ok "Driver supports CUDA >= 12.1 — cu121 wheels are safe to use."
        fi

        gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
        [[ -n "${gpu_name}" ]] && info "Detected GPU: ${gpu_name}"
    fi
else
    warn "Skipping driver/CUDA compatibility check (--skip-driver-check)."
fi

# ---------------------------------------------------------------------------
# 3. Create / reuse venv
# ---------------------------------------------------------------------------
if [[ -d "${VENV_DIR}" ]]; then
    if [[ "${FORCE_RECREATE}" -eq 1 ]]; then
        warn "Removing existing venv at ${VENV_DIR} (--force-recreate)."
        rm -rf "${VENV_DIR}"
    else
        info "Reusing existing venv at ${VENV_DIR} (use --force-recreate to rebuild)."
    fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    info "Creating venv at ${VENV_DIR} ..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
    error "venv python not found at ${VENV_PY} — venv creation likely failed."
    exit 1
fi

"${VENV_PY}" -m pip install --quiet --upgrade pip setuptools wheel
ok "venv ready: ${VENV_PY}"

# ---------------------------------------------------------------------------
# 4. Install torch (cu121, pinned) — the load-bearing constraint of this box
# ---------------------------------------------------------------------------
info "Installing torch==${TORCH_VERSION}+${CUDA_TAG} (this is the last torch release"
info "with an official ${CUDA_TAG} + cp39 wheel; do not bump without re-checking"
info "https://download.pytorch.org/whl/${CUDA_TAG}/torch/ for cp39 availability)."

"${VENV_PY}" -m pip install \
    "torch==${TORCH_VERSION}+${CUDA_TAG}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# ---------------------------------------------------------------------------
# 5. Install remaining runtime deps (optionally via a mirror index)
# ---------------------------------------------------------------------------
pip_extra_args=()
if [[ -n "${PIP_INDEX_URL}" ]]; then
    info "Using pip index mirror: ${PIP_INDEX_URL}"
    pip_extra_args+=(-i "${PIP_INDEX_URL}")
fi

info "Installing transformers==${TRANSFORMERS_VERSION}, safetensors, xxhash, pytest ..."
"${VENV_PY}" -m pip install --quiet "${pip_extra_args[@]}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "safetensors>=0.5" \
    "xxhash>=3.5" \
    "pytest>=8.0" \
    "pytest-timeout>=2.3" \
    numpy

# ---------------------------------------------------------------------------
# 6. Install mini-vllm itself (editable, no-deps — deps are already pinned above)
# ---------------------------------------------------------------------------
info "Installing mini-vllm in editable mode ..."
"${VENV_PY}" -m pip install --quiet -e "${PROJECT_ROOT}" --no-deps

# ---------------------------------------------------------------------------
# 7. Verification pass + environment snapshot
# ---------------------------------------------------------------------------
SNAPSHOT_DIR="${PROJECT_ROOT}/results/env_snapshot"
mkdir -p "${SNAPSHOT_DIR}"

info "Running environment verification ..."
"${VENV_PY}" "${SCRIPT_DIR}/verify_env.py" --out "${SNAPSHOT_DIR}/env_report.json"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q > "${SNAPSHOT_DIR}/nvidia_smi.txt" 2>&1 || true
fi
"${VENV_PY}" -m pip freeze > "${SNAPSHOT_DIR}/pip_freeze.txt"
if command -v git >/dev/null 2>&1 && git -C "${PROJECT_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
    git -C "${PROJECT_ROOT}" rev-parse HEAD > "${SNAPSHOT_DIR}/git_commit.txt"
    git -C "${PROJECT_ROOT}" status --short > "${SNAPSHOT_DIR}/git_status.txt"
fi

# ---------------------------------------------------------------------------
# 8. Optional: HF mirror for later model downloads
# ---------------------------------------------------------------------------
if [[ -n "${HF_ENDPOINT_OVERRIDE}" ]]; then
    info "Remember to export HF_ENDPOINT=${HF_ENDPOINT_OVERRIDE} in your shell before"
    info "downloading models (this script does not persist shell env vars for you)."
fi

echo
ok "Environment ready."
echo
echo "Next steps:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_correctness.py tests/test_kv_fp8.py"
echo "  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_h20_serving.py --help"
echo
echo "Snapshot saved to: ${SNAPSHOT_DIR}/"
