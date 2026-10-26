#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=================================="
echo " ThousandML CQ Unified Installer"
echo "=================================="

if ! python -c 'import sys; assert sys.version_info[:2] == (3, 12)' ; then
    echo "ERROR: ThousandML currently requires Python 3.12."
    exit 1
fi

echo "Fetching ThousandML native libraries..."
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Installing Git LFS..."
    sudo apt-get update
    sudo apt-get install -y git-lfs
fi

git lfs install
git lfs pull

python -m pip install --upgrade pip setuptools wheel
python -m  pip install --ignore-installed PyJWT

echo "[1/6] Checking protected binaries..."

python - <<'PY'
from pathlib import Path
import sysconfig
suffix = sysconfig.get_config_var("EXT_SUFFIX")
required = {
    "Quantization": Path("lib/utils") / f"_cq_core{suffix}",
    "Inference runtime": Path("inference/cq") / f"runtime_ops{suffix}",
    "Inference kernels": Path("inference/cq") / f"cq_kernels{suffix}",
}

training = [
    "cq_linear",
    "kernel_backend",
    "runtime_geometry",
    "checkpoint_layout",
    "_cq_training_license",
]

for name in training:
    required["Training " + name] = (
        Path("training") / f"{name}{suffix}"
    )
missing = []
for label, path in required.items():
    if path.is_file():
        print(f"OK: {label}")
    else:
        print(f"MISSING: {label} ({path})")
        missing.append(str(path))

if missing:
    raise SystemExit(
        "Missing compiled binaries for this Python ABI. "
        "Build and distribute the matching protected customer binaries."
    )
PY

echo "[2/6] Installing CQ inference + CUDA support..."
python -m pip install ./inference

echo "[3/6] Installing shared training dependencies..."
# Latest Transformers release allowed by unsloth-zoo 2026.10.7.
# Pin explicitly so inference/PEFT resolution cannot retain 5.20.2.
python -m pip install "transformers==5.17.0"
python -m pip install \
    "peft>=0.20,<0.22" \
    "accelerate==1.10.1" \
    "datasets>=4.7.0,<5"

echo "[4/6] Installing CQ-compatible training acceleration..."
# Build CUDA extensions against the PyTorch already installed by CQ inference.
# Avoid a dependency resolver upgrading/replacing the CUDA-enabled torch stack.
python -m pip install ninja packaging einops

# Discover CUDA compiler, including the pip-provided nvidia/cu13 toolkit.
# A PyTorch CUDA runtime package does not guarantee a matching nvcc exists.
# If necessary, install ONLY the matching compiler (never replace torch).
find_matching_cuda_root() {
python - <<'PYCUDA'
import os, re, site, subprocess, sys
from pathlib import Path
import torch
want = torch.version.cuda
if not want:
    sys.exit('ERROR: CUDA-enabled PyTorch required')
major_minor = tuple(map(int, want.split('.')[:2]))
roots = []
for env in ('CUDA_HOME', 'CUDA_PATH'):
    if os.environ.get(env):
        roots.append(Path(os.environ[env]))
roots += [Path('/usr/local') / ('cuda-' + want), Path('/usr/local/cuda')]
for root in site.getsitepackages() + [site.getusersitepackages()]:
    roots.extend(Path(root).glob('nvidia/cu*/'))
seen = set()
for root in roots:
    if str(root) in seen:
        continue
    seen.add(str(root))
    nvcc = root / 'bin' / 'nvcc'
    if not nvcc.is_file():
        continue
    try:
        proc = subprocess.run([str(nvcc), '--version'], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        continue
    match = re.search(r'release\s+(\d+)\.(\d+)', proc.stdout + proc.stderr)
    if proc.returncode == 0 and match and tuple(map(int, match.groups())) == major_minor:
        print(root)
        sys.exit(0)
sys.exit(1)
PYCUDA
}

if ! CUDA_TOOLKIT_ROOT="$(find_matching_cuda_root)"; then
    TORCH_CUDA_VERSION="$(python -c 'import torch; print(torch.version.cuda)')"
    echo "No matching nvcc found for PyTorch CUDA $TORCH_CUDA_VERSION."
    echo "Installing matching CUDA compiler via pip (without modifying PyTorch)..."
    python -m pip install --no-deps "nvidia-cuda-nvcc==${TORCH_CUDA_VERSION}.*"
    if ! CUDA_TOOLKIT_ROOT="$(find_matching_cuda_root)"; then
        echo "ERROR: No working nvcc matching PyTorch CUDA $TORCH_CUDA_VERSION."
        echo "CUDA candidates:"
        find /usr/local/cuda* /usr/local/lib/python3.12/dist-packages/nvidia -path '*/bin/nvcc' -print 2>/dev/null || true
        echo "Check: python -m pip show nvidia-cuda-nvcc"
        exit 1
    fi
fi
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "Using CUDA toolkit: $CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version

# A40 is SM 8.6; override TORCH_CUDA_ARCH_LIST for other GPUs.
# Build from source because a matching cu130/torch2.13 wheel may not exist.
# MAX_JOBS="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" \
# CAUSAL_CONV1D_FORCE_BUILD=TRUE \
#     python -m pip install causal-conv1d --no-build-isolation --no-cache-dir

# Keep torch/triton/transformers and CQ dependencies under CQ's control.
python -m pip install flash-linear-attention --no-deps
# Unsloth Zoo requires the parent `unsloth` package to be imported first.
# Install without resolving dependencies to protect CQ's torch/CUDA stack.
python -m pip install unsloth --no-deps

# Unsloth Zoo provides optional smart gradient checkpointing. Full Unsloth
# model patching is NOT enabled: it can replace CQ quantized projections.
# Do not allow the resolver to replace the CUDA 13 PyTorch stack.
python -m pip install "unsloth-zoo==2026.10.7" --no-deps

# Install Zoo's declared auxiliary dependencies without allowing their resolver
# to replace CQ's CUDA-enabled torch/triton or pinned Transformers stack.

# Keep CQ's PyTorch, CUDA, Triton and Transformers installations intact.
python -m pip install --no-deps \
    "unsloth==2026.10.8" \
    "unsloth-zoo==2026.10.8"

# Install the missing Unsloth dependencies without replacing CQ's stack.
python -m pip install \
    bitsandbytes \
    importlib-metadata \
    diffusers \
    nest-asyncio \
    structlog \
    xformers \
    "certifi>=2026.10.19"

# Keep your existing auxiliary packages.
python -m pip install --no-deps \
    "cut-cross-entropy==25.1.1" \
    "torchao==0.18.0" \
    "trl==1.15.0" \
    "tyro==1.0.16"

python -m pip install "typeguard>=4.0"

# Reassert the CQ-compatible Transformers version.

# python -m pip install --no-deps "transformers==5.17.0"
python -m pip install "transformers==5.14.1"

python -m pip install rouge-score sacrebleu

python -m pip install --no-deps fla-core

echo "[5/6] Checking dependency consistency..."
python -m pip check

echo "[6/6] Running installation checks..."
python - <<'PY'
import torch
import transformers
from packaging.version import Version
assert Version(transformers.__version__) == Version("5.17.0"), transformers.__version__
import peft
import cq.cq_kernels
import cq.runtime_ops
import fla
# Import the parent package first: Zoo deliberately rejects standalone imports.
# Do not call FastLanguageModel.from_pretrained or apply Unsloth model patches;
# CQ remains responsible for its own quantized projection modules.
import unsloth
import unsloth_zoo.gradient_checkpointing as gc
assert callable(getattr(gc, 'patch_unsloth_smart_gradient_checkpointing', None)), (
    'Installed unsloth-zoo lacks the checkpointing API expected by training/unsloth_compat.py'
)
from lib.utils import _cq_core
from peft import LoraConfig, get_peft_model
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("flash-linear-attention (fla) import: OK")
print("Unsloth + Zoo checkpointing API: OK")
print("ThousandML CQ imports: OK")
PY

echo "=================================="
echo " ThousandML installation complete"
echo "=================================="
echo "Quantize: ./quantize-model"
echo "Inference: vllm serve MODEL --quantization cq"
echo "Training: python training/train.py --help"
echo "CQ-compatible acceleration: --checkpointing unsloth --optimizer adamw_torch_fused"
echo "NOTE: Unsloth fused LoRA, loss, RMSNorm and SwiGLU kernels are NOT wired into CQ training."
echo "      Installing a package alone cannot enable these kernels."
