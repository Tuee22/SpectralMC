#!/bin/bash
# Strict error handling
set -euo pipefail

echo "=== SpectralMC GPU Detection & Build ==="
echo ""

# CRITICAL: Check nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found!"
    echo ""
    echo "SpectralMC requires GPU access during Docker build."
    echo ""
    echo "Possible causes:"
    echo "  1. nvidia-docker2 not installed on host"
    echo "  2. Docker build not using legacy builder (DOCKER_BUILDKIT=0 required)"
    echo "  3. GPU not accessible to Docker daemon"
    echo ""
    echo "Solutions:"
    echo "  Install nvidia-docker2:"
    echo "    sudo apt-get install -y nvidia-docker2"
    echo "    sudo systemctl restart docker"
    echo ""
    echo "  Build with legacy builder:"
    echo "    cd docker && DOCKER_BUILDKIT=0 docker compose build spectralmc"
    echo ""
    echo "  Verify GPU access:"
    echo "    docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi"
    echo ""
    echo "See docs/GPU_BUILD_TROUBLESHOOTING.md for detailed help."
    echo ""
    exit 1
fi

# Detect GPU compute capability
echo "Querying GPU information..."
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

# Validate GPU detection results
if [ -z "$COMPUTE_CAP" ] || [ -z "$GPU_NAME" ]; then
    echo "ERROR: Failed to query GPU information from nvidia-smi"
    echo "nvidia-smi output:"
    nvidia-smi
    exit 1
fi

# Remove decimal point for numeric comparison
COMPUTE_CAP_INT=$(echo "$COMPUTE_CAP" | tr -d '.')

# Validate compute capability is a number
if ! [[ "$COMPUTE_CAP_INT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid compute capability detected: $COMPUTE_CAP"
    exit 1
fi

echo "✓ Detected GPU: ${GPU_NAME}"
echo "✓ Compute capability: ${COMPUTE_CAP} (sm_${COMPUTE_CAP_INT})"
echo "✓ GPU Memory: ${GPU_MEM}"
echo ""

# Validate minimum compute capability (3.0 required for CUDA 12.8)
if [ "${COMPUTE_CAP_INT}" -lt 30 ]; then
    echo "ERROR: GPU compute capability too old!"
    echo ""
    echo "Detected: ${COMPUTE_CAP} (sm_${COMPUTE_CAP_INT})"
    echo "Minimum: 3.0 (Kepler architecture)"
    echo ""
    echo "CUDA 12.8 requires compute capability >= 3.0"
    echo "Your GPU is not supported."
    echo ""
    exit 1
fi

# Decision logic: < 6.0 requires source build
if [ "${COMPUTE_CAP_INT}" -lt 60 ]; then
    echo ">>> Building PyTorch and CuPy from source for legacy GPU (sm_${COMPUTE_CAP_INT})"
    echo "    This will take 2-4 hours. Please be patient."
    BUILD_FROM_SOURCE=1
else
    echo ">>> Using pre-compiled binaries for modern GPU (sm_${COMPUTE_CAP_INT})"
    echo "    This will take 5-10 minutes."
    BUILD_FROM_SOURCE=0
fi

# Export for subsequent build steps
export BUILD_FROM_SOURCE
export COMPUTE_CAP
export COMPUTE_CAP_INT
export GPU_NAME
export GPU_MEM

echo ""

# Build PyTorch from source if needed
if [ "${BUILD_FROM_SOURCE}" = "1" ]; then
    echo "=== Building PyTorch 2.7.0 from source (this will take 2-4 hours) ==="

    # Set CUDA architecture for compilation
    export TORCH_CUDA_ARCH_LIST="${COMPUTE_CAP}"
    export MAX_JOBS=4  # Limit parallel jobs to avoid OOM during compilation
    export USE_CUDA=1
    export USE_CUDNN=1
    export USE_NCCL=1

    # Install build dependencies
    echo "Installing build dependencies..."
    apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        ccache \
        libjpeg-dev \
        libpng-dev \
        && rm -rf /var/lib/apt/lists/*

    pip install --no-cache-dir setuptools wheel pyyaml typing_extensions

    # Clone PyTorch 2.7.0
    echo "Cloning PyTorch 2.7.0..."
    git clone --depth 1 --branch v2.7.0 --recursive \
        https://github.com/pytorch/pytorch.git /tmp/pytorch

    cd /tmp/pytorch

    # Build and install
    echo "Building PyTorch (this takes 2-4 hours, please be patient)..."
    if ! python setup.py bdist_wheel; then
        echo "ERROR: PyTorch build failed!"
        echo "Check build logs above for errors."
        echo "Common issues:"
        echo "  - Out of memory (reduce MAX_JOBS in detect_gpu_and_build.sh)"
        echo "  - Missing build dependencies"
        exit 1
    fi

    echo "Installing PyTorch wheel..."
    if ! pip install --no-cache-dir dist/*.whl; then
        echo "ERROR: PyTorch wheel installation failed!"
        exit 1
    fi

    # Verify PyTorch installation (from outside source directory)
    echo "Verifying PyTorch installation..."
    cd /tmp
    if ! python -c "import torch; print(f'PyTorch {torch.__version__} installed')"; then
        echo "ERROR: PyTorch import failed after installation!"
        exit 1
    fi

    # Cleanup
    cd /
    rm -rf /tmp/pytorch

    echo "✓ PyTorch built successfully for sm_${COMPUTE_CAP_INT}"
    echo ""

    # Build CuPy from source
    echo "=== Building CuPy 13.x from source (this will take 30-60 minutes) ==="
    echo ""

    export CUPY_NVCC_GENERATE_CODE="arch=compute_${COMPUTE_CAP_INT},code=sm_${COMPUTE_CAP_INT}"
    export CUPY_COMPILE_WITH_PTX=1

    # Clone CuPy 13.3.0
    echo "Cloning CuPy 13.3.0..."
    git clone --depth 1 --branch v13.3.0 --recursive \
        https://github.com/cupy/cupy.git /tmp/cupy

    cd /tmp/cupy

    # Build and install
    echo "Building CuPy (this takes 30-60 minutes, please be patient)..."
    if ! python setup.py install; then
        echo "ERROR: CuPy build failed!"
        echo "Check build logs above for errors."
        exit 1
    fi

    # Install NumPy and fastrlock (required for CuPy verification)
    echo "Installing NumPy and fastrlock for CuPy verification..."
    pip install --no-cache-dir numpy fastrlock

    # Verify CuPy installation (from outside source directory)
    echo "Verifying CuPy installation..."
    cd /tmp
    if ! python -c "import cupy; print(f'CuPy {cupy.__version__} installed')"; then
        echo "ERROR: CuPy import failed after installation!"
        exit 1
    fi

    # Cleanup
    cd /
    rm -rf /tmp/cupy

    echo "✓ CuPy built successfully for sm_${COMPUTE_CAP_INT}"
    echo ""

    # Install remaining dependencies via Poetry (excluding GPU packages)
    echo "=== Installing remaining SpectralMC dependencies ==="
    if ! poetry install --with dev --no-interaction; then
        echo "ERROR: Poetry dependency installation failed!"
        exit 1
    fi
    echo "✓ Dependencies installed"

else
    # Use Poetry to install all dependencies including binary wheels
    echo "=== Installing all dependencies via Poetry (including binary PyTorch/CuPy) ==="
    if ! poetry install --with dev --extras binary-gpu --no-interaction; then
        echo "ERROR: Poetry installation failed!"
        echo "This may indicate:"
        echo "  - Network issues downloading packages"
        echo "  - Incompatible package versions"
        echo "  - Missing system dependencies"
        exit 1
    fi

    # Verify installations
    echo "Verifying installations..."
    if ! python -c "import torch; print(f'PyTorch {torch.__version__}')"; then
        echo "ERROR: PyTorch import failed!"
        exit 1
    fi
    if ! python -c "import cupy; print(f'CuPy {cupy.__version__}')"; then
        echo "ERROR: CuPy import failed!"
        exit 1
    fi

    echo "✓ Binary packages installed and verified"
fi

echo ""
echo "=== Final Verification ==="
echo "Checking CUDA availability..."

# Final verification that CUDA is accessible
if ! python -c "
import torch
import cupy as cp

# Check PyTorch CUDA
if not torch.cuda.is_available():
    print('ERROR: PyTorch CUDA not available!')
    exit(1)

# Check CuPy CUDA
if not cp.cuda.is_available():
    print('ERROR: CuPy CUDA not available!')
    exit(1)

# Check compute capability matches
cap = torch.cuda.get_device_capability(0)
detected_cap = f'{cap[0]}.{cap[1]}'
print(f'✓ PyTorch CUDA available (compute {detected_cap})')
print(f'✓ CuPy CUDA available')
"; then
    echo "ERROR: Final CUDA verification failed!"
    echo "GPU libraries installed but CUDA not accessible."
    exit 1
fi

echo ""
echo "=== Build Complete & Verified ==="
echo "GPU: ${GPU_NAME}"
echo "Compute capability: ${COMPUTE_CAP} (sm_${COMPUTE_CAP_INT})"
echo "GPU Memory: ${GPU_MEM}"
echo "Build method: $([ "${BUILD_FROM_SOURCE}" = "1" ] && echo "Source compilation" || echo "Pre-compiled binaries")"
echo ""
echo "PyTorch and CuPy are installed and CUDA-enabled."
echo "Ready for GPU-accelerated SpectralMC operations!"
echo ""
