#!/bin/bash
# Strict error handling
set -euo pipefail

echo "=== GPU Build Verification ==="
echo ""

# Check nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found!"
    echo ""
    echo "This script requires GPU access to verify the build."
    echo "Make sure you're running inside the SpectralMC container."
    echo ""
    echo "Usage:"
    echo "  docker compose exec spectralmc bash scripts/verify_gpu_build.sh"
    echo ""
    exit 1
fi

# Detect GPU
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

# Validate GPU detection
if [ -z "$COMPUTE_CAP" ] || [ -z "$GPU_NAME" ]; then
    echo "ERROR: Failed to query GPU information"
    nvidia-smi
    exit 1
fi

echo "Hardware Information:"
echo "  GPU: ${GPU_NAME}"
echo "  Compute Capability: ${COMPUTE_CAP}"
echo "  Memory: ${GPU_MEM}"
echo ""

# Test PyTorch
echo "1. Testing PyTorch Installation"
echo "--------------------------------"
if ! python3 -c "
import sys
try:
    import torch
except ImportError as e:
    print(f'ERROR: Failed to import PyTorch: {e}')
    sys.exit(1)

print(f'  PyTorch version: {torch.__version__}')

if not torch.cuda.is_available():
    print('  ERROR: CUDA not available in PyTorch!')
    print('  PyTorch was not built with CUDA support.')
    sys.exit(1)

print(f'  ✓ CUDA available: True')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  GPU count: {torch.cuda.device_count()}')
print(f'  GPU 0: {torch.cuda.get_device_name(0)}')
cap = torch.cuda.get_device_capability(0)
print(f'  Compute capability: {cap[0]}.{cap[1]}')
"; then
    echo ""
    echo "PyTorch verification FAILED!"
    exit 1
fi

echo ""

# Test CuPy
echo "2. Testing CuPy Installation"
echo "-----------------------------"
if ! python3 -c "
import sys
try:
    import cupy as cp
except ImportError as e:
    print(f'ERROR: Failed to import CuPy: {e}')
    sys.exit(1)

print(f'  CuPy version: {cp.__version__}')

if not cp.cuda.is_available():
    print('  ERROR: CUDA not available in CuPy!')
    print('  CuPy was not built with CUDA support.')
    sys.exit(1)

print(f'  ✓ CUDA available: True')
print(f'  CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}')
print(f'  Device 0: {cp.cuda.Device(0).name.decode()}')
"; then
    echo ""
    echo "CuPy verification FAILED!"
    exit 1
fi

echo ""

# Test actual GPU operations
echo "3. Testing GPU Kernel Execution"
echo "--------------------------------"
if ! python3 -c "
import sys
import torch
import cupy as cp

print('  Testing PyTorch GPU matmul...')
try:
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x.T)

    if not torch.isfinite(y).all():
        print('  ERROR: PyTorch GPU matmul produced NaN/Inf')
        sys.exit(1)

    print(f'  ✓ PyTorch GPU matmul successful: {y.shape}')
except RuntimeError as e:
    if 'no kernel image is available' in str(e):
        print('  ERROR: CUDA kernel not available for this GPU!')
        print('  PyTorch was not compiled for your compute capability.')
        print('  Rebuild required with source compilation.')
    else:
        print(f'  ERROR: PyTorch GPU test failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'  ERROR: Unexpected error in PyTorch test: {e}')
    sys.exit(1)

print('')
print('  Testing CuPy GPU dot product...')
try:
    a = cp.random.randn(1000, 1000)
    b = cp.dot(a, a.T)

    if not cp.isfinite(b).all():
        print('  ERROR: CuPy GPU dot product produced NaN/Inf')
        sys.exit(1)

    print(f'  ✓ CuPy GPU dot product successful: {b.shape}')
except RuntimeError as e:
    if 'no kernel image is available' in str(e):
        print('  ERROR: CUDA kernel not available for this GPU!')
        print('  CuPy was not compiled for your compute capability.')
        print('  Rebuild required with source compilation.')
    else:
        print(f'  ERROR: CuPy GPU test failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'  ERROR: Unexpected error in CuPy test: {e}')
    sys.exit(1)

print('')
print('  ✓ All kernel executions successful')
"; then
    echo ""
    echo "GPU kernel execution FAILED!"
    echo ""
    echo "This usually means:"
    echo "  - PyTorch/CuPy were not compiled for your GPU architecture"
    echo "  - Rebuild the Docker image from scratch"
    echo "  - Ensure DOCKER_BUILDKIT=0 during build"
    echo ""
    exit 1
fi

echo ""
echo "=== All Verifications Passed ==="
echo ""
echo "Summary:"
echo "  ✓ GPU detected and accessible"
echo "  ✓ PyTorch ${COMPUTE_CAP} CUDA support verified"
echo "  ✓ CuPy ${COMPUTE_CAP} CUDA support verified"
echo "  ✓ GPU kernel execution successful"
echo ""
echo "SpectralMC is ready for GPU-accelerated operations!"
echo ""
