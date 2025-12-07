# tests/test_gpu_memory.py
"""
GPU memory and compute capability validation tests.

Ensures that the GPU has sufficient resources for the test suite and that
PyTorch/CuPy were compiled for the correct compute capability.

All tests require GPU - missing GPU is a hard failure, not a skip.
"""

from __future__ import annotations

import cupy as cp
import torch


# Module-level GPU requirement - test file fails immediately without GPU
assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"



def test_gpu_memory_sufficient() -> None:
    """Verify GPU has sufficient memory for test suite."""
    # Use torch.cuda module's get_device_properties function
    cuda_module = torch.cuda
    props = cuda_module.get_device_properties(0)
    total_memory: int = props.total_memory
    min_required = 1.5 * (1024**3)  # 1.5GB minimum

    assert total_memory >= min_required, (
        f"GPU has {total_memory / (1024**3):.2f}GB, "
        f"need at least {min_required / (1024**3):.2f}GB for tests"
    )


def test_gpu_compute_capability() -> None:
    """Verify GPU compute capability is detected and supported."""
    # Use torch.cuda module's get_device_capability function
    cuda_module = torch.cuda
    compute_cap: tuple[int, int] = cuda_module.get_device_capability(0)
    major, minor = compute_cap

    # Maxwell (GTX 970) is 5.2, should work with source build
    # Kepler (GTX 780 Ti) is 3.5, also supported
    assert major >= 3, f"Compute capability {major}.{minor} too old (need 3.0+)"

    print(f"GPU compute capability: {major}.{minor}")


def test_gpu_operations_basic() -> None:
    """Verify basic GPU operations work (PyTorch kernel execution)."""
    # Test basic tensor operations
    x = torch.randn(100, 100, device=GPU_DEV)
    y = torch.matmul(x, x.T)

    # Verify result is finite (no NaN/Inf)
    assert torch.isfinite(y).all(), "GPU matmul produced NaN or Inf"

    # Verify result shape
    assert y.shape == (100, 100), f"Expected (100, 100), got {y.shape}"


def test_cupy_available() -> None:
    """Verify CuPy is available and can run GPU operations."""
    # Use cupy.cuda.runtime to check availability
    cuda_runtime = cp.cuda.runtime
    device_count: int = cuda_runtime.getDeviceCount()
    assert device_count > 0, "CuPy CUDA not available"

    # Test basic CuPy operations using cp.random module
    random_module = cp.random
    a = random_module.randn(100, 100)
    b = cp.tensordot(a, a.T, axes=1)

    # Verify result is finite
    finite_check = cp.isfinite(b)
    assert finite_check.all(), "CuPy dot product produced NaN or Inf"

    # Verify result shape
    assert b.shape == (100, 100), f"Expected (100, 100), got {b.shape}"
