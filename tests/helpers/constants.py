# tests/helpers/constants.py
"""Shared test constants for SpectralMC test suite.

These constants provide defaults for common test scenarios. Tests can
override these values when specific values are needed. Consolidates
magic numbers duplicated across test files.
"""

from __future__ import annotations

from typing import Literal

# ============================================================================
# Simulation Parameters
# ============================================================================

DEFAULT_TIMESTEPS = 100
"""Default number of timesteps for Monte Carlo simulations in tests."""

DEFAULT_NETWORK_SIZE = 1024
"""Default neural network hidden layer size for CVNN models in tests."""

DEFAULT_BATCHES_PER_RUN = 8
"""Default number of batches per Monte Carlo run in tests."""


DEFAULT_THREADS_PER_BLOCK: Literal[32, 64, 128, 256, 512, 1024] = 256
"""Default CUDA threads per block for GPU kernels in tests."""

DEFAULT_MC_SEED = 42
"""Default random seed for Monte Carlo simulations (reproducibility)."""

DEFAULT_BUFFER_SIZE = 10000
"""Default buffer size for asynchronous normal distribution generation."""

# ============================================================================
# Numerical Tolerances (Precision-Specific)
# ============================================================================

# Absolute tolerances (atol)
ATOL_FLOAT32 = 1e-8
"""Absolute tolerance for float32 comparisons.

Use with torch.allclose or assert_tensors_close for float32 numerical tests.
This value balances precision and robustness for single-precision arithmetic.
"""

ATOL_FLOAT64 = 1e-10
"""Absolute tolerance for float64 comparisons.

Use with torch.allclose or assert_tensors_close for float64 numerical tests.
This tighter tolerance leverages double-precision accuracy.
"""

# Relative tolerances (rtol)
RTOL_FLOAT32 = 1e-5
"""Relative tolerance for float32 comparisons.

Use with torch.allclose or assert_tensors_close for float32 numerical tests.
Relative tolerance scales with magnitude of values being compared.
"""

RTOL_FLOAT64 = 1e-8
"""Relative tolerance for float64 comparisons.

Use with torch.allclose or assert_tensors_close for float64 numerical tests.
Relative tolerance scales with magnitude of values being compared.
"""

# Epsilon values (machine precision)
EPS_FLOAT32 = 1.192092896e-07
"""Machine epsilon for float32 (torch.finfo(torch.float32).eps).

Use for numerical stability tests or when checking if values are "close to zero"
relative to single-precision floating point limits. This is the smallest value
such that 1.0 + eps != 1.0 in float32 arithmetic.
"""

EPS_FLOAT64 = 2.220446049250313e-16
"""Machine epsilon for float64 (torch.finfo(torch.float64).eps).

Use for numerical stability tests or when checking if values are "close to zero"
relative to double-precision floating point limits. This is the smallest value
such that 1.0 + eps != 1.0 in float64 arithmetic.
"""

# ============================================================================
# Training Parameters
# ============================================================================

DEFAULT_LEARNING_RATE = 1e-3
"""Default learning rate for optimizer tests."""

DEFAULT_MAX_ITERATIONS = 1000
"""Default maximum iterations for training loops in tests."""

CONVERGENCE_THRESHOLD = 1e-6
"""Default convergence threshold for loss values."""

# ============================================================================
# Model Sizes
# ============================================================================

SMALL_MODEL_SIZE = 256
"""Small model size for quick tests."""

MEDIUM_MODEL_SIZE = 1024
"""Medium model size for standard tests (same as DEFAULT_NETWORK_SIZE)."""

LARGE_MODEL_SIZE = 4096
"""Large model size for stress tests or GPU memory tests."""
