# tests/helpers/dtype_constants.py
"""
Shared dtype/precision constants for parametrized testing.

Use these constants with @pytest.mark.parametrize when you need decorator-based
parametrization instead of fixture-based parametrization.

For fixture-based parametrization (recommended for most cases), use the fixtures
from tests.helpers.fixtures instead.
"""

from __future__ import annotations

import torch

from spectralmc.models.torch import FullPrecisionDType
from spectralmc.models.numerical import Precision


# ============================================================================
# PyTorch dtype tuples
# ============================================================================

FULL_PRECISION_DTYPES: tuple[FullPrecisionDType, ...] = (
    FullPrecisionDType.float32,
    FullPrecisionDType.float64,
)
"""FullPrecisionDType tuple for decorator parametrization.

Use with @pytest.mark.parametrize for PyTorch model tests that need explicit
enum control without the automatic default_dtype context manager.

Example:
    @pytest.mark.parametrize("dtype", FULL_PRECISION_DTYPES)
    def test_device_placement(dtype: FullPrecisionDType) -> None:
        model = create_model(dtype=dtype)
        for p in model.parameters():
            assert p.dtype == dtype.to_torch()
"""

TORCH_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.float64,
)
"""torch.dtype tuple for decorator parametrization.

Use with @pytest.mark.parametrize when you need raw torch.dtype values
without enum wrappers.

Example:
    @pytest.mark.parametrize("dtype", TORCH_DTYPES)
    def test_tensor_ops(dtype: torch.dtype) -> None:
        x = torch.randn(10, dtype=dtype)
        assert x.dtype == dtype
"""


# ============================================================================
# Numerical precision tuples
# ============================================================================

PRECISIONS: tuple[Precision, ...] = (
    Precision.float32,
    Precision.float64,
)
"""Precision enum tuple for numerical simulation tests.

Use with @pytest.mark.parametrize for tests involving SimulationParams, GBM,
or other numerical components that use the Precision enum.

Example:
    @pytest.mark.parametrize("precision", PRECISIONS)
    def test_simulation(precision: Precision) -> None:
        params = make_simulation_params(dtype=precision)
        result = run_simulation(params)
        assert result.dtype == precision
"""


# ============================================================================
# Complex dtypes (for specialized tests)
# ============================================================================

FULL_PRECISION_COMPLEX_DTYPES: tuple[FullPrecisionDType, ...] = (
    FullPrecisionDType.complex64,
    FullPrecisionDType.complex128,
)
"""FullPrecisionDType complex tuple for complex-valued neural network tests.

Use when testing CVNN components that require complex dtypes.

Example:
    @pytest.mark.parametrize("dtype", FULL_PRECISION_COMPLEX_DTYPES)
    def test_complex_activation(dtype: FullPrecisionDType) -> None:
        layer = ComplexActivation(dtype=dtype)
        assert layer.dtype == dtype
"""

PRECISION_COMPLEX: tuple[Precision, ...] = (
    Precision.complex64,
    Precision.complex128,
)
"""Precision enum complex tuple for complex-valued simulation tests.

Use when testing numerical simulations that use complex precision.

Example:
    @pytest.mark.parametrize("precision", PRECISION_COMPLEX)
    def test_fft_simulation(precision: Precision) -> None:
        params = make_fft_params(dtype=precision)
        result = run_fft(params)
        assert result.dtype == precision
"""


__all__ = [
    "FULL_PRECISION_DTYPES",
    "TORCH_DTYPES",
    "PRECISIONS",
    "FULL_PRECISION_COMPLEX_DTYPES",
    "PRECISION_COMPLEX",
]
