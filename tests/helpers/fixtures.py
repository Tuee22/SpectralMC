# tests/helpers/fixtures.py
"""
Shared pytest fixtures for dtype/precision parametrization.

Provides DRY fixtures for mixed precision testing across float32/float64.
All fixtures automatically parametrize tests to run twice (once per precision).
"""

from __future__ import annotations

from typing import Iterator

import pytest
import torch

from spectralmc.models.torch import FullPrecisionDType, default_dtype
from spectralmc.models.numerical import Precision


@pytest.fixture(
    params=[FullPrecisionDType.float32, FullPrecisionDType.float64], ids=["f32", "f64"]
)
def full_dtype(request: pytest.FixtureRequest) -> Iterator[torch.dtype]:
    """Parametrize PyTorch tests across float32/float64 with default_dtype context.

    Use this fixture for PyTorch model tests. The fixture installs the dtype as
    the default inside a context manager, so all torch.tensor literals and module
    construction automatically use the correct precision.

    The fixture yields the torch.dtype itself, which can be used for assertions
    or explicit dtype control within the test body.

    Example:
        def test_model_forward(full_dtype: torch.dtype) -> None:
            '''Test model forward pass - runs twice (f32, f64).'''
            model = MyModel()  # Uses full_dtype as default
            x = torch.randn(10)  # Uses full_dtype
            y = model(x)
            assert y.dtype == full_dtype

    Yields:
        torch.dtype: The active precision (torch.float32 or torch.float64)
    """
    enum = request.param
    with default_dtype(enum.to_torch()):
        yield enum.to_torch()


@pytest.fixture(
    params=[FullPrecisionDType.float32, FullPrecisionDType.float64], ids=["f32", "f64"]
)
def full_dtype_enum(request: pytest.FixtureRequest) -> FullPrecisionDType:
    """Parametrize tests across FullPrecisionDType enum values (no context manager).

    Use this when you need the enum itself rather than torch.dtype, or when you
    need explicit dtype conversion control without default_dtype context.

    This fixture does NOT install a default dtype context manager. Tests using
    this fixture must explicitly convert the enum to torch.dtype and pass it to
    tensor creation operations.

    Example:
        def test_dtype_roundtrip(full_dtype_enum: FullPrecisionDType) -> None:
            '''Test dtype enum roundtrip conversion.'''
            torch_dt = full_dtype_enum.to_torch()
            roundtrip = FullPrecisionDType.from_torch(torch_dt)
            match roundtrip:
                case Success(value):
                    assert value == full_dtype_enum
                case Failure(error):
                    pytest.fail(f"Roundtrip failed: {error}")

    Returns:
        FullPrecisionDType: The enum variant (float32 or float64)
    """
    return request.param


@pytest.fixture(params=[Precision.float32, Precision.float64], ids=["f32", "f64"])
def precision(request: pytest.FixtureRequest) -> Precision:
    """Parametrize numerical simulation tests across float32/float64.

    Use this fixture for tests involving SimulationParams, GBM, or other
    numerical components that use the Precision enum (not PyTorch models).

    This fixture returns the Precision enum directly for use in simulation
    parameter construction. It does NOT install any global dtype context.

    Example:
        def test_simulation(precision: Precision) -> None:
            '''Test Monte Carlo simulation across precisions.'''
            params = make_simulation_params(dtype=precision)
            result = run_simulation(params)
            assert result.dtype == precision
            assert_no_nan_inf(result)

    Returns:
        Precision: The enum variant (Precision.float32 or Precision.float64)
    """
    return request.param


__all__ = [
    "full_dtype",
    "full_dtype_enum",
    "precision",
]
