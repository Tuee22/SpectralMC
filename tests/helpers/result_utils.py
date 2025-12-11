# tests/helpers/result_utils.py
"""Result type unwrapping utilities for SpectralMC tests.

This module provides helpers for unwrapping Result types in tests,
consolidating the _expect_success pattern duplicated across 9+ test files.
"""

from __future__ import annotations

from typing import TypeVar

from spectralmc.result import Failure, Result, Success

T = TypeVar("T")
E = TypeVar("E")


def expect_success(result: Result[T, E]) -> T:
    """Unwrap Success or fail test with error message.

    This replaces the _expect_success pattern duplicated across test files.
    Use this when you expect a Result to be Success and want to unwrap the
    value, failing the test with a clear message if it's actually a Failure.

    Args:
        result: Result to unwrap

    Returns:
        The success value

    Raises:
        AssertionError: If result is Failure with the error message

    Example:
        >>> from spectralmc.gbm import build_simulation_params
        >>> from tests.helpers import expect_success
        >>>
        >>> params = expect_success(build_simulation_params(...))
        >>> # If build_simulation_params returns Failure, test fails immediately
    """
    match result:
        case Success(value):
            return value
        case Failure(error):
            raise AssertionError(f"Unexpected failure: {error}")


def expect_failure(result: Result[T, E]) -> E:
    """Unwrap Failure or fail test.

    Use this when you expect a Result to be Failure and want to unwrap the
    error value, failing the test if it's actually a Success.

    Args:
        result: Result to unwrap

    Returns:
        The failure error

    Raises:
        AssertionError: If result is Success

    Example:
        >>> from tests.helpers import expect_failure
        >>>
        >>> error = expect_failure(some_operation_that_should_fail())
        >>> assert "invalid" in str(error).lower()
    """
    match result:
        case Failure(error):
            return error
        case Success(value):
            raise AssertionError(f"Expected failure but got success: {value}")
