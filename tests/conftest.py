# tests/conftest.py
"""Global PyTest fixtures for the test-suite.

CuPy is imported unconditionally—if it is not installed the test session
will fail immediately, making the missing dependency obvious.
"""

from __future__ import annotations

import gc
import warnings
from typing import Generator

import cupy as cp  # type: ignore[import-untyped]
import pytest
import torch


def _free_cupy() -> None:
    """Release CuPy memory pools.

    Any exception is turned into a *RuntimeWarning* so users see the problem
    instead of silently proceeding.
    """
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"CuPy memory-pool cleanup failed: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )


@pytest.fixture(autouse=True)
def cleanup_gpu() -> Generator[None, None, None]:
    """Auto-fixture that frees GPU memory after *every* test."""
    yield
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"torch.cuda.empty_cache() failed: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )

    _free_cupy()