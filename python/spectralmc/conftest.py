from __future__ import annotations

import gc
from typing import Generator, TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    # Gives mypy a name for type hints without importing CuPy at runtime
    import cupy as cp  # type: ignore[import-untyped]


def _free_cupy() -> None:
    """Release CuPy memory pools if CuPy is available at runtime."""
    try:
        import cupy as cp  # pylint: disable=import-error
    except ModuleNotFoundError:
        return
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@pytest.fixture(autouse=True)
def cleanup_gpu() -> Generator[None, None, None]:
    """Auto‑fixture: free GPU memory after each test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _free_cupy()
