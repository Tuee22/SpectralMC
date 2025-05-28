"""Slim CuPy stubs for spectralmc (generated 2025-05-28)."""

from __future__ import annotations
from typing import Any, Tuple
from numpy.typing import NDArray

# ── dtypes ──────────────────────────────────────────────────────────────────
class dtype:
    itemsize: int
    def __init__(self, obj: object, align: bool | None = ...) -> None: ...

float32: dtype
float64: dtype

# ── ndarray ────────────────────────────────────────────────────────────────
class ndarray(NDArray[Any]):
    def __mul__(self, other: object) -> "ndarray": ...
    def __imul__(self, other: object) -> "ndarray": ...

# ── random namespace ───────────────────────────────────────────────────────
class _Generator:
    def standard_normal(
        self, shape: Tuple[int, ...], *, dtype: dtype | None = ...
    ) -> ndarray: ...

class _RandomNS:
    def default_rng(self, seed: int) -> _Generator: ...

random: _RandomNS

# ── array helpers ──────────────────────────────────────────────────────────
def linspace(
    start: float, stop: float, num: int = ..., *, dtype: dtype | None = ...
) -> ndarray: ...
def exp(x: object, /) -> ndarray: ...
def mean(
    a: Any, *, axis: int | Tuple[int, ...] | None = ..., keepdims: bool = ...
) -> ndarray: ...
def maximum(a: object, b: object, /) -> ndarray: ...
def asarray(obj: object, *, dtype: dtype | None = ...) -> ndarray: ...
def expand_dims(a: Any, axis: int) -> ndarray: ...

newaxis: object

# ── cuda re-export ─────────────────────────────────────────────────────────
from . import cuda as cuda

Stream = cuda.Stream
Event = cuda.Event

__all__ = [
    "dtype",
    "float32",
    "float64",
    "ndarray",
    "random",
    "linspace",
    "exp",
    "mean",
    "maximum",
    "asarray",
    "expand_dims",
    "newaxis",
    "cuda",
    "Stream",
    "Event",
]

# ------------------------------------------------------------------
# >>> ADDED BY patch_async_normals.sh (async_normals support) <<<
# NumPy-compatible helpers required by the test-suite
# ------------------------------------------------------------------
from typing import Any
import numpy as np

def allclose(
    a: Any,
    b: Any,
    *,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> bool: ...

# ------------------------------------------------------------------
# Re-export the cuda sub-module so attribute access ``cp.cuda`` works
# ------------------------------------------------------------------
from . import cuda as cuda  # noqa: E402

# ------------------------------------------------------------------
# <<< END OF PATCH
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# >>> ADDED BY fix_mypy.sh (async_normals test support) <<<
# Minimal memory-pool API required by tests/conftest.py
# ------------------------------------------------------------------
class _MemoryPool:
    def free_all_blocks(self) -> None: ...

def get_default_memory_pool() -> _MemoryPool: ...
def get_default_pinned_memory_pool() -> _MemoryPool: ...

# ------------------------------------------------------------------
# <<< END OF PATCH
# ------------------------------------------------------------------
