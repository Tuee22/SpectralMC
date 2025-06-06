"""Strict, minimal CuPy subset for spectralmc (non‑generic ``ndarray`` to satisfy
``mypy --strict`` without forcing callers to write type parameters)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike
from typing import Tuple

# ---------------------------------------------------------------------------
# dtype – trivial wrapper so we can construct cp.dtype("float32") in code
# ---------------------------------------------------------------------------
class dtype:  # noqa: D101
    itemsize: int
    def __init__(self, obj: object, align: bool | None = ...) -> None: ...

float32: dtype
float64: dtype
complex64: dtype
complex128: dtype

# ---------------------------------------------------------------------------
# ndarray – we treat every CuPy array as ``NDArray[Any]`` so using plain
# ``cp.ndarray`` in type annotations doesn’t raise “missing type parameters”.
# ---------------------------------------------------------------------------
class ndarray(NDArray[np.generic]):
    # minimal arithmetic we rely on --------------------------------------
    def __mul__(self, other: ArrayLike | "ndarray") -> "ndarray": ...
    def __rmul__(self, other: ArrayLike | "ndarray") -> "ndarray": ...
    def __imul__(self, other: ArrayLike | "ndarray") -> "ndarray": ...
    def __truediv__(self, other: ArrayLike | "ndarray") -> "ndarray": ...
    def __rtruediv__(self, other: ArrayLike | "ndarray") -> "ndarray": ...
    def __itruediv__(self, other: ArrayLike | "ndarray") -> "ndarray": ...

    # tiny helper needed by spectralmc -----------------------------------
    def toDlpack(self) -> object: ...

# ---------------------------------------------------------------------------
# Functional helpers referenced in code/tests
# ---------------------------------------------------------------------------

def zeros(
    shape: int | Tuple[int, ...],
    dtype: dtype | DTypeLike | None = ...,
    order: str | None = ...,
) -> ndarray: ...
def linspace(
    start: float, stop: float, num: int = ..., *, dtype: dtype | DTypeLike | None = ...
) -> ndarray: ...
def exp(x: ArrayLike, /) -> ndarray: ...
def mean(
    a: ArrayLike, *, axis: int | Tuple[int, ...] | None = ..., keepdims: bool = ...
) -> ndarray: ...
def maximum(a: ArrayLike, b: ArrayLike, /) -> ndarray: ...
def asarray(x: ArrayLike, *, dtype: dtype | DTypeLike | None = ...) -> ndarray: ...
def expand_dims(a: ArrayLike, axis: int) -> ndarray: ...
def allclose(
    a: ArrayLike,
    b: ArrayLike,
    *,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> bool: ...

# ---------------------------------------------------------------------------
# Random namespace -----------------------------------------------------------
class _Generator:
    def standard_normal(
        self, shape: Tuple[int, ...], *, dtype: dtype | DTypeLike | None = ...
    ) -> ndarray: ...

class _RandomNS:
    def default_rng(self, seed: int) -> _Generator: ...

random: _RandomNS

# ---------------------------------------------------------------------------
# FFT helper -----------------------------------------------------------------
class _FFTModule:
    def fft(self, a: ArrayLike, axis: int = ..., n: int | None = ...) -> ndarray: ...

fft: _FFTModule

# ---------------------------------------------------------------------------
# CUDA sub‑module re-export + memory‑pool helpers expected at top level
# ---------------------------------------------------------------------------
from .cuda import _MemoryPool

def get_default_memory_pool() -> _MemoryPool: ...
def get_default_pinned_memory_pool() -> _MemoryPool: ...
