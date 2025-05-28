"""Minimal CuPy stubs that satisfy spectralmc & its test-suite.

* No `type: ignore` comments.
* Everything lives in a single declaration block, so classes are defined once.
"""

from __future__ import annotations
from typing import Any, Tuple
from numpy.typing import NDArray

# ── dtypes ────────────────────────────────────────────────────────────────
class dtype:
    itemsize: int
    def __init__(self, obj: object, align: bool | None = ...) -> None: ...

float32: dtype
float64: dtype
complex64: dtype
complex128: dtype

# ── core ndarray ──────────────────────────────────────────────────────────
class ndarray(NDArray[Any]):
    # Arithmetic subset we touch
    def __mul__(self, other: object) -> "ndarray": ...
    def __imul__(self, other: object) -> "ndarray": ...

    # Needed by gbm_trainer.py for zero-copy transfer
    def toDlpack(self) -> Any: ...

# ── array helpers used in code/tests ──────────────────────────────────────
def zeros(
    shape: Any, dtype: dtype | None = ..., order: str | None = ...
) -> ndarray: ...
def linspace(
    start: float, stop: float, num: int = ..., *, dtype: dtype | None = ...
) -> ndarray: ...
def exp(x: Any, /) -> ndarray: ...
def mean(a: Any, *, axis: Any = ..., keepdims: bool = ...) -> ndarray: ...
def maximum(a: Any, b: Any, /) -> ndarray: ...
def asarray(x: Any, *, dtype: dtype | None = ...) -> ndarray: ...
def expand_dims(a: Any, axis: int) -> ndarray: ...

# numpy-compat helper required by tests
def allclose(
    a: Any,
    b: Any,
    *,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> bool: ...

# ── random namespace ──────────────────────────────────────────────────────
class _Generator:
    def standard_normal(
        self, shape: Tuple[int, ...], *, dtype: dtype | None = ...
    ) -> ndarray: ...

class _RandomNS:
    def default_rng(self, seed: int) -> _Generator: ...

random: _RandomNS

# ── cp.fft.fft helper ─────────────────────────────────────────────────────
class _FFTModule:
    def fft(self, a: Any, axis: int = ..., n: int | None = ...) -> ndarray: ...

fft: _FFTModule

# ── cuda sub-module stubs ─────────────────────────────────────────────────
class Stream:
    null: "Stream"
    def __init__(self, non_blocking: bool | None = ...) -> None: ...
    def synchronize(self) -> None: ...
    def __enter__(self) -> "Stream": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None: ...

class Event:
    ptr: int
    def __init__(self, disable_timing: bool = ...) -> None: ...
    def record(self) -> None: ...

class runtime:
    @staticmethod
    def eventQuery(ptr: int) -> int: ...

class Device:
    def __init__(self, id: int = ...) -> None: ...
    def synchronize(self) -> None: ...

# memory-pool API used in tests/conftest.py
class _MemoryPool:
    def free_all_blocks(self) -> None: ...

def get_default_memory_pool() -> _MemoryPool: ...
def get_default_pinned_memory_pool() -> _MemoryPool: ...

# Re-export cuda as a sub-module (cp.cuda)
import types as _t

# re-export sub-module so mypy finds cp.cuda
from . import cuda as cuda
