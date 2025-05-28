"""
Minimal CuPy stubs needed by spectralmc.async_normals.
Last generated: 2025-05-28
"""

from __future__ import annotations
from typing import Any, Tuple
from numpy.typing import NDArray

# ── fundamental types ──────────────────────────────────────────────────────
class dtype:
    def __init__(self, obj: object, align: bool | None = ...) -> None: ...

float32: dtype
float64: dtype

class ndarray(NDArray[Any]): ...

# allow `cp.ndarray` annotations

# ── random namespace ───────────────────────────────────────────────────────
class Generator:
    def standard_normal(
        self,
        shape: Tuple[int, ...],
        *,
        dtype: dtype | None = ...,
    ) -> ndarray: ...

class _RandomNS:
    def default_rng(self, seed: int) -> Generator: ...

# numerical helper

def allclose(
    a: ndarray,
    b: ndarray,
    *,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> bool: ...

random: _RandomNS  # instance, not class

# ── cuda sub-module re-export ───────────────────────────────────────────────
from . import cuda as cuda

Stream = cuda.Stream
Event = cuda.Event

__all__ = [
    "dtype",
    "float32",
    "float64",
    "ndarray",
    "Generator",
    "random",
    "cuda",
    "Stream",
    "Event",
]
