"""
/spectralmc/python/stubs/cupy/__init__.pyi

Partial type hints for CuPy, overshadowing any official stubs so Mypy picks ours.
We have specialized & fallback overloads for random.Generator.normal(...).

No final implementation line; that would cause "implementation not allowed" in stubs.
"""

from __future__ import annotations

from typing import Any, Tuple, overload, Type
import numpy
from spectralmc import GPUArray as _GPUArray

class ndarray(_GPUArray):
    """CuPy array type fulfilling the GPUArray protocol."""

    shape: Tuple[int, ...]
    dtype: numpy.dtype[Any]
    def __init__(self, shape: Tuple[int, ...], dtype: Any = ...) -> None: ...
    def get(self) -> numpy.ndarray[Any, Any]: ...
    def astype(self, dtype: Any, *, copy: bool = True) -> ndarray: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...

class random:
    class Generator:
        @overload
        def normal(
            self,
            loc: float | int,
            scale: float | int,
            size: tuple[int, int],
            dtype: Type[numpy.float32],
        ) -> ndarray:
            """
            Specialized overload for usage: normal(0.0, dt_sqrt, (N-1, paths), cp.float32)
            => returns a float32 array matching your user call in sde.py
            """
            ...

        @overload
        def normal(
            self,
            loc: float | int = ...,
            scale: float | int = ...,
            size: tuple[int, ...] | None = ...,
            dtype: Any = ...,
        ) -> ndarray:
            """
            Fallback overload for all other combos => returns generic ndarray
            """
            ...
        # No final "def normal(...): ..." line—stub files cannot have real implementations.

        def standard_normal(
            self, size: tuple[int, ...] | None = None, dtype: Any = ...
        ) -> ndarray:
            """No single-float return overload. Always returns an ndarray."""
            ...

    @staticmethod
    def default_rng(seed: int = 0) -> random.Generator: ...

class cuda:
    @staticmethod
    def stream() -> Any: ...
    @staticmethod
    def grid(dim: int) -> int: ...
    @staticmethod
    def as_cuda_array(obj: Any, stream: Any = ...) -> ndarray: ...

def zeros(shape: Tuple[int, ...], dtype: Any = ...) -> ndarray: ...
def array(obj: Any, dtype: Any = ..., copy: bool = True) -> ndarray: ...
