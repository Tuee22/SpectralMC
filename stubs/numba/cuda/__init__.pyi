"""Strict subset of Numba CUDA API used by spectralmc (Any‑free)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, ParamSpec, TypeVar, Protocol, Tuple

P = ParamSpec("P")
R = TypeVar("R")
P_kernel = ParamSpec("P_kernel")

# CUDA device array protocol
class DeviceNDArray(Protocol):
    """Protocol for Numba CUDA device arrays with minimal required interface."""

    @property
    def shape(self) -> Tuple[int, ...]: ...
    def __getitem__(self, key: object) -> float: ...
    def __setitem__(self, key: object, value: float) -> None: ...

# Kernel-launch protocol returned by .jit
class _CUDALaunchable(Protocol[P_kernel]):
    def __getitem__(
        self, launch_cfg: tuple[int, int, "Stream"]
    ) -> Callable[P_kernel, None]: ...

# decorators ------------------------------------------------------------------

def jit(func: Callable[P, R], /) -> _CUDALaunchable[P]: ...

# helpers ---------------------------------------------------------------------

def stream() -> "Stream": ...
def as_cuda_array(x: object, /) -> DeviceNDArray: ...
def grid(ndim: int) -> int: ...  # we use only 1‑D grids in this code‑base

# classes ---------------------------------------------------------------------

class Stream:  # bare‑minimum wrapper
    def synchronize(self) -> None: ...
