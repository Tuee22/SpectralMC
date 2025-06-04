"""Strict subset of Numba CUDA API used by spectralmc (Any‑free)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Callable, ParamSpec, TypeVar, Protocol

P = ParamSpec("P")
R = TypeVar("R")

# Kernel-launch protocol returned by .jit
class _CUDALaunchable(Protocol):
    def __getitem__(
        self, launch_cfg: tuple[int, int, "Stream"]
    ) -> Callable[..., None]: ...

# decorators ------------------------------------------------------------------

def jit(func: Callable[P, R], /) -> _CUDALaunchable: ...

# helpers ---------------------------------------------------------------------

def stream() -> "Stream": ...
def as_cuda_array(x: ArrayLike, /) -> NDArray[np.generic]: ...
def grid(ndim: int) -> int: ...  # we use only 1‑D grids in this code‑base

# classes ---------------------------------------------------------------------

class Stream:  # bare‑minimum wrapper
    def synchronize(self) -> None: ...
