"""Strict subset of Numba CUDA API used by spectralmc (Any‑free)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# decorators ------------------------------------------------------------------

def jit(func: Callable[P, R], /) -> Callable[P, R]: ...

# helpers ---------------------------------------------------------------------

def stream() -> "Stream": ...
def as_cuda_array(x: ArrayLike, /) -> NDArray[np.generic]: ...
def grid(ndim: int) -> int: ...  # we use only 1‑D grids in this code‑base

# classes ---------------------------------------------------------------------

class Stream:  # bare‑minimum wrapper
    def synchronize(self) -> None: ...
