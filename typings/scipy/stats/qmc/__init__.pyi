"""Minimal stubs for scipy.stats.qmc.Sobol (just enough for SobolSampler)."""

from __future__ import annotations
from numpy.typing import NDArray
from typing import Any

class Sobol:
    def __init__(
        self, d: int, *, scramble: bool = ..., seed: int | None = ...
    ) -> None: ...
    def random(self, n: int) -> NDArray[Any]: ...
    def fast_forward(self, n: int) -> None: ...

__all__ = ["Sobol"]
