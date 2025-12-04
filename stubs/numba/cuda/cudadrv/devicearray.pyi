from __future__ import annotations

from typing import Protocol, Tuple


class DeviceNDArray(Protocol):
    """Protocol for Numba CUDA device arrays with minimal interface."""

    @property
    def shape(self) -> Tuple[int, ...]: ...
    def __getitem__(self, key: object) -> float: ...
    def __setitem__(self, key: object, value: float) -> None: ...
