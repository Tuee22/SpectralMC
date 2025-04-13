"""
/spectralmc/python/spectralmc/__init__.py

Root package init for SpectralMC library.
Exposes cp (CuPy alias) and GPUArray protocol for type checking.

No placeholders or ignoring imports. Must pass mypy --strict.
Imports at top. from __future__ import annotations so no quotes needed.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import numpy as cp
else:
    import cupy as cp

__all__ = ["cp", "GPUArray"]

from typing import Protocol, Any, Tuple


class GPUArray(Protocol):
    """
    Minimal interface for GPU arrays. shape, dtype, get().
    """

    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...
    def get(self) -> cp.ndarray[Any, Any]: ...
