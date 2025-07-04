# src/spectralmc/models/numerical.py
from __future__ import annotations

"""
`spectralmc.models.numerical`

`Precision` enumerates the high-precision floating-point formats we support
(`float32`, `float64`) and provides strictly-typed, map-based helpers for
NumPy and CuPy.  The module passes **mypy --strict** with zero ignores,
casts, or Anys.

Public helpers
--------------
* `Precision.to_numpy()  -> numpy.dtype`
* `Precision.from_numpy(dtype) -> Precision`
* `Precision.to_cupy()   -> cupy.dtype`
* `Precision.from_cupy(dtype)  -> Precision`
"""

from enum import Enum
from typing import Type, TypeAlias, Union

import numpy as np
import cupy as cp

__all__ = ["Precision"]

# --------------------------------------------------------------------------- #
# NumPy typing aliases
# --------------------------------------------------------------------------- #
_NPDTypeF32: TypeAlias = np.dtype[np.float32]
_NPDTypeF64: TypeAlias = np.dtype[np.float64]
_NPDTypeRet: TypeAlias = Union[_NPDTypeF32, _NPDTypeF64]

_NPDTypeLike: TypeAlias = Union[
    Type[np.float32],
    Type[np.float64],
    _NPDTypeF32,
    _NPDTypeF64,
]

# --------------------------------------------------------------------------- #
# CuPy typing aliases
# --------------------------------------------------------------------------- #
# Cupy stubs do not expose `cp.float32` / `cp.float64` as usable *types* in
# annotations, so we restrict the type alias to `cp.dtype` plus the NumPy
# scalar types (which `cp.float32` & `cp.float64` alias at runtime).
_CPDTypeRet: TypeAlias = cp.dtype
_CPDTypeLike: TypeAlias = Union[
    cp.dtype,
    Type[np.float32],
    Type[np.float64],
]

# --------------------------------------------------------------------------- #
# Mapping tables (functional style, no conditional branching)
# --------------------------------------------------------------------------- #
_PRECISION_STR_TO_NP: dict[str, _NPDTypeRet] = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}

_NP_TO_PRECISION_STR: dict[_NPDTypeLike, str] = {
    np.float32: "float32",
    np.dtype(np.float32): "float32",
    np.float64: "float64",
    np.dtype(np.float64): "float64",
}

_PRECISION_STR_TO_CP: dict[str, _CPDTypeRet] = {
    "float32": cp.dtype(cp.float32),
    "float64": cp.dtype(cp.float64),
}

_CP_TO_PRECISION_STR: dict[_CPDTypeLike, str] = {
    cp.float32: "float32",
    cp.dtype(cp.float32): "float32",
    cp.float64: "float64",
    cp.dtype(cp.float64): "float64",
    # NumPy scalar classes are aliases for the CuPy ones at runtime,
    # so include them for completeness.
    np.float32: "float32",
    np.float64: "float64",
}


class Precision(str, Enum):
    """High-precision numeric dtypes."""

    float32 = "float32"
    float64 = "float64"

    # ------------------------------------------------------------------ #
    # NumPy helpers
    # ------------------------------------------------------------------ #
    def to_numpy(self) -> _NPDTypeRet:
        """Return the equivalent `numpy.dtype` (float32 or float64)."""
        return _PRECISION_STR_TO_NP[self.value]

    @classmethod
    def from_numpy(cls, dtype: _NPDTypeLike) -> "Precision":
        """
        Convert a NumPy dtype or scalar type to :class:`Precision`.

        Raises
        ------
        ValueError
            If *dtype* is not one of the supported formats.
        """
        try:
            return cls(_NP_TO_PRECISION_STR[dtype])
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unsupported NumPy dtype: {dtype!r}") from exc

    # ------------------------------------------------------------------ #
    # CuPy helpers
    # ------------------------------------------------------------------ #
    def to_cupy(self) -> _CPDTypeRet:
        """Return the equivalent `cupy.dtype` (float32 or float64)."""
        return _PRECISION_STR_TO_CP[self.value]

    @classmethod
    def from_cupy(cls, dtype: _CPDTypeLike) -> "Precision":
        """
        Convert a CuPy dtype or scalar type to :class:`Precision`.

        Raises
        ------
        ValueError
            If *dtype* is not one of the supported formats.
        """
        try:
            return cls(_CP_TO_PRECISION_STR[dtype])
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unsupported CuPy dtype: {dtype!r}") from exc
