"""
`spectralmc.models.numerical`
--------------------------------
A single, strictly-typed source of truth for the numeric *scalar* formats
(`float32`, `float64`, `complex64`, `complex128`) that *SpectralMC* supports.

The :class:`Precision` enum exposes loss-free conversions to / from both
NumPy and CuPy **without** any conditional branching at runtime.  All lookup
operations are *O(1)* dict accesses, so every helper is a pure function with
zero side-effects.

Why do we appear to map *both* ``np.float32`` **and** ``np.dtype(np.float32)``
(or their CuPy equivalents)?
    * The first is the *scalar class* (the type you get from ``type(x)``)
    * The second is the *dtype object* used when constructing arrays

They are distinct objects and users routinely pass *either* form.  We accept
this micro-duplication (two extra dict entries per dtype) to guarantee
constant-time look-ups **without** coercions like ``np.dtype(x)`` or
``isinstance`` checks.  The overhead is negligible (<100 B per table) while
preserving functional purity and mypy --strict compatibility.

Public helpers
--------------
* ``Precision.to_numpy()        -> numpy.dtype``
* ``Precision.from_numpy(...)   -> Precision``
* ``Precision.to_cupy()         -> cupy.dtype``
* ``Precision.from_cupy(...)    -> Precision``
* ``Precision.to_complex()      -> Precision``
* ``Precision.from_complex(p)   -> Precision``  (where *p* **is already** a
  :class:`Precision`)
"""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias, Union

import cupy as cp
import numpy as np


__all__ = ["Precision"]

# --------------------------------------------------------------------------- #
# Typing aliases
# --------------------------------------------------------------------------- #
_NPDTypeF32: TypeAlias = np.dtype[np.float32]
_NPDTypeF64: TypeAlias = np.dtype[np.float64]
_NPDTypeC64: TypeAlias = np.dtype[np.complex64]
_NPDTypeC128: TypeAlias = np.dtype[np.complex128]
_NPDTypeRet: TypeAlias = Union[_NPDTypeF32, _NPDTypeF64, _NPDTypeC64, _NPDTypeC128]

_NPDTypeLike: TypeAlias = Union[
    type[np.float32],
    type[np.float64],
    type[np.complex64],
    type[np.complex128],
    _NPDTypeF32,
    _NPDTypeF64,
    _NPDTypeC64,
    _NPDTypeC128,
]

# CuPy stub types do not expose scalar classes as valid annotation targets,
# so we fall back to NumPy scalar types plus `cp.dtype`.
_CPDTypeRet: TypeAlias = cp.dtype
_CPDTypeLike: TypeAlias = Union[
    cp.dtype,
    type[np.float32],
    type[np.float64],
    type[np.complex64],
    type[np.complex128],
]

# --------------------------------------------------------------------------- #
# Mapping tables - generated *functionally* to avoid repetition
# --------------------------------------------------------------------------- #
_PRECISION_NAMES: tuple[str, ...] = (
    "float32",
    "float64",
    "complex64",
    "complex128",
)

# str  -> numpy.dtype
_PRECISION_STR_TO_NP: dict[str, _NPDTypeRet] = {
    name: np.dtype(getattr(np, name)) for name in _PRECISION_NAMES
}

# numpy scalar *or* numpy.dtype  -> str
_NP_TO_PRECISION_STR: dict[_NPDTypeLike, str] = {
    obj: name
    for name in _PRECISION_NAMES
    for obj in (getattr(np, name), np.dtype(getattr(np, name)))
}

# str  -> cupy.dtype
_PRECISION_STR_TO_CP: dict[str, _CPDTypeRet] = {
    name: cp.dtype(getattr(cp, name)) for name in _PRECISION_NAMES
}

# cupy scalar *or* cupy.dtype *or* numpy scalar  -> str
_CP_TO_PRECISION_STR: dict[_CPDTypeLike, str] = {
    obj: name
    for name in _PRECISION_NAMES
    for obj in (getattr(cp, name), cp.dtype(getattr(cp, name)), getattr(np, name))
}

# float → complex bijection and its programmatic inverse
_PRECISION_TO_COMPLEX_STR: dict[str, str] = {
    "float32": "complex64",
    "float64": "complex128",
}
_COMPLEX_TO_FLOAT_STR: dict[str, str] = {v: k for k, v in _PRECISION_TO_COMPLEX_STR.items()}


# --------------------------------------------------------------------------- #
# Enum definition
# --------------------------------------------------------------------------- #
class Precision(str, Enum):
    """Scalar floating / complex-floating formats supported by SpectralMC."""

    float32 = "float32"
    float64 = "float64"
    complex64 = "complex64"
    complex128 = "complex128"

    # ------------------------------------------------------------------ #
    # NumPy helpers
    # ------------------------------------------------------------------ #
    def to_numpy(self) -> _NPDTypeRet:
        """Return the corresponding ``numpy.dtype``."""
        return _PRECISION_STR_TO_NP[self.value]

    @classmethod
    def from_numpy(cls, dtype: _NPDTypeLike) -> Precision:
        """Map a NumPy *dtype* or scalar class back to :class:`Precision`."""
        try:
            return cls(_NP_TO_PRECISION_STR[dtype])
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unsupported NumPy dtype: {dtype!r}") from exc

    # ------------------------------------------------------------------ #
    # CuPy helpers
    # ------------------------------------------------------------------ #
    def to_cupy(self) -> _CPDTypeRet:
        """Return the corresponding ``cupy.dtype``."""
        return _PRECISION_STR_TO_CP[self.value]

    @classmethod
    def from_cupy(cls, dtype: _CPDTypeLike) -> Precision:
        """Map a CuPy *dtype* or scalar class back to :class:`Precision`."""
        try:
            return cls(_CP_TO_PRECISION_STR[dtype])
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unsupported CuPy dtype: {dtype!r}") from exc

    # ------------------------------------------------------------------ #
    # Complex helpers (pure, branch-free)
    # ------------------------------------------------------------------ #
    def to_complex(self) -> Precision:
        """Return the complex counterpart (idempotent for complex inputs)."""
        return Precision(_PRECISION_TO_COMPLEX_STR[self.value])

    @classmethod
    def from_complex(cls, prec: Precision) -> Precision:
        """Return the *real* precision behind a complex format (idempotent)."""
        return cls(_COMPLEX_TO_FLOAT_STR[prec.value])
