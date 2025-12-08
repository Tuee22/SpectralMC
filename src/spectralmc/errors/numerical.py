"""ADTs for numerical conversion failures."""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "UnsupportedNumPyDType",
    "UnsupportedCuPyDType",
    "InvalidComplexConversion",
    "NumericalError",
]


@dataclass(frozen=True)
class UnsupportedNumPyDType:
    """NumPy dtype is not supported by SpectralMC."""

    dtype_repr: str
    kind: Literal["UnsupportedNumPyDType"] = "UnsupportedNumPyDType"


@dataclass(frozen=True)
class UnsupportedCuPyDType:
    """CuPy dtype is not supported by SpectralMC."""

    dtype_repr: str
    kind: Literal["UnsupportedCuPyDType"] = "UnsupportedCuPyDType"


@dataclass(frozen=True)
class InvalidComplexConversion:
    """Precision cannot be converted to/from complex."""

    precision: str
    kind: Literal["InvalidComplexConversion"] = "InvalidComplexConversion"


NumericalError = UnsupportedNumPyDType | UnsupportedCuPyDType | InvalidComplexConversion
