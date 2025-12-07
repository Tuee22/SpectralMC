"""Error ADTs used by the torch façade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar

from spectralmc.result import Result


@dataclass(frozen=True)
class UnsupportedTorchDType:
    """torch.dtype cannot be represented in SpectralMC."""

    dtype: str
    kind: Literal["UnsupportedTorchDType"] = "UnsupportedTorchDType"


@dataclass(frozen=True)
class UnsupportedTorchDevice:
    """Device is unsupported (non-CPU/CUDA0)."""

    device: str
    kind: Literal["UnsupportedTorchDevice"] = "UnsupportedTorchDevice"


@dataclass(frozen=True)
class TensorStateConversionFailed:
    """SafeTensor serialization/deserialization failed."""

    message: str
    kind: Literal["TensorStateConversionFailed"] = "TensorStateConversionFailed"


@dataclass(frozen=True)
class InvalidAdamState:
    """Optimizer state dict was malformed."""

    message: str
    kind: Literal["InvalidAdamState"] = "InvalidAdamState"


TorchFacadeError = (
    UnsupportedTorchDType | UnsupportedTorchDevice | TensorStateConversionFailed | InvalidAdamState
)

T = TypeVar("T")
TorchFacadeResult = Result[T, TorchFacadeError]

__all__ = [
    "InvalidAdamState",
    "TensorStateConversionFailed",
    "TorchFacadeError",
    "TorchFacadeResult",
    "UnsupportedTorchDevice",
    "UnsupportedTorchDType",
]
