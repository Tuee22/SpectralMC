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


@dataclass(frozen=True)
class EmptyTensorTree:
    """TensorTree contains no tensors."""

    kind: Literal["EmptyTensorTree"] = "EmptyTensorTree"


@dataclass(frozen=True)
class HeterogeneousTensorTree:
    """TensorTree contains tensors with different device/dtype."""

    kind: Literal["HeterogeneousTensorTree"] = "HeterogeneousTensorTree"


@dataclass(frozen=True)
class NoOpTransfer:
    """Attempted to transfer tensor to its current device."""

    device: str
    kind: Literal["NoOpTransfer"] = "NoOpTransfer"


@dataclass(frozen=True)
class CudaUnavailable:
    """CUDA destination requested but CUDA is not available."""

    kind: Literal["CudaUnavailable"] = "CudaUnavailable"


@dataclass(frozen=True)
class TransferRejected:
    """Transfer planner rejected the request (e.g., unsupported path)."""

    reason: str
    kind: Literal["TransferRejected"] = "TransferRejected"


TorchFacadeError = (
    UnsupportedTorchDType
    | UnsupportedTorchDevice
    | TensorStateConversionFailed
    | InvalidAdamState
    | EmptyTensorTree
    | HeterogeneousTensorTree
    | NoOpTransfer
    | CudaUnavailable
    | TransferRejected
)

T = TypeVar("T")
TorchFacadeResult = Result[T, TorchFacadeError]

__all__ = [
    "CudaUnavailable",
    "EmptyTensorTree",
    "HeterogeneousTensorTree",
    "InvalidAdamState",
    "NoOpTransfer",
    "TensorStateConversionFailed",
    "TorchFacadeError",
    "TorchFacadeResult",
    "TransferRejected",
    "UnsupportedTorchDevice",
    "UnsupportedTorchDType",
]
