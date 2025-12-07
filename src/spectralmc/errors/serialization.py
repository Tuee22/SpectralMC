"""ADTs for serialization failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import ValidationError

from spectralmc.result import Result


@dataclass(frozen=True)
class UnsupportedPrecision:
    """Precision value is not one of the supported enums."""

    proto_value: int
    kind: Literal["UnsupportedPrecision"] = "UnsupportedPrecision"


@dataclass(frozen=True)
class UnknownThreadsPerBlock:
    """Proto specifies an invalid CUDA thread block size."""

    value: int
    kind: Literal["UnknownThreadsPerBlock"] = "UnknownThreadsPerBlock"


@dataclass(frozen=True)
class InvalidWidthSpecProto:
    """WidthSpec proto has an unknown variant."""

    variant: str | None
    kind: Literal["InvalidWidthSpecProto"] = "InvalidWidthSpecProto"


@dataclass(frozen=True)
class UnknownActivationKind:
    """Activation proto contains an unsupported kind."""

    value: int
    kind: Literal["UnknownActivationKind"] = "UnknownActivationKind"


@dataclass(frozen=True)
class UnknownDType:
    """Tensor/protobuf dtype is not recognised."""

    proto_value: int
    kind: Literal["UnknownDType"] = "UnknownDType"


@dataclass(frozen=True)
class InvalidTensorState:
    """Tensor state proto cannot be reconstructed as a torch.Tensor."""

    message: str
    kind: Literal["InvalidTensorState"] = "InvalidTensorState"


@dataclass(frozen=True)
class ValidationFailed:
    """Pydantic validation failed while building a model from proto."""

    error: ValidationError
    kind: Literal["ValidationFailed"] = "ValidationFailed"


@dataclass(frozen=True)
class TensorProtoMismatch:
    """Protobuf message does not match the expected shape or dtype semantics."""

    message: str
    kind: Literal["TensorProtoMismatch"] = "TensorProtoMismatch"


SerializationError = (
    UnsupportedPrecision
    | UnknownThreadsPerBlock
    | InvalidWidthSpecProto
    | UnknownActivationKind
    | UnknownDType
    | InvalidTensorState
    | TensorProtoMismatch
    | ValidationFailed
)

T = TypeVar("T")
SerializationResult = Result[T, SerializationError]

__all__ = [
    "InvalidTensorState",
    "SerializationError",
    "SerializationResult",
    "TensorProtoMismatch",
    "UnknownActivationKind",
    "UnknownDType",
    "UnsupportedPrecision",
    "UnknownThreadsPerBlock",
    "InvalidWidthSpecProto",
    "ValidationFailed",
]
