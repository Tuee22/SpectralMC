# src/spectralmc/serialization/common.py
"""Converters for common enums and basic types."""

from __future__ import annotations

from spectralmc.errors.serialization import SerializationResult, UnknownDType, UnsupportedPrecision
from spectralmc.models.numerical import Precision
from spectralmc.models.torch import (
    AnyDType,
    Device,
    FullPrecisionDType,
    ReducedPrecisionDType,
)
from spectralmc.proto import common_pb2
from spectralmc.result import Failure, Success


# Unified dtype mapping (avoids if/elif chains)
_DTYPE_TO_PROTO: dict[AnyDType, int] = {
    FullPrecisionDType.float32: common_pb2.DTYPE_FLOAT32,
    FullPrecisionDType.float64: common_pb2.DTYPE_FLOAT64,
    FullPrecisionDType.complex64: common_pb2.DTYPE_COMPLEX64,
    FullPrecisionDType.complex128: common_pb2.DTYPE_COMPLEX128,
    ReducedPrecisionDType.float16: common_pb2.DTYPE_FLOAT16,
    ReducedPrecisionDType.bfloat16: common_pb2.DTYPE_BFLOAT16,
}

_PROTO_TO_DTYPE: dict[int, AnyDType] = {
    common_pb2.DTYPE_FLOAT32: FullPrecisionDType.float32,
    common_pb2.DTYPE_FLOAT64: FullPrecisionDType.float64,
    common_pb2.DTYPE_COMPLEX64: FullPrecisionDType.complex64,
    common_pb2.DTYPE_COMPLEX128: FullPrecisionDType.complex128,
    common_pb2.DTYPE_FLOAT16: ReducedPrecisionDType.float16,
    common_pb2.DTYPE_BFLOAT16: ReducedPrecisionDType.bfloat16,
}


class PrecisionConverter:
    """Convert between Pydantic Precision and Protobuf PrecisionProto."""

    @staticmethod
    def to_proto(precision: Precision) -> int:
        """Convert Precision enum to proto enum value."""
        mapping = {
            Precision.float32: common_pb2.PRECISION_FLOAT32,
            Precision.float64: common_pb2.PRECISION_FLOAT64,
        }
        return mapping[precision]

    @staticmethod
    def from_proto(proto_value: int) -> SerializationResult[Precision]:
        """Convert proto enum value to Precision enum."""
        mapping = {
            common_pb2.PRECISION_FLOAT32: Precision.float32,
            common_pb2.PRECISION_FLOAT64: Precision.float64,
        }
        match mapping.get(proto_value):
            case None:
                return Failure(UnsupportedPrecision(proto_value=proto_value))
            case precision:
                return Success(precision)


class DeviceConverter:
    """Convert between Pydantic Device and Protobuf DeviceProto."""

    @staticmethod
    def to_proto(device: Device) -> int:
        """Convert Device enum to proto enum value."""
        mapping = {
            Device.cpu: common_pb2.DEVICE_CPU,
            Device.cuda: common_pb2.DEVICE_CUDA,
        }
        return mapping[device]

    @staticmethod
    def from_proto(proto_value: int) -> Device:
        """Convert proto enum value to Device enum."""
        mapping = {
            common_pb2.DEVICE_CPU: Device.cpu,
            common_pb2.DEVICE_CUDA: Device.cuda,
        }
        return mapping[proto_value]


class DTypeConverter:
    """Convert between Pydantic DType enums and Protobuf DTypeProto."""

    @staticmethod
    def to_proto(dtype: AnyDType) -> SerializationResult[int]:
        """Convert FullPrecisionDType or ReducedPrecisionDType to proto enum value."""
        match _DTYPE_TO_PROTO.get(dtype):
            case None:
                return Failure(
                    UnknownDType(proto_value=0)
                )  # No valid proto value for unknown dtype
            case proto_value:
                return Success(proto_value)

    @staticmethod
    def from_proto(proto_value: int) -> SerializationResult[AnyDType]:
        """Convert proto enum value to FullPrecisionDType or ReducedPrecisionDType."""
        match _PROTO_TO_DTYPE.get(proto_value):
            case None:
                return Failure(UnknownDType(proto_value=proto_value))
            case dtype:
                return Success(dtype)


__all__ = [
    "PrecisionConverter",
    "DeviceConverter",
    "DTypeConverter",
]
