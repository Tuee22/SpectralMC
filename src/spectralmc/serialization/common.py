# src/spectralmc/serialization/common.py
"""Converters for common enums and basic types."""

from __future__ import annotations

from spectralmc.models.numerical import Precision
from spectralmc.models.torch import (
    Device,
    FullPrecisionDType,
    ReducedPrecisionDType,
    AnyDType,
)
from spectralmc.proto import common_pb2


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
    def from_proto(proto_value: int) -> Precision:
        """Convert proto enum value to Precision enum."""
        mapping = {
            common_pb2.PRECISION_FLOAT32: Precision.float32,
            common_pb2.PRECISION_FLOAT64: Precision.float64,
        }
        return mapping[proto_value]


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
    def to_proto(dtype: AnyDType) -> int:
        """Convert FullPrecisionDType or ReducedPrecisionDType to proto enum value."""
        if isinstance(dtype, FullPrecisionDType):
            mapping_full = {
                FullPrecisionDType.float32: common_pb2.DTYPE_FLOAT32,
                FullPrecisionDType.float64: common_pb2.DTYPE_FLOAT64,
                FullPrecisionDType.complex64: common_pb2.DTYPE_COMPLEX64,
                FullPrecisionDType.complex128: common_pb2.DTYPE_COMPLEX128,
            }
            return mapping_full[dtype]
        elif isinstance(dtype, ReducedPrecisionDType):
            mapping_reduced = {
                ReducedPrecisionDType.float16: common_pb2.DTYPE_FLOAT16,
                ReducedPrecisionDType.bfloat16: common_pb2.DTYPE_BFLOAT16,
            }
            return mapping_reduced[dtype]
        else:
            raise TypeError(f"Unknown dtype type: {type(dtype)}")

    @staticmethod
    def from_proto(proto_value: int) -> AnyDType:
        """Convert proto enum value to FullPrecisionDType or ReducedPrecisionDType."""
        mapping: dict[int, AnyDType] = {
            common_pb2.DTYPE_FLOAT32: FullPrecisionDType.float32,
            common_pb2.DTYPE_FLOAT64: FullPrecisionDType.float64,
            common_pb2.DTYPE_COMPLEX64: FullPrecisionDType.complex64,
            common_pb2.DTYPE_COMPLEX128: FullPrecisionDType.complex128,
            common_pb2.DTYPE_FLOAT16: ReducedPrecisionDType.float16,
            common_pb2.DTYPE_BFLOAT16: ReducedPrecisionDType.bfloat16,
        }
        return mapping[proto_value]


__all__ = [
    "PrecisionConverter",
    "DeviceConverter",
    "DTypeConverter",
]
