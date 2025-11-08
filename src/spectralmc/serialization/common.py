# src/spectralmc/serialization/common.py
"""Converters for common enums and basic types."""

from __future__ import annotations

from spectralmc.models.numerical import Precision
from spectralmc.models.torch import Device, DType as TorchDType
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
    """Convert between Pydantic TorchDType and Protobuf DTypeProto."""

    @staticmethod
    def to_proto(dtype: TorchDType) -> int:
        """Convert TorchDType enum to proto enum value."""
        mapping = {
            TorchDType.float32: common_pb2.DTYPE_FLOAT32,
            TorchDType.float64: common_pb2.DTYPE_FLOAT64,
            TorchDType.complex64: common_pb2.DTYPE_COMPLEX64,
            TorchDType.complex128: common_pb2.DTYPE_COMPLEX128,
        }
        return mapping[dtype]

    @staticmethod
    def from_proto(proto_value: int) -> TorchDType:
        """Convert proto enum value to TorchDType enum."""
        mapping = {
            common_pb2.DTYPE_FLOAT32: TorchDType.float32,
            common_pb2.DTYPE_FLOAT64: TorchDType.float64,
            common_pb2.DTYPE_COMPLEX64: TorchDType.complex64,
            common_pb2.DTYPE_COMPLEX128: TorchDType.complex128,
        }
        return mapping[proto_value]


__all__ = [
    "PrecisionConverter",
    "DeviceConverter",
    "DTypeConverter",
]
