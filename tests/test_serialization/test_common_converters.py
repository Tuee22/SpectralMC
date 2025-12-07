# tests/test_serialization/test_common_converters.py
"""Round-trip tests for common enum converters."""

from __future__ import annotations

import torch

from spectralmc.models.numerical import Precision
from spectralmc.models.torch import Device, DType
from spectralmc.serialization.common import (
    DeviceConverter,
    DTypeConverter,
    PrecisionConverter,
)

assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"


def test_precision_round_trip() -> None:
    """Test Precision enum round-trip."""
    for precision in (Precision.float32, Precision.float64):
        proto = PrecisionConverter.to_proto(precision)
        result = PrecisionConverter.from_proto(proto)
        assert result == precision


def test_device_round_trip() -> None:
    """Test Device enum round-trip."""
    for device in (Device.cpu, Device.cuda):
        proto = DeviceConverter.to_proto(device)
        result = DeviceConverter.from_proto(proto)
        assert result == device


def test_dtype_round_trip() -> None:
    """Test DType enum round-trip."""
    for dtype in (
        DType.float32,
        DType.float64,
        DType.complex64,
        DType.complex128,
    ):
        proto = DTypeConverter.to_proto(dtype)
        result = DTypeConverter.from_proto(proto)
        assert result == dtype
