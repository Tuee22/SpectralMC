# tests/test_serialization/test_common_converters.py
"""Round-trip tests for common enum converters."""

from __future__ import annotations

import pytest


from spectralmc.models.numerical import Precision
from spectralmc.models.torch import Device, FullPrecisionDType
from spectralmc.result import Failure, Success
from spectralmc.serialization.common import (
    DeviceConverter,
    DTypeConverter,
    PrecisionConverter,
)


def test_precision_round_trip() -> None:
    """Test Precision enum round-trip."""
    for precision in (Precision.float32, Precision.float64):
        proto = PrecisionConverter.to_proto(precision)
        result = PrecisionConverter.from_proto(proto)
        match result:
            case Success(value):
                assert value == precision
            case Failure(error):
                pytest.fail(f"precision conversion failed: {error}")


def test_device_round_trip() -> None:
    """Test Device enum round-trip."""
    for device in (Device.cpu, Device.cuda):
        proto = DeviceConverter.to_proto(device)
        result = DeviceConverter.from_proto(proto)
        assert result == device


def test_dtype_round_trip() -> None:
    """Test dtype enum round-trip."""
    for dtype in (
        FullPrecisionDType.float32,
        FullPrecisionDType.float64,
        FullPrecisionDType.complex64,
        FullPrecisionDType.complex128,
    ):
        proto_result = DTypeConverter.to_proto(dtype)
        match proto_result:
            case Success(proto):
                result = DTypeConverter.from_proto(proto)
                match result:
                    case Success(value):
                        assert value == dtype
                    case Failure(error):
                        pytest.fail(f"from_proto failed: {error}")
            case Failure(error):
                pytest.fail(f"to_proto failed: {error}")
