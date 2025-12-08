# tests/test_serialization/test_simulation_converters.py
"""Tests for simulation serialization converters."""

from __future__ import annotations

from spectralmc.errors.serialization import (
    UnsupportedPrecision,
    UnknownThreadsPerBlock,
    ValidationFailed,
)
from spectralmc.proto import common_pb2, simulation_pb2
from spectralmc.result import Failure, Success
from spectralmc.serialization.simulation import (
    BoundSpecConverter,
    SimulationParamsConverter,
)


def _make_base_params_proto() -> simulation_pb2.SimulationParamsProto:  # type: ignore[no-any-unimported]
    proto = simulation_pb2.SimulationParamsProto()
    proto.skip = 0
    proto.timesteps = 4
    proto.network_size = 2
    proto.batches_per_mc_run = 8
    proto.threads_per_block = 256
    proto.mc_seed = 11
    proto.buffer_size = 1
    proto.dtype = common_pb2.PRECISION_FLOAT32
    return proto


def test_simulation_params_round_trip_success() -> None:
    """Valid proto converts successfully to SimulationParams."""
    proto = _make_base_params_proto()
    result = SimulationParamsConverter.from_proto(proto)
    match result:
        case Success(params):
            assert params.threads_per_block == 256
        case Failure(err):
            raise AssertionError(f"expected Success, got {err}")


def test_simulation_params_invalid_threads() -> None:
    """Invalid thread block size yields UnknownThreadsPerBlock."""
    proto = _make_base_params_proto()
    proto.threads_per_block = 7
    result = SimulationParamsConverter.from_proto(proto)
    match result:
        case Failure(error):
            assert isinstance(error, UnknownThreadsPerBlock)
        case Success(_):
            raise AssertionError("expected UnknownThreadsPerBlock")


def test_simulation_params_invalid_dtype() -> None:
    """Unknown dtype proto yields UnsupportedPrecision."""
    proto = _make_base_params_proto()
    proto.dtype = 999
    result = SimulationParamsConverter.from_proto(proto)
    match result:
        case Failure(error):
            assert isinstance(error, UnsupportedPrecision)
        case Success(_):
            raise AssertionError("expected UnsupportedPrecision")


def test_bound_spec_validation_failure() -> None:
    """BoundSpec proto with invalid bounds yields ValidationFailed."""
    proto = simulation_pb2.BoundSpecProto()
    proto.lower = 1.0
    proto.upper = 0.5
    result = BoundSpecConverter.from_proto(proto)
    match result:
        case Failure(error):
            assert isinstance(error, ValidationFailed)
        case Success(_):
            raise AssertionError("expected ValidationFailed")
