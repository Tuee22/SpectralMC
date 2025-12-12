# src/spectralmc/serialization/simulation.py
"""Converters for simulation-related Pydantic models."""

from __future__ import annotations

from typing import Literal

from spectralmc.effects import ForwardNormalization, PathScheme
from spectralmc.errors.serialization import SerializationResult, UnknownThreadsPerBlock
from spectralmc.gbm import (
    BlackScholesConfig,
    SimulationParams,
    build_black_scholes_config,
    build_simulation_params,
)
from spectralmc.proto import simulation_pb2
from spectralmc.sobol_sampler import BoundSpec, build_bound_spec
from spectralmc.result import Failure, Success

from .common import PrecisionConverter


ThreadsPerBlock = Literal[32, 64, 128, 256, 512, 1024]


class SimulationParamsConverter:
    """Convert SimulationParams."""

    @staticmethod
    def to_proto(params: SimulationParams) -> simulation_pb2.SimulationParamsProto:
        """Convert to proto."""
        proto = simulation_pb2.SimulationParamsProto()
        proto.skip = params.skip
        proto.timesteps = params.timesteps
        proto.network_size = params.network_size
        proto.batches_per_mc_run = params.batches_per_mc_run
        proto.threads_per_block = params.threads_per_block
        proto.mc_seed = params.mc_seed
        proto.buffer_size = params.buffer_size
        proto.dtype = PrecisionConverter.to_proto(params.dtype)
        return proto

    @staticmethod
    def from_proto(
        proto: simulation_pb2.SimulationParamsProto,
    ) -> SerializationResult[SimulationParams]:
        """Convert from proto."""
        precision_result = PrecisionConverter.from_proto(proto.dtype)
        match precision_result:
            case Failure(error):
                return Failure(error)
            case Success(precision):
                pass

        # Validate and map threads_per_block to Literal type
        # mypy can't narrow int to Literal, so we use explicit mapping
        threads_mapping: dict[int, ThreadsPerBlock] = {
            32: 32,
            64: 64,
            128: 128,
            256: 256,
            512: 512,
            1024: 1024,
        }
        match threads_mapping.get(proto.threads_per_block):
            case None:
                return Failure(UnknownThreadsPerBlock(value=proto.threads_per_block))
            case threads:
                params_result = build_simulation_params(
                    skip=proto.skip,
                    timesteps=proto.timesteps,
                    network_size=proto.network_size,
                    batches_per_mc_run=proto.batches_per_mc_run,
                    threads_per_block=threads,
                    mc_seed=proto.mc_seed,
                    buffer_size=proto.buffer_size,
                    dtype=precision,
                )
                match params_result:
                    case Failure(params_err):
                        return Failure(params_err)
                    case Success(params):
                        return Success(params)


class BlackScholesConfigConverter:
    """Convert BlackScholesConfig."""

    @staticmethod
    def to_proto(config: BlackScholesConfig) -> simulation_pb2.BlackScholesConfigProto:
        """Convert to proto."""
        proto = simulation_pb2.BlackScholesConfigProto()
        proto.sim_params.CopyFrom(SimulationParamsConverter.to_proto(config.sim_params))
        proto.simulate_log_return = config.path_scheme is PathScheme.LOG_EULER
        proto.normalize_forwards = config.normalization is ForwardNormalization.NORMALIZE
        return proto

    @staticmethod
    def from_proto(
        proto: simulation_pb2.BlackScholesConfigProto,
    ) -> SerializationResult[BlackScholesConfig]:
        """Convert from proto."""
        sim_result = SimulationParamsConverter.from_proto(proto.sim_params)
        match sim_result:
            case Failure(error):
                return Failure(error)
            case Success(sim_params):
                pass

        config_result = build_black_scholes_config(
            sim_params=sim_params,
            path_scheme=(
                PathScheme.LOG_EULER if proto.simulate_log_return else PathScheme.SIMPLE_EULER
            ),
            normalization=(
                ForwardNormalization.NORMALIZE
                if proto.normalize_forwards
                else ForwardNormalization.RAW
            ),
        )
        match config_result:
            case Failure(config_err):
                return Failure(config_err)
            case Success(config):
                return Success(config)


class BoundSpecConverter:
    """Convert BoundSpec."""

    @staticmethod
    def to_proto(bound: BoundSpec) -> simulation_pb2.BoundSpecProto:
        """Convert to proto."""
        proto = simulation_pb2.BoundSpecProto()
        proto.lower = bound.lower
        proto.upper = bound.upper
        return proto

    @staticmethod
    def from_proto(proto: simulation_pb2.BoundSpecProto) -> SerializationResult[BoundSpec]:
        """Convert from proto."""
        match build_bound_spec(lower=proto.lower, upper=proto.upper):
            case Failure() as f:
                return f
            case Success(bound):
                return Success(bound)


__all__ = [
    "SimulationParamsConverter",
    "BlackScholesConfigConverter",
    "BoundSpecConverter",
]
