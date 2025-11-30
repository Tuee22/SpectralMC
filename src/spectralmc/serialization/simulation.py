# src/spectralmc/serialization/simulation.py
"""Converters for simulation-related Pydantic models."""

from __future__ import annotations

from typing import Literal

from spectralmc.gbm import BlackScholesConfig, SimulationParams
from spectralmc.proto import simulation_pb2
from spectralmc.sobol_sampler import BoundSpec

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
    def from_proto(proto: simulation_pb2.SimulationParamsProto) -> SimulationParams:
        """Convert from proto."""
        precision = PrecisionConverter.from_proto(proto.dtype)
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
        threads = threads_mapping.get(proto.threads_per_block)
        if threads is None:
            raise ValueError(
                f"Invalid threads_per_block: {proto.threads_per_block}. "
                f"Must be one of {list(threads_mapping.keys())}"
            )
        return SimulationParams(
            skip=proto.skip,
            timesteps=proto.timesteps,
            network_size=proto.network_size,
            batches_per_mc_run=proto.batches_per_mc_run,
            threads_per_block=threads,
            mc_seed=proto.mc_seed,
            buffer_size=proto.buffer_size,
            dtype=precision,
        )


class BlackScholesConfigConverter:
    """Convert BlackScholesConfig."""

    @staticmethod
    def to_proto(config: BlackScholesConfig) -> simulation_pb2.BlackScholesConfigProto:
        """Convert to proto."""
        proto = simulation_pb2.BlackScholesConfigProto()
        proto.sim_params.CopyFrom(SimulationParamsConverter.to_proto(config.sim_params))
        proto.simulate_log_return = config.simulate_log_return
        proto.normalize_forwards = config.normalize_forwards
        return proto

    @staticmethod
    def from_proto(proto: simulation_pb2.BlackScholesConfigProto) -> BlackScholesConfig:
        """Convert from proto."""
        return BlackScholesConfig(
            sim_params=SimulationParamsConverter.from_proto(proto.sim_params),
            simulate_log_return=proto.simulate_log_return,
            normalize_forwards=proto.normalize_forwards,
        )


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
    def from_proto(proto: simulation_pb2.BoundSpecProto) -> BoundSpec:
        """Convert from proto."""
        return BoundSpec(lower=proto.lower, upper=proto.upper)


__all__ = [
    "SimulationParamsConverter",
    "BlackScholesConfigConverter",
    "BoundSpecConverter",
]
