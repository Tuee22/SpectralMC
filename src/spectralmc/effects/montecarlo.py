"""
Monte Carlo Effect ADTs for simulation and FFT operations.

This module defines frozen dataclasses representing all Monte Carlo-related side effects,
enabling exhaustive pattern matching and type-safe effect composition.

Type Safety:
    - All effect types are frozen dataclasses (immutable)
    - Literal discriminators enable exhaustive pattern matching
    - Union types define closed sets of effect variants

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GenerateNormals:
    """Request to generate standard normal random matrix on GPU.

    Attributes:
        kind: Discriminator for pattern matching. Always "GenerateNormals".
        rows: Number of rows in the output matrix.
        cols: Number of columns in the output matrix.
        seed: Random seed for reproducibility.
        skip: Number of random values to skip (for resuming).
        output_tensor_id: Identifier for storing the generated normals in registry.
    """

    kind: Literal["GenerateNormals"] = "GenerateNormals"
    rows: int = 0
    cols: int = 0
    seed: int = 0
    skip: int = 0
    output_tensor_id: str = "normals"


@dataclass(frozen=True)
class SimulatePaths:
    """Request to simulate GBM price paths on GPU.

    Attributes:
        kind: Discriminator for pattern matching. Always "SimulatePaths".
        spot: Initial spot price.
        strike: Strike price.
        rate: Risk-free interest rate.
        dividend: Continuous dividend yield.
        vol: Volatility.
        expiry: Time to expiration in years.
        timesteps: Number of timesteps to simulate.
        batches: Number of parallel simulations.
        simulate_log_return: Use log-Euler scheme for variance reduction.
        normalize_forwards: Normalize each time-slice to analytic forward.
        input_normals_id: Identifier for pre-generated normal random matrix.
        output_tensor_id: Identifier for storing the simulated paths in registry.
    """

    kind: Literal["SimulatePaths"] = "SimulatePaths"
    spot: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    dividend: float = 0.0
    vol: float = 0.2
    expiry: float = 1.0
    timesteps: int = 252
    batches: int = 1024
    simulate_log_return: bool = True
    normalize_forwards: bool = True
    input_normals_id: str = ""
    output_tensor_id: str = "paths"


@dataclass(frozen=True)
class ComputeFFT:
    """Request to compute FFT on GPU tensor.

    Attributes:
        kind: Discriminator for pattern matching. Always "ComputeFFT".
        input_tensor_id: Identifier for the input tensor.
        axis: Axis along which to compute FFT.
        output_tensor_id: Identifier for storing the FFT result in registry.
    """

    kind: Literal["ComputeFFT"] = "ComputeFFT"
    input_tensor_id: str = ""
    axis: int = -1
    output_tensor_id: str = "fft"


# Monte Carlo Effect Union
MonteCarloEffect = GenerateNormals | SimulatePaths | ComputeFFT
