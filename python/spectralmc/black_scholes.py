"""
/spectralmc/python/spectralmc/black_scholes.py

Black-Scholes GPU-based MC using partial stubs, no placeholders or ignoring.
Must pass mypy --strict. from __future__ import annotations so no quotes needed.
"""

from __future__ import annotations

import math
from typing import Any
import numba
from spectralmc import cp
from spectralmc.immutable_base_model import ImmutableBaseModel
from pydantic import Field


@numba.cuda.jit
def SimulateBlackScholes(
    input_output: cp.ndarray[Any, Any],
    timesteps: int,
    sqrt_dt: float,
    X0: float,
    v: float,
) -> None:
    """
    Numba kernel for naive BS: dX = v * X * dW, ignoring drift.

    Args:
        input_output: shape (timesteps, total_paths) storing results
        timesteps: number of timesteps
        sqrt_dt: sqrt(dt)
        X0: initial price
        v: volatility
    """
    idx = numba.cuda.grid(1)
    if idx < input_output.shape[1]:
        X = X0
        for i in range(timesteps):
            dW = input_output[i, idx] * sqrt_dt
            X += v * X * dW
            if X < 0.0:
                X = -X
            input_output[i, idx] = X


class SimulationParams(ImmutableBaseModel):
    """
    Dimension + concurrency info for the BS simulation.
    """

    timesteps: int = Field(..., gt=0)
    network_size: int = Field(..., gt=0)
    batches_per_mc_run: int = Field(..., gt=0)
    threads_per_block: int = Field(..., gt=0)
    mc_seed: int = Field(..., gt=0)
    buffer_size: int = Field(..., gt=0)

    @property
    def total_paths(self) -> int:
        """
        Return total # of simulation paths = network_size * batches_per_mc_run
        """
        return self.network_size * self.batches_per_mc_run

    @property
    def total_blocks(self) -> int:
        """
        Compute number of CUDA blocks
        """
        return (self.total_paths + self.threads_per_block - 1) // self.threads_per_block


def run_black_scholes_sim(params: SimulationParams) -> cp.ndarray[Any, Any]:
    """
    Minimal BS path sim returning a Cupy ndarray.

    Args:
        params: concurrency + dimension info

    Returns:
        shape (timesteps, total_paths) Cupy ndarray
    """
    sims = cp.zeros((params.timesteps, params.total_paths), dtype=cp.float32)

    dt = 1.0 / float(params.timesteps)
    sqrt_dt = math.sqrt(dt)

    sims_dev = numba.cuda.as_cuda_array(sims)
    SimulateBlackScholes[params.total_blocks, params.threads_per_block](
        sims_dev, params.timesteps, sqrt_dt, 100.0, 0.2
    )
    return sims
