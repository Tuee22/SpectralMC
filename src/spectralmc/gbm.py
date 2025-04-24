# python/spectralmc/gbm.py

"""
Simulating and pricing using a simple geometric Brownian motion (GBM)
model with Numba+CuPy.
 - SimulationParams: defines layout, seeds
 - BlackScholes: GPU-based simulation + payoffs
"""

from __future__ import annotations

import cupy as cp  # type: ignore[import-untyped]
from numba import cuda  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Annotated
from math import sqrt

from spectralmc.async_normals import ConcurrentNormGenerator

# --------------------------------------------------------------------------- #
#                               Type Helpers                                  #
# --------------------------------------------------------------------------- #

PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]

# --------------------------------------------------------------------------- #
#                          Simulation Parameter Model                         #
# --------------------------------------------------------------------------- #


class SimulationParams(BaseModel):
    """
    Defines simulation parameters for GPU‑based Monte‑Carlo.
    """

    timesteps: int = Field(..., gt=0)
    network_size: int = Field(..., gt=0)
    batches_per_mc_run: int = Field(..., gt=0)
    threads_per_block: int = Field(..., gt=0)
    mc_seed: int = Field(..., gt=0)
    buffer_size: int = Field(..., gt=0)

    # ---------------------- convenience helpers --------------------------- #

    def total_paths(self) -> int:
        """Total simulated paths (network_size × batches_per_mc_run)."""
        return self.network_size * self.batches_per_mc_run

    def total_blocks(self) -> int:
        """
        CUDA blocks = ceil(total_paths / threads_per_block)

        We implement   (n + t − 1) // t   for that ceiling division.
        """
        return (
            self.total_paths() + self.threads_per_block - 1
        ) // self.threads_per_block

    def memory_footprint_bytes(self) -> int:
        """Returns total memory footprint in bytes"""
        bytes_per_float = 8  # double precision floats
        return bytes_per_float * self.total_paths() * self.timesteps * self.buffer_size


# --------------------------------------------------------------------------- #
#                              CUDA Kernel                                    #
# --------------------------------------------------------------------------- #


@cuda.jit  # type: ignore[misc]
def SimulateBlackScholes(
    input_output: cp.ndarray, timesteps: int, sqrt_dt: float, X0: float, v: float
) -> None:
    """
    Evolves *Geometric* Brownian motion paths **in‑place** on the GPU.
    Each column = one path; each row = time‑step.
    """
    idx = cuda.grid(1)
    if idx < input_output.shape[1]:
        X = X0
        for i in range(timesteps):
            dW = input_output[i, idx] * sqrt_dt
            X += v * X * dW
            X = abs(X)  # numerical guard
            input_output[i, idx] = X


# --------------------------------------------------------------------------- #
#                           Black–Scholes Engine                              #
# --------------------------------------------------------------------------- #


class BlackScholes:
    """
    GPU Monte‑Carlo engine for a vanilla Black–Scholes world.
    """

    # ------------------------------ inputs -------------------------------- #

    class Inputs(BaseModel):
        X0: PosFloat
        K: PosFloat
        T: NonNegFloat
        r: float
        d: float
        v: NonNegFloat

    # ---------------------------- results --------------------------------- #

    class SimResults(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        times: cp.ndarray
        sims: cp.ndarray
        forwards: cp.ndarray
        df: cp.ndarray

    class PricingResults(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        call_price_intrinsic: cp.ndarray
        put_price_intrinsic: cp.ndarray
        underlying: cp.ndarray
        put_convexity: cp.ndarray
        call_convexity: cp.ndarray

    class HostPricingResults(BaseModel):
        call_price_intrinsic: float
        put_price_intrinsic: float
        underlying: float
        put_convexity: float
        call_convexity: float
        call_price: float
        put_price: float

    # -------------------------- constructor ------------------------------- #

    def __init__(self, sp: SimulationParams) -> None:
        self._sp = sp
        self._cp_stream = cp.cuda.Stream(non_blocking=True)
        self._numba_stream = cuda.stream()

        # Asynchronous normal generator pool
        self._normal_gen = ConcurrentNormGenerator(
            rows=self._sp.timesteps,
            cols=self._sp.total_paths(),
            seed=self._sp.mc_seed,
            buffer_size=self._sp.buffer_size,
        )

    # ------------------------ path simulation ----------------------------- #

    def _simulate(self, inputs: Inputs) -> SimResults:
        # 1) Fetch normals (already on‑device)
        sims = self._normal_gen.get_matrix()

        # 2) Launch kernel on Numba stream
        dt = inputs.T / self._sp.timesteps
        sqrt_dt = sqrt(dt)
        sims_numba = cuda.as_cuda_array(sims)  # same underlying memory
        SimulateBlackScholes[
            self._sp.total_blocks(), self._sp.threads_per_block, self._numba_stream
        ](sims_numba, self._sp.timesteps, sqrt_dt, inputs.X0, inputs.v)

        # 3) Concurrent CuPy work
        with self._cp_stream:
            times = cp.linspace(dt, inputs.T, num=self._sp.timesteps)
            forwards = inputs.X0 * cp.exp((inputs.r - inputs.d) * times)
            df = cp.exp(-inputs.r * times)

            self._numba_stream.synchronize()  # wait for kernel
            row_means = cp.mean(sims, axis=1, keepdims=True)
            sims *= forwards[:, None] / row_means

        self._cp_stream.synchronize()
        return self.SimResults(times=times, sims=sims, forwards=forwards, df=df)

    # ----------------------------- pricing -------------------------------- #

    def price(self, inputs: Inputs, sr: Optional[SimResults] = None) -> PricingResults:
        """generates a pricing sample on the device"""
        sr = sr or self._simulate(inputs)

        with self._cp_stream:
            F = sr.forwards[-1]
            df = sr.df[-1]
            K = cp.array(inputs.K)

            put_intr = df * cp.maximum(K - F, 0)
            call_intr = df * cp.maximum(F - K, 0)

            underlying_terminal = sr.sims[-1, :].reshape(
                (self._sp.network_size, self._sp.batches_per_mc_run)
            )
            put_conv = df * cp.maximum(K - underlying_terminal, 0) - put_intr
            call_conv = df * cp.maximum(underlying_terminal - K, 0) - call_intr

        self._cp_stream.synchronize()
        return self.PricingResults(
            call_price_intrinsic=call_intr,
            put_price_intrinsic=put_intr,
            underlying=underlying_terminal,
            put_convexity=put_conv,
            call_convexity=call_conv,
        )

    def get_host_price(self, pr: PricingResults) -> HostPricingResults:
        """takes the pricing sample on the device, averages it, and moves it to the host"""
        with self._cp_stream:
            call_price_intrinsic = pr.call_price_intrinsic.item()
            put_price_intrinsic = pr.put_price_intrinsic.item()
            underlying = pr.underlying.mean().item()
            put_convexity = pr.put_convexity.mean().item()
            call_convexity = pr.call_convexity.mean().item()
        self._cp_stream.synchronize()

        return self.HostPricingResults(
            call_price_intrinsic=call_price_intrinsic,
            put_price_intrinsic=put_price_intrinsic,
            underlying=underlying,
            put_convexity=put_convexity,
            call_convexity=call_convexity,
            call_price=call_price_intrinsic + call_convexity,
            put_price=put_price_intrinsic + put_convexity,
        )

    def price_to_host(self, inputs: Inputs) -> HostPricingResults:
        """prices on device and returns price to the host"""
        pr = self.price(inputs=inputs)
        return self.get_host_price(pr=pr)
