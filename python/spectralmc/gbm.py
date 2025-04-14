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
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Annotated
from math import sqrt

PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]


class SimulationParams(BaseModel):
    """
    Defines simulation parameters for GPU-based Monte Carlo.
    """

    timesteps: int = Field(..., gt=0)
    network_size: int = Field(..., gt=0)
    batches_per_mc_run: int = Field(..., gt=0)
    threads_per_block: int = Field(..., gt=0)
    mc_seed: int = Field(..., gt=0)
    buffer_size: int = Field(..., gt=0)

    def total_paths(self) -> int:
        return self.network_size * self.batches_per_mc_run

    def total_blocks(self) -> int:
        return (
            self.total_paths() + self.threads_per_block - 1
        ) // self.threads_per_block


@cuda.jit  # type: ignore[misc]
def SimulateBlackScholes(
    input_output: cp.ndarray, timesteps: int, sqrt_dt: float, X0: float, v: float
) -> None:
    """
    GPU kernel for GBM paths.
    Each column in input_output is a path; each row is a time step.
    """
    idx = cuda.grid(1)
    if idx < input_output.shape[1]:
        X = X0
        for i in range(timesteps):
            dW = input_output[i, idx] * sqrt_dt
            X += v * X * dW
            X = abs(X)
            input_output[i, idx] = X


class BlackScholes:
    """
    Manages a GPU-based Black-Scholes simulation and payoff computation.
    """

    class Inputs(BaseModel):
        X0: PosFloat
        K: PosFloat
        T: NonNegFloat
        r: float
        d: float
        v: PosFloat

    class SimResults(BaseModel):
        times: cp.ndarray
        sims: cp.ndarray
        forwards: cp.ndarray
        df: cp.ndarray

        @field_validator("times", mode="before")
        def check_times(cls, val: cp.ndarray) -> cp.ndarray:
            if not isinstance(val, cp.ndarray):
                raise ValueError("Expected CuPy array for times")
            return val

        @field_validator("sims", mode="before")
        def check_sims(cls, val: cp.ndarray) -> cp.ndarray:
            if not isinstance(val, cp.ndarray):
                raise ValueError("Expected CuPy array for sims")
            return val

        @field_validator("forwards", mode="before")
        def check_forwards(cls, val: cp.ndarray) -> cp.ndarray:
            if not isinstance(val, cp.ndarray):
                raise ValueError("Expected CuPy array for forwards")
            return val

        @field_validator("df", mode="before")
        def check_df(cls, val: cp.ndarray) -> cp.ndarray:
            if not isinstance(val, cp.ndarray):
                raise ValueError("Expected CuPy array for df")
            return val

    class PricingResults(BaseModel):
        call_price_intrinsic: cp.ndarray
        put_price_intrinsic: cp.ndarray
        underlying: cp.ndarray
        put_convexity: cp.ndarray
        call_convexity: cp.ndarray

        @field_validator("call_price_intrinsic", mode="before")
        def check_call_intr(cls, val: cp.ndarray) -> cp.ndarray:
            if not isinstance(val, cp.ndarray):
                raise ValueError("call_price_intrinsic must be CuPy array")
            return val

        @field_validator("put_price_intrinsic", mode="before")
        def check_put_intr(cls, val: cp.ndarray) -> cp.ndarray:
            if not isinstance(val, cp.ndarray):
                raise ValueError("put_price_intrinsic must be CuPy array")
            return val

        @field_validator("underlying", mode="before")
        def check_underlying(cls, val: cp.ndarray) -> cp.ndarray:
            if not isinstance(val, cp.ndarray):
                raise ValueError("underlying must be CuPy array")
            return val

        @field_validator("put_convexity", mode="before")
        def check_put_conv(cls, val: cp.ndarray) -> cp.ndarray:
            if not isinstance(val, cp.ndarray):
                raise ValueError("put_convexity must be CuPy array")
            return val

        @field_validator("call_convexity", mode="before")
        def check_call_conv(cls, val: cp.ndarray) -> cp.ndarray:
            if not isinstance(val, cp.ndarray):
                raise ValueError("call_convexity must be CuPy array")
            return val

    def __init__(self, sp: SimulationParams) -> None:
        self._sp = sp
        self._cp_stream = cp.cuda.Stream(non_blocking=True)
        self._numba_stream = cuda.stream()

    def _simulate(self, inputs: BlackScholes.Inputs) -> BlackScholes.SimResults:
        sims = cp.random.randn(self._sp.timesteps, self._sp.total_paths())
        dt = inputs.T / self._sp.timesteps if self._sp.timesteps > 0 else 0.0
        sqrt_dt = sqrt(dt) if dt > 0 else 0.0

        sims_dev = cuda.to_device(sims, stream=self._numba_stream)
        SimulateBlackScholes[
            self._sp.total_blocks(), self._sp.threads_per_block, self._numba_stream
        ](sims_dev, self._sp.timesteps, sqrt_dt, inputs.X0, inputs.v)

        with self._cp_stream:
            times = (
                cp.linspace(dt, inputs.T, num=self._sp.timesteps)
                if self._sp.timesteps > 0
                else cp.array([])
            )
            forwards = (
                inputs.X0 * cp.exp((inputs.r - inputs.d) * times)
                if self._sp.timesteps > 0
                else cp.array([])
            )
            df = cp.exp(-inputs.r * times) if self._sp.timesteps > 0 else cp.array([])

            self._numba_stream.synchronize()
            row_means = (
                cp.mean(sims, axis=1, keepdims=True)
                if self._sp.timesteps > 0
                else cp.array([[1]])
            )
            factors = forwards[:, None] / row_means if self._sp.timesteps > 0 else 1
            sims = sims * factors

        self._cp_stream.synchronize()
        return self.SimResults(times=times, sims=sims, forwards=forwards, df=df)

    def price(
        self, inputs: BlackScholes.Inputs, sr: Optional[BlackScholes.SimResults] = None
    ) -> BlackScholes.PricingResults:
        if sr is None:
            sr = self._simulate(inputs)

        with self._cp_stream:
            if sr.forwards.size == 0:
                # Edge case: no timesteps
                return self.PricingResults(
                    call_price_intrinsic=cp.array([]),
                    put_price_intrinsic=cp.array([]),
                    underlying=cp.array([]),
                    put_convexity=cp.array([]),
                    call_convexity=cp.array([]),
                )

            F = sr.forwards[-1]
            df = sr.df[-1]
            K = cp.array(inputs.K)
            put_intrinsic = df * cp.maximum(K - F, 0)
            call_intrinsic = df * cp.maximum(F - K, 0)

            underlying_terminal = sr.sims[-1, :].reshape(
                (self._sp.network_size, self._sp.batches_per_mc_run)
            )
            put_convexity = df * cp.maximum(K - underlying_terminal, 0) - put_intrinsic
            call_convexity = (
                df * cp.maximum(underlying_terminal - K, 0) - call_intrinsic
            )

        self._cp_stream.synchronize()
        return self.PricingResults(
            call_price_intrinsic=call_intrinsic,
            put_price_intrinsic=put_intrinsic,
            underlying=underlying_terminal,
            put_convexity=put_convexity,
            call_convexity=call_convexity,
        )
