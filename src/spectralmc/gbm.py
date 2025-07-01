"""GPU-accelerated Geometric-Brownian-Motion Monte-Carlo engine (mypy-clean)."""

from __future__ import annotations

from math import exp, sqrt
from typing import Annotated, Literal, Optional, TypeAlias

import cupy as cp
import numpy as np
from numba import cuda
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from spectralmc.async_normals import (
    ConcurrentNormGenerator,
    ConcurrentNormGeneratorConfig,
)

# --------------------------------------------------------------------------- #
# Type aliases                                                                #
# --------------------------------------------------------------------------- #

PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]
# DtypeLiteral = Literal["float32", "float64"]
DeviceNDArray: TypeAlias = NDArray[np.generic]

# --------------------------------------------------------------------------- #
# Simulation-parameter model                                                  #
# --------------------------------------------------------------------------- #


class SimulationParams(BaseModel):
    timesteps: int = Field(..., gt=0)
    network_size: int = Field(..., gt=0)
    batches_per_mc_run: int = Field(..., gt=0)
    threads_per_block: int = Field(..., gt=0)
    mc_seed: int = Field(..., gt=0)
    buffer_size: int = Field(..., gt=0)
    skip: int = Field(0, ge=0)
    dtype: DtypeLiteral

    model_config = ConfigDict(frozen=True)

    # helpers
    @property
    def cp_dtype(self) -> cp.dtype:  # noqa: D401
        return cp.dtype(self.dtype)

    def total_paths(self) -> int:
        return self.network_size * self.batches_per_mc_run

    def total_blocks(self) -> int:
        return (
            self.total_paths() + self.threads_per_block - 1
        ) // self.threads_per_block


# --------------------------------------------------------------------------- #
# Engine config                                                               #
# --------------------------------------------------------------------------- #


class BlackScholesConfig(BaseModel):
    sim_params: SimulationParams
    simulate_log_return: bool = True
    normalize_forwards: bool = True

    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- #
# CUDA kernel (typed Any)                                                     #
# --------------------------------------------------------------------------- #


@cuda.jit
def SimulateBlackScholes(  # noqa: N802
    io: DeviceNDArray,
    timesteps: int,
    dt: float,
    X0: float,
    r: float,
    d: float,
    v: float,
    simulate_log_return: bool,
) -> None:
    """In-place GBM evolution."""
    idx = cuda.grid(1)
    if idx < io.shape[1]:
        sqrt_dt = sqrt(dt)
        X = X0
        if simulate_log_return:
            drift = r - d - 0.5 * v * v
            for i in range(timesteps):
                dW = io[i, idx] * sqrt_dt
                X *= exp(drift * dt + v * dW)
                io[i, idx] = X
        else:
            drift = r - d
            for i in range(timesteps):
                dW = io[i, idx] * sqrt_dt
                X += drift * X * dt + v * X * dW
                X = abs(X)
                io[i, idx] = X


# --------------------------------------------------------------------------- #
# Monte-Carlo engine                                                          #
# --------------------------------------------------------------------------- #


class BlackScholes:
    """GPU Monte-Carlo engine driven by CuPy + Numba."""

    # nested Pydantic models ------------------------------------------------ #
    class Inputs(BaseModel):
        X0: PosFloat
        K: PosFloat
        T: NonNegFloat
        r: float
        d: float
        v: NonNegFloat

    class SimResults(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        times: cp.ndarray
        sims: cp.ndarray
        forwards: cp.ndarray
        df: cp.ndarray

    class PricingResults(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        put_price_intrinsic: cp.ndarray
        call_price_intrinsic: cp.ndarray
        underlying: cp.ndarray
        put_price: cp.ndarray
        call_price: cp.ndarray

    class HostPricingResults(BaseModel):
        put_price_intrinsic: float
        call_price_intrinsic: float
        underlying: float
        put_convexity: float
        call_convexity: float
        put_price: float
        call_price: float

    # ------------------------------------------------------------------ #

    def __init__(self, cfg: BlackScholesConfig) -> None:
        self._cfg = cfg
        sp = cfg.sim_params
        self._cp_dtype = sp.cp_dtype

        self._cp_stream = cp.cuda.Stream(non_blocking=True)
        self._numba_stream = cuda.stream()

        ngen_cfg = ConcurrentNormGeneratorConfig(
            rows=sp.timesteps,
            cols=sp.total_paths(),
            seed=sp.mc_seed,
            dtype=sp.dtype,
            skips=sp.skip,
        )
        self._ngen = ConcurrentNormGenerator(sp.buffer_size, ngen_cfg)
        self._sp = sp

    # ------------------------------------------------------------------ #

    def snapshot(self) -> BlackScholesConfig:
        sp = self._sp.model_copy(
            update={"skip": self._ngen.snapshot().skips},
            deep=True,
        )
        return BlackScholesConfig(
            sim_params=sp,
            simulate_log_return=self._cfg.simulate_log_return,
            normalize_forwards=self._cfg.normalize_forwards,
        )

    # ------------------------------------------------------------------ #

    def _simulate(self, inputs: Inputs) -> SimResults:
        sims = self._ngen.get_matrix()
        dt = inputs.T / self._sp.timesteps

        SimulateBlackScholes[
            self._sp.total_blocks(),
            self._sp.threads_per_block,
            self._numba_stream,
        ](
            cuda.as_cuda_array(sims),
            self._sp.timesteps,
            dt,
            inputs.X0,
            inputs.r,
            inputs.d,
            inputs.v,
            self._cfg.simulate_log_return,
        )

        with self._cp_stream:
            times = cp.linspace(dt, inputs.T, self._sp.timesteps, dtype=self._cp_dtype)
            forwards = inputs.X0 * cp.exp((inputs.r - inputs.d) * times)
            forwards_arr = cp.asarray(forwards)
            df = cp.exp(-inputs.r * times)

        self._numba_stream.synchronize()
        if self._cfg.normalize_forwards:
            row_means = cp.mean(sims, axis=1, keepdims=True).squeeze()
            sims *= cp.expand_dims(forwards / row_means, 1)

        self._cp_stream.synchronize()
        return self.SimResults(times=times, sims=sims, forwards=forwards_arr, df=df)

    # ------------------------------------------------------------------ #

    def price(
        self, *, inputs: Inputs, sr: Optional[SimResults] = None
    ) -> PricingResults:
        sr = sr or self._simulate(inputs)
        with self._cp_stream:
            F = sr.forwards[-1]
            df_last = sr.df[-1]
            K = cp.asarray(inputs.K, dtype=self._cp_dtype)

            put_intr = df_last * cp.maximum(K - F, 0)
            call_intr = df_last * cp.maximum(F - K, 0)

            terminal = sr.sims[-1]
            put_price = df_last * cp.maximum(K - terminal, 0)
            call_price = df_last * cp.maximum(terminal - K, 0)

        self._cp_stream.synchronize()
        return self.PricingResults(
            put_price_intrinsic=put_intr,
            call_price_intrinsic=call_intr,
            underlying=terminal,
            put_price=put_price,
            call_price=call_price,
        )

    # ------------------------------------------------------------------ #

    def get_host_price(self, pr: PricingResults) -> HostPricingResults:
        with self._cp_stream:
            put_intr = float(pr.put_price_intrinsic.item())
            call_intr = float(pr.call_price_intrinsic.item())
            underlying = float(pr.underlying.mean().item())
            put_price = float(pr.put_price.mean().item())
            call_price = float(pr.call_price.mean().item())

        self._cp_stream.synchronize()
        return self.HostPricingResults(
            put_price_intrinsic=put_intr,
            call_price_intrinsic=call_intr,
            underlying=underlying,
            put_convexity=put_price - put_intr,
            call_convexity=call_price - call_intr,
            put_price=put_price,
            call_price=call_price,
        )

    # ------------------------------------------------------------------ #

    def price_to_host(self, inputs: Inputs) -> HostPricingResults:
        return self.get_host_price(self.price(inputs=inputs))
