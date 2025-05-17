# spectralmc/gbm.py
"""GPU-accelerated Geometric-Brownian Monte-Carlo paths.

The module integrates tightly with :pymod:`spectralmc.async_normals` and passes
``mypy --strict``.  The only ``Any`` annotations refer to the compiled CUDA
kernel dispatcher whose launch-syntax cannot be expressed in the type system.
"""

from __future__ import annotations

from math import exp, sqrt
from typing import Annotated, Any, List, Literal, Optional, TypeAlias

import cupy as cp  # type: ignore[import-untyped]
import numpy as np
from numba import cuda  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from spectralmc.async_normals import ConcurrentNormGenerator, ConcurrentNormGeneratorConfig

# --------------------------------------------------------------------------- #
# Helper type aliases                                                         #
# --------------------------------------------------------------------------- #

PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]
DtypeLiteral = Literal["float32", "float64"]

DeviceNDArray: TypeAlias = NDArray[np.floating]

# --------------------------------------------------------------------------- #
# Simulation-parameter model                                                  #
# --------------------------------------------------------------------------- #


class SimulationParams(BaseModel):
    """Layout and RNG settings (immutable)."""

    timesteps: int = Field(..., gt=0)
    network_size: int = Field(..., gt=0)
    batches_per_mc_run: int = Field(..., gt=0)
    threads_per_block: int = Field(..., gt=0)
    mc_seed: int = Field(..., gt=0)
    buffer_size: int = Field(..., gt=0)
    dtype: DtypeLiteral
    simulate_log_return: bool = True
    normalize_forwards: bool = True

    # ---------------- convenience ----------------------------------- #

    @property
    def cp_dtype(self) -> cp.dtype:  # noqa: D401
        return cp.dtype(self.dtype)

    def total_paths(self) -> int:
        return self.network_size * self.batches_per_mc_run

    def total_blocks(self) -> int:
        return (self.total_paths() + self.threads_per_block - 1) // self.threads_per_block

    def memory_footprint_bytes(self) -> int:
        return (
            int(self.cp_dtype.itemsize)
            * self.total_paths()
            * self.timesteps
            * self.buffer_size
        )


# --------------------------------------------------------------------------- #
# CUDA kernel (typed as Any)                                                  #
# --------------------------------------------------------------------------- #


@cuda.jit  # type: ignore[misc]
def _simulate_black_scholes(  # noqa: N802
    input_output: DeviceNDArray,
    timesteps: int,
    dt: float,
    X0: float,
    r: float,
    d: float,
    v: float,
    simulate_log_return: bool,
) -> None:
    """In-place GBM evolution kernel (dtype derived from *input_output*)."""

    idx = cuda.grid(1)
    if idx < input_output.shape[1]:
        sqrt_dt = sqrt(dt)
        X = X0
        if simulate_log_return:
            drift = r - d - 0.5 * v * v
            for i in range(timesteps):
                dW = input_output[i, idx] * sqrt_dt
                log_v = drift * dt + v * dW
                X *= exp(log_v)
                input_output[i, idx] = X
        else:
            drift = r - d
            for i in range(timesteps):
                dW = input_output[i, idx] * sqrt_dt
                X += drift * X * dt + v * X * dW
                X = abs(X)
                input_output[i, idx] = X


SimulateBlackScholes: Any = _simulate_black_scholes  # noqa: N802

# --------------------------------------------------------------------------- #
# Monte-Carlo engine                                                          #
# --------------------------------------------------------------------------- #


class BlackScholes:
    """GPU Monte-Carlo engine obeying :class:`SimulationParams` precision."""

    # ---------------- nested models ----------------------------------- #

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

    # ---------------- constructor ------------------------------------ #

    def __init__(self, sp: SimulationParams) -> None:
        self._sp = sp
        self._cp_dtype: cp.dtype = sp.cp_dtype
        self._cp_stream: cp.cuda.Stream = cp.cuda.Stream(non_blocking=True)
        self._numba_stream: cuda.cudadrv.driver.Stream = cuda.stream()

        # Build NormGenConfig (skips = 0 for fresh run)
        cfg = ConcurrentNormGeneratorConfig(
            rows=sp.timesteps,
            cols=sp.total_paths(),
            seed=sp.mc_seed,
            dtype=sp.dtype,
            skips=0,
        )
        self._normal_gen = ConcurrentNormGenerator(sp.buffer_size, cfg)

    # ---------------- simulation pipeline --------------------------- #

    def _simulate(self, inputs: Inputs) -> SimResults:
        sims: cp.ndarray = self._normal_gen.get_matrix()
        dt: float = inputs.T / self._sp.timesteps

        sims_numba = cuda.as_cuda_array(sims)
        SimulateBlackScholes[
            self._sp.total_blocks(),
            self._sp.threads_per_block,
            self._numba_stream,
        ](
            sims_numba,
            self._sp.timesteps,
            dt,
            inputs.X0,
            inputs.r,
            inputs.d,
            inputs.v,
            self._sp.simulate_log_return,
        )

        with self._cp_stream:  # concurrent with kernel
            times = cp.linspace(dt, inputs.T, self._sp.timesteps, dtype=self._cp_dtype)
            forwards = inputs.X0 * cp.exp((inputs.r - inputs.d) * times)
            df = cp.exp(-inputs.r * times)

        # Wait for kernel completion before optional normalisation
        self._numba_stream.synchronize()
        if self._sp.normalize_forwards:
            row_means = cp.mean(sims, axis=1, keepdims=True)
            sims *= forwards[:, cp.newaxis] / row_means

        self._cp_stream.synchronize()
        return self.SimResults(times=times, sims=sims, forwards=forwards, df=df)

    # ---------------- pricing --------------------------------------- #

    def price(self, *, inputs: Inputs, sr: Optional[SimResults] = None) -> PricingResults:
        sr = sr or self._simulate(inputs)

        with self._cp_stream:
            F = sr.forwards[-1]
            df_last = sr.df[-1]
            K = cp.asarray(inputs.K, dtype=self._cp_dtype)

            put_intrinsic = df_last * cp.maximum(K - F, 0)
            call_intrinsic = df_last * cp.maximum(F - K, 0)

            terminal = sr.sims[-1, :]
            put_price = df_last * cp.maximum(K - terminal, 0)
            call_price = df_last * cp.maximum(terminal - K, 0)

        self._cp_stream.synchronize()
        return self.PricingResults(
            put_price_intrinsic=put_intrinsic,
            call_price_intrinsic=call_intrinsic,
            underlying=terminal,
            put_price=put_price,
            call_price=call_price,
        )

    # ---------------- host aggregation ------------------------------- #

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

    # ---------------- convenience wrapper --------------------------- #

    def price_to_host(self, inputs: Inputs) -> HostPricingResults:
        return self.get_host_price(self.price(inputs=inputs))
