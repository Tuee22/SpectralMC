"""
spectralmc.gbm
---------------
GPU‑accelerated Geometric Brownian‑motion Monte‑Carlo with selectable
precision.  The file passes **mypy --strict** while deliberately marking
only the compiled CUDA kernel as ``Any`` (Python’s type system cannot
describe the square‑bracket launch syntax).
"""

from __future__ import annotations

from math import sqrt
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    Union,
    TypeAlias,
)

import cupy as cp  # type: ignore[import-untyped]
import numpy as np
from numba import cuda  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from spectralmc.async_normals import ConcurrentNormGenerator

# ---------------------------------------------------------------------------#
# Helper type aliases                                                        #
# ---------------------------------------------------------------------------#

PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]
DtypeLiteral = Literal["float32", "float64"]

Scalar32 = np.float32
Scalar64 = np.float64
Scalar: TypeAlias = Union[Scalar32, Scalar64]

DeviceNDArray: TypeAlias = NDArray[np.floating]

# ---------------------------------------------------------------------------#
# Simulation‑parameter model                                                 #
# ---------------------------------------------------------------------------#


class SimulationParams(BaseModel):
    """Immutable layout + RNG settings; precision is mandatory."""

    timesteps: int = Field(..., gt=0)
    network_size: int = Field(..., gt=0)
    batches_per_mc_run: int = Field(..., gt=0)
    threads_per_block: int = Field(..., gt=0)
    mc_seed: int = Field(..., gt=0)
    buffer_size: int = Field(..., gt=0)

    dtype: DtypeLiteral

    # ---------------- convenience --------------------------------------

    @property
    def cp_dtype(self) -> cp.dtype:  # noqa: D401
        return cp.dtype(self.dtype)

    def total_paths(self) -> int:
        return self.network_size * self.batches_per_mc_run

    def total_blocks(self) -> int:
        return (
            self.total_paths() + self.threads_per_block - 1
        ) // self.threads_per_block

    def memory_footprint_bytes(self) -> int:
        return (
            int(self.cp_dtype.itemsize)
            * self.total_paths()
            * self.timesteps
            * self.buffer_size
        )


# ---------------------------------------------------------------------------#
# CUDA kernel — compiled object typed as Any                                 #
# ---------------------------------------------------------------------------#


@cuda.jit  # type: ignore[misc]
def _simulate_black_scholes(  # noqa: N802 (CUDA naming)
    input_output: DeviceNDArray,
    timesteps: int,
    dt: float,
    X0: float,
    r: float,
    d: float,
    v: float,
) -> None:
    """In‑place GBM evolution kernel (dtype comes from *input_output*)."""

    idx = cuda.grid(1)
    if idx < input_output.shape[1]:
        sqrt_dt = sqrt(dt)
        X = X0
        for i in range(timesteps):
            dW = input_output[i, idx] * sqrt_dt
            # log_v = (r - d - 0.5*v*v) * dt + v * dW
            # X *= exp(log_v)
            X += (r - d) * X * dt + v * X * dW
            X = abs(X)
            input_output[i, idx] = X


# Treat dispatcher as Any so mypy ignores launch syntax
SimulateBlackScholes: Any = _simulate_black_scholes  # noqa: N802

# ---------------------------------------------------------------------------#
# Black‑Scholes Monte‑Carlo engine                                           #
# ---------------------------------------------------------------------------#


class BlackScholes:
    """GPU Monte‑Carlo engine obeying the precision in *SimulationParams*."""

    # ---------------- nested models ------------------------------------ #

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

    # ---------------- constructor -------------------------------------- #

    def __init__(self, sp: SimulationParams) -> None:
        self._sp = sp
        self._cp_dtype: cp.dtype = sp.cp_dtype
        self._cp_stream: cp.cuda.Stream = cp.cuda.Stream(non_blocking=True)
        self._numba_stream = cuda.stream()

        self._normal_gen = ConcurrentNormGenerator(
            rows=sp.timesteps,
            cols=sp.total_paths(),
            seed=sp.mc_seed,
            buffer_size=sp.buffer_size,
            dtype=self._cp_dtype,
        )

    # ---------------- simulation pipeline ------------------------------ #

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
        )

        # concurrent wrt kernel above
        with self._cp_stream:
            times = cp.linspace(
                dt,
                inputs.T,
                num=self._sp.timesteps,
                dtype=self._cp_dtype,
            )
            forwards = inputs.X0 * cp.exp((inputs.r - inputs.d) * times)
            df = cp.exp(-inputs.r * times)

        self._numba_stream.synchronize()
        self._cp_stream.synchronize()
        return self.SimResults(times=times, sims=sims, forwards=forwards, df=df)

    # ---------------- pricing ------------------------------------------ #

    def price(
        self,
        *,
        inputs: Inputs,
        sr: Optional[SimResults] = None,
    ) -> PricingResults:
        sr = sr or self._simulate(inputs)

        with self._cp_stream:
            F = sr.forwards[-1]
            df_last = sr.df[-1]
            K = cp.asarray(inputs.K, dtype=self._cp_dtype)

            put_intr = df_last * cp.maximum(K - F, 0)
            call_intr = df_last * cp.maximum(F - K, 0)

            terminal = sr.sims[-1, :].reshape(
                (self._sp.network_size, self._sp.batches_per_mc_run)
            )
            put_conv = df_last * cp.maximum(K - terminal, 0) - put_intr
            call_conv = df_last * cp.maximum(terminal - K, 0) - call_intr

        self._cp_stream.synchronize()
        return self.PricingResults(
            call_price_intrinsic=call_intr,
            put_price_intrinsic=put_intr,
            underlying=terminal,
            put_convexity=put_conv,
            call_convexity=call_conv,
        )

    # ---------------- host aggregation --------------------------------- #

    def get_host_price(self, pr: PricingResults) -> HostPricingResults:
        with self._cp_stream:
            call_intr = float(pr.call_price_intrinsic.item())
            put_intr = float(pr.put_price_intrinsic.item())
            underlying = float(pr.underlying.mean().item())
            put_conv = float(pr.put_convexity.mean().item())
            call_conv = float(pr.call_convexity.mean().item())
        self._cp_stream.synchronize()

        return self.HostPricingResults(
            call_price_intrinsic=call_intr,
            put_price_intrinsic=put_intr,
            underlying=underlying,
            put_convexity=put_conv,
            call_convexity=call_conv,
            call_price=call_intr + call_conv,
            put_price=put_intr + put_conv,
        )

    # ---------------- convenience wrapper ------------------------------ #

    def price_to_host(self, inputs: Inputs) -> HostPricingResults:
        return self.get_host_price(self.price(inputs=inputs))
