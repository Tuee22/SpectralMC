"""
spectralmc.gbm
==============

GPU Monte-Carlo simulator and pricer for **Black–Scholes**.
Now supports single-precision or double-precision device buffers via
``SimulationParams.dtype``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Final, Literal, Protocol, TypeAlias

import cupy as cp  # type: ignore[import-untyped]
from numba import cuda  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

__all__ = ["SimulationParams", "BlackScholesPricer"]

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]
NDArrayF: TypeAlias = "cp.ndarray"  # forward-ref keeps mypy happy

# ---------------------------------------------------------------------------
# Async normal generator
# ---------------------------------------------------------------------------

from spectralmc.async_normals import ConcurrentNormGenerator

# ---------------------------------------------------------------------------
# Simulation-grid parameters
# ---------------------------------------------------------------------------


class SimulationParams(BaseModel):
    """
    Grid / RNG configuration.

    Parameters
    ----------
    dtype
        ``"float32"``  → all device buffers use ``cp.float32``
        ``"float64"``  → … use ``cp.float64`` (default)
    """

    timesteps: int = Field(..., gt=0)
    network_size: int = Field(..., gt=0)
    batches_per_mc_run: int = Field(..., gt=0)
    threads_per_block: int = Field(..., gt=0)
    mc_seed: int = Field(..., gt=0)
    buffer_size: int = Field(..., gt=0)
    dtype: Literal["float32", "float64"] = "float64"

    # ---------- derived helpers -----------------------------------------

    def total_paths(self) -> int:
        return self.network_size * self.batches_per_mc_run

    def total_blocks(self) -> int:
        return (
            self.total_paths() + self.threads_per_block - 1
        ) // self.threads_per_block

    # dtype helpers ------------------------------------------------------

    @property
    def cp_dtype(self) -> cp.dtype:  # noqa: D401
        return cp.float32 if self.dtype == "float32" else cp.float64

    @property
    def itemsize(self) -> int:  # noqa: D401
        return int(cp.dtype(self.cp_dtype).itemsize)

    def memory_footprint_bytes(self) -> int:
        """Approximate bytes used by the async normal pool."""
        return self.itemsize * self.total_paths() * self.timesteps * self.buffer_size


# ---------------------------------------------------------------------------
# Forward-process abstraction + constant-σ implementation
# ---------------------------------------------------------------------------


class ForwardProcess(Protocol):
    def sigma(self, t_idx: int, forward: float) -> float: ...


@dataclass(slots=True)
class BlackScholesForward(ForwardProcess):
    sigma_const: float

    def sigma(self, _t_idx: int, _forward: float) -> float:
        return self.sigma_const


# ---------------------------------------------------------------------------
# CUDA kernel – exact log update
# ---------------------------------------------------------------------------


@cuda.jit  # type: ignore[misc]
def _simulate_forward_kernel(
    normals: NDArrayF,
    sigmas: NDArrayF,
    F0: float,
    dt: float,
    sqrt_dt: float,
) -> None:
    j = cuda.grid(1)
    if j >= normals.shape[1]:
        return

    F_val = F0
    for i in range(normals.shape[0]):
        sigma_val = sigmas[i, j]
        z = normals[i, j]
        dlog = -0.5 * sigma_val * sigma_val * dt + sigma_val * sqrt_dt * z
        F_val *= math.exp(dlog)
        normals[i, j] = F_val


# ---------------------------------------------------------------------------
# Generic forward-path engine
# ---------------------------------------------------------------------------


class MCForwardEngine:
    def __init__(self, sim: SimulationParams) -> None:
        self.sp: Final = sim
        self._cuda_stream: Final = cuda.stream()
        self._norm_pool: Final = ConcurrentNormGenerator(
            rows=sim.timesteps,
            cols=sim.total_paths(),
            seed=sim.mc_seed,
            buffer_size=sim.buffer_size,
            dtype=sim.cp_dtype,  # precision choice
        )

    def simulate(self, F0: float, model: ForwardProcess) -> NDArrayF:
        normals: NDArrayF = self._norm_pool.get_matrix()

        with self._cuda_stream.use():
            sigmas: NDArrayF = cp.full_like(normals, model.sigma(0, F0))

            dt = 1.0 / self.sp.timesteps
            _simulate_forward_kernel[
                self.sp.total_blocks(),
                self.sp.threads_per_block,
                self._cuda_stream,
            ](normals, sigmas, F0, dt, math.sqrt(dt))

        self._cuda_stream.synchronize()
        return normals


# ---------------------------------------------------------------------------
# Black-Scholes pricer
# ---------------------------------------------------------------------------


class BlackScholesPricer:
    # --------------------------- inputs ---------------------------------

    class Inputs(BaseModel):
        X0: PosFloat
        K: PosFloat
        T: NonNegFloat
        r: float
        d: float
        v: NonNegFloat

    # --------------- device-side per-path results -----------------------

    class PricingResults(BaseModel):
        model_config = dict(arbitrary_types_allowed=True)

        call_price_intrinsic: NDArrayF
        put_price_intrinsic: NDArrayF
        underlying: NDArrayF
        call_convexity: NDArrayF
        put_convexity: NDArrayF

    # ---------------- host-side scalar aggregates ----------------------

    class HostPricingResults(BaseModel):
        call_price_intrinsic: float
        put_price_intrinsic: float
        underlying: float
        call_convexity: float
        put_convexity: float
        call_price: float
        put_price: float

    # ------------------------------------------------------------------

    def __init__(self, sim: SimulationParams) -> None:
        self._sim: Final = sim
        self._cp_stream: Final = cp.cuda.Stream(non_blocking=True)
        self._engine: Final = MCForwardEngine(sim)

    # ------------------------------------------------------------------

    def _simulate_forward(self, inp: Inputs) -> NDArrayF:
        return self._engine.simulate(inp.X0, BlackScholesForward(inp.v))

    def price(
        self,
        inp: Inputs,
        forwards: NDArrayF | None = None,
    ) -> PricingResults:
        forwards = forwards or self._simulate_forward(inp)

        with self._cp_stream:
            F_T: NDArrayF = forwards[-1]
            S_T: NDArrayF = F_T * math.exp((inp.r - inp.d) * inp.T)
            df = math.exp(-inp.r * inp.T)

            call_price = df * cp.maximum(S_T - inp.K, 0.0)
            put_price = df * cp.maximum(inp.K - S_T, 0.0)

            F_exp = inp.X0 * math.exp((inp.r - inp.d) * inp.T)
            call_intr_val = df * max(F_exp - inp.K, 0.0)
            put_intr_val = df * max(inp.K - F_exp, 0.0)

            call_intr = cp.asarray(call_intr_val, dtype=self._sim.cp_dtype)
            put_intr = cp.asarray(put_intr_val, dtype=self._sim.cp_dtype)

            call_conv = call_price - call_intr
            put_conv = put_price - put_intr

            underlying = S_T.reshape(
                (self._sim.network_size, self._sim.batches_per_mc_run)
            )

        self._cp_stream.synchronize()
        return self.PricingResults(
            call_price_intrinsic=call_intr,
            put_price_intrinsic=put_intr,
            underlying=underlying,
            call_convexity=call_conv,
            put_convexity=put_conv,
        )

    def to_host(self, pr: PricingResults) -> HostPricingResults:
        with self._cp_stream:
            call_intr = float(pr.call_price_intrinsic.item())
            put_intr = float(pr.put_price_intrinsic.item())
            call_conv = float(pr.call_convexity.mean().item())
            put_conv = float(pr.put_convexity.mean().item())
            underlying = float(pr.underlying.mean().item())
        self._cp_stream.synchronize()

        return self.HostPricingResults(
            call_price_intrinsic=call_intr,
            put_price_intrinsic=put_intr,
            underlying=underlying,
            call_convexity=call_conv,
            put_convexity=put_conv,
            call_price=call_intr + call_conv,
            put_price=put_intr + put_conv,
        )
