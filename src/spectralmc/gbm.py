# src/spectralmc/gbm.py
"""
GPU-accelerated Monte-Carlo engine for a *Geometric Brownian Motion*
under Black-Scholes assumptions.

Key features
------------
* **Pure-GPU** workflow - Sobol/Box-Muller normal generation, Numba-CUDA
  path simulation and CuPy reductions never leave the device.
* **Deterministic snapshots** - :class:`BlackScholesConfig` contains
  *every* mutable degree of freedom so a simulation can be resumed
  bit-exactly on any host.  All snapshot tensors are forced onto the
  CPU to avoid device-specific artefacts.
* **Strict typing** - the project-wide :class:`spectralmc.models.numerical.Precision`
  replaces the old ``Literal["float32","float64"]`` helper.  The file is
  clean under ``mypy --strict`` with **zero** `Any`, casts or ignore
  comments.

Public API
----------
`BlackScholes`
    The single-GPU Monte-Carlo engine.
`BlackScholes.Inputs`
    Per-contract parameters (Pydantic model).
`BlackScholes.SimResults / PricingResults / HostPricingResults`
    Strongly-typed outputs returned by :meth:`_simulate`, :meth:`price`
    and :meth:`price_to_host`, respectively.
"""

from __future__ import annotations

from math import exp, sqrt
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import cupy as cp
from numba import cuda
from pydantic import BaseModel, ConfigDict, Field, model_validator


if TYPE_CHECKING:
    from numba.cuda import DeviceNDArray

from spectralmc.async_normals import (
    BufferConfig,
    ConcurrentNormGenerator,
    ConcurrentNormGeneratorConfig,
)
from spectralmc.effects import (
    EffectSequence,
    GenerateNormals,
    SimulatePaths,
    StreamSync,
    sequence_effects,
)
from spectralmc.models.numerical import Precision


# ──────────────────────────────── typing helpers ─────────────────────────────
PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]

# Valid CUDA thread block sizes (must be power of 2, range [32, 1024])
ThreadsPerBlock: TypeAlias = Literal[32, 64, 128, 256, 512, 1024]

# ───────────────────────── simulation-parameter schema ──────────────────────


class SimulationParams(BaseModel):
    """Immutable run-time parameters for one engine instance."""

    timesteps: int = Field(..., gt=0)
    network_size: int = Field(..., gt=0)
    batches_per_mc_run: int = Field(..., gt=0)
    threads_per_block: ThreadsPerBlock
    mc_seed: int = Field(..., gt=0)
    buffer_size: int = Field(..., gt=0)
    skip: int = Field(0, ge=0)
    dtype: Precision

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def validate_gpu_memory(self) -> SimulationParams:
        """
        Validate that GPU memory requirements are reasonable.

        The total memory footprint is roughly:
        network_size * batches_per_mc_run * timesteps * sizeof(dtype)

        This is a soft limit to catch configuration errors early.
        """
        total_paths = self.network_size * self.batches_per_mc_run
        # Rough estimate: 1 billion paths for float32, 500M for float64
        max_paths = 1_000_000_000 if self.dtype.value == "float32" else 500_000_000
        if total_paths > max_paths:
            raise ValueError(
                f"GPU memory limit exceeded: "
                f"{total_paths:,} paths > {max_paths:,} (network_size={self.network_size}, "
                f"batches_per_mc_run={self.batches_per_mc_run})"
            )
        return self

    # ..................................... convenience ......................
    # @property
    # def cp_dtype(self) -> cp.dtype:
    #    """Return the equivalent :class:`cupy.dtype`."""
    #    return cp.dtype(self.dtype.value)

    def total_paths(self) -> int:
        """Total number of simulated paths."""
        return self.network_size * self.batches_per_mc_run

    def total_blocks(self) -> int:
        """CUDA grid size for the Numba kernel."""
        return (self.total_paths() + self.threads_per_block - 1) // self.threads_per_block


# ────────────────────────────── engine configuration ────────────────────────


class BlackScholesConfig(BaseModel):
    """
    Complete, **frozen** configuration of a Monte-Carlo engine.

    Parameters
    ----------
    sim_params
        Numerical parameters & precision.
    simulate_log_return
        Use a log-Euler scheme (variance reduction).
    normalize_forwards
        Normalise each time-slice to the analytic forward (bias
        reduction).
    """

    sim_params: SimulationParams
    simulate_log_return: bool = True
    normalize_forwards: bool = True

    model_config = ConfigDict(frozen=True, extra="forbid")


# ─────────────────────────────── CUDA path kernel ───────────────────────────


@cuda.jit
def SimulateBlackScholes(
    io: DeviceNDArray,
    timesteps: int,
    dt: float,
    X0: float,
    r: float,
    d: float,
    v: float,
    simulate_log_return: bool,
) -> None:
    """Advance each path *in-place*.  One CUDA thread ≙ one path."""
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


# ─────────────────────────────── main engine ────────────────────────────────


class BlackScholes:
    """Single-GPU Monte-Carlo pricing engine."""

    # ....................................... nested Pydantic models .........
    class Inputs(BaseModel):
        """Parameters of one European option contract."""

        X0: PosFloat
        K: PosFloat
        T: NonNegFloat
        r: float
        d: float
        v: NonNegFloat

        model_config = ConfigDict(frozen=True, extra="forbid")

    class SimResults(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
        times: cp.ndarray
        sims: cp.ndarray
        forwards: cp.ndarray
        df: cp.ndarray

    class PricingResults(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
        put_price_intrinsic: cp.ndarray
        call_price_intrinsic: cp.ndarray
        underlying: cp.ndarray
        put_price: cp.ndarray
        call_price: cp.ndarray

    class HostPricingResults(BaseModel):
        """CPU copy of :class:`PricingResults` (scalars only)."""

        put_price_intrinsic: float
        call_price_intrinsic: float
        underlying: float
        put_convexity: float
        call_convexity: float
        put_price: float
        call_price: float

        model_config = ConfigDict(frozen=True, extra="forbid")

    # ....................................... construction ...................
    def __init__(self, cfg: BlackScholesConfig) -> None:
        self._cfg: BlackScholesConfig = cfg
        self._sp: SimulationParams = cfg.sim_params
        self._cp_dtype = self._sp.dtype.to_cupy()

        self._cp_stream = cp.cuda.Stream(non_blocking=True)
        self._numba_stream = cuda.stream()

        ngen_cfg = ConcurrentNormGeneratorConfig(
            rows=self._sp.timesteps,
            cols=self._sp.total_paths(),
            seed=self._sp.mc_seed,
            dtype=self._sp.dtype,
            skips=self._sp.skip,
        )
        buffer_cfg = BufferConfig.create(self._sp.buffer_size, ngen_cfg.rows, ngen_cfg.cols)
        self._ngen = ConcurrentNormGenerator(buffer_cfg, ngen_cfg)

    # ....................................... snapshot interface .............
    def snapshot(self) -> BlackScholesConfig:
        """Return a *deep* copy capturing current RNG skip offset."""
        sp = self._sp.model_copy(update={"skip": self._ngen.snapshot().skips}, deep=True)
        return BlackScholesConfig(
            sim_params=sp,
            simulate_log_return=self._cfg.simulate_log_return,
            normalize_forwards=self._cfg.normalize_forwards,
        )

    # ....................................... effect builders .................
    def build_simulation_effects(self, inputs: Inputs) -> EffectSequence[list[object]]:
        """Build pure effect sequence describing a complete simulation.

        This method produces an immutable effect description that can be:
        - Inspected and tested without GPU hardware
        - Serialized for reproducibility tracking
        - Composed with other effects in larger workflows

        The actual execution happens when the interpreter processes these effects.

        Args:
            inputs: Contract parameters (spot, strike, rate, dividend, vol, expiry).

        Returns:
            EffectSequence describing: normal generation → path simulation → stream sync.

        Example:
            >>> effects = bs.build_simulation_effects(inputs)
            >>> # Pure description - no side effects yet
            >>> result = await interpreter.interpret_sequence(effects)
        """
        ngen_snapshot = self._ngen.snapshot()
        return sequence_effects(
            GenerateNormals(
                rows=self._sp.timesteps,
                cols=self._sp.total_paths(),
                seed=ngen_snapshot.seed,
                skip=ngen_snapshot.skips,
            ),
            SimulatePaths(
                spot=inputs.X0,
                strike=inputs.K,
                rate=inputs.r,
                dividend=inputs.d,
                vol=inputs.v,
                expiry=inputs.T,
                timesteps=self._sp.timesteps,
                batches=self._sp.total_paths(),
                simulate_log_return=self._cfg.simulate_log_return,
                normalize_forwards=self._cfg.normalize_forwards,
                input_normals_id="generated_normals",
            ),
            StreamSync(stream_type="numba"),
            StreamSync(stream_type="cupy"),
        )

    # ....................................... internals ......................
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
            df = cp.exp(-inputs.r * times)

        # optional forward normalisation
        self._numba_stream.synchronize()
        if self._cfg.normalize_forwards:
            row_means = cp.mean(sims, axis=1, keepdims=True).squeeze()
            sims *= cp.expand_dims(forwards / row_means, 1)

        self._cp_stream.synchronize()
        return self.SimResults(times=times, sims=sims, forwards=forwards, df=df)

    # ....................................... pricing .......................
    def price(self, *, inputs: Inputs, sr: SimResults | None = None) -> PricingResults:
        """Return CuPy tensors with full MC results (no CPU sync)."""
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

    # ....................................... host helpers ...................
    def get_host_price(self, pr: PricingResults) -> HostPricingResults:
        """Synchronise & return scalar host prices."""
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

    def price_to_host(self, inputs: Inputs) -> HostPricingResults:
        """Convenience wrapper - GPU price → CPU scalars."""
        return self.get_host_price(self.price(inputs=inputs))


__all__: tuple[str, ...] = (
    "BlackScholes",
    "BlackScholesConfig",
    "SimulationParams",
)
