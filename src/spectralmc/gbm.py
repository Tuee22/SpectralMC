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
from typing import Annotated, Literal, TypeAlias

import cupy as cp
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from pydantic import BaseModel, ConfigDict, Field

from spectralmc.async_normals import (
    BufferConfig,
    ConcurrentNormGenerator,
    ConcurrentNormGeneratorConfig,
)
from spectralmc.effects import (
    EffectSequence,
    ForwardNormalization,
    GenerateNormals,
    PathScheme,
    SimulatePaths,
    StreamSync,
    sequence_effects,
)
from spectralmc.errors.gbm import (
    GPUMemoryLimitExceeded,
    InvalidBlackScholesConfig,
    InvalidSimulationParams,
    NormalsGenerationFailed,
    NormalsUnavailable,
)
from spectralmc.models.numerical import Precision
from spectralmc.result import Failure, Result, Success
from spectralmc.validation import validate_model


# ──────────────────────────────── typing helpers ─────────────────────────────
PosFloat = Annotated[float, Field(gt=0)]
NonNegFloat = Annotated[float, Field(ge=0)]

# Valid CUDA thread block sizes (must be power of 2, range [32, 1024])
ThreadsPerBlock: TypeAlias = Literal[32, 64, 128, 256, 512, 1024]
NormalsError: TypeAlias = NormalsUnavailable | NormalsGenerationFailed

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


def validate_simulation_params_memory(
    params: SimulationParams,
) -> Result[SimulationParams, GPUMemoryLimitExceeded]:
    """
    Validate GPU memory requirements for simulation parameters.

    The total memory footprint is roughly:
    network_size * batches_per_mc_run * timesteps * sizeof(dtype)

    This is a soft limit to catch configuration errors early.

    Args:
        params: Parameters to validate

    Returns:
        Success(params) if memory within limits, else Failure(GPUMemoryLimitExceeded)
    """
    total_paths = params.network_size * params.batches_per_mc_run
    max_paths = 1_000_000_000 if params.dtype.value == "float32" else 500_000_000

    return (
        Failure(
            GPUMemoryLimitExceeded(
                total_paths=total_paths,
                max_paths=max_paths,
                network_size=params.network_size,
                batches_per_mc_run=params.batches_per_mc_run,
            )
        )
        if total_paths > max_paths
        else Success(params)
    )


# ────────────────────────────── engine configuration ────────────────────────


class BlackScholesConfig(BaseModel):
    """
    Complete, **frozen** configuration of a Monte-Carlo engine.

    Parameters
    ----------
    sim_params
        Numerical parameters & precision.
    path_scheme
        Explicit path scheme (log-Euler vs simple Euler).
    normalization
        Forward normalization intent (normalize vs raw paths).
    """

    sim_params: SimulationParams
    path_scheme: PathScheme = PathScheme.LOG_EULER
    normalization: ForwardNormalization = ForwardNormalization.NORMALIZE

    model_config = ConfigDict(frozen=True, extra="forbid")


def build_simulation_params(
    *,
    timesteps: int,
    network_size: int,
    batches_per_mc_run: int,
    threads_per_block: ThreadsPerBlock,
    mc_seed: int,
    buffer_size: int,
    dtype: Precision,
    skip: int = 0,
) -> Result[SimulationParams, InvalidSimulationParams | GPUMemoryLimitExceeded]:
    """Create SimulationParams via pure validation."""
    params_result = validate_model(
        SimulationParams,
        timesteps=timesteps,
        network_size=network_size,
        batches_per_mc_run=batches_per_mc_run,
        threads_per_block=threads_per_block,
        mc_seed=mc_seed,
        buffer_size=buffer_size,
        skip=skip,
        dtype=dtype,
    )
    match params_result:
        case Failure(error):
            return Failure(InvalidSimulationParams(error=error))
        case Success(params):
            # Chain additional validation after Pydantic checks pass
            memory_result = validate_simulation_params_memory(params)
            match memory_result:
                case Failure(gpu_error):
                    return Failure(gpu_error)
                case Success(validated_params):
                    return Success(validated_params)


def build_black_scholes_config(
    *,
    sim_params: SimulationParams,
    path_scheme: PathScheme = PathScheme.LOG_EULER,
    normalization: ForwardNormalization = ForwardNormalization.NORMALIZE,
) -> Result[BlackScholesConfig, InvalidBlackScholesConfig]:
    """Construct a validated BlackScholesConfig."""
    config_result = validate_model(
        BlackScholesConfig,
        sim_params=sim_params,
        path_scheme=path_scheme,
        normalization=normalization,
    )
    match config_result:
        case Failure(error):
            return Failure(InvalidBlackScholesConfig(error=error))
        case Success(config):
            return Success(config)


# ─────────────────────────────── CUDA path kernel ───────────────────────────


# TIER 3 BOUNDARY: CUDA kernel - imperative patterns acceptable for GPU efficiency
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
    """Advance each path *in-place*.  One CUDA thread ≙ one path.

    NOTE: This kernel uses imperative patterns (if statements, for loops) which are
    acceptable at the GPU compute boundary (Tier 3). Purity rules apply to the
    calling code, not kernel internals.
    """
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

        ngen_cfg_result = ConcurrentNormGeneratorConfig.create(
            rows=self._sp.timesteps,
            cols=self._sp.total_paths(),
            seed=self._sp.mc_seed,
            dtype=self._sp.dtype,
            skips=self._sp.skip,
        )
        match ngen_cfg_result:
            case Failure(err):
                raise AssertionError(f"Invalid norm generator config: {err}")
            case Success(ngen_cfg):
                pass
        buffer_cfg = BufferConfig.create(self._sp.buffer_size, ngen_cfg.rows, ngen_cfg.cols)
        self._ngen_result = ConcurrentNormGenerator.create(buffer_cfg, ngen_cfg)

    # ....................................... snapshot interface .............
    def snapshot(self) -> Result[BlackScholesConfig, NormalsUnavailable]:
        """Return a *deep* copy capturing current RNG skip offset."""
        match self._ngen_result:
            case Failure(error):
                return Failure(NormalsUnavailable(error=error))
            case Success(gen):
                sp = self._sp.model_copy(update={"skip": gen.snapshot().skips}, deep=True)
                return Success(self._cfg.model_copy(update={"sim_params": sp}, deep=True))

    # ....................................... effect builders .................
    def build_simulation_effects(
        self, inputs: Inputs
    ) -> Result[EffectSequence[list[object]], NormalsUnavailable]:
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
            >>> effects_result = bs.build_simulation_effects(inputs)
            >>> match effects_result:
            ...     case Success(effects):
            ...         result = await interpreter.interpret_sequence(effects)
            ...     case Failure(error):
            ...         ...
        """
        match self._ngen_result:
            case Failure(error):
                return Failure(NormalsUnavailable(error=error))
            case Success(gen):
                ngen_snapshot = gen.snapshot()
                return Success(
                    sequence_effects(
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
                            path_scheme=self._cfg.path_scheme,
                            normalization=self._cfg.normalization,
                            input_normals_id="generated_normals",
                        ),
                        StreamSync(stream_type="numba"),
                        StreamSync(stream_type="cupy"),
                    )
                )

    # ....................................... internals ......................
    def _simulate(self, inputs: Inputs) -> Result[SimResults, NormalsError]:
        match self._ngen_result:
            case Failure(ngen_error):
                return Failure(NormalsUnavailable(error=ngen_error))
            case Success(gen):
                sims_result = gen.get_matrix()
                match sims_result:
                    case Success(sims):
                        pass
                    case Failure(matrix_error):
                        return Failure(NormalsGenerationFailed(error=matrix_error))
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
            self._cfg.path_scheme is PathScheme.LOG_EULER,
        )

        with self._cp_stream:
            times = cp.linspace(dt, inputs.T, self._sp.timesteps, dtype=self._cp_dtype)
            forwards = inputs.X0 * cp.exp((inputs.r - inputs.d) * times)
            df = cp.exp(-inputs.r * times)

        # optional forward normalisation
        self._numba_stream.synchronize()
        match self._cfg.normalization:
            case ForwardNormalization.NORMALIZE:
                row_means = cp.mean(sims, axis=1, keepdims=True).squeeze()
                sims *= cp.expand_dims(forwards / row_means, 1)
            case ForwardNormalization.RAW:
                pass

        self._cp_stream.synchronize()
        match validate_model(self.SimResults, times=times, sims=sims, forwards=forwards, df=df):
            case Success(sr):
                return Success(sr)
            case Failure(err):
                raise AssertionError(f"SimResults validation failed: {err}")

    # ....................................... pricing .......................
    def price(
        self,
        *,
        inputs: Inputs,
        sr_result: Result["BlackScholes.SimResults", NormalsError] | None = None,
    ) -> Result[PricingResults, NormalsError]:
        """Return CuPy tensors with full MC results (no CPU sync)."""
        sim_result: Result["BlackScholes.SimResults", NormalsError] = sr_result or self._simulate(
            inputs
        )
        match sim_result:
            case Failure(error):
                return Failure(error)
            case Success(sr):
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
                match validate_model(
                    self.PricingResults,
                    put_price_intrinsic=put_intr,
                    call_price_intrinsic=call_intr,
                    underlying=terminal,
                    put_price=put_price,
                    call_price=call_price,
                ):
                    case Success(pr):
                        return Success(pr)
                    case Failure(err):
                        raise AssertionError(f"PricingResults validation failed: {err}")

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
        match validate_model(
            self.HostPricingResults,
            put_price_intrinsic=put_intr,
            call_price_intrinsic=call_intr,
            underlying=underlying,
            put_convexity=put_price - put_intr,
            call_convexity=call_price - call_intr,
            put_price=put_price,
            call_price=call_price,
        ):
            case Success(host):
                return host
            case Failure(err):
                raise AssertionError(f"HostPricingResults validation failed: {err}")

    def price_to_host(self, inputs: Inputs) -> Result[HostPricingResults, NormalsError]:
        """Convenience wrapper - GPU price → CPU scalars."""
        match self.price(inputs=inputs):
            case Failure(error):
                return Failure(error)
            case Success(pricing):
                return Success(self.get_host_price(pricing))


__all__: tuple[str, ...] = (
    "BlackScholes",
    "BlackScholesConfig",
    "SimulationParams",
    "build_black_scholes_config",
    "build_simulation_params",
)
