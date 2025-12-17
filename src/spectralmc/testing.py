"""Testing utilities for SpectralMC."""

from __future__ import annotations

import random
from typing import Final

import numpy as np
import torch

from spectralmc.effects import ForwardNormalization, PathScheme
from spectralmc.gbm import (
    BlackScholes,
    BlackScholesConfig,
    SimulationParams,
    ThreadsPerBlock,
    build_black_scholes_config,
    build_simulation_params,
)
from spectralmc.gbm_trainer import ComplexValuedModel, GbmCVNNPricerConfig
from spectralmc.models.torch import AdamOptimizerState
from spectralmc.models.numerical import Precision
from spectralmc.result import Failure, Success
from spectralmc.sobol_sampler import (
    BoundSpec,
    DomainBounds,
    build_bound_spec,
    build_domain_bounds,
)
from spectralmc.validation import validate_model

_DEFAULT_TIMESTEPS: Final[int] = 100
_DEFAULT_NETWORK_SIZE: Final[int] = 1024
_DEFAULT_BATCHES_PER_RUN: Final[int] = 8
_DEFAULT_THREADS_PER_BLOCK: Final[ThreadsPerBlock] = 256
_DEFAULT_MC_SEED: Final[int] = 42
_DEFAULT_BUFFER_SIZE: Final[int] = 10_000
_DEFAULT_SKIP: Final[int] = 0
_DEFAULT_PRECISION: Final[Precision] = Precision.float32


def seed_all_rngs(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for deterministic tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_domain_bounds(
    *,
    x0: tuple[float, float] = (0.001, 10_000.0),
    k: tuple[float, float] = (0.001, 20_000.0),
    t: tuple[float, float] = (0.0, 10.0),
    r: tuple[float, float] = (-0.20, 0.20),
    d: tuple[float, float] = (-0.20, 0.20),
    v: tuple[float, float] = (0.0, 2.0),
) -> DomainBounds[BlackScholes.Inputs]:
    """Provide default Sobol bounds for Black-Scholes inputs."""

    def _bound(name: str, bounds: tuple[float, float]) -> BoundSpec:
        lower, upper = bounds
        match build_bound_spec(lower=lower, upper=upper):
            case Success(spec):
                return spec
            case Failure(error):
                raise AssertionError(f"Invalid bounds for {name}: {error}")

    bound_map = {
        "X0": _bound("X0", x0),
        "K": _bound("K", k),
        "T": _bound("T", t),
        "r": _bound("r", r),
        "d": _bound("d", d),
        "v": _bound("v", v),
    }
    match build_domain_bounds(BlackScholes.Inputs, bound_map):
        case Success(domain_bounds):
            return domain_bounds
        case Failure(error):
            raise AssertionError(f"Invalid domain bounds: {error}")


def make_test_simulation_params(
    *,
    timesteps: int = _DEFAULT_TIMESTEPS,
    network_size: int = _DEFAULT_NETWORK_SIZE,
    batches_per_mc_run: int = _DEFAULT_BATCHES_PER_RUN,
    threads_per_block: ThreadsPerBlock = _DEFAULT_THREADS_PER_BLOCK,
    mc_seed: int = _DEFAULT_MC_SEED,
    buffer_size: int = _DEFAULT_BUFFER_SIZE,
    skip: int = _DEFAULT_SKIP,
    dtype: Precision = _DEFAULT_PRECISION,
) -> SimulationParams:
    """Create validated SimulationParams with safe defaults for tests."""
    match build_simulation_params(
        timesteps=timesteps,
        network_size=network_size,
        batches_per_mc_run=batches_per_mc_run,
        threads_per_block=threads_per_block,
        mc_seed=mc_seed,
        buffer_size=buffer_size,
        skip=skip,
        dtype=dtype,
    ):
        case Success(params):
            return params
        case Failure(error):
            raise AssertionError(f"Failed to build SimulationParams: {error}")


def make_test_black_scholes_config(
    *,
    sim_params: SimulationParams | None = None,
    path_scheme: PathScheme = PathScheme.LOG_EULER,
    normalization: ForwardNormalization = ForwardNormalization.NORMALIZE,
) -> BlackScholesConfig:
    """Build a validated BlackScholesConfig with default simulation params."""
    resolved_sim_params = sim_params or make_test_simulation_params()
    match build_black_scholes_config(
        sim_params=resolved_sim_params,
        path_scheme=path_scheme,
        normalization=normalization,
    ):
        case Success(config):
            return config
        case Failure(error):
            raise AssertionError(f"Failed to build BlackScholesConfig: {error}")


def make_gbm_cvnn_config(
    model: ComplexValuedModel,
    *,
    global_step: int = 0,
    sim_params: SimulationParams | None = None,
    bs_config: BlackScholesConfig | None = None,
    domain_bounds: DomainBounds[BlackScholes.Inputs] | None = None,
    sobol_skip: int = 0,
    optimizer_state: AdamOptimizerState | None = None,
) -> GbmCVNNPricerConfig:
    """Create a deterministic GbmCVNNPricerConfig for testing."""
    resolved_sim_params = sim_params or make_test_simulation_params()
    resolved_bs_config = bs_config or make_test_black_scholes_config(sim_params=resolved_sim_params)
    cpu_rng_state = torch.get_rng_state().numpy().tobytes()
    cuda_states: list[bytes] = (
        [
            torch.cuda.get_rng_state(device=device).numpy().tobytes()
            for device in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else []
    )

    cfg_result = validate_model(
        GbmCVNNPricerConfig,
        cfg=resolved_bs_config,
        domain_bounds=domain_bounds or default_domain_bounds(),
        cvnn=model,
        optimizer_state=optimizer_state,
        global_step=global_step,
        sobol_skip=sobol_skip,
        torch_cpu_rng_state=cpu_rng_state,
        torch_cuda_rng_states=cuda_states,
    )
    match cfg_result:
        case Success(cfg):
            return cfg
        case Failure(error):
            raise AssertionError(f"Failed to build GbmCVNNPricerConfig: {error}")
