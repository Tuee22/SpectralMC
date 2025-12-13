# tests/helpers/factories.py
"""Test data factories for SpectralMC tests.

This module consolidates config builder patterns duplicated across 6+ test files.
Provides reusable factories for simulation parameters, BlackScholes configs,
and GbmCVNNPricerConfig instances.
"""

from __future__ import annotations

from typing import Literal

from spectralmc.effects import ForwardNormalization, PathScheme
from spectralmc.gbm import (
    BlackScholesConfig,
    SimulationParams,
    build_black_scholes_config,
    build_simulation_params,
)
from spectralmc.gbm_trainer import GbmCVNNPricerConfig
from spectralmc.models.numerical import Precision
import torch

from tests.helpers.constants import (
    DEFAULT_BATCHES_PER_RUN,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_MC_SEED,
    DEFAULT_NETWORK_SIZE,
    DEFAULT_THREADS_PER_BLOCK,
    DEFAULT_TIMESTEPS,
)
from tests.helpers.result_utils import expect_success

ThreadsPerBlock = Literal[32, 64, 128, 256, 512, 1024]


def make_simulation_params(
    timesteps: int = DEFAULT_TIMESTEPS,
    network_size: int = DEFAULT_NETWORK_SIZE,
    batches_per_mc_run: int = DEFAULT_BATCHES_PER_RUN,
    threads_per_block: ThreadsPerBlock = DEFAULT_THREADS_PER_BLOCK,
    mc_seed: int = DEFAULT_MC_SEED,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    skip: int = 0,
    dtype: Precision = Precision.float32,
) -> SimulationParams:
    """Create SimulationParams with defaults for testing.

    Consolidates the build_simulation_params boilerplate repeated across
    multiple test files. Uses DEFAULT_* constants for common scenarios.

    Args:
        timesteps: Number of timesteps (default: 100)
        network_size: Neural network hidden layer size (default: 1024)
        batches_per_mc_run: Batches per MC run (default: 8)
        threads_per_block: CUDA threads per block (default: 256)
        mc_seed: Random seed for reproducibility (default: 42)
        buffer_size: Async buffer size (default: 10000)
        skip: Sobol sequence skip (default: 0)
        dtype: Numerical precision (default: float32)

    Returns:
        Validated SimulationParams instance

    Raises:
        AssertionError: If build_simulation_params returns Failure

    Example:
        >>> params = make_simulation_params(timesteps=200, network_size=512)
        >>> assert params.timesteps == 200
    """
    return expect_success(
        build_simulation_params(
            timesteps=timesteps,
            network_size=network_size,
            batches_per_mc_run=batches_per_mc_run,
            threads_per_block=threads_per_block,
            mc_seed=mc_seed,
            buffer_size=buffer_size,
            skip=skip,
            dtype=dtype,
        )
    )


def make_black_scholes_config(
    sim_params: SimulationParams | None = None,
    path_scheme: PathScheme = PathScheme.LOG_EULER,
    normalization: ForwardNormalization = ForwardNormalization.NORMALIZE,
) -> BlackScholesConfig:
    """Create BlackScholesConfig with defaults for testing.

    Consolidates the build_black_scholes_config boilerplate. Creates default
    SimulationParams if none provided.

    Args:
        sim_params: Simulation parameters (creates default if None)
        path_scheme: Path computation scheme (default: log-Euler)
        normalization: Forward normalization intent (default: normalize)

    Returns:
        Validated BlackScholesConfig instance

    Raises:
        AssertionError: If build_black_scholes_config returns Failure

    Example:
        >>> config = make_black_scholes_config()
        >>> assert config.path_scheme is PathScheme.LOG_EULER
    """
    if sim_params is None:
        sim_params = make_simulation_params()

    return expect_success(
        build_black_scholes_config(
            sim_params=sim_params,
            path_scheme=path_scheme,
            normalization=normalization,
        )
    )


def make_gbm_cvnn_config(
    model: torch.nn.Module,
    global_step: int = 0,
    sim_params: SimulationParams | None = None,
    bs_config: BlackScholesConfig | None = None,
) -> GbmCVNNPricerConfig:
    """Create GbmCVNNPricerConfig for testing.

    Consolidates the config builder pattern duplicated in:
    - tests/test_storage/test_inference_client.py
    - tests/test_storage/test_e2e_storage.py
    - tests/test_storage/test_training_integration.py
    - tests/test_integrity/test_blockchain_integrity.py
    - And others

    Creates default SimulationParams and BlackScholesConfig if not provided.
    Captures current CPU RNG state for reproducibility.

    Args:
        model: CVNN model (typically torch.nn.Linear for tests)
        global_step: Training step counter (default: 0)
        sim_params: Optional simulation parameters (creates default if None)
        bs_config: Optional BlackScholes config (creates default if None)

    Returns:
        Frozen GbmCVNNPricerConfig ready for testing

    Example:
        >>> model = torch.nn.Linear(5, 5)
        >>> config = make_gbm_cvnn_config(model, global_step=100)
        >>> assert config.global_step == 100
        >>> assert config.cvnn is model
    """
    if bs_config is None:
        bs_config = make_black_scholes_config(sim_params)

    cpu_rng_state = torch.get_rng_state().numpy().tobytes()

    return GbmCVNNPricerConfig(
        cfg=bs_config,
        domain_bounds={},
        cvnn=model,
        optimizer_state=None,
        global_step=global_step,
        sobol_skip=0,
        torch_cpu_rng_state=cpu_rng_state,
        torch_cuda_rng_states=[],
    )
