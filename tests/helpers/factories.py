# tests/helpers/factories.py
"""Test data factories for SpectralMC tests.

This module consolidates config builder patterns duplicated across 6+ test files.
Provides reusable factories for simulation parameters, BlackScholes configs,
and GbmCVNNPricerConfig instances.
"""

from __future__ import annotations

from typing import Literal

from spectralmc.effects import ForwardNormalization, PathScheme
from spectralmc.cvnn_factory import (
    ActivationCfg,
    ActivationKind,
    ExplicitWidth,
    LayerCfg,
    LinearCfg,
    build_cvnn_config,
    build_model,
)
from spectralmc.gbm import BlackScholes, BlackScholesConfig, SimulationParams
from spectralmc.gbm_trainer import (
    ComplexValuedModel,
    GbmCVNNPricerConfig,
    TrainingConfig,
    build_training_config,
)
from spectralmc.models.torch import AdamOptimizerState
from spectralmc.models.numerical import Precision
from spectralmc.models.torch import Device, FullPrecisionDType
from spectralmc.sobol_sampler import DomainBounds
from spectralmc.testing import (
    default_domain_bounds,
    make_gbm_cvnn_config as _make_core_gbm_cvnn_config,
    make_test_black_scholes_config,
    make_test_simulation_params,
)
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


def make_test_cvnn(
    *,
    n_inputs: int,
    n_outputs: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device | str = Device.cuda.to_torch(),
    hidden_width: int = 32,
    add_output_layer: bool = True,
) -> ComplexValuedModel:
    """Create a deterministic CVNN for tests (DRY wrapper around build_model).

    Args:
        n_inputs: Number of input features
        n_outputs: Number of output features
        seed: RNG seed for deterministic weights
        dtype: Torch dtype (float32/float64)
        device: Target device (default: cuda:0)
        hidden_width: Width of the hidden layer (default: 32)
        add_output_layer: Whether to append an explicit output Linear layer

    Returns:
        CVNN on the requested device/dtype
    """
    enum_dtype = expect_success(FullPrecisionDType.from_torch(dtype))
    layers: list[LayerCfg] = [
        LinearCfg(
            width=ExplicitWidth(value=hidden_width),
            activation=ActivationCfg(kind=ActivationKind.MOD_RELU),
        ),
    ]
    if add_output_layer:
        layers.append(LinearCfg(width=ExplicitWidth(value=n_outputs)))

    cfg = expect_success(build_cvnn_config(dtype=enum_dtype, layers=layers, seed=seed))
    model = expect_success(build_model(n_inputs=n_inputs, n_outputs=n_outputs, cfg=cfg))
    return model.to(device, dtype)


def make_domain_bounds() -> DomainBounds[BlackScholes.Inputs]:
    """Create default Black-Scholes domain bounds for tests."""
    return default_domain_bounds()


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
    return make_test_simulation_params(
        timesteps=timesteps,
        network_size=network_size,
        batches_per_mc_run=batches_per_mc_run,
        threads_per_block=threads_per_block,
        mc_seed=mc_seed,
        buffer_size=buffer_size,
        skip=skip,
        dtype=dtype,
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
    return make_test_black_scholes_config(
        sim_params=sim_params,
        path_scheme=path_scheme,
        normalization=normalization,
    )


def make_gbm_cvnn_config(
    model: ComplexValuedModel,
    global_step: int = 0,
    sim_params: SimulationParams | None = None,
    bs_config: BlackScholesConfig | None = None,
    domain_bounds: DomainBounds[BlackScholes.Inputs] | None = None,
    sobol_skip: int = 0,
    optimizer_state: AdamOptimizerState | None = None,
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
    return _make_core_gbm_cvnn_config(
        model,
        global_step=global_step,
        sim_params=sim_params,
        bs_config=bs_config,
        domain_bounds=domain_bounds or default_domain_bounds(),
        sobol_skip=sobol_skip,
        optimizer_state=optimizer_state,
    )


def make_training_config(
    *, num_batches: int, batch_size: int, learning_rate: float = 1.0e-2
) -> TrainingConfig:
    """Create a small TrainingConfig for deterministic smoke tests."""
    return expect_success(
        build_training_config(
            num_batches=num_batches,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    )


def max_param_diff(a: ComplexValuedModel, b: ComplexValuedModel) -> float:
    """Return the L∞-norm between two models' parameters."""
    return max(
        (
            float(torch.abs(pa - pb).max().item())
            for pa, pb in zip(a.parameters(), b.parameters(), strict=True)
        ),
        default=0.0,
    )
