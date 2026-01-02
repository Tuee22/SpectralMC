# tests/helpers/factories.py
"""Test data factories for SpectralMC tests.

This module consolidates config builder patterns duplicated across 6+ test files.
Provides reusable factories for simulation parameters, BlackScholes configs,
and GbmCVNNPricerConfig instances.
"""

from __future__ import annotations

import random

import numpy as np
import torch

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
from spectralmc.gbm import (
    BlackScholes,
    BlackScholesConfig,
    SimulationParams,
    ThreadsPerBlock,
    build_black_scholes_config,
    build_simulation_params,
)
from spectralmc.gbm_trainer import (
    ComplexValuedModel,
    GbmCVNNPricerConfig,
    TrainingConfig,
    build_training_config,
)
from spectralmc.models.torch import AdamOptimizerState, Device, FullPrecisionDType
from spectralmc.models.numerical import Precision
from spectralmc.result import Failure, Success
from spectralmc.sobol_sampler import (
    BoundSpec,
    DomainBounds,
    build_bound_spec,
    build_domain_bounds,
)
from spectralmc.validation import validate_model

from tests.helpers.constants import (
    DEFAULT_BATCHES_PER_RUN,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_MC_SEED,
    DEFAULT_NETWORK_SIZE,
    DEFAULT_THREADS_PER_BLOCK,
    DEFAULT_TIMESTEPS,
)
from tests.helpers.result_utils import expect_success


def seed_all_rngs(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for deterministic tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def make_domain_bounds(
    *,
    x0: tuple[float, float] = (0.001, 10_000.0),
    k: tuple[float, float] = (0.001, 20_000.0),
    t: tuple[float, float] = (0.0, 10.0),
    r: tuple[float, float] = (-0.20, 0.20),
    d: tuple[float, float] = (-0.20, 0.20),
    v: tuple[float, float] = (0.0, 2.0),
) -> DomainBounds[BlackScholes.Inputs]:
    """Provide default Sobol bounds for Black-Scholes inputs.

    Creates validated domain bounds for all 6 Black-Scholes input parameters.
    Tests can override specific bounds while keeping others as defaults.

    Args:
        x0: Spot price bounds (default: 0.001 to 10,000)
        k: Strike price bounds (default: 0.001 to 20,000)
        t: Time to maturity bounds (default: 0 to 10 years)
        r: Risk-free rate bounds (default: -20% to 20%)
        d: Dividend yield bounds (default: -20% to 20%)
        v: Volatility bounds (default: 0 to 200%)

    Returns:
        Validated DomainBounds instance

    Raises:
        AssertionError: If any bound specification is invalid

    Example:
        >>> bounds = make_domain_bounds(v=(0.1, 0.5))  # Override volatility
        >>> assert bounds["v"].lower == 0.1
    """

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
    resolved_sim_params = sim_params or make_simulation_params()
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
    resolved_sim_params = sim_params or make_simulation_params()
    resolved_bs_config = bs_config or make_black_scholes_config(sim_params=resolved_sim_params)

    # Capture current RNG state for deterministic testing
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
        domain_bounds=domain_bounds or make_domain_bounds(),
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
