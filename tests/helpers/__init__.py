# tests/helpers/__init__.py
"""Shared test utilities for SpectralMC test suite.

This package provides DRY helpers for Result unwrapping, config factories,
assertions, and constants. Consolidates patterns duplicated across 15+ test files.

See testing_architecture.md for usage patterns and design rationale.

Usage:
    >>> from tests.helpers import expect_success, make_gbm_cvnn_config
    >>> from tests.helpers import DEFAULT_TIMESTEPS, RTOL_FLOAT32
    >>> from tests.helpers import assert_tensors_close, assert_no_nan_inf
    >>>
    >>> # Result unwrapping
    >>> params = expect_success(build_simulation_params(...))
    >>>
    >>> # Config factory
    >>> model = torch.nn.Linear(5, 5)
    >>> config = make_gbm_cvnn_config(model, global_step=100)
    >>>
    >>> # Enhanced assertions
    >>> assert_tensors_close(actual, expected, rtol=RTOL_FLOAT32)
"""

from __future__ import annotations

from tests.helpers.assertions import (
    assert_converged,
    assert_no_nan_inf,
    assert_tensors_close,
)
from tests.helpers.constants import (
    ATOL_FLOAT32,
    ATOL_FLOAT64,
    CONVERGENCE_THRESHOLD,
    DEFAULT_BATCHES_PER_RUN,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MC_SEED,
    DEFAULT_NETWORK_SIZE,
    DEFAULT_THREADS_PER_BLOCK,
    DEFAULT_TIMESTEPS,
    LARGE_MODEL_SIZE,
    MEDIUM_MODEL_SIZE,
    RTOL_FLOAT32,
    RTOL_FLOAT64,
    SMALL_MODEL_SIZE,
)
from spectralmc.gbm import ThreadsPerBlock
from spectralmc.testing import seed_all_rngs
from tests.helpers.factories import (
    make_domain_bounds,
    make_black_scholes_config,
    make_gbm_cvnn_config,
    make_simulation_params,
    make_test_cvnn,
    make_training_config,
    max_param_diff,
)
from tests.helpers.result_utils import E, T, expect_failure, expect_success

__all__ = [
    # Result unwrapping
    "expect_success",
    "expect_failure",
    "T",
    "E",
    # Config factories
    "make_simulation_params",
    "make_black_scholes_config",
    "make_gbm_cvnn_config",
    "make_test_cvnn",
    "make_domain_bounds",
    "make_training_config",
    "max_param_diff",
    "seed_all_rngs",
    "ThreadsPerBlock",
    # Assertions
    "assert_tensors_close",
    "assert_no_nan_inf",
    "assert_converged",
    # Simulation constants
    "DEFAULT_TIMESTEPS",
    "DEFAULT_NETWORK_SIZE",
    "DEFAULT_BATCHES_PER_RUN",
    "DEFAULT_THREADS_PER_BLOCK",
    "DEFAULT_MC_SEED",
    "DEFAULT_BUFFER_SIZE",
    # Tolerance constants
    "RTOL_FLOAT32",
    "RTOL_FLOAT64",
    "ATOL_FLOAT32",
    "ATOL_FLOAT64",
    # Training constants
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MAX_ITERATIONS",
    "CONVERGENCE_THRESHOLD",
    # Model size constants
    "SMALL_MODEL_SIZE",
    "MEDIUM_MODEL_SIZE",
    "LARGE_MODEL_SIZE",
]
