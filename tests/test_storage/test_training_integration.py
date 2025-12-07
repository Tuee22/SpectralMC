# tests/test_storage/test_training_integration.py
"""Integration tests for GbmCVNNPricer training with blockchain storage.

All tests require GPU - missing GPU is a hard failure, not a skip.
"""

from __future__ import annotations

from typing import TypeVar, Union

import pytest
import torch

from spectralmc.cvnn_factory import (
    ActivationCfg,
    ActivationKind,
    CVNNConfig,
    ExplicitWidth,
    LinearCfg,
    build_model,
)
from spectralmc.gbm import build_black_scholes_config, build_simulation_params
from spectralmc.gbm_trainer import (
    ComplexValuedModel,
    GbmCVNNPricer,
    GbmCVNNPricerConfig,
    TrainingConfig,
)
from spectralmc.models.numerical import Precision
from spectralmc.models.torch import DType as TorchDTypeEnum
from spectralmc.result import Failure, Result, Success
from spectralmc.sobol_sampler import BoundSpec
from spectralmc.storage import AsyncBlockchainModelStore, commit_snapshot

T = TypeVar("T")


def _expect_success(result: Result[T, object]) -> T:
    match result:
        case Success(value):
            return value
        case Failure(error):
            raise AssertionError(f"Unexpected CVNN factory failure: {error}")


# Module-level GPU requirement - test file fails immediately without GPU
assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"


def _make_test_cvnn(
    n_inputs: int,
    n_outputs: int,
    *,
    seed: int,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> ComplexValuedModel:
    """Create a simple CVNN for testing (matches test_gbm_trainer.py pattern)."""
    enum_dtype = TorchDTypeEnum.from_torch(dtype)
    cfg = CVNNConfig(
        dtype=enum_dtype,
        layers=[
            LinearCfg(
                width=ExplicitWidth(value=32),
                activation=ActivationCfg(kind=ActivationKind.MOD_RELU),
            ),
        ],
        seed=seed,
    )
    return _expect_success(build_model(n_inputs=n_inputs, n_outputs=n_outputs, cfg=cfg)).to(
        device, dtype
    )


def make_test_config(model: ComplexValuedModel, global_step: int = 0) -> GbmCVNNPricerConfig:
    """Factory to create test configurations."""
    match build_simulation_params(
        timesteps=10,  # Reduced from 100: match test_e2e_storage.py pattern
        network_size=128,  # Reduced from 1024: sufficient for CVNN operation
        batches_per_mc_run=2,  # Reduced from 8: minimum for mean calculation
        threads_per_block=256,
        mc_seed=42,
        buffer_size=256,  # Reduced from 10000: conservative async buffer (was 3.82 GB, now 1.26 MB)
        skip=0,
        dtype=Precision.float32,
    ):
        case Failure(err):
            raise AssertionError(f"SimulationParams creation failed: {err}")
        case Success(sim_params):
            pass

    match build_black_scholes_config(
        sim_params=sim_params,
        simulate_log_return=True,
        normalize_forwards=True,
    ):
        case Failure(err):
            raise AssertionError(f"BlackScholesConfig creation failed: {err}")
        case Success(bs_config):
            pass

    cpu_rng_state = torch.get_rng_state().numpy().tobytes()
    cuda_rng_states = [state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()]

    # Black-Scholes parameter bounds (required for SobolSampler)
    domain_bounds: dict[str, BoundSpec] = {
        "X0": BoundSpec(lower=50.0, upper=150.0),  # Initial spot price
        "K": BoundSpec(lower=50.0, upper=150.0),  # Strike price
        "T": BoundSpec(lower=0.1, upper=2.0),  # Time to maturity
        "r": BoundSpec(lower=0.0, upper=0.1),  # Risk-free rate
        "d": BoundSpec(lower=0.0, upper=0.05),  # Dividend yield
        "v": BoundSpec(lower=0.1, upper=0.5),  # Volatility
    }

    return GbmCVNNPricerConfig(
        cfg=bs_config,
        domain_bounds=domain_bounds,
        cvnn=model,
        optimizer_state=None,
        global_step=global_step,
        sobol_skip=0,
        torch_cpu_rng_state=cpu_rng_state,
        torch_cuda_rng_states=cuda_rng_states,
    )


@pytest.mark.asyncio
async def test_training_with_auto_commit(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test training with manual commit after completion (auto_commit skipped in async context)."""
    model = _make_test_cvnn(6, 128, seed=42, device="cuda", dtype=torch.float32)
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=5,
        batch_size=16,
        learning_rate=0.001,
    )

    # Train (auto_commit will be skipped due to async context)
    pricer.train(training_config)

    # Manually commit after training
    snapshot = pricer.snapshot()
    version = await commit_snapshot(
        async_store, snapshot, f"Final checkpoint: step={snapshot.global_step}"
    )

    # Verify commit was created
    assert version.counter == 0  # First version
    assert "Final checkpoint" in version.commit_message


@pytest.mark.asyncio
async def test_training_with_commit_interval(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test training with manual periodic commits (simulating commit_interval behavior)."""
    model = _make_test_cvnn(6, 128, seed=42, device="cuda", dtype=torch.float32)
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=15,
        batch_size=16,
        learning_rate=0.001,
    )

    # Train without blockchain integration
    pricer.train(training_config)

    # Manually commit final state
    snapshot = pricer.snapshot()
    version = await commit_snapshot(
        async_store, snapshot, f"Checkpoint at step={snapshot.global_step}"
    )

    # Verify commit was created
    assert version.counter == 0
    assert "Checkpoint" in version.commit_message


@pytest.mark.asyncio
async def test_training_without_storage_backward_compat(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that training without blockchain_store still works (backward compatibility)."""
    model = _make_test_cvnn(6, 128, seed=42, device="cuda", dtype=torch.float32)
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=5,
        batch_size=16,
        learning_rate=0.001,
    )

    # Train without any blockchain storage arguments
    pricer.train(training_config)

    # Verify no commits were created
    head_result = await async_store.get_head()
    match head_result:
        case Failure(_):
            pass  # Expected - no versions in store
        case Success(_):
            pytest.fail("Expected no HEAD since we didn't commit to store")


@pytest.mark.asyncio
async def test_training_validation_auto_commit_requires_store(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that auto_commit=True without blockchain_store raises error."""
    model = _make_test_cvnn(6, 128, seed=42, device="cuda", dtype=torch.float32)
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=5,
        batch_size=16,
        learning_rate=0.001,
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="requires blockchain_store"):
        pricer.train(
            training_config,
            auto_commit=True,  # Without blockchain_store
        )


@pytest.mark.asyncio
async def test_training_validation_commit_interval_requires_store(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that commit_interval without blockchain_store raises error."""
    model = _make_test_cvnn(6, 128, seed=42, device="cuda", dtype=torch.float32)
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=5,
        batch_size=16,
        learning_rate=0.001,
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="requires blockchain_store"):
        pricer.train(
            training_config,
            commit_interval=3,  # Without blockchain_store
        )


@pytest.mark.asyncio
async def test_training_commit_message_template(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that commit message can be formatted with training details."""
    model = _make_test_cvnn(6, 128, seed=42, device="cuda", dtype=torch.float32)
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=5,
        batch_size=16,
        learning_rate=0.001,
    )

    # Train
    pricer.train(training_config)

    # Manually commit with custom message
    snapshot = pricer.snapshot()
    message = f"Training: step={snapshot.global_step}, batch={training_config.num_batches}"
    version = await commit_snapshot(async_store, snapshot, message)

    # Verify message was formatted
    assert "Training: step=" in version.commit_message
    assert "batch=" in version.commit_message


@pytest.mark.asyncio
async def test_training_commit_preserves_optimizer_state(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that committing after training preserves optimizer state."""
    model = _make_test_cvnn(6, 128, seed=42, device="cuda", dtype=torch.float32)
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=10,
        batch_size=16,
        learning_rate=0.001,
    )

    # Train
    pricer.train(training_config)

    # Get snapshot and commit
    snapshot = pricer.snapshot()

    # Verify optimizer state exists before commit
    assert snapshot.optimizer_state is not None
    assert snapshot.global_step == 10
    assert snapshot.sobol_skip > 0

    # Commit and verify
    version = await commit_snapshot(async_store, snapshot, "Final state")
    assert version.counter == 0
