# tests/test_storage/test_training_integration.py
"""Integration tests for GbmCVNNPricer training with blockchain storage."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectralmc.gbm import BlackScholesConfig, SimulationParams
from spectralmc.gbm_trainer import GbmCVNNPricerConfig, GbmCVNNPricer, TrainingConfig
from spectralmc.models.numerical import Precision
from spectralmc.storage import AsyncBlockchainModelStore, commit_snapshot


def make_test_config(
    model: torch.nn.Module, global_step: int = 0
) -> GbmCVNNPricerConfig:
    """Factory to create test configurations."""
    sim_params = SimulationParams(
        timesteps=100,
        network_size=1024,
        batches_per_mc_run=8,
        threads_per_block=256,
        mc_seed=42,
        buffer_size=10000,
        skip=0,
        dtype=Precision.float32,
    )

    bs_config = BlackScholesConfig(
        sim_params=sim_params,
        simulate_log_return=True,
        normalize_forwards=True,
    )

    cpu_rng_state = torch.get_rng_state().numpy().tobytes()
    cuda_rng_states = []
    if torch.cuda.is_available():
        cuda_rng_states = [
            state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()
        ]

    return GbmCVNNPricerConfig(
        cfg=bs_config,
        domain_bounds={},
        cvnn=model,
        optimizer_state=None,
        global_step=global_step,
        sobol_skip=0,
        torch_cpu_rng_state=cpu_rng_state,
        torch_cuda_rng_states=cuda_rng_states,
    )


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_training_with_auto_commit(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test training with auto_commit=True creates final checkpoint."""
    model = torch.nn.Linear(5, 5).cuda()
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=5,
        batch_size=16,
        learning_rate=0.001,
    )

    # Train with auto_commit
    pricer.train(
        training_config,
        blockchain_store=async_store,
        auto_commit=True,
        commit_message_template="Final checkpoint: step={step}",
    )

    # Verify commit was created
    head = await async_store.get_head()
    assert head is not None
    assert head.counter == 0  # First version
    assert "Final checkpoint" in head.commit_message


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_training_with_commit_interval(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test training with commit_interval creates periodic checkpoints."""
    model = torch.nn.Linear(5, 5).cuda()
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=15,  # Will create commits at steps 5, 10, 15
        batch_size=16,
        learning_rate=0.001,
    )

    # Train with periodic commits every 5 batches
    pricer.train(
        training_config,
        blockchain_store=async_store,
        auto_commit=True,
        commit_interval=5,
        commit_message_template="Checkpoint at step={step}",
    )

    # Verify multiple commits were created (3 periodic + 1 final = 4 total)
    head = await async_store.get_head()
    assert head is not None
    # Should have at least 3 versions (periodic commits at 5, 10, 15 + final)
    # But final might coincide with last periodic, so at least 3
    assert head.counter >= 2


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_training_without_storage_backward_compat(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that training without blockchain_store still works (backward compatibility)."""
    model = torch.nn.Linear(5, 5).cuda()
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
    head = await async_store.get_head()
    assert head is None  # No versions in store


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_training_validation_auto_commit_requires_store(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that auto_commit=True without blockchain_store raises error."""
    model = torch.nn.Linear(5, 5).cuda()
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


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_training_validation_commit_interval_requires_store(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that commit_interval without blockchain_store raises error."""
    model = torch.nn.Linear(5, 5).cuda()
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


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_training_commit_message_template(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that commit message template variables are interpolated correctly."""
    model = torch.nn.Linear(5, 5).cuda()
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=5,
        batch_size=16,
        learning_rate=0.001,
    )

    # Train with custom commit message template
    pricer.train(
        training_config,
        blockchain_store=async_store,
        auto_commit=True,
        commit_message_template="Training: step={step}, loss={loss:.4f}, batch={batch}",
    )

    # Verify message was formatted
    head = await async_store.get_head()
    assert head is not None
    assert "Training: step=" in head.commit_message
    assert "loss=" in head.commit_message
    assert "batch=" in head.commit_message


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_training_commit_preserves_optimizer_state(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that committing during training preserves optimizer state."""
    model = torch.nn.Linear(5, 5).cuda()
    config = make_test_config(model)
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=10,
        batch_size=16,
        learning_rate=0.001,
    )

    # Train with periodic commits
    pricer.train(
        training_config,
        blockchain_store=async_store,
        auto_commit=True,
        commit_interval=5,
    )

    # Get final snapshot
    snapshot = pricer.snapshot()

    # Verify optimizer state exists
    assert snapshot.optimizer_state is not None
    assert snapshot.global_step == 10
    assert snapshot.sobol_skip > 0

    # Verify version was committed
    head = await async_store.get_head()
    assert head is not None
