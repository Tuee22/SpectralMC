# tests/test_storage/test_training_integration.py
"""Integration tests for GbmCVNNPricer training with blockchain storage.

All tests require GPU - missing GPU is a hard failure, not a skip.
"""

from __future__ import annotations

import pytest
import torch


from spectralmc.errors.trainer import InvalidTrainerConfig
from spectralmc.gbm_trainer import (
    ComplexValuedModel,
    FinalCommit,
    IntervalCommit,
    GbmCVNNPricer,
    GbmCVNNPricerConfig,
)
from spectralmc.models.numerical import Precision
from spectralmc.storage import AsyncBlockchainModelStore, commit_snapshot
from tests.helpers import (
    expect_failure,
    expect_success,
    make_black_scholes_config,
    make_domain_bounds,
    make_simulation_params,
    make_training_config,
    make_test_cvnn,
    make_gbm_cvnn_config,
)

DOMAIN_BOUNDS = make_domain_bounds()
SIM_PARAMS = make_simulation_params(
    timesteps=10,  # Reduced from 100: match test_e2e_storage.py pattern
    network_size=128,  # Reduced from 1024: sufficient for CVNN operation
    batches_per_mc_run=2,  # Reduced from 8: minimum for mean calculation
    threads_per_block=256,
    mc_seed=42,
    buffer_size=256,  # Reduced from 10000: conservative async buffer (was 3.82 GB, now 1.26 MB)
    skip=0,
    dtype=Precision.float32,
)
BS_CONFIG = make_black_scholes_config(sim_params=SIM_PARAMS)

# Module-level GPU requirement - test file fails immediately without GPU


def _make_pricer_config(model: ComplexValuedModel, *, global_step: int = 0) -> GbmCVNNPricerConfig:
    """Shared GbmCVNNPricerConfig using the standardized training defaults."""
    return make_gbm_cvnn_config(
        model,
        global_step=global_step,
        sim_params=SIM_PARAMS,
        bs_config=BS_CONFIG,
        domain_bounds=DOMAIN_BOUNDS,
    )


@pytest.mark.asyncio
async def test_training_manual_commit_after_training(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Train without commit plan and manually commit afterwards."""
    model = make_test_cvnn(
        n_inputs=6,
        n_outputs=128,
        seed=42,
        dtype=torch.float32,
        add_output_layer=False,
    )
    config = _make_pricer_config(model)
    pricer = expect_success(GbmCVNNPricer.create(config))

    training_config = make_training_config(num_batches=5, batch_size=16, learning_rate=0.001)

    # Train with no blockchain integration
    pricer.train(training_config)

    # Manually commit after training
    snapshot = expect_success(pricer.snapshot())
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
    model = make_test_cvnn(
        n_inputs=6,
        n_outputs=128,
        seed=42,
        dtype=torch.float32,
        add_output_layer=False,
    )
    config = _make_pricer_config(model)
    pricer = expect_success(GbmCVNNPricer.create(config))

    training_config = make_training_config(num_batches=15, batch_size=16, learning_rate=0.001)

    # Train without blockchain integration
    pricer.train(training_config)

    # Manually commit final state
    snapshot = expect_success(pricer.snapshot())
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
    model = make_test_cvnn(
        n_inputs=6,
        n_outputs=128,
        seed=42,
        dtype=torch.float32,
        add_output_layer=False,
    )
    config = _make_pricer_config(model)
    pricer = expect_success(GbmCVNNPricer.create(config))

    training_config = make_training_config(num_batches=5, batch_size=16, learning_rate=0.001)

    # Train without any blockchain storage arguments
    pricer.train(training_config)

    # Verify no commits were created
    head_result = await async_store.get_head()
    expect_failure(head_result)


@pytest.mark.asyncio
async def test_training_validation_commit_plan_requires_store(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that commit plan requiring storage fails without blockchain_store."""
    model = make_test_cvnn(
        n_inputs=6,
        n_outputs=128,
        seed=42,
        dtype=torch.float32,
        add_output_layer=False,
    )
    config = _make_pricer_config(model)
    pricer = expect_success(GbmCVNNPricer.create(config))

    training_config = make_training_config(num_batches=5, batch_size=16, learning_rate=0.001)

    # Should return Failure with InvalidTrainerConfig
    result = pricer.train(training_config, commit_plan=FinalCommit())
    error = expect_failure(result)
    assert isinstance(error, InvalidTrainerConfig)
    assert "commit_plan requires blockchain_store" in error.message


@pytest.mark.asyncio
async def test_training_validation_interval_commit_requires_store(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that interval commit plan without blockchain_store raises error."""
    model = make_test_cvnn(
        n_inputs=6,
        n_outputs=128,
        seed=42,
        dtype=torch.float32,
        add_output_layer=False,
    )
    config = _make_pricer_config(model)
    pricer = expect_success(GbmCVNNPricer.create(config))

    training_config = make_training_config(num_batches=5, batch_size=16, learning_rate=0.001)

    # Should return Failure with InvalidTrainerConfig
    result = pricer.train(training_config, commit_plan=IntervalCommit(interval=3))
    error = expect_failure(result)
    assert isinstance(error, InvalidTrainerConfig)
    assert "commit_plan requires blockchain_store" in error.message


@pytest.mark.asyncio
async def test_training_commit_message_template(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that commit message can be formatted with training details."""
    model = make_test_cvnn(
        n_inputs=6,
        n_outputs=128,
        seed=42,
        dtype=torch.float32,
        add_output_layer=False,
    )
    config = _make_pricer_config(model)
    pricer = expect_success(GbmCVNNPricer.create(config))

    training_config = make_training_config(num_batches=5, batch_size=16, learning_rate=0.001)

    # Train
    pricer.train(training_config)

    # Manually commit with custom message
    snapshot = expect_success(pricer.snapshot())
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
    model = make_test_cvnn(
        n_inputs=6,
        n_outputs=128,
        seed=42,
        dtype=torch.float32,
        add_output_layer=False,
    )
    config = _make_pricer_config(model)
    pricer = expect_success(GbmCVNNPricer.create(config))

    training_config = make_training_config(num_batches=10, batch_size=16, learning_rate=0.001)

    # Train
    pricer.train(training_config)

    # Get snapshot and commit
    snapshot = expect_success(pricer.snapshot())

    # Verify optimizer state exists before commit
    assert snapshot.optimizer_state is not None, "Optimizer state should be preserved"
    assert snapshot.global_step == 10
    assert snapshot.sobol_skip > 0

    # Commit and verify
    version = await commit_snapshot(async_store, snapshot, "Final state")
    assert version.counter == 0
