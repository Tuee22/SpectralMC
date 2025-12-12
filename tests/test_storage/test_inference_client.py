# tests/test_storage/test_inference_client.py
"""Tests for InferenceClient with pinned and tracking modes."""

from __future__ import annotations

import asyncio

import pytest
import torch

from spectralmc.effects import ForwardNormalization, PathScheme
from spectralmc.gbm import build_black_scholes_config, build_simulation_params
from spectralmc.result import Failure, Success
from spectralmc.gbm_trainer import GbmCVNNPricerConfig
from spectralmc.models.numerical import Precision
from spectralmc.storage import (
    AsyncBlockchainModelStore,
    InferenceClient,
    PinnedMode,
    TrackingMode,
    commit_snapshot,
)

assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"


def make_test_config(model: torch.nn.Module, global_step: int = 0) -> GbmCVNNPricerConfig:
    """Factory to create test configurations (GbmCVNNPricerConfig is frozen)."""
    match build_simulation_params(
        timesteps=100,
        network_size=1024,
        batches_per_mc_run=8,
        threads_per_block=256,
        mc_seed=42,
        buffer_size=10000,
        skip=0,
        dtype=Precision.float32,
    ):
        case Failure(sim_err):
            pytest.fail(f"SimulationParams creation failed: {sim_err}")
        case Success(sim_params):
            pass

    match build_black_scholes_config(
        sim_params=sim_params,
        path_scheme=PathScheme.LOG_EULER,
        normalization=ForwardNormalization.NORMALIZE,
    ):
        case Failure(bs_err):
            pytest.fail(f"BlackScholesConfig creation failed: {bs_err}")
        case Success(bs_config):
            pass

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


@pytest.mark.asyncio
async def test_pinned_mode_basic(async_store: AsyncBlockchainModelStore) -> None:
    """Test pinned mode loads specific version."""
    # Create 2 versions
    model1 = torch.nn.Linear(5, 5)
    config1 = make_test_config(model1, global_step=100)
    _v1 = await commit_snapshot(async_store, config1, "V1")

    model2 = torch.nn.Linear(5, 5)
    config2 = make_test_config(model2, global_step=200)
    _v2 = await commit_snapshot(async_store, config2, "V2")

    # Pin to version 0
    template = make_test_config(torch.nn.Linear(5, 5))
    client = InferenceClient(
        mode=PinnedMode(counter=0),
        poll_interval=0.1,
        store=async_store,
        model_template=torch.nn.Linear(5, 5),
        config_template=template,
    )

    async with client:
        snapshot = client.get_model()
        assert snapshot.global_step == 100

        version = client.get_current_version()
        assert version is not None, "Version should be loaded in pinned mode"
        assert isinstance(version.counter, int), "Version counter should be an integer"
        assert version.counter == 0, "Should load version 0 in pinned mode"
        assert len(version.content_hash) == 64, "Content hash should be SHA256 (64 hex chars)"


@pytest.mark.asyncio
async def test_tracking_mode_basic(async_store: AsyncBlockchainModelStore) -> None:
    """Test tracking mode loads latest version."""
    # Create version 0
    config = make_test_config(torch.nn.Linear(5, 5), global_step=42)
    await commit_snapshot(async_store, config, "V0")

    # Start tracking mode
    template = make_test_config(torch.nn.Linear(5, 5))
    client = InferenceClient(
        mode=TrackingMode(),
        poll_interval=0.5,
        store=async_store,
        model_template=torch.nn.Linear(5, 5),
        config_template=template,
    )

    async with client:
        snapshot = client.get_model()
        assert snapshot.global_step == 42

        version = client.get_current_version()
        assert version is not None, "Version should be loaded"
        assert version.counter == 0


@pytest.mark.asyncio
async def test_tracking_mode_auto_update(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test tracking mode automatically updates to new versions."""
    # Create initial version
    config = make_test_config(torch.nn.Linear(5, 5), global_step=0)
    await commit_snapshot(async_store, config, "V0")

    # Start tracking with fast polling
    template = make_test_config(torch.nn.Linear(5, 5))
    client = InferenceClient(
        mode=TrackingMode(),
        poll_interval=0.1,
        store=async_store,
        model_template=torch.nn.Linear(5, 5),
        config_template=template,
    )

    async with client:
        # Add explicit None check for type narrowing
        version = client.get_current_version()
        assert version is not None, "Expected version to be loaded"
        assert version.counter == 0

        # Create new version
        config2 = make_test_config(torch.nn.Linear(5, 5), global_step=100)
        await commit_snapshot(async_store, config2, "V1")

        # Wait for poll to detect
        await asyncio.sleep(0.3)

        # Should have upgraded
        version = client.get_current_version()
        assert version is not None, "Version should be loaded"
        assert version.counter == 1
        assert client.get_model().global_step == 100


@pytest.mark.asyncio
async def test_get_model_before_start(async_store: AsyncBlockchainModelStore) -> None:
    """Test that get_model() raises error before start."""
    config = make_test_config(torch.nn.Linear(5, 5))
    await commit_snapshot(async_store, config, "V0")

    template = make_test_config(torch.nn.Linear(5, 5))
    client = InferenceClient(
        mode=PinnedMode(counter=0),
        poll_interval=1.0,
        store=async_store,
        model_template=torch.nn.Linear(5, 5),
        config_template=template,
    )

    with pytest.raises(RuntimeError, match="not started"):
        client.get_model()


@pytest.mark.asyncio
async def test_context_manager_lifecycle(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test context manager starts and stops client."""
    config = make_test_config(torch.nn.Linear(5, 5))
    await commit_snapshot(async_store, config, "V0")

    template = make_test_config(torch.nn.Linear(5, 5))
    client = InferenceClient(
        mode=PinnedMode(counter=0),
        poll_interval=1.0,
        store=async_store,
        model_template=torch.nn.Linear(5, 5),
        config_template=template,
    )

    assert client.get_current_version() is None

    async with client:
        assert client.get_current_version() is not None, "Current version should be available"

    # After exit, version is cached
    assert client.get_current_version() is not None, "Current version should be available"


@pytest.mark.asyncio
async def test_manual_start_stop(async_store: AsyncBlockchainModelStore) -> None:
    """Test manual start/stop methods."""
    config = make_test_config(torch.nn.Linear(5, 5))
    await commit_snapshot(async_store, config, "V0")

    template = make_test_config(torch.nn.Linear(5, 5))
    client = InferenceClient(
        mode=PinnedMode(counter=0),
        poll_interval=1.0,
        store=async_store,
        model_template=torch.nn.Linear(5, 5),
        config_template=template,
    )

    await client.start()
    assert client.get_current_version() is not None, "Current version should be available"

    await client.stop()
    assert client.get_current_version() is not None, "Current version should be available"


@pytest.mark.asyncio
async def test_empty_store_tracking_mode(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test tracking mode fails on empty store."""
    template = make_test_config(torch.nn.Linear(5, 5))
    client = InferenceClient(
        mode=TrackingMode(),
        poll_interval=1.0,
        store=async_store,
        model_template=torch.nn.Linear(5, 5),
        config_template=template,
    )

    with pytest.raises(RuntimeError, match="no versions"):
        async with client:
            pass


@pytest.mark.asyncio
async def test_graceful_shutdown_tracking(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test polling task is cancelled on shutdown."""
    config = make_test_config(torch.nn.Linear(5, 5))
    await commit_snapshot(async_store, config, "V0")

    template = make_test_config(torch.nn.Linear(5, 5))
    client = InferenceClient(
        mode=TrackingMode(),
        poll_interval=0.1,
        store=async_store,
        model_template=torch.nn.Linear(5, 5),
        config_template=template,
    )

    async with client:
        assert client._polling_task is not None, "Polling task should be running in tracking mode"
        assert not client._polling_task.done(), "Polling task should still be active"
        assert not client._polling_task.cancelled(), "Polling task should not be cancelled"

        # Save task reference before exit
        polling_task = client._polling_task

    # After exit, task should be cancelled
    assert polling_task.done()
    assert polling_task.cancelled()
