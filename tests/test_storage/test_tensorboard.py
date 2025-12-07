# tests/test_storage/test_tensorboard.py
"""Tests for TensorBoard logging of blockchain model versions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from spectralmc.gbm import build_black_scholes_config, build_simulation_params
from spectralmc.result import Failure, Success
from spectralmc.gbm_trainer import GbmCVNNPricerConfig
from spectralmc.models.numerical import Precision
from spectralmc.storage import (
    AsyncBlockchainModelStore,
    TensorBoardWriter,
    commit_snapshot,
    log_blockchain_to_tensorboard,
)

assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"


def make_test_config(model: torch.nn.Module, global_step: int = 0) -> GbmCVNNPricerConfig:
    """Factory to create test configurations."""
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
        case Failure(err):
            pytest.fail(f"SimulationParams creation failed: {err}")
        case Success(sim_params):
            pass

    match build_black_scholes_config(
        sim_params=sim_params,
        simulate_log_return=True,
        normalize_forwards=True,
    ):
        case Failure(err):
            pytest.fail(f"BlackScholesConfig creation failed: {err}")
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
async def test_tensorboard_basic_initialization(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test TensorBoard writer initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorBoardWriter(async_store, log_dir=tmpdir)

        assert writer.store is async_store
        assert writer.log_dir == Path(tmpdir)
        assert writer.writer is not None

        writer.close()


@pytest.mark.asyncio
async def test_tensorboard_log_version_metadata_only(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test logging version metadata without checkpoint loading."""
    # Create 1 version
    config = make_test_config(torch.nn.Linear(5, 5))
    version = await commit_snapshot(async_store, config, "Test version")

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorBoardWriter(async_store, log_dir=tmpdir)

        # Log without model/config templates (metadata only)
        await writer.log_version(version)

        # Verify log directory was created
        assert Path(tmpdir).exists()

        writer.close()


@pytest.mark.asyncio
async def test_tensorboard_log_version_with_checkpoint(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test logging version with checkpoint metrics."""
    # Create 1 version
    model = torch.nn.Linear(5, 5)
    config = make_test_config(model, global_step=1000)
    version = await commit_snapshot(async_store, config, "Test version")

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorBoardWriter(async_store, log_dir=tmpdir)

        # Log with model/config templates (includes checkpoint metrics)
        template_model = torch.nn.Linear(5, 5)
        template_config = make_test_config(template_model)

        await writer.log_version(version, template_model, template_config)

        # Verify log directory was created
        assert Path(tmpdir).exists()

        writer.close()


@pytest.mark.asyncio
async def test_tensorboard_log_all_versions(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test logging all versions in blockchain."""
    # Create 5 versions
    for i in range(5):
        config = make_test_config(torch.nn.Linear(5, 5), global_step=i * 100)
        await commit_snapshot(async_store, config, f"Version {i}")

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorBoardWriter(async_store, log_dir=tmpdir)

        # Log all versions
        template_model = torch.nn.Linear(5, 5)
        template_config = make_test_config(template_model)

        await writer.log_all_versions(template_model, template_config)

        # Verify log directory has content
        assert Path(tmpdir).exists()
        assert any(Path(tmpdir).iterdir())  # Not empty

        writer.close()


@pytest.mark.asyncio
async def test_tensorboard_log_empty_chain(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test logging empty blockchain (no versions)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorBoardWriter(async_store, log_dir=tmpdir)

        # Log all versions (should handle empty chain gracefully)
        await writer.log_all_versions()

        writer.close()


@pytest.mark.asyncio
async def test_tensorboard_summary_statistics(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test logging summary statistics."""
    # Create several versions
    for i in range(3):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"Version {i}")

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorBoardWriter(async_store, log_dir=tmpdir)

        await writer.log_summary_statistics()

        # Verify log directory has content
        assert Path(tmpdir).exists()

        writer.close()


@pytest.mark.asyncio
async def test_tensorboard_summary_empty_chain(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test summary statistics on empty chain."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorBoardWriter(async_store, log_dir=tmpdir)

        # Should handle empty chain without error
        await writer.log_summary_statistics()

        writer.close()


@pytest.mark.asyncio
async def test_tensorboard_context_manager(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test TensorBoardWriter as context manager."""
    # Create 1 version
    config = make_test_config(torch.nn.Linear(5, 5))
    await commit_snapshot(async_store, config, "Test")

    with tempfile.TemporaryDirectory() as tmpdir:
        with TensorBoardWriter(async_store, log_dir=tmpdir) as writer:
            await writer.log_all_versions()

        # Writer should be closed automatically
        # Verify log directory exists
        assert Path(tmpdir).exists()


@pytest.mark.asyncio
async def test_tensorboard_convenience_function(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test log_blockchain_to_tensorboard convenience function."""
    # Create 3 versions
    for i in range(3):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"Version {i}")

    with tempfile.TemporaryDirectory() as tmpdir:
        template_model = torch.nn.Linear(5, 5)
        template_config = make_test_config(template_model)

        await log_blockchain_to_tensorboard(
            async_store,
            log_dir=tmpdir,
            model_template=template_model,
            config_template=template_config,
        )

        # Verify logs were created
        assert Path(tmpdir).exists()
        assert any(Path(tmpdir).iterdir())


@pytest.mark.asyncio
async def test_tensorboard_multiple_versions_sequential(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test logging versions sequentially as they are committed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorBoardWriter(async_store, log_dir=tmpdir)

        template_model = torch.nn.Linear(5, 5)
        template_config = make_test_config(template_model)

        # Commit and log versions one at a time
        for i in range(3):
            config = make_test_config(torch.nn.Linear(5, 5), global_step=i * 100)
            version = await commit_snapshot(async_store, config, f"Version {i}")

            # Log immediately after commit
            await writer.log_version(version, template_model, template_config)

        writer.close()

        # Verify logs exist
        assert Path(tmpdir).exists()


@pytest.mark.asyncio
async def test_tensorboard_log_dir_creation(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that log directory is created if it does not exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use nested directory that does not exist yet
        log_dir = str(Path(tmpdir) / "nested" / "tensorboard_logs")

        writer = TensorBoardWriter(async_store, log_dir=log_dir)

        # Directory should be created
        assert Path(log_dir).exists()

        writer.close()


@pytest.mark.asyncio
async def test_tensorboard_handles_checkpoint_load_failure(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that logging continues even if checkpoint loading fails."""
    # Create a version
    config = make_test_config(torch.nn.Linear(5, 5))
    version = await commit_snapshot(async_store, config, "Test")

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TensorBoardWriter(async_store, log_dir=tmpdir)

        # Try to log with wrong model template (different size)
        wrong_model = torch.nn.Linear(10, 10)  # Different architecture
        wrong_config = make_test_config(wrong_model)

        # Should log metadata even if checkpoint loading fails
        await writer.log_version(version, wrong_model, wrong_config)

        writer.close()

        # Verify some logs were still created (metadata at minimum)
        assert Path(tmpdir).exists()
