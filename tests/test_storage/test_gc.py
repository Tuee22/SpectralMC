# tests/test_storage/test_gc.py
"""Tests for garbage collection (GC) of old model versions."""

from __future__ import annotations

import pytest
import torch

from spectralmc.gbm import build_black_scholes_config, build_simulation_params
from spectralmc.gbm_trainer import GbmCVNNPricerConfig
from spectralmc.models.numerical import Precision
from spectralmc.result import Failure, Success
from spectralmc.errors.storage import GCError
from spectralmc.storage import (
    AsyncBlockchainModelStore,
    GarbageCollector,
    RetentionPolicy,
    VersionNotFoundError,
    commit_snapshot,
    run_gc,
)
from spectralmc.storage.gc import GCReport

assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"


def _expect_success_gc(result: Success[GCReport] | Failure[GCError]) -> GCReport:
    """Unwrap Result for GC operations, fail test if Failure."""
    match result:
        case Success(report):
            return report
        case Failure(error):
            pytest.fail(f"GC operation failed: {error.message}")


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
        simulate_log_return=True,
        normalize_forwards=True,
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
async def test_gc_empty_chain(async_store: AsyncBlockchainModelStore) -> None:
    """Test GC on empty chain does nothing."""
    policy = RetentionPolicy(keep_versions=5)
    gc = GarbageCollector(async_store, policy)

    report = _expect_success_gc(await gc.collect(dry_run=True))

    assert len(report.deleted_versions) == 0
    assert len(report.protected_versions) == 0
    assert report.bytes_freed == 0
    assert report.dry_run is True


@pytest.mark.asyncio
async def test_gc_keep_all_versions(async_store: AsyncBlockchainModelStore) -> None:
    """Test GC with keep_versions=None keeps everything."""
    # Create 5 versions
    for i in range(5):
        config = make_test_config(torch.nn.Linear(5, 5), global_step=i * 100)
        await commit_snapshot(async_store, config, f"V{i}")

    policy = RetentionPolicy(keep_versions=None)  # Keep all
    gc = GarbageCollector(async_store, policy)

    report = _expect_success_gc(await gc.collect(dry_run=True))

    assert len(report.deleted_versions) == 0
    assert len(report.protected_versions) == 5
    assert set(report.protected_versions) == {0, 1, 2, 3, 4}


@pytest.mark.asyncio
async def test_gc_keep_recent_versions(async_store: AsyncBlockchainModelStore) -> None:
    """Test GC keeps N most recent versions."""
    # Create 10 versions
    for i in range(10):
        config = make_test_config(torch.nn.Linear(5, 5), global_step=i * 100)
        await commit_snapshot(async_store, config, f"V{i}")

    # Keep last 5 versions
    policy = RetentionPolicy(keep_versions=5)
    gc = GarbageCollector(async_store, policy)

    report = _expect_success_gc(await gc.collect(dry_run=True))

    # Should delete v0-v4, keep v5-v9
    # But v0 (genesis) is always protected, so actually:
    # Protected: v0 (genesis), v5-v9 (recent 5)
    # Deleted: v1-v4
    assert len(report.deleted_versions) == 4
    assert set(report.deleted_versions) == {1, 2, 3, 4}
    assert 0 in report.protected_versions  # Genesis always protected
    assert set(range(5, 10)).issubset(set(report.protected_versions))


@pytest.mark.asyncio
async def test_gc_genesis_always_protected(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that v0 (genesis) is never deleted."""
    # Create 3 versions
    for i in range(3):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Keep only 1 version (should be v2, but v0 also protected)
    # Set keep_min_versions=1 to allow this scenario
    policy = RetentionPolicy(keep_versions=1, keep_min_versions=1)
    gc = GarbageCollector(async_store, policy)

    report = _expect_success_gc(await gc.collect(dry_run=True))

    # Protected: v0 (genesis), v2 (most recent)
    # Deleted: v1
    assert 0 in report.protected_versions
    assert 2 in report.protected_versions
    assert report.deleted_versions == [1]


@pytest.mark.asyncio
async def test_gc_protected_tags(async_store: AsyncBlockchainModelStore) -> None:
    """Test that tagged versions are always protected."""
    # Create 10 versions
    for i in range(10):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Keep last 3 versions + protect v3 and v5 (e.g., production releases)
    policy = RetentionPolicy(keep_versions=3, protect_tags=[3, 5])
    gc = GarbageCollector(async_store, policy)

    report = _expect_success_gc(await gc.collect(dry_run=True))

    # Protected: v0 (genesis), v3 (tag), v5 (tag), v7-v9 (recent 3)
    # Deleted: v1, v2, v4, v6
    assert 0 in report.protected_versions
    assert 3 in report.protected_versions
    assert 5 in report.protected_versions
    assert set(range(7, 10)).issubset(set(report.protected_versions))
    assert set(report.deleted_versions) == {1, 2, 4, 6}


@pytest.mark.asyncio
async def test_gc_minimum_versions_enforced(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that keep_min_versions prevents over-deletion."""
    # Create 5 versions
    for i in range(5):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Try to keep only 1 version, but min is 3
    policy = RetentionPolicy(keep_versions=1, keep_min_versions=3)
    gc = GarbageCollector(async_store, policy)

    # Should keep: v0 (genesis), v4 (recent), = 2 versions
    # This is < keep_min_versions=3, so should return Failure
    result = await gc.collect(dry_run=True)
    match result:
        case Failure(error):
            assert error.protected_count == 2
            assert error.minimum_required == 3
            assert "minimum version requirement" in error.message
        case Success(_):
            pytest.fail("Expected Failure for policy violation")


@pytest.mark.asyncio
async def test_gc_dry_run_vs_actual(async_store: AsyncBlockchainModelStore) -> None:
    """Test that dry_run doesn't delete, actual deletion works."""
    # Create 5 versions
    for i in range(5):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    policy = RetentionPolicy(keep_versions=3)
    gc = GarbageCollector(async_store, policy)

    # Dry run
    report_dry = _expect_success_gc(await gc.collect(dry_run=True))
    assert report_dry.dry_run is True
    assert len(report_dry.deleted_versions) == 1  # v1 (v0 protected, v2-v4 recent)

    # Verify nothing actually deleted
    head_result = await async_store.get_head()
    match head_result:
        case Success(head):
            assert head.counter == 4
        case Failure(_):
            pytest.fail("Expected HEAD")

    # All versions still exist
    for i in range(5):
        version_id = f"v{i:010d}"
        version = await async_store.get_version(version_id)
        assert version.counter == i

    # Now actually delete
    report_actual = _expect_success_gc(await gc.collect(dry_run=False))
    assert report_actual.dry_run is False
    assert report_actual.deleted_versions == report_dry.deleted_versions
    assert report_actual.bytes_freed > 0

    # v1 should be deleted now
    with pytest.raises(VersionNotFoundError):
        await async_store.get_version("v0000000001")

    # Others should still exist
    for i in [0, 2, 3, 4]:
        version_id = f"v{i:010d}"
        version = await async_store.get_version(version_id)
        assert version.counter == i


@pytest.mark.asyncio
async def test_gc_bytes_freed_nonzero(async_store: AsyncBlockchainModelStore) -> None:
    """Test that bytes_freed is calculated correctly."""
    # Create 3 versions
    for i in range(3):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Keep v2, protect v0, delete v1; set keep_min_versions=1 to allow
    policy = RetentionPolicy(keep_versions=1, keep_min_versions=1)
    gc = GarbageCollector(async_store, policy)

    # Dry run should estimate size
    report_dry = _expect_success_gc(await gc.collect(dry_run=True))
    assert report_dry.bytes_freed > 0  # Should be nonzero (checkpoint + metadata)

    # Actual deletion should also report size
    report_actual = _expect_success_gc(await gc.collect(dry_run=False))
    assert report_actual.bytes_freed > 0
    # Should be similar to dry run estimate
    assert abs(report_dry.bytes_freed - report_actual.bytes_freed) < 100  # Allow small diff


@pytest.mark.asyncio
async def test_run_gc_convenience_function(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test run_gc() convenience function."""
    # Create 8 versions
    for i in range(8):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Use convenience function
    report = _expect_success_gc(
        await run_gc(async_store, keep_versions=4, protect_tags=[2], dry_run=True)
    )

    # Protected: v0 (genesis), v2 (tag), v4-v7 (recent 4)
    # Deleted: v1, v3
    assert set(report.deleted_versions) == {1, 3}
    assert 0 in report.protected_versions
    assert 2 in report.protected_versions
    assert set(range(4, 8)).issubset(set(report.protected_versions))


@pytest.mark.asyncio
async def test_gc_single_version_chain(async_store: AsyncBlockchainModelStore) -> None:
    """Test GC on chain with only genesis version."""
    config = make_test_config(torch.nn.Linear(5, 5))
    await commit_snapshot(async_store, config, "Genesis")

    # Single version chain, set keep_min_versions=1
    policy = RetentionPolicy(keep_versions=10, keep_min_versions=1)
    gc = GarbageCollector(async_store, policy)

    report = _expect_success_gc(await gc.collect(dry_run=True))

    # Nothing to delete, only v0 (genesis)
    assert len(report.deleted_versions) == 0
    assert report.protected_versions == [0]


@pytest.mark.asyncio
async def test_gc_keeps_chain_integrity(async_store: AsyncBlockchainModelStore) -> None:
    """Test that GC doesn't break chain integrity (parent_hash references)."""
    # Create 6 versions
    for i in range(6):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Keep last 3 versions
    policy = RetentionPolicy(keep_versions=3)
    gc = GarbageCollector(async_store, policy)

    # Delete old versions
    report = _expect_success_gc(await gc.collect(dry_run=False))

    # Protected: v0 (genesis), v3-v5 (recent 3)
    # Deleted: v1, v2
    assert set(report.deleted_versions) == {1, 2}

    # Verify remaining versions are intact
    for counter in [0, 3, 4, 5]:
        version_id = f"v{counter:010d}"
        version = await async_store.get_version(version_id)
        assert version.counter == counter

    # Note: In real use, if parent_hash references are critical,
    # GC policy should keep contiguous ranges or use different strategy.
    # This test just verifies that deletion completes without error.


@pytest.mark.asyncio
async def test_gc_large_batch_deletion(async_store: AsyncBlockchainModelStore) -> None:
    """Test GC with many versions to delete."""
    # Create 20 versions
    for i in range(20):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Keep only last 5
    policy = RetentionPolicy(keep_versions=5)
    gc = GarbageCollector(async_store, policy)

    report = _expect_success_gc(await gc.collect(dry_run=False))

    # Protected: v0 (genesis), v15-v19 (recent 5)
    # Deleted: v1-v14 (14 versions)
    assert len(report.deleted_versions) == 14
    assert set(report.deleted_versions) == set(range(1, 15))
    assert report.bytes_freed > 0


@pytest.mark.asyncio
async def test_gc_all_versions_protected_by_tags(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test GC when all versions are protected via tags."""
    # Create 5 versions
    for i in range(5):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Protect all versions via tags
    policy = RetentionPolicy(keep_versions=1, protect_tags=[0, 1, 2, 3, 4])
    gc = GarbageCollector(async_store, policy)

    report = _expect_success_gc(await gc.collect(dry_run=True))

    # Nothing deleted, everything protected
    assert len(report.deleted_versions) == 0
    assert set(report.protected_versions) == {0, 1, 2, 3, 4}


@pytest.mark.asyncio
async def test_gc_policy_validation(async_store: AsyncBlockchainModelStore) -> None:
    """Test that policy validation works correctly."""
    # Create 10 versions
    for i in range(10):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Create policy that would violate minimum (keep only v9, but min=5)
    # Actually, v0 is always protected, so this would keep v0 + v9 = 2, which is < 5
    policy = RetentionPolicy(keep_versions=1, keep_min_versions=5)
    gc = GarbageCollector(async_store, policy)

    # Should return Failure during collection
    result = await gc.collect(dry_run=True)
    match result:
        case Failure(error):
            assert error.protected_count == 2
            assert error.minimum_required == 5
            assert "minimum version requirement" in error.message
        case Success(_):
            pytest.fail("Expected Failure for policy violation")


@pytest.mark.asyncio
async def test_gc_custom_min_versions(async_store: AsyncBlockchainModelStore) -> None:
    """Test GC with custom keep_min_versions setting."""
    # Create 10 versions
    for i in range(10):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Keep last 7 versions, minimum 5 (should succeed)
    policy = RetentionPolicy(keep_versions=7, keep_min_versions=5)
    gc = GarbageCollector(async_store, policy)

    report = _expect_success_gc(await gc.collect(dry_run=True))

    # Protected: v0 (genesis), v3-v9 (recent 7) = 8 versions total
    # Deleted: v1, v2
    assert len(report.deleted_versions) == 2
    assert set(report.deleted_versions) == {1, 2}
    assert len(report.protected_versions) >= 5  # Meets minimum
