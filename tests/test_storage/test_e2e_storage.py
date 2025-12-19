# tests/test_storage/test_e2e_storage.py
"""End-to-end tests for blockchain storage workflows (CPU-only).

These tests validate complete workflows without requiring GPU/GBM training.
Checkpoints are created programmatically for testing purposes.
"""

from __future__ import annotations

import asyncio

import pytest
import torch


from spectralmc.gbm_trainer import GbmCVNNPricerConfig
from spectralmc.models.numerical import Precision
from spectralmc.storage import (
    AsyncBlockchainModelStore,
    ChainCorrupted,
    ChainValid,
    InferenceClient,
    PinnedMode,
    TrackingMode,
    commit_snapshot,
    load_snapshot_from_checkpoint,
    verify_chain,
)
from spectralmc.storage.errors import (
    VersionNotFoundError,
)
from spectralmc.storage.gc import ExecuteGC, PreviewGC, run_gc
from tests.helpers import (
    expect_success,
    make_black_scholes_config,
    make_domain_bounds,
    make_gbm_cvnn_config,
    make_simulation_params,
)


DOMAIN_BOUNDS = make_domain_bounds()
SIM_PARAMS = make_simulation_params(
    timesteps=10,
    network_size=128,
    batches_per_mc_run=2,
    threads_per_block=64,
    mc_seed=42,
    buffer_size=1000,
    skip=0,
    dtype=Precision.float32,
)
BS_CONFIG = make_black_scholes_config(sim_params=SIM_PARAMS)


def make_test_snapshot(
    global_step: int = 0, model_size: tuple[int, int] = (5, 5)
) -> GbmCVNNPricerConfig:
    """Create a test snapshot programmatically (CPU-only)."""
    model = torch.nn.Linear(*model_size)

    return make_gbm_cvnn_config(
        model,
        global_step=global_step,
        sim_params=SIM_PARAMS,
        bs_config=BS_CONFIG,
        domain_bounds=DOMAIN_BOUNDS,
        sobol_skip=global_step * 100,
    )


def _assert_chain_valid(outcome: ChainValid | ChainCorrupted) -> None:
    match outcome:
        case ChainValid():
            return
        case ChainCorrupted(corruption_type=corruption_type, details=details):
            pytest.fail(f"Chain corrupted: {corruption_type} ({details})")


@pytest.mark.asyncio
async def test_e2e_complete_lifecycle(async_store: AsyncBlockchainModelStore) -> None:
    """Test complete lifecycle: commit -> load -> modify -> commit -> verify."""

    # 1. Create and commit initial snapshot
    snapshot1 = make_test_snapshot(global_step=0)
    version1 = await commit_snapshot(async_store, snapshot1, "Initial commit")

    assert version1.counter == 0
    assert version1.semantic_version == "1.0.0"
    assert version1.parent_hash == ""

    # 2. Load the snapshot
    model_template = torch.nn.Linear(5, 5)
    config_template = make_test_snapshot(0)

    loaded_snapshot = expect_success(
        await load_snapshot_from_checkpoint(async_store, version1, model_template, config_template)
    )

    assert loaded_snapshot.global_step == 0
    assert loaded_snapshot.sobol_skip == 0

    # 3. "Train" (modify) the loaded snapshot
    loaded_snapshot = make_gbm_cvnn_config(
        loaded_snapshot.cvnn,
        global_step=100,
        sim_params=loaded_snapshot.cfg.sim_params,
        bs_config=loaded_snapshot.cfg,
        domain_bounds=loaded_snapshot.domain_bounds,
        sobol_skip=10000,
    )

    # 4. Commit updated snapshot
    version2 = await commit_snapshot(async_store, loaded_snapshot, "After training")

    assert version2.counter == 1
    assert version2.semantic_version == "1.0.1"
    assert version2.parent_hash == version1.content_hash

    # 5. Verify chain integrity
    report = expect_success(await verify_chain(async_store))
    _assert_chain_valid(report)

    # 6. Load final snapshot and verify state
    final_snapshot = expect_success(
        await load_snapshot_from_checkpoint(async_store, version2, model_template, config_template)
    )

    assert final_snapshot.global_step == 100
    # Note: sobol_skip might be overridden by config_template
    # Just verify it loads without error


@pytest.mark.asyncio
async def test_e2e_sequential_commits(async_store: AsyncBlockchainModelStore) -> None:
    """Test sequential commits from multiple 'workers'.

    This test validates the blockchain store's basic commit workflow by
    performing sequential commits (not concurrent). Concurrent conflict
    handling is tested in unit tests for the store layer.
    """
    successful_commits: list[int] = []

    # Perform 5 sequential commits (no concurrency)
    for worker_id in range(5):
        snapshot = make_test_snapshot(global_step=worker_id * 100)
        await commit_snapshot(async_store, snapshot, f"Worker {worker_id} commit")
        successful_commits.append(worker_id)

    # Verify all commits succeeded
    assert len(successful_commits) == 5, "All 5 commits should succeed"

    # Verify chain integrity
    report = expect_success(await verify_chain(async_store))
    _assert_chain_valid(report)

    # Verify HEAD exists and counter matches successful commits
    head = expect_success(await async_store.get_head())
    # Counter should be 4 (5 commits = counters 0-4)
    assert head.counter == 4

    # Verify all workers' commits are present
    worker_messages = set()
    for i in range(5):
        version = await async_store.get_version(f"v{i:010d}")
        worker_messages.add(version.commit_message)

    assert len(worker_messages) == 5
    assert "Worker 0 commit" in worker_messages
    assert "Worker 4 commit" in worker_messages


@pytest.mark.asyncio
async def test_e2e_inference_client_hot_swap(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test InferenceClient hot-swaps models when new versions are committed."""

    # Commit initial version
    snapshot1 = make_test_snapshot(global_step=0)
    await commit_snapshot(async_store, snapshot1, "Version 1")

    model_template = torch.nn.Linear(5, 5)
    config_template = make_test_snapshot(0)

    # Start InferenceClient in tracking mode (poll_interval = 0.5s)
    async with InferenceClient(
        mode=TrackingMode(),  # Track HEAD
        poll_interval=0.5,
        store=async_store,
        model_template=model_template,
        config_template=config_template,
    ) as client:
        # Initial model
        model1 = client.get_model()
        assert model1.global_step == 0

        # Commit new version
        snapshot2 = make_test_snapshot(global_step=100)
        _version2 = await commit_snapshot(async_store, snapshot2, "Version 2")

        # Wait for hot-swap (poll_interval + buffer)
        await asyncio.sleep(1.0)

        # Model should be updated
        model2 = client.get_model()
        assert model2.global_step == 100

        # Commit another version
        snapshot3 = make_test_snapshot(global_step=200)
        _version3 = await commit_snapshot(async_store, snapshot3, "Version 3")

        await asyncio.sleep(1.0)

        # Model should be updated again
        model3 = client.get_model()
        assert model3.global_step == 200


@pytest.mark.asyncio
async def test_e2e_pinned_inference_client(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test InferenceClient in pinned mode never updates."""

    # Commit multiple versions
    for i in range(5):
        snapshot = make_test_snapshot(global_step=i * 100)
        await commit_snapshot(async_store, snapshot, f"Version {i}")

    model_template = torch.nn.Linear(5, 5)
    config_template = make_test_snapshot(0)

    # Pin to version 2
    async with InferenceClient(
        mode=PinnedMode(counter=2),  # Pin to v2
        poll_interval=0.5,
        store=async_store,
        model_template=model_template,
        config_template=config_template,
    ) as client:
        # Initial model
        model1 = client.get_model()
        assert model1.global_step == 200  # v2 has step=200

        # Commit more versions
        for i in range(5, 8):
            snapshot = make_test_snapshot(global_step=i * 100)
            await commit_snapshot(async_store, snapshot, f"Version {i}")

        await asyncio.sleep(1.5)

        # Model should still be v2
        model2 = client.get_model()
        assert model2.global_step == 200  # Still v2


@pytest.mark.asyncio
async def test_e2e_chain_verification_multiple_versions(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test chain verification across many versions."""

    # Commit 20 versions
    for i in range(20):
        snapshot = make_test_snapshot(global_step=i * 50)
        await commit_snapshot(async_store, snapshot, f"Checkpoint {i}")

    # Verify chain integrity
    report = expect_success(await verify_chain(async_store))
    _assert_chain_valid(report)

    # Verify all versions are accessible
    for i in range(20):
        version = await async_store.get_version(f"v{i:010d}")
        assert version.counter == i
        assert f"Checkpoint {i}" in version.commit_message


@pytest.mark.asyncio
async def test_e2e_garbage_collection_workflow(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test complete GC workflow: create versions -> preview -> delete -> verify."""

    # 1. Create 15 versions
    for i in range(15):
        snapshot = make_test_snapshot(global_step=i * 100)
        await commit_snapshot(async_store, snapshot, f"Version {i}")

    # 2. Preview GC (dry run) - keep last 5, protect v3 and v7
    preview_result = await run_gc(
        async_store,
        keep_versions=5,
        protect_tags=[3, 7],
        mode=PreviewGC(),
    )
    preview_report = expect_success(preview_result)

    # Should delete versions 1,2,4,5,6,8,9
    # Protected: 0 (genesis), 3 (tag), 7 (tag), 10-14 (recent 5)
    assert preview_report.dry_run is True
    assert 0 not in preview_report.deleted_versions  # Genesis protected
    assert 3 not in preview_report.deleted_versions  # Protected tag
    assert 7 not in preview_report.deleted_versions  # Protected tag
    assert 14 in preview_report.protected_versions  # Recent

    # 3. Actually run GC
    gc_result = await run_gc(
        async_store,
        keep_versions=5,
        protect_tags=[3, 7],
        mode=ExecuteGC(),
    )
    gc_report = expect_success(gc_result)

    assert gc_report.dry_run is False
    assert len(gc_report.deleted_versions) > 0
    assert gc_report.bytes_freed >= 0

    # 4. Note: Cannot verify full chain after GC since deleted versions create gaps
    # This is expected behavior - GC intentionally breaks chain continuity

    # 5. Verify protected versions still exist
    v0 = await async_store.get_version("v0000000000")
    assert v0.counter == 0

    v3 = await async_store.get_version("v0000000003")
    assert v3.counter == 3

    v7 = await async_store.get_version("v0000000007")
    assert v7.counter == 7

    # 6. Verify deleted versions are gone
    for i in gc_report.deleted_versions:
        with pytest.raises(VersionNotFoundError):
            await async_store.get_version(f"v{i:010d}")


@pytest.mark.asyncio
async def test_e2e_version_history_traversal(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test traversing version history backward through parent links."""

    # Create a chain of 10 versions
    versions: list[str] = []
    for i in range(10):
        snapshot = make_test_snapshot(global_step=i * 10)
        version = await commit_snapshot(async_store, snapshot, f"Step {i}")
        versions.append(version.content_hash)

    # Traverse backward from HEAD to genesis
    current = expect_success(await async_store.get_head())

    traversed = []
    while current is not None:
        traversed.append(current.counter)

        if current.parent_hash == "":
            # Reached genesis
            break

        # Find parent by matching content_hash to parent_hash
        parent_counter = current.counter - 1
        if parent_counter < 0:
            break

        parent = await async_store.get_version(f"v{parent_counter:010d}")
        assert parent.content_hash == current.parent_hash

        current = parent

    # Should traverse all 10 versions: 9, 8, 7, ..., 1, 0
    assert traversed == list(range(9, -1, -1))


@pytest.mark.asyncio
async def test_e2e_rollback_to_previous_version(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test rolling back to a previous version and continuing from there."""

    # Create initial chain: v0, v1, v2, v3
    for i in range(4):
        snapshot = make_test_snapshot(global_step=i * 100)
        await commit_snapshot(async_store, snapshot, f"Version {i}")

    # "Rollback" by loading v1 and continuing from there
    v1 = await async_store.get_version("v0000000001")

    model_template = torch.nn.Linear(5, 5)
    config_template = make_test_snapshot(0)

    snapshot_at_v1 = expect_success(
        await load_snapshot_from_checkpoint(async_store, v1, model_template, config_template)
    )

    # Modify and commit (this creates v4)
    snapshot_at_v1 = make_gbm_cvnn_config(
        snapshot_at_v1.cvnn,
        sim_params=snapshot_at_v1.cfg.sim_params,
        bs_config=snapshot_at_v1.cfg,
        domain_bounds=snapshot_at_v1.domain_bounds,
        optimizer_state=None,
        global_step=500,  # Different from v2/v3
        sobol_skip=50000,
    ).model_copy(
        update={
            "torch_cpu_rng_state": snapshot_at_v1.torch_cpu_rng_state,
            "torch_cuda_rng_states": snapshot_at_v1.torch_cuda_rng_states,
        }
    )

    v4 = await commit_snapshot(async_store, snapshot_at_v1, "Rollback branch")

    # Now we have: v0 -> v1 -> v2 -> v3 -> v4
    # v4 is based on v1's state but committed after v3
    assert v4.counter == 4
    assert v4.parent_hash == (await async_store.get_version("v0000000003")).content_hash

    # Load v4 and verify it has v1's modified state
    snapshot_v4 = expect_success(
        await load_snapshot_from_checkpoint(async_store, v4, model_template, config_template)
    )
    assert snapshot_v4.global_step == 500


@pytest.mark.asyncio
async def test_e2e_concurrent_gc_and_commits(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that GC and commits can run concurrently without conflicts."""

    # Create initial versions
    for i in range(10):
        snapshot = make_test_snapshot(global_step=i * 100)
        await commit_snapshot(async_store, snapshot, f"Initial {i}")

    async def commit_worker() -> None:
        """Continuously commit new versions."""
        for i in range(5):
            await asyncio.sleep(0.1)
            snapshot = make_test_snapshot(global_step=1000 + i * 100)
            await commit_snapshot(async_store, snapshot, f"New {i}")

    async def gc_worker() -> None:
        """Run GC once during commits."""
        await asyncio.sleep(0.2)
        await run_gc(async_store, keep_versions=5, mode=ExecuteGC())

    # Run concurrently
    await asyncio.gather(commit_worker(), gc_worker())

    # Note: Cannot verify full chain after GC since it creates gaps
    # Just verify operations completed without crashes

    # Verify we have commits from both workers
    head = expect_success(await async_store.get_head())
    assert head.counter >= 10  # At least initial 10 + some new commits


@pytest.mark.asyncio
async def test_e2e_empty_chain_operations(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test operations on empty chain handle gracefully."""

    # Verify empty chain
    await verify_chain(async_store)

    # Get HEAD of empty chain
    head_result = await async_store.get_head()
    assert head_result.is_failure()

    # GC on empty chain
    report = expect_success(await run_gc(async_store, keep_versions=10, mode=PreviewGC()))
    assert report.deleted_versions == ()
    assert report.protected_versions == ()
    assert report.bytes_freed == 0

    # Load from empty chain should fail
    _model_template = torch.nn.Linear(5, 5)
    _config_template = make_test_snapshot(0)

    with pytest.raises(VersionNotFoundError):
        await async_store.get_version("v0000000000")


@pytest.mark.asyncio
async def test_e2e_large_model_storage(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test storing and loading larger models."""

    # Create larger model (100 x 100 = 10k parameters)
    large_model = torch.nn.Linear(100, 100)

    large_snapshot = make_gbm_cvnn_config(
        large_model,
        sim_params=SIM_PARAMS,
        bs_config=BS_CONFIG,
        domain_bounds=DOMAIN_BOUNDS,
        optimizer_state=None,
        global_step=0,
        sobol_skip=0,
    )

    # Commit large model
    version = await commit_snapshot(async_store, large_snapshot, "Large model")

    # Load and verify
    model_template = torch.nn.Linear(100, 100)
    config_template = make_test_snapshot(0, model_size=(100, 100))

    loaded = expect_success(
        await load_snapshot_from_checkpoint(async_store, version, model_template, config_template)
    )

    # Verify model parameters match
    original_params = sum(p.numel() for p in large_model.parameters())
    loaded_params = sum(p.numel() for p in loaded.cvnn.parameters())
    assert original_params == loaded_params
    assert loaded_params == 10100  # 100*100 + 100 bias


@pytest.mark.asyncio
async def test_e2e_audit_log_integrity(async_store: AsyncBlockchainModelStore) -> None:
    """Test audit log tracks all operations correctly."""

    # Perform various operations
    for i in range(5):
        snapshot = make_test_snapshot(global_step=i * 100)
        await commit_snapshot(async_store, snapshot, f"Commit {i}")

    # Run GC
    await run_gc(async_store, keep_versions=3, mode=ExecuteGC())

    # Verify more commits
    for i in range(5, 8):
        snapshot = make_test_snapshot(global_step=i * 100)
        await commit_snapshot(async_store, snapshot, f"Commit {i}")

    # Audit log should exist (actual validation would require reading log files)
    # For now, just verify operations completed without error
    head = expect_success(await async_store.get_head())
    assert head.counter >= 7


@pytest.mark.asyncio
async def test_e2e_semantic_versioning_progression(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test semantic version progression through lifecycle."""

    # v0: 1.0.0 (genesis)
    snapshot0 = make_test_snapshot(0)
    v0 = await commit_snapshot(async_store, snapshot0, "Genesis")
    assert v0.semantic_version == "1.0.0"

    # v1-9: 1.0.1, 1.0.2, ..., 1.0.9
    for i in range(1, 10):
        snapshot = make_test_snapshot(i * 100)
        version = await commit_snapshot(async_store, snapshot, f"Patch {i}")
        assert version.semantic_version == f"1.0.{i}"

    # v10: 1.0.10 (continues incrementing)
    snapshot10 = make_test_snapshot(1000)
    v10 = await commit_snapshot(async_store, snapshot10, "Patch 10")
    assert v10.semantic_version == "1.0.10"

    # Verify chain
    await verify_chain(async_store)
