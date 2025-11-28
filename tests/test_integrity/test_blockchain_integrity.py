# tests/test_integrity/test_blockchain_integrity.py
"""Integrity tests for blockchain storage (ModelVersion only)."""

from __future__ import annotations

import pytest

from spectralmc.storage.chain import ModelVersion
from spectralmc.storage import (
    AsyncBlockchainModelStore,
    verify_chain,
    verify_chain_detailed,
    find_corruption,
    commit_snapshot,
)
from spectralmc.storage.errors import ChainCorruptionError
from spectralmc.gbm_trainer import GbmCVNNPricerConfig
from spectralmc.gbm import BlackScholesConfig, SimulationParams
from spectralmc.models.numerical import Precision
import torch


def test_content_hash_integrity() -> None:
    """Test that changing content invalidates hash."""
    version = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent",
        content_hash="original_hash",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    hash1 = version.compute_hash()

    # Different content hash should produce different version hash
    version2 = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent",
        content_hash="modified_hash",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    hash2 = version2.compute_hash()

    assert hash1 != hash2


def test_parent_hash_integrity() -> None:
    """Test that parent hash changes affect version hash."""
    version1 = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent1",
        content_hash="content",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    version2 = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent2",
        content_hash="content",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    assert version1.compute_hash() != version2.compute_hash()


def test_timestamp_integrity() -> None:
    """Test that timestamp changes affect version hash."""
    version1 = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent",
        content_hash="content",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    version2 = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent",
        content_hash="content",
        commit_timestamp="2025-01-02T00:00:00Z",
        commit_message="Test",
    )

    assert version1.compute_hash() != version2.compute_hash()


def test_directory_name_uniqueness() -> None:
    """Test version directory names are unique."""
    versions = [
        ModelVersion(
            counter=i,
            semantic_version=f"1.0.{i}",
            parent_hash="",
            content_hash=f"hash{i}",
            commit_timestamp="2025-01-01T00:00:00Z",
            commit_message="Test",
        )
        for i in range(100)
    ]

    dir_names = [v.directory_name for v in versions]

    # All directory names should be unique
    assert len(dir_names) == len(set(dir_names))


def test_version_immutability() -> None:
    """Test ModelVersion is immutable (frozen dataclass)."""
    version = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent",
        content_hash="content",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    # Should raise FrozenInstanceError when attempting to mutate frozen field
    # Use object.__setattr__ to bypass type checking for testing immutability
    with pytest.raises(Exception):
        object.__setattr__(version, "counter", 999)


def test_hash_length_consistency() -> None:
    """Test all hashes are 64-character SHA256 hex."""
    versions = [
        ModelVersion(
            counter=i,
            semantic_version=f"1.0.{i}",
            parent_hash=f"parent{i}" * 8,
            content_hash=f"content{i}" * 8,
            commit_timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            commit_message=f"Message {i}",
        )
        for i in range(20)
    ]

    for version in versions:
        version_hash = version.compute_hash()
        assert len(version_hash) == 64
        assert all(c in "0123456789abcdef" for c in version_hash)


# ============================================================================
# Helper function for creating test configs
# ============================================================================


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


# ============================================================================
# Chain Verification Tests
# ============================================================================


@pytest.mark.asyncio
async def test_verify_empty_chain(async_store: AsyncBlockchainModelStore) -> None:
    """Test verify_chain with empty store (no corruption)."""
    # Empty chain should be valid
    report = await verify_chain_detailed(async_store)
    assert report.is_valid
    assert report.corrupted_version is None
    assert "Empty chain" in report.details


@pytest.mark.asyncio
async def test_verify_single_version_chain(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test verify_chain with genesis only."""
    config = make_test_config(torch.nn.Linear(5, 5))
    await commit_snapshot(async_store, config, "Genesis")

    # Single genesis version should be valid
    is_valid = await verify_chain(async_store)
    assert is_valid


@pytest.mark.asyncio
async def test_verify_multi_version_chain(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test verify_chain with multiple valid versions."""
    # Create 5 versions
    for i in range(5):
        config = make_test_config(torch.nn.Linear(5, 5), global_step=i * 100)
        await commit_snapshot(async_store, config, f"Version {i}")

    # Chain should be valid
    is_valid = await verify_chain(async_store)
    assert is_valid

    report = await verify_chain_detailed(async_store)
    assert report.is_valid
    assert "5 versions intact" in report.details


@pytest.mark.asyncio
async def test_find_corruption_on_valid_chain(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test find_corruption returns None for valid chain."""
    # Create valid chain
    for i in range(3):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Should find no corruption
    corrupted = await find_corruption(async_store)
    assert corrupted is None


@pytest.mark.asyncio
async def test_detect_broken_merkle_chain(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test detection of broken Merkle chain (tampered parent_hash)."""
    # Create 2 versions
    for i in range(2):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Tamper with version 1's metadata.json parent_hash
    import json

    # Find v1 metadata
    assert async_store._s3_client is not None
    paginator = async_store._s3_client.get_paginator("list_objects_v2")
    async for page in paginator.paginate(
        Bucket=async_store.bucket_name, Prefix="versions/v0000000001"
    ):
        if isinstance(page, dict) and "Contents" in page:
            contents = page["Contents"]
            assert isinstance(contents, list)
            for obj in contents:
                if "metadata.json" in obj["Key"]:
                    metadata_key = obj["Key"]

                    assert async_store._s3_client is not None
                    response = await async_store._s3_client.get_object(
                        Bucket=async_store.bucket_name, Key=metadata_key
                    )
                    body = response["Body"]
                    assert hasattr(body, "read")
                    metadata = json.loads((await body.read()).decode("utf-8"))

                    # Corrupt parent_hash (should match v0's content_hash but we change it)
                    metadata["parent_hash"] = "tampered_parent_hash_12345"

                    assert async_store._s3_client is not None
                    await async_store._s3_client.put_object(
                        Bucket=async_store.bucket_name,
                        Key=metadata_key,
                        Body=json.dumps(metadata).encode("utf-8"),
                    )
                    break

    # Should detect corruption
    report = await verify_chain_detailed(async_store)
    assert not report.is_valid
    assert report.corruption_type == "broken_merkle_chain"
    assert "parent_hash" in report.details


@pytest.mark.asyncio
async def test_detect_invalid_genesis_counter(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test detection of invalid genesis counter."""
    # Create version
    config = make_test_config(torch.nn.Linear(5, 5))
    await commit_snapshot(async_store, config, "Genesis")

    # Tamper with counter in metadata
    import json

    version_dir = "v0000000000_1.0.0_"  # Starts with this

    # List objects to find exact version dir
    assert async_store._s3_client is not None
    paginator = async_store._s3_client.get_paginator("list_objects_v2")
    async for page in paginator.paginate(
        Bucket=async_store.bucket_name, Prefix="versions/"
    ):
        if isinstance(page, dict) and "Contents" in page:
            contents = page["Contents"]
            assert isinstance(contents, list)
            for obj in contents:
                if "metadata.json" in obj["Key"]:
                    metadata_key = obj["Key"]

                    # Read metadata
                    assert async_store._s3_client is not None
                    response = await async_store._s3_client.get_object(
                        Bucket=async_store.bucket_name, Key=metadata_key
                    )
                    body = response["Body"]
                    assert hasattr(body, "read")
                    metadata = json.loads((await body.read()).decode("utf-8"))

                    # Corrupt counter
                    metadata["counter"] = 999

                    # Write back
                    assert async_store._s3_client is not None
                    await async_store._s3_client.put_object(
                        Bucket=async_store.bucket_name,
                        Key=metadata_key,
                        Body=json.dumps(metadata).encode("utf-8"),
                    )
                    break

    # Should detect corruption
    report = await verify_chain_detailed(async_store)
    assert not report.is_valid
    assert report.corruption_type == "invalid_genesis_counter"


@pytest.mark.asyncio
async def test_detect_non_sequential_counters(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test detection of non-sequential counters."""
    # Create 3 versions
    for i in range(3):
        config = make_test_config(torch.nn.Linear(5, 5))
        await commit_snapshot(async_store, config, f"V{i}")

    # Tamper with version 2's counter
    import json

    # Find v2 metadata
    assert async_store._s3_client is not None
    paginator = async_store._s3_client.get_paginator("list_objects_v2")
    async for page in paginator.paginate(
        Bucket=async_store.bucket_name, Prefix="versions/v0000000002"
    ):
        if isinstance(page, dict) and "Contents" in page:
            contents = page["Contents"]
            assert isinstance(contents, list)
            for obj in contents:
                if "metadata.json" in obj["Key"]:
                    metadata_key = obj["Key"]

                    assert async_store._s3_client is not None
                    response = await async_store._s3_client.get_object(
                        Bucket=async_store.bucket_name, Key=metadata_key
                    )
                    body = response["Body"]
                    assert hasattr(body, "read")
                    metadata = json.loads((await body.read()).decode("utf-8"))

                    # Skip counter (2 → 5)
                    metadata["counter"] = 5

                    assert async_store._s3_client is not None
                    await async_store._s3_client.put_object(
                        Bucket=async_store.bucket_name,
                        Key=metadata_key,
                        Body=json.dumps(metadata).encode("utf-8"),
                    )
                    break

    # Should detect corruption (Note: will fail to fetch v3, v4 first)
    report = await verify_chain_detailed(async_store)
    assert not report.is_valid
    assert report.corruption_type is not None
    assert (
        "missing" in report.corruption_type.lower()
        or "non_sequential" in report.corruption_type.lower()
    )


@pytest.mark.asyncio
async def test_chain_corruption_error_raised(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that verify_chain raises ChainCorruptionError on corruption."""
    # Create version
    config = make_test_config(torch.nn.Linear(5, 5))
    await commit_snapshot(async_store, config, "Genesis")

    # Corrupt genesis parent_hash
    import json

    # Find metadata
    assert async_store._s3_client is not None
    paginator = async_store._s3_client.get_paginator("list_objects_v2")
    async for page in paginator.paginate(
        Bucket=async_store.bucket_name, Prefix="versions/v0000000000"
    ):
        if isinstance(page, dict) and "Contents" in page:
            contents = page["Contents"]
            assert isinstance(contents, list)
            for obj in contents:
                if "metadata.json" in obj["Key"]:
                    metadata_key = obj["Key"]

                    assert async_store._s3_client is not None
                    response = await async_store._s3_client.get_object(
                        Bucket=async_store.bucket_name, Key=metadata_key
                    )
                    body = response["Body"]
                    assert hasattr(body, "read")
                    metadata = json.loads((await body.read()).decode("utf-8"))

                    # Corrupt parent_hash
                    metadata["parent_hash"] = "invalid"

                    assert async_store._s3_client is not None
                    await async_store._s3_client.put_object(
                        Bucket=async_store.bucket_name,
                        Key=metadata_key,
                        Body=json.dumps(metadata).encode("utf-8"),
                    )
                    break

    # Should raise ChainCorruptionError
    with pytest.raises(ChainCorruptionError, match="corruption detected"):
        await verify_chain(async_store)


@pytest.mark.asyncio
async def test_verify_chain_with_long_history(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test verify_chain with many versions (performance test)."""
    # Create 20 versions
    for i in range(20):
        config = make_test_config(torch.nn.Linear(5, 5), global_step=i)
        await commit_snapshot(async_store, config, f"V{i}")

    # Should verify successfully
    is_valid = await verify_chain(async_store)
    assert is_valid

    report = await verify_chain_detailed(async_store)
    assert "20 versions intact" in report.details
