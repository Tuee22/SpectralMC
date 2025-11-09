# tests/test_storage/test_store.py
"""Tests for blockchain model store."""

from __future__ import annotations

import pytest

from spectralmc.result import Success, Failure
from spectralmc.storage import AsyncBlockchainModelStore
from spectralmc.storage.chain import ModelVersion
from spectralmc.storage.errors import VersionNotFoundError, HeadNotFoundError


@pytest.mark.asyncio
async def test_store_initialization(async_store: AsyncBlockchainModelStore) -> None:
    """Test store initializes correctly."""
    # S3 store doesn't have filesystem paths, but should be usable
    assert async_store.bucket_name.startswith("test-")
    assert async_store._s3_client is not None


@pytest.mark.asyncio
async def test_get_head_empty(async_store: AsyncBlockchainModelStore) -> None:
    """Test get_head returns None when no commits exist."""
    head_result = await async_store.get_head()
    assert head_result.is_failure()


@pytest.mark.asyncio
async def test_commit_genesis(async_store: AsyncBlockchainModelStore) -> None:
    """Test genesis commit creation."""
    checkpoint_data = b"test checkpoint data"
    content_hash = "abc123def456"

    version = await async_store.commit(checkpoint_data, content_hash, "Initial commit")

    assert version.counter == 0
    assert version.semantic_version == "1.0.0"
    assert version.parent_hash == ""
    assert version.content_hash == content_hash
    assert version.commit_message == "Initial commit"


@pytest.mark.asyncio
async def test_commit_incremental(async_store: AsyncBlockchainModelStore) -> None:
    """Test incremental commits."""
    # First commit
    checkpoint1 = b"checkpoint 1"
    hash1 = "hash1"
    version1 = await async_store.commit(checkpoint1, hash1, "First")

    # Second commit
    checkpoint2 = b"checkpoint 2"
    hash2 = "hash2"
    version2 = await async_store.commit(checkpoint2, hash2, "Second")

    assert version2.counter == 1
    assert version2.semantic_version == "1.0.1"
    assert version2.parent_hash == hash1
    assert version2.content_hash == hash2
    assert version2.commit_message == "Second"


@pytest.mark.asyncio
async def test_get_head_after_commit(async_store: AsyncBlockchainModelStore) -> None:
    """Test get_head returns latest commit."""
    checkpoint1 = b"checkpoint 1"
    version1 = await async_store.commit(checkpoint1, "hash1", "First")

    head_result = await async_store.get_head()
    match head_result:
        case Success(head):
            assert head.counter == version1.counter
            assert head.semantic_version == version1.semantic_version
        case Failure(_):
            pytest.fail("Expected HEAD to exist")


@pytest.mark.asyncio
async def test_commit_chain(async_store: AsyncBlockchainModelStore) -> None:
    """Test multiple commits form proper chain."""
    versions = []
    for i in range(5):
        checkpoint = f"checkpoint {i}".encode()
        content_hash = f"hash{i}"
        message = f"Commit {i}"
        version = await async_store.commit(checkpoint, content_hash, message)
        versions.append(version)

    # Verify chain properties
    for i, version in enumerate(versions):
        assert version.counter == i
        if i == 0:
            assert version.parent_hash == ""
        else:
            assert version.parent_hash == versions[i - 1].content_hash

    # Verify HEAD points to latest
    head_result = await async_store.get_head()
    match head_result:
        case Success(head):
            assert head.counter == 4
        case Failure(_):
            pytest.fail("Expected HEAD to exist")


@pytest.mark.asyncio
async def test_get_version_by_id(async_store: AsyncBlockchainModelStore) -> None:
    """Test retrieving version by ID."""
    checkpoint = b"test checkpoint"
    content_hash = "abcdef123456"
    version = await async_store.commit(checkpoint, content_hash, "Test")

    retrieved = await async_store.get_version("v0000000000")
    assert retrieved.counter == version.counter
    assert retrieved.content_hash == version.content_hash


@pytest.mark.asyncio
async def test_get_version_not_found(async_store: AsyncBlockchainModelStore) -> None:
    """Test VersionNotFoundError for non-existent version."""
    with pytest.raises(VersionNotFoundError) as exc_info:
        await async_store.get_version("v9999999999")

    assert "v9999999999" in str(exc_info.value)


@pytest.mark.asyncio
async def test_load_checkpoint(async_store: AsyncBlockchainModelStore) -> None:
    """Test loading checkpoint data."""
    checkpoint_data = b"test checkpoint binary data"
    content_hash = "checksum123"
    version = await async_store.commit(checkpoint_data, content_hash)

    loaded_data = await async_store.load_checkpoint(version)
    assert loaded_data == checkpoint_data


@pytest.mark.asyncio
async def test_semantic_versioning_progression(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test semantic version increments correctly."""
    versions = []
    for i in range(10):
        checkpoint = f"checkpoint {i}".encode()
        version = await async_store.commit(checkpoint, f"hash{i}")
        versions.append(version)

    # All should be patch increments (1.0.0 -> 1.0.1 -> ... -> 1.0.9)
    assert versions[0].semantic_version == "1.0.0"
    assert versions[1].semantic_version == "1.0.1"
    assert versions[5].semantic_version == "1.0.5"
    assert versions[9].semantic_version == "1.0.9"
