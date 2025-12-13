# tests/test_storage/test_fast_forward.py
"""Tests for fast-forward validation during commits."""

from __future__ import annotations

from unittest.mock import patch

import pytest


from spectralmc.result import Failure, Success
from spectralmc.storage import AsyncBlockchainModelStore
from spectralmc.storage.errors import ConflictError


@pytest.mark.asyncio
async def test_fast_forward_validation(async_store: AsyncBlockchainModelStore) -> None:
    """
    Test that non-fast-forward commits are rejected.

    If chain.json changes between get_head() and CAS write (indicating
    a concurrent commit), the fast-forward check should detect this and
    raise ConflictError with rollback.
    """
    # Create genesis commit
    checkpoint1 = b"checkpoint 1"
    hash1 = "hash1"
    version1 = await async_store.commit(checkpoint1, hash1, "First")
    assert version1.counter == 0

    # Create second commit normally
    checkpoint2 = b"checkpoint 2"
    hash2 = "hash2"
    version2 = await async_store.commit(checkpoint2, hash2, "Second")
    assert version2.counter == 1

    # Now simulate a concurrent commit by patching get_head to return stale data
    # while chain.json has actually moved forward
    _original_get_head = async_store.get_head

    async def get_stale_head() -> object:
        """Return version1 as HEAD (stale)."""
        return Success(version1)  # Return stale version wrapped in Success

    # Prepare third checkpoint
    checkpoint3 = b"checkpoint 3"
    hash3 = "hash3"

    # Patch get_head to return stale version
    with patch.object(async_store, "get_head", side_effect=get_stale_head):
        # Attempt to commit - should fail fast-forward check
        with pytest.raises(ConflictError, match="Concurrent commit detected"):
            await async_store.commit(checkpoint3, hash3, "Third - should fail")

    # Verify rollback occurred - no version 3 artifacts
    # (would be v2_1.0.2 if it had succeeded)

    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    paginator = async_store._s3_client.get_paginator("list_objects_v2")
    version_dirs = []
    async for page in paginator.paginate(Bucket=async_store.bucket_name, Prefix="versions/"):
        if isinstance(page, dict) and "Contents" in page:
            contents = page["Contents"]
            assert isinstance(contents, list)
            for obj in contents:
                key = obj["Key"]
                # Extract version directory
                parts = key.split("/")
                if len(parts) >= 2:
                    version_dirs.append(parts[1])

    version_dirs = list(set(version_dirs))  # Deduplicate

    # Should only have v0 and v1, not v2
    assert (
        len(version_dirs) == 2
    ), f"Expected 2 version directories, got {len(version_dirs)}: {version_dirs}"
    assert "v0000000000_1.0.0_hash1" in version_dirs
    assert "v0000000001_1.0.1_hash2" in version_dirs

    # Verify HEAD is still at version 2
    head_result = await async_store.get_head()
    match head_result:
        case Success(head):
            assert head.counter == 1
            assert head.content_hash == hash2
        case Failure(_):
            pytest.fail("Expected HEAD")
