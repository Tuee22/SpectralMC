# tests/test_storage/test_rollback.py
"""Tests for automatic rollback on commit failures."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError


from spectralmc.result import Failure, Success
from spectralmc.storage import AsyncBlockchainModelStore
from spectralmc.storage.errors import CommitError, ConflictError
from spectralmc.storage.protocols import S3ResponseProtocol
from spectralmc.storage.store import JsonDict


@pytest.mark.asyncio
async def test_rollback_on_cas_failure(async_store: AsyncBlockchainModelStore) -> None:
    """
    Test that artifacts are rolled back when CAS write fails.

    When a conflict is detected during the CAS write, the system should:
    1. Detect the ETag mismatch (PreconditionFailed)
    2. Clean up the uploaded artifacts (checkpoint.pb, metadata.json, etc.)
    3. Raise ConflictError
    """
    # Create genesis commit
    checkpoint1 = b"checkpoint 1"
    hash1 = "hash1"
    version1 = await async_store.commit(checkpoint1, hash1, "First")
    assert version1.counter == 0

    # Get current ETag
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    response = await async_store._s3_client.get_object(
        Bucket=async_store.bucket_name,
        Key="chain.json",
    )
    etag1_raw = response["ETag"]
    assert isinstance(etag1_raw, str)
    etag1 = etag1_raw.strip('"')

    # Simulate concurrent modification by committing again
    checkpoint2 = b"checkpoint 2"
    hash2 = "hash2"
    _version2 = await async_store.commit(checkpoint2, hash2, "Second")

    # Get new ETag (should be different)
    response = await async_store._s3_client.get_object(
        Bucket=async_store.bucket_name,
        Key="chain.json",
    )
    etag2_raw = response["ETag"]
    assert isinstance(etag2_raw, str)
    etag2 = etag2_raw.strip('"')
    assert etag1 != etag2

    # Now attempt a commit that will fail due to stale ETag
    # We'll patch the commit method to simulate reading stale ETag
    original_get_object = async_store._s3_client.get_object

    async def get_object_with_stale_etag(*args: object, **kwargs: object) -> S3ResponseProtocol:
        """Return stale ETag to simulate race condition."""
        result = await original_get_object(*args, **kwargs)
        if kwargs.get("Key") == "chain.json":
            # Return the old ETag to force conflict - we need to mutate the response
            if not isinstance(result, dict):
                raise TypeError("Expected dict response from get_object in test double")
            mutable_result = dict(result)
            mutable_result["ETag"] = f'"{etag1}"'
            return mutable_result
        return result

    # Patch get_object to return stale ETag
    with patch.object(async_store._s3_client, "get_object", side_effect=get_object_with_stale_etag):
        checkpoint3 = b"checkpoint 3"
        hash3 = "hash3"

        # This should fail with ConflictError
        with pytest.raises(ConflictError, match="Concurrent commit detected"):
            await async_store.commit(checkpoint3, hash3, "Third - should fail")

    # Verify rollback: version 3 artifacts should not exist
    version_dir = "v2_1.0.2"  # Would be version 2's directory

    # Check that artifacts were cleaned up
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    with pytest.raises(ClientError) as exc_info:
        await async_store._s3_client.head_object(
            Bucket=async_store.bucket_name, Key=f"versions/{version_dir}/checkpoint.pb"
        )
    assert exc_info.value.response["Error"]["Code"] == "404"

    # Verify chain.json is still at version 2 (not corrupted)
    head_result = await async_store.get_head()
    match head_result:
        case Success(value=head):
            assert head.counter == 1
            assert head.content_hash == hash2
        case Failure(error=_):
            pytest.fail("Expected get_head() to return Success")


@pytest.mark.asyncio
async def test_rollback_on_upload_failure(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """
    Test that partial uploads are cleaned up when artifact upload fails.

    If uploading checkpoint.pb succeeds but metadata.json fails,
    the system should clean up the partial artifacts.
    """
    checkpoint = b"test checkpoint"
    content_hash = "test_hash"

    # Mock upload to fail on metadata.json
    original_upload_json = async_store._upload_json
    upload_call_count = 0

    async def upload_json_with_failure(key: str, data: JsonDict) -> None:
        """Fail on first metadata.json upload."""
        nonlocal upload_call_count
        upload_call_count += 1
        if upload_call_count == 1 and key.endswith("metadata.json"):
            raise ClientError(
                {"Error": {"Code": "InternalError", "Message": "Simulated failure"}},
                "PutObject",
            )
        await original_upload_json(key, data)

    with patch.object(async_store, "_upload_json", side_effect=upload_json_with_failure):
        # Commit should fail
        with pytest.raises(CommitError, match="Failed to upload artifacts"):
            await async_store.commit(checkpoint, content_hash, "Should fail on upload")

    # Verify no artifacts were left behind
    # List all objects in versions/
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    paginator = async_store._s3_client.get_paginator("list_objects_v2")
    all_objects = []
    async for page in paginator.paginate(Bucket=async_store.bucket_name, Prefix="versions/"):
        if isinstance(page, dict) and "Contents" in page:
            contents = page["Contents"]
            assert isinstance(contents, list)
            all_objects.extend([obj["Key"] for obj in contents])

    # Should be empty (genesis commit was never completed)
    assert len(all_objects) == 0, f"Found leftover artifacts: {all_objects}"

    # Verify chain.json doesn't exist (genesis never succeeded)
    with pytest.raises(ClientError) as exc_info:
        await async_store._s3_client.head_object(Bucket=async_store.bucket_name, Key="chain.json")
    assert exc_info.value.response["Error"]["Code"] == "404"
