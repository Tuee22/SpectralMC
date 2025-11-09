# tests/test_storage/test_atomic_cas.py
"""Tests for atomic Compare-And-Swap (CAS) commit logic."""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from spectralmc.result import Success, Failure
from spectralmc.storage import AsyncBlockchainModelStore


@pytest.mark.asyncio
async def test_concurrent_commit_etag_changes(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """
    Test that ETag changes with each commit, enabling CAS detection.

    This verifies the fundamental mechanism that enables atomic commits.
    """
    # First commit
    checkpoint1 = b"checkpoint 1"
    hash1 = "hash1"
    version1 = await async_store.commit(checkpoint1, hash1, "First")
    assert version1.counter == 0

    # Get ETag after first commit
    assert async_store._s3_client is not None
    response1 = await async_store._s3_client.get_object(
        Bucket=async_store.bucket_name,
        Key="chain.json",
    )
    etag1_raw = response1["ETag"]
    assert isinstance(etag1_raw, str)
    etag1 = etag1_raw.strip('"')

    # Second commit
    checkpoint2 = b"checkpoint 2"
    hash2 = "hash2"
    version2 = await async_store.commit(checkpoint2, hash2, "Second")
    assert version2.counter == 1

    # Get ETag after second commit
    assert async_store._s3_client is not None
    response2 = await async_store._s3_client.get_object(
        Bucket=async_store.bucket_name,
        Key="chain.json",
    )
    etag2_raw = response2["ETag"]
    assert isinstance(etag2_raw, str)
    etag2 = etag2_raw.strip('"')

    # ETags should be different (this enables conflict detection)
    assert etag1 != etag2, "ETag must change with each commit for CAS to work"

    # Verify we can't write with stale ETag
    with pytest.raises(ClientError) as exc_info:
        assert async_store._s3_client is not None
        await async_store._s3_client.put_object(
            Bucket=async_store.bucket_name,
            Key="chain.json",
            Body=b"should fail",
            IfMatch=etag1,  # Stale ETag
        )

    assert exc_info.value.response["Error"]["Code"] == "PreconditionFailed"


@pytest.mark.asyncio
async def test_cas_etag_mismatch() -> None:
    """
    Test CAS failure when ETag doesn't match.

    Simulates external modification of chain.json between
    fetch and write operations.
    """
    # Create two separate stores to simulate concurrent access
    bucket_name = "test-cas-etag"

    async with AsyncBlockchainModelStore(bucket_name) as store1:
        # Create bucket
        assert store1._s3_client is not None
        try:
            await store1._s3_client.create_bucket(Bucket=bucket_name)
        except Exception:
            pass

        # First commit
        checkpoint1 = b"checkpoint 1"
        hash1 = "hash1"
        version1 = await store1.commit(checkpoint1, hash1, "First")

        assert version1.counter == 0

        # Get current ETag
        response = await store1._s3_client.get_object(
            Bucket=bucket_name,
            Key="chain.json",
        )
        old_etag_raw = response["ETag"]
        assert isinstance(old_etag_raw, str)
        old_etag = old_etag_raw.strip('"')

        # Another process commits (changes ETag)
        checkpoint2 = b"checkpoint 2"
        hash2 = "hash2"
        version2 = await store1.commit(checkpoint2, hash2, "Second")

        # Get new ETag (should be different)
        response = await store1._s3_client.get_object(
            Bucket=bucket_name,
            Key="chain.json",
        )
        new_etag_raw = response["ETag"]
        assert isinstance(new_etag_raw, str)
        new_etag = new_etag_raw.strip('"')

        assert old_etag != new_etag, "ETag should change after commit"

        # Try to write with old ETag - should fail
        with pytest.raises(ClientError) as exc_info:
            await store1._s3_client.put_object(
                Bucket=bucket_name,
                Key="chain.json",
                Body=b"fake data",
                IfMatch=old_etag,
            )

        assert exc_info.value.response["Error"]["Code"] == "PreconditionFailed"

        # Cleanup: Delete all objects first, then bucket
        paginator = store1._s3_client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket_name):
            if isinstance(page, dict) and "Contents" in page:
                contents = page["Contents"]
                assert isinstance(contents, list)
                objects = [{"Key": obj["Key"]} for obj in contents]
                if objects:
                    await store1._s3_client.delete_objects(
                        Bucket=bucket_name, Delete={"Objects": objects}
                    )
        await store1._s3_client.delete_bucket(Bucket=bucket_name)


@pytest.mark.asyncio
async def test_genesis_commit_no_etag(async_store: AsyncBlockchainModelStore) -> None:
    """
    Test that genesis commit works without ETag.

    The first commit to an empty bucket should not require
    an ETag for the CAS operation.
    """
    # Verify bucket is empty
    head_result = await async_store.get_head()
    match head_result:
        case Success(_):
            pytest.fail("Bucket should start empty")
        case Failure(_):
            pass  # Expected - no HEAD yet

    # Genesis commit should succeed without ETag
    checkpoint = b"genesis checkpoint"
    content_hash = "genesis_hash"

    version = await async_store.commit(checkpoint, content_hash, "Genesis")

    assert version.counter == 0
    assert version.semantic_version == "1.0.0"
    assert version.parent_hash == ""
    assert version.commit_message == "Genesis"

    # Verify HEAD was set
    head_result = await async_store.get_head()
    match head_result:
        case Success(head):
            assert head.counter == 0
            assert head.content_hash == content_hash
        case Failure(_):
            pytest.fail("Expected HEAD to exist")
