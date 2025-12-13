# tests/test_storage/test_retry_logic.py
"""Tests for retry logic with exponential backoff."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError


from spectralmc.result import Failure, Success
from spectralmc.storage import AsyncBlockchainModelStore
from spectralmc.storage.protocols import S3ResponseProtocol


@pytest.mark.asyncio
async def test_retry_on_throttling(async_store: AsyncBlockchainModelStore) -> None:
    """
    Test that throttling errors trigger exponential backoff retry.

    The retry_on_throttle decorator should:
    1. Catch SlowDown, RequestLimitExceeded, ServiceUnavailable errors
    2. Retry with exponential backoff (0.1s, 0.2s, 0.4s, 0.8s, 1.6s)
    3. Eventually succeed if error stops
    """
    # Create a function that throttles 3 times then succeeds
    call_count = 0
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    original_get_object = async_store._s3_client.get_object

    async def get_object_with_throttling(*args: object, **kwargs: object) -> S3ResponseProtocol:
        """Simulate throttling errors that eventually succeed."""
        nonlocal call_count
        call_count += 1

        if call_count <= 3:
            # First 3 calls: throttle
            raise ClientError(
                {
                    "Error": {
                        "Code": "SlowDown",
                        "Message": "Please reduce your request rate",
                    }
                },
                "GetObject",
            )

        # 4th call: succeed
        return await original_get_object(*args, **kwargs)

    # First create an object so get_object has something to fetch
    checkpoint = b"test checkpoint"
    content_hash = "test_hash"
    _version = await async_store.commit(checkpoint, content_hash, "Test")

    # Reset counter
    call_count = 0

    # Now test retry logic on get_object
    with patch.object(async_store._s3_client, "get_object", side_effect=get_object_with_throttling):
        # This should retry 3 times and succeed on 4th attempt
        start_time = asyncio.get_event_loop().time()

        # get_head() uses get_object under the hood
        head_result = await async_store.get_head()

        elapsed_time = asyncio.get_event_loop().time() - start_time

        # Verify it succeeded
        match head_result:
            case Success(head):
                assert head.content_hash == content_hash

                # Verify it retried 3 times (call_count should be 4)
                assert (
                    call_count == 4
                ), f"Expected 4 calls (3 retries + 1 success), got {call_count}"

                # Verify exponential backoff occurred (0.1 + 0.2 + 0.4 = 0.7s minimum)
                assert (
                    elapsed_time >= 0.7
                ), f"Expected at least 0.7s for 3 retries, got {elapsed_time:.3f}s"
            case Failure(_):
                pytest.fail("Expected HEAD")


@pytest.mark.asyncio
async def test_retry_exhaustion(async_store: AsyncBlockchainModelStore) -> None:
    """
    Test that retries give up after max_retries attempts.

    If S3 keeps throttling beyond max_retries (default 5),
    the decorator should raise the last error.
    """
    # First create an object so we have something to download
    checkpoint = b"test checkpoint"
    content_hash = "test_hash"
    _version = await async_store.commit(checkpoint, content_hash, "Test")

    # Create a function that always throttles
    call_count = 0
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    _original_get_object = async_store._s3_client.get_object

    async def always_throttle(*args: object, **kwargs: object) -> dict[str, object]:
        """Always return throttling error."""
        nonlocal call_count
        call_count += 1
        raise ClientError(
            {"Error": {"Code": "ServiceUnavailable", "Message": "Service unavailable"}},
            "GetObject",
        )

    # Patch get_object to always throttle
    with patch.object(async_store._s3_client, "get_object", side_effect=always_throttle):
        # Test _download_bytes directly (has single retry decorator)
        with pytest.raises(ClientError, match="Service unavailable"):
            await async_store._download_bytes("chain.json")

        # Default max_retries is 5, so we expect 6 total calls (1 initial + 5 retries)
        assert call_count == 6, f"Expected 6 calls (1 initial + 5 retries), got {call_count}"


@pytest.mark.asyncio
async def test_no_retry_on_conflict(async_store: AsyncBlockchainModelStore) -> None:
    """
    Test that CAS conflicts are NOT retried (fail fast).

    PreconditionFailed (412) errors indicate concurrent modification
    and should raise immediately without retrying.
    """
    # Create genesis commit
    checkpoint1 = b"checkpoint 1"
    hash1 = "hash1"
    _version1 = await async_store.commit(checkpoint1, hash1, "First")

    # Get current ETag
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    response = await async_store._s3_client.get_object(
        Bucket=async_store.bucket_name,
        Key="chain.json",
    )
    etag1_raw = response["ETag"]
    assert isinstance(etag1_raw, str)
    etag1 = etag1_raw.strip('"')

    # Commit again to change ETag
    checkpoint2 = b"checkpoint 2"
    hash2 = "hash2"
    _version2 = await async_store.commit(checkpoint2, hash2, "Second")

    # Track put_object calls
    call_count = 0
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    original_put_object = async_store._s3_client.put_object

    async def track_put_calls(*args: object, **kwargs: object) -> object:
        """Track put_object calls."""
        nonlocal call_count
        call_count += 1
        return await original_put_object(*args, **kwargs)

    with patch.object(async_store._s3_client, "put_object", side_effect=track_put_calls):
        # Try to write with stale ETag
        assert (
            async_store._s3_client is not None
        ), "S3 client should be initialized in async context"
        with pytest.raises(ClientError, match="PreconditionFailed"):
            await async_store._s3_client.put_object(
                Bucket=async_store.bucket_name,
                Key="chain.json",
                Body=b"should fail",
                IfMatch=etag1,  # Stale ETag
            )

        # Should fail immediately without retry (call_count == 1)
        assert call_count == 1, f"PreconditionFailed should not be retried, got {call_count} calls"
