# mypy: warn_unreachable=False
# tests/test_storage/test_context_manager.py
"""Tests for async context manager functionality."""

from __future__ import annotations

import pytest
import torch

from spectralmc.storage import AsyncBlockchainModelStore

assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"


@pytest.mark.asyncio
async def test_context_manager_cleanup() -> None:
    """
    Test that async context manager properly initializes and cleans up.

    The context manager should:
    1. Create aioboto3 session on __aenter__
    2. Create S3 client on __aenter__
    3. Close S3 client on __aexit__
    4. Set client to None after cleanup
    """
    bucket_name = "test-context-cleanup"

    store = AsyncBlockchainModelStore(bucket_name)

    # Before entering context, client should not exist
    assert not hasattr(store, "_s3_client") or store._s3_client is None

    async with store:
        # Inside context, client should exist
        assert store._s3_client is not None
        assert store._client_context is not None

        # Should be able to create bucket
        await store._s3_client.create_bucket(Bucket=bucket_name)

        # Verify bucket exists
        response = await store._s3_client.list_buckets()
        assert isinstance(response, dict)
        assert "Buckets" in response
        buckets = response["Buckets"]
        assert isinstance(buckets, list)
        bucket_names = [b["Name"] for b in buckets]
        assert bucket_name in bucket_names

        # Cleanup bucket
        await store._s3_client.delete_bucket(Bucket=bucket_name)

    # After exiting context, client should be None
    assert store._s3_client is None and store._client_context is None


@pytest.mark.asyncio
async def test_context_manager_exception_handling() -> None:
    """
    Test that cleanup happens even if an exception occurs inside the context.

    The __aexit__ method should be called even when an exception is raised,
    ensuring proper resource cleanup.
    """
    bucket_name = "test-context-exception"

    store = AsyncBlockchainModelStore(bucket_name)

    # Simulate an exception inside the context
    async with store:
        # Verify client was created
        assert store._s3_client is not None

        # Create bucket for cleanup test
        await store._s3_client.create_bucket(Bucket=bucket_name)

    # Re-enter context and raise exception
    with pytest.raises(ValueError, match="Simulated error"):
        async with store:
            raise ValueError("Simulated error")

    # Even after exception, cleanup should have occurred
    assert store._s3_client is None and store._client_context is None

    # Verify we can still use a new context with the same store instance
    async with store:
        assert store._s3_client is not None

        # Bucket should still exist from before exception
        response = await store._s3_client.list_buckets()
        bucket_names = [b["Name"] for b in response["Buckets"]]
        assert bucket_name in bucket_names

        # Cleanup
        await store._s3_client.delete_bucket(Bucket=bucket_name)
