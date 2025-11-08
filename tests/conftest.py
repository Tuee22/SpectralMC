# tests/conftest.py
"""Global PyTest fixtures for the test-suite.

CuPy is imported unconditionally—if it is not installed the test session
will fail immediately, making the missing dependency obvious.
"""

from __future__ import annotations

import spectralmc.models.torch

import gc
import warnings
from typing import Generator

import cupy as cp
import pytest
import torch


def _free_cupy() -> None:
    """Release CuPy memory pools.

    Any exception is turned into a *RuntimeWarning* so users see the problem
    instead of silently proceeding.
    """
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"CuPy memory-pool cleanup failed: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )


@pytest.fixture(autouse=True)
def cleanup_gpu() -> Generator[None, None, None]:
    """Auto-fixture that frees GPU memory after *every* test."""
    yield
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"torch.cuda.empty_cache() failed: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )

    _free_cupy()


# =========================================================================== #
#                   ASYNC STORAGE TEST FIXTURES                               #
# =========================================================================== #

import asyncio
import uuid
from typing import AsyncGenerator

from spectralmc.storage import AsyncBlockchainModelStore


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create event loop for async tests (session scope).

    pytest-asyncio requires an event_loop fixture for async tests.
    Using session scope allows reusing the loop across tests for efficiency.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_store() -> AsyncGenerator[AsyncBlockchainModelStore, None]:
    """
    Create AsyncBlockchainModelStore with unique test bucket.

    Creates a unique bucket for each test to ensure isolation.
    Uses MinIO from docker-compose (opt-models bucket).
    Automatically cleans up all objects and bucket after test.

    Usage:
        @pytest.mark.asyncio
        async def test_something(async_store: AsyncBlockchainModelStore) -> None:
            version = await async_store.commit(data, hash, "msg")
            assert version.counter == 0
    """
    # Use unique test bucket per test for isolation
    bucket_name = f"test-{uuid.uuid4().hex[:12]}"

    async with AsyncBlockchainModelStore(bucket_name) as store:
        # Create test bucket
        assert store._s3_client is not None
        try:
            await store._s3_client.create_bucket(Bucket=bucket_name)
        except Exception as e:
            # Bucket might already exist, that's ok
            if "BucketAlreadyOwnedByYou" not in str(e):
                raise

        yield store

        # Cleanup: Delete all objects in bucket, then delete bucket
        assert store._s3_client is not None
        try:
            # List and delete all objects
            paginator = store._s3_client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket_name):
                if isinstance(page, dict) and "Contents" in page:
                    contents = page["Contents"]
                    assert isinstance(contents, list)
                    objects = [{"Key": obj["Key"]} for obj in contents]
                    if objects:
                        await store._s3_client.delete_objects(
                            Bucket=bucket_name, Delete={"Objects": objects}
                        )

            # Delete bucket
            await store._s3_client.delete_bucket(Bucket=bucket_name)
        except Exception:
            # Best-effort cleanup - don't fail test if cleanup fails
            pass
