# tests/conftest.py
"""Global PyTest fixtures for the test-suite.

CuPy is imported unconditionally-if it is not installed the test session
will fail immediately, making the missing dependency obvious.

All tests require GPU - missing GPU is a hard failure, not a skip.
"""

from __future__ import annotations

import asyncio
import gc
import signal
import uuid
import warnings
from types import FrameType
from typing import AsyncGenerator, Callable, Generator


# CRITICAL: Import facade BEFORE torch for deterministic algorithms
# isort: off
import spectralmc.models.torch  # noqa: F401
import torch

# isort: on

import cupy as cp
import pytest
import botocore.exceptions

from spectralmc.storage import AsyncBlockchainModelStore

# Module-level GPU requirement - test suite fails immediately without GPU
assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"

GPU_DEV: torch.device = torch.device("cuda:0")

DEFAULT_TEST_TIMEOUT_SECONDS = 60.0


def _build_timeout_handler(
    timeout_seconds: float,
) -> Callable[[int, FrameType | None], None]:
    """Create SIGALRM handler that fails the test when timeout is reached."""

    def _handle_timeout(signum: int, frame: FrameType | None) -> None:
        pytest.fail(
            f"Test exceeded {timeout_seconds:.0f}s timeout (includes setup/teardown)",
            pytrace=True,
        )

    return _handle_timeout


def _resolve_timeout_seconds(request: pytest.FixtureRequest) -> float:
    """Return timeout for current test (marker override allowed)."""
    marker = request.node.get_closest_marker("timeout")
    if marker is None:
        return DEFAULT_TEST_TIMEOUT_SECONDS

    raw_value = marker.kwargs.get("seconds", marker.args[0] if marker.args else None)
    if raw_value is None:
        pytest.fail("timeout marker requires seconds argument", pytrace=True)

    seconds = float(raw_value)
    if seconds <= 0:
        pytest.fail("timeout marker must be positive seconds", pytrace=True)

    return seconds


@pytest.fixture(autouse=True)
def per_test_timeout(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Fail any test that runs longer than the default timeout."""
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    timeout_seconds = _resolve_timeout_seconds(request)
    handler = _build_timeout_handler(timeout_seconds)
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


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
        except botocore.exceptions.ClientError as e:
            # Bucket might already exist, that's ok
            if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
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
        except botocore.exceptions.ClientError:
            # Best-effort cleanup - don't fail test if S3 cleanup fails
            pass
