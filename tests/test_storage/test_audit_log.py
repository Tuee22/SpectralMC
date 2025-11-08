# tests/test_storage/test_audit_log.py
"""Tests for audit log functionality."""

from __future__ import annotations

import json
import pytest
from botocore.exceptions import ClientError
from unittest.mock import patch

from spectralmc.storage import AsyncBlockchainModelStore


@pytest.mark.asyncio
async def test_audit_log_append(async_store: AsyncBlockchainModelStore) -> None:
    """
    Test that audit log entries are created after successful commits.

    Each commit should create a timestamped JSONL file in audit_log/
    containing the version metadata.
    """
    # Create 3 commits
    versions = []
    for i in range(3):
        checkpoint = f"checkpoint {i}".encode()
        content_hash = f"hash{i}"
        version = await async_store.commit(checkpoint, content_hash, f"Commit {i}")
        versions.append(version)

    # List all audit log entries
    assert async_store._s3_client is not None
    paginator = async_store._s3_client.get_paginator("list_objects_v2")
    audit_entries = []
    async for page in paginator.paginate(
        Bucket=async_store.bucket_name, Prefix="audit_log/"
    ):
        if isinstance(page, dict) and "Contents" in page:
            contents = page["Contents"]
            assert isinstance(contents, list)
            audit_entries.extend([obj["Key"] for obj in contents])

    # Should have 3 audit log entries
    assert (
        len(audit_entries) == 3
    ), f"Expected 3 audit log entries, got {len(audit_entries)}"

    # Verify each entry contains correct metadata
    for i, entry_key in enumerate(sorted(audit_entries)):
        # Download and parse the log entry
        assert async_store._s3_client is not None
        response = await async_store._s3_client.get_object(
            Bucket=async_store.bucket_name,
            Key=entry_key,
        )
        body = response["Body"]
        assert hasattr(body, "read")
        log_data = await body.read()
        log_entry = json.loads(log_data.decode("utf-8"))

        # Verify log entry matches version metadata
        expected_version = versions[i]
        assert log_entry["version_id"] == expected_version.version_id
        assert log_entry["counter"] == expected_version.counter
        assert log_entry["semantic_version"] == expected_version.semantic_version
        assert log_entry["parent_hash"] == expected_version.parent_hash
        assert log_entry["content_hash"] == expected_version.content_hash
        assert log_entry["commit_message"] == expected_version.commit_message


@pytest.mark.asyncio
async def test_audit_log_failure_non_blocking(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """
    Test that audit log failures don't prevent commits from succeeding.

    Audit logging is best-effort - if it fails, the commit should still complete.
    """
    checkpoint = b"test checkpoint"
    content_hash = "test_hash"

    # Mock _append_audit_log to fail
    async def failing_audit_log(version: object) -> None:
        """Simulate audit log failure."""
        raise ClientError(
            {
                "Error": {
                    "Code": "InternalError",
                    "Message": "Audit log service unavailable",
                }
            },
            "PutObject",
        )

    with patch.object(async_store, "_append_audit_log", side_effect=failing_audit_log):
        # Commit should succeed despite audit log failure
        version = await async_store.commit(checkpoint, content_hash, "Test commit")

        # Verify commit succeeded
        assert version.counter == 0
        assert version.semantic_version == "1.0.0"
        assert version.content_hash == content_hash

        # Verify HEAD was updated
        head = await async_store.get_head()
        assert head is not None
        assert head.counter == 0
        assert head.content_hash == content_hash

    # Verify no audit log entries were created (because it failed)
    assert async_store._s3_client is not None
    paginator = async_store._s3_client.get_paginator("list_objects_v2")
    audit_entries = []
    async for page in paginator.paginate(
        Bucket=async_store.bucket_name, Prefix="audit_log/"
    ):
        if isinstance(page, dict) and "Contents" in page:
            contents = page["Contents"]
            assert isinstance(contents, list)
            audit_entries.extend([obj["Key"] for obj in contents])

    assert (
        len(audit_entries) == 0
    ), f"Expected no audit entries after failure, got {len(audit_entries)}"
