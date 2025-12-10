# tests/test_storage/test_cli.py
"""Tests for blockchain storage CLI commands."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import torch

from spectralmc.result import Failure, Success
from spectralmc.storage import AsyncBlockchainModelStore

assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """
    Run CLI command and return result.

    Args:
        *args: CLI arguments (e.g., "verify", "test-bucket")

    Returns:
        CompletedProcess with stdout, stderr, returncode
    """
    cmd = [sys.executable, "-m", "spectralmc.storage", *args]
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,  # Prevent hanging
        encoding="utf-8",  # Handle Unicode characters (checkmarks) in CLI output
    )


@pytest.mark.asyncio
async def test_cli_verify_empty_chain(async_store: AsyncBlockchainModelStore) -> None:
    """Test verify command on empty chain."""
    result = run_cli("verify", async_store.bucket_name)

    assert result.returncode == 0
    assert "Chain integrity verified" in result.stdout
    assert async_store.bucket_name in result.stdout


@pytest.mark.asyncio
async def test_cli_verify_valid_chain(async_store: AsyncBlockchainModelStore) -> None:
    """Test verify command on valid chain with commits."""
    # Create a few commits
    await async_store.commit(b"data1", "hash1", "First commit")
    await async_store.commit(b"data2", "hash2", "Second commit")
    await async_store.commit(b"data3", "hash3", "Third commit")

    result = run_cli("verify", async_store.bucket_name)

    assert result.returncode == 0
    assert "Chain integrity verified" in result.stdout


@pytest.mark.asyncio
async def test_cli_verify_detailed_valid(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test verify --detailed on valid chain."""
    # Create a commit
    await async_store.commit(b"data1", "hash1", "Test commit")

    result = run_cli("verify", async_store.bucket_name, "--detailed")

    assert result.returncode == 0

    # Parse JSON output
    output = json.loads(result.stdout)
    assert output["is_valid"] is True
    assert output["corrupted_version"] is None
    assert output["corruption_type"] is None


@pytest.mark.asyncio
async def test_cli_verify_corrupted_chain(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test verify command detects corruption."""
    # Create commits
    v1 = await async_store.commit(b"data1", "hash1", "First")
    await async_store.commit(b"data2", "hash2", "Second")

    # Corrupt first version by directly modifying metadata
    # This simulates tampering with the blockchain
    metadata_key = f"versions/{v1.directory_name}/metadata.json"

    # Get current metadata
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    response = await async_store._s3_client.get_object(
        Bucket=async_store.bucket_name, Key=metadata_key
    )
    body = response["Body"]
    assert hasattr(body, "read")
    metadata = json.loads(await body.read())

    # Tamper with content_hash
    metadata["content_hash"] = "corrupted_hash_xxx"

    # Put back corrupted metadata
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    await async_store._s3_client.put_object(
        Bucket=async_store.bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata).encode("utf-8"),
        ContentType="application/json",
    )

    # Verify should detect corruption
    result = run_cli("verify", async_store.bucket_name)

    assert result.returncode == 1
    assert "corruption detected" in result.stderr.lower()


@pytest.mark.asyncio
async def test_cli_find_corruption_clean(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test find-corruption on clean chain."""
    # Create commits
    await async_store.commit(b"data1", "hash1", "First")
    await async_store.commit(b"data2", "hash2", "Second")

    result = run_cli("find-corruption", async_store.bucket_name)

    assert result.returncode == 0
    assert "No corruption found" in result.stdout


@pytest.mark.asyncio
async def test_cli_find_corruption_detects(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test find-corruption detects corrupted version."""
    # Create commits
    v1 = await async_store.commit(b"data1", "hash1", "First")
    await async_store.commit(b"data2", "hash2", "Second")
    await async_store.commit(b"data3", "hash3", "Third")

    # Corrupt first version
    metadata_key = f"versions/{v1.directory_name}/metadata.json"
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    response = await async_store._s3_client.get_object(
        Bucket=async_store.bucket_name, Key=metadata_key
    )
    body = response["Body"]
    assert hasattr(body, "read")
    metadata = json.loads(await body.read())
    metadata["content_hash"] = "corrupted"
    assert async_store._s3_client is not None, "S3 client should be initialized in async context"
    await async_store._s3_client.put_object(
        Bucket=async_store.bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata).encode("utf-8"),
        ContentType="application/json",
    )

    # Should find corruption in first version
    result = run_cli("find-corruption", async_store.bucket_name)

    assert result.returncode == 1

    # Parse JSON output
    output = json.loads(result.stdout)
    assert output["corrupted"] is True
    # Corruption detected at version 1 (child detects parent tampering)
    assert output["version_counter"] == 1


@pytest.mark.asyncio
async def test_cli_list_versions_empty(async_store: AsyncBlockchainModelStore) -> None:
    """Test list-versions on empty bucket."""
    result = run_cli("list-versions", async_store.bucket_name)

    assert result.returncode == 0
    assert "No versions" in result.stdout


@pytest.mark.asyncio
async def test_cli_list_versions_multiple(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test list-versions with multiple commits."""
    # Create commits
    await async_store.commit(b"data1", "hash1abc", "First commit")
    await async_store.commit(b"data2", "hash2def", "Second commit")
    await async_store.commit(b"data3", "hash3ghi", "Third commit")

    result = run_cli("list-versions", async_store.bucket_name)

    assert result.returncode == 0
    assert "Total: 3 versions" in result.stdout
    assert "v0000000000" in result.stdout
    assert "v0000000001" in result.stdout
    assert "v0000000002" in result.stdout
    assert "hash1abc" in result.stdout
    assert "hash2def" in result.stdout
    assert "hash3ghi" in result.stdout


@pytest.mark.asyncio
async def test_cli_inspect_version(async_store: AsyncBlockchainModelStore) -> None:
    """Test inspect command on specific version."""
    # Create commits
    await async_store.commit(b"data1", "hash1", "First commit")
    _v2 = await async_store.commit(b"data2", "hash2abc", "Second commit message")

    result = run_cli("inspect", async_store.bucket_name, "v0000000001")

    assert result.returncode == 0

    # Parse JSON output
    output = json.loads(result.stdout)
    assert output["counter"] == 1
    assert output["version_id"] == "v0000000001"
    assert output["semantic_version"] == "1.0.1"
    assert output["content_hash"] == "hash2abc"
    assert output["commit_message"] == "Second commit message"
    assert output["parent_hash"] != ""  # Should have parent


@pytest.mark.asyncio
async def test_cli_inspect_genesis(async_store: AsyncBlockchainModelStore) -> None:
    """Test inspect on genesis version."""
    await async_store.commit(b"data1", "genesis_hash", "Genesis commit")

    result = run_cli("inspect", async_store.bucket_name, "v0000000000")

    assert result.returncode == 0

    output = json.loads(result.stdout)
    assert output["counter"] == 0
    assert output["parent_hash"] == ""  # Genesis has no parent
    assert output["content_hash"] == "genesis_hash"


@pytest.mark.asyncio
async def test_cli_inspect_nonexistent(async_store: AsyncBlockchainModelStore) -> None:
    """Test inspect on nonexistent version."""
    result = run_cli("inspect", async_store.bucket_name, "v0000000099")

    assert result.returncode == 2
    assert "Error" in result.stderr


@pytest.mark.asyncio
async def test_cli_gc_preview_empty(async_store: AsyncBlockchainModelStore) -> None:
    """Test gc-preview on empty chain."""
    result = run_cli("gc-preview", async_store.bucket_name, "5")

    assert result.returncode == 0

    # Parse JSON output
    output = json.loads(result.stdout)
    assert output["dry_run"] is True
    assert output["deleted_versions"] == []
    assert output["bytes_freed"] == 0


@pytest.mark.asyncio
async def test_cli_gc_preview_keeps_recent(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test gc-preview keeps recent versions."""
    # Create 10 versions
    for i in range(10):
        await async_store.commit(b"data" * 100, f"hash{i}", f"Commit {i}")

    # Preview keeping last 5
    result = run_cli("gc-preview", async_store.bucket_name, "5")

    assert result.returncode == 0

    output = json.loads(result.stdout)
    assert output["dry_run"] is True
    # Genesis (v0) is always protected, so delete 1-4, keep 0,5-9
    assert len(output["deleted_versions"]) == 4
    assert output["deleted_versions"] == [1, 2, 3, 4]
    assert len(output["protected_versions"]) == 6  # 0 (genesis) + 5-9 (recent)
    assert output["bytes_freed"] > 0


@pytest.mark.asyncio
async def test_cli_gc_preview_with_protected_tags(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test gc-preview respects protected tags."""
    # Create 10 versions
    for i in range(10):
        await async_store.commit(b"data" * 100, f"hash{i}", f"Commit {i}")

    # Preview keeping last 3, but protect versions 2 and 5
    result = run_cli("gc-preview", async_store.bucket_name, "3", "--protect-tags", "2,5")

    assert result.returncode == 0

    output = json.loads(result.stdout)
    assert 2 in output["protected_versions"]  # Protected by tag
    assert 5 in output["protected_versions"]  # Protected by tag
    assert 7 in output["protected_versions"]  # Recent (keep last 3: 7,8,9)
    assert 8 in output["protected_versions"]
    assert 9 in output["protected_versions"]

    # Should delete 0, 1, 3, 4, 6 (not protected by tag or recent)
    assert 2 not in output["deleted_versions"]
    assert 5 not in output["deleted_versions"]


@pytest.mark.asyncio
async def test_cli_gc_run_with_yes_flag(async_store: AsyncBlockchainModelStore) -> None:
    """Test gc-run with --yes flag skips confirmation."""
    # Create 5 versions
    for i in range(5):
        await async_store.commit(b"data" * 100, f"hash{i}", f"Commit {i}")

    # Run GC with --yes (skip confirmation)
    result = run_cli("gc-run", async_store.bucket_name, "2", "--yes")

    assert result.returncode == 0
    assert "Garbage collection completed" in result.stdout
    # Genesis (v0) is always protected, so delete 1,2; keep 0,3,4
    assert "Deleted 2 versions" in result.stdout

    # Verify versions were actually deleted
    head_result = await async_store.get_head()
    match head_result:
        case Success(head):
            assert head.counter == 4  # Last version still exists
        case Failure(_):
            pytest.fail("Expected HEAD to exist")

    # Version 0 should still exist (genesis is always protected)
    v0 = await async_store.get_version("v0000000000")
    assert v0.counter == 0

    # Version 1 should be deleted
    with pytest.raises(Exception):  # VersionNotFoundError
        await async_store.get_version("v0000000001")


@pytest.mark.asyncio
async def test_cli_gc_run_protects_tags(async_store: AsyncBlockchainModelStore) -> None:
    """Test gc-run respects protected tags."""
    # Create 6 versions
    for i in range(6):
        await async_store.commit(b"data" * 100, f"hash{i}", f"Commit {i}")

    # Run GC keeping last 2, but protect version 1
    result = run_cli(
        "gc-run",
        async_store.bucket_name,
        "2",
        "--protect-tags",
        "1",
        "--yes",
    )

    assert result.returncode == 0

    # Version 1 should still exist (protected by tag)
    v1 = await async_store.get_version("v0000000001")
    assert v1.counter == 1

    # Version 0 should still exist (genesis is always protected)
    v0 = await async_store.get_version("v0000000000")
    assert v0.counter == 0

    # Version 2 should be deleted (not protected by tag, not recent)
    with pytest.raises(Exception):
        await async_store.get_version("v0000000002")

    # Versions 4,5 should exist (recent)
    v4 = await async_store.get_version("v0000000004")
    assert v4.counter == 4


@pytest.mark.asyncio
async def test_cli_tensorboard_log(async_store: AsyncBlockchainModelStore, tmp_path: Path) -> None:
    """Test tensorboard-log command."""
    # Create some commits
    for i in range(3):
        await async_store.commit(b"data" * 100, f"hash{i}", f"Commit {i}")

    # Log to TensorBoard
    log_dir = str(tmp_path / "test_tb_logs")
    result = run_cli("tensorboard-log", async_store.bucket_name, "--log-dir", log_dir)

    assert result.returncode == 0
    assert "Blockchain logged to TensorBoard" in result.stdout
    assert f"tensorboard --logdir={log_dir}" in result.stdout


@pytest.mark.asyncio
async def test_cli_verify_nonexistent_bucket() -> None:
    """Test verify on nonexistent bucket returns error."""
    # Use a bucket that definitely doesn't exist
    fake_bucket = f"nonexistent-bucket-{uuid.uuid4().hex}"

    result = run_cli("verify", fake_bucket)

    assert result.returncode == 2
    assert "Error" in result.stderr


@pytest.mark.asyncio
async def test_cli_invalid_command() -> None:
    """Test invalid command returns error."""
    result = run_cli("invalid-command", "some-bucket")

    # argparse exits with code 2 for invalid commands
    assert result.returncode == 2


@pytest.mark.asyncio
async def test_cli_missing_required_argument() -> None:
    """Test missing required argument returns error."""
    # verify requires bucket_name
    result = run_cli("verify")

    assert result.returncode == 2
    assert result.stderr  # Should have error message


@pytest.mark.asyncio
async def test_cli_gc_invalid_keep_versions() -> None:
    """Test gc-run with invalid keep_versions argument."""
    result = run_cli("gc-run", "test-bucket", "not-a-number")

    assert result.returncode == 2


@pytest.mark.asyncio
async def test_cli_inspect_requires_version_id() -> None:
    """Test inspect without version_id returns error."""
    result = run_cli("inspect", "test-bucket")

    assert result.returncode == 2
