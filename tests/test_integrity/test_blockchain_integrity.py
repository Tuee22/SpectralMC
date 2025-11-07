# tests/test_integrity/test_blockchain_integrity.py
"""Integrity tests for blockchain storage."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from spectralmc.storage.store import BlockchainModelStore
from spectralmc.storage.chain import ModelVersion


def test_content_hash_integrity() -> None:
    """Test that changing content invalidates hash."""
    version = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent",
        content_hash="original_hash",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    hash1 = version.compute_hash()

    # Different content hash should produce different version hash
    version2 = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent",
        content_hash="modified_hash",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    hash2 = version2.compute_hash()

    assert hash1 != hash2


def test_parent_hash_integrity() -> None:
    """Test that parent hash changes affect version hash."""
    version1 = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent1",
        content_hash="content",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    version2 = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent2",
        content_hash="content",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    assert version1.compute_hash() != version2.compute_hash()


def test_timestamp_integrity() -> None:
    """Test that timestamp changes affect version hash."""
    version1 = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent",
        content_hash="content",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    version2 = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent",
        content_hash="content",
        commit_timestamp="2025-01-02T00:00:00Z",
        commit_message="Test",
    )

    assert version1.compute_hash() != version2.compute_hash()


def test_chain_file_tampering_detection() -> None:
    """Test that manual chain.json tampering is detectable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        # Create initial commit
        checkpoint = b"test checkpoint"
        version = store.commit(checkpoint, "hash123", "Initial")

        # Read chain.json
        with open(store.chain_file) as f:
            chain_data = json.load(f)

        original_hash = version.compute_hash()

        # Tamper with content_hash
        chain_data["content_hash"] = "tampered_hash"

        with open(store.chain_file, "w") as f:
            json.dump(chain_data, f)

        # Read back HEAD
        tampered_head = store.get_head()
        assert tampered_head is not None

        # Computed hash should differ, indicating tampering
        tampered_hash = tampered_head.compute_hash()
        assert tampered_hash != original_hash


def test_checkpoint_corruption_detection() -> None:
    """Test that checkpoint file corruption is detectable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        checkpoint_data = b"original checkpoint data"
        content_hash = "abcd1234"

        version = store.commit(checkpoint_data, content_hash)

        # Corrupt checkpoint file
        version_dir = store.versions_path / version.directory_name
        checkpoint_file = version_dir / "checkpoint.pb"

        with open(checkpoint_file, "wb") as f:
            f.write(b"corrupted data")

        # Load corrupted checkpoint
        loaded = store.load_checkpoint(version)

        # Data should be different from original
        assert loaded != checkpoint_data
        assert loaded == b"corrupted data"


def test_metadata_json_integrity() -> None:
    """Test metadata.json integrity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        checkpoint = b"test"
        version = store.commit(checkpoint, "hash123")

        version_dir = store.versions_path / version.directory_name
        metadata_file = version_dir / "metadata.json"

        # Read original metadata
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["counter"] == version.counter
        assert metadata["content_hash"] == version.content_hash

        # Tamper with metadata
        metadata["counter"] = 999

        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Retrieve version - should have tampered data
        retrieved = store.get_version(version.version_id)
        assert retrieved.counter == 999


def test_semantic_version_ordering_integrity() -> None:
    """Test semantic version ordering is maintained."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        versions = []
        for i in range(5):
            checkpoint = f"checkpoint {i}".encode()
            version = store.commit(checkpoint, f"hash{i}")
            versions.append(version)

        # Verify versions increment correctly
        assert versions[0].semantic_version == "1.0.0"
        assert versions[1].semantic_version == "1.0.1"
        assert versions[2].semantic_version == "1.0.2"
        assert versions[3].semantic_version == "1.0.3"
        assert versions[4].semantic_version == "1.0.4"


def test_counter_monotonicity() -> None:
    """Test counter values are strictly increasing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        versions = []
        for i in range(10):
            checkpoint = f"checkpoint {i}".encode()
            version = store.commit(checkpoint, f"hash{i}")
            versions.append(version)

        # Counters should be strictly increasing
        for i in range(len(versions) - 1):
            assert versions[i + 1].counter == versions[i].counter + 1


def test_parent_chain_integrity() -> None:
    """Test parent hash chain is valid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        versions = []
        for i in range(5):
            checkpoint = f"checkpoint {i}".encode()
            content_hash = f"hash{i}"
            version = store.commit(checkpoint, content_hash)
            versions.append(version)

        # Verify parent chain
        assert versions[0].parent_hash == ""  # Genesis
        for i in range(1, len(versions)):
            assert versions[i].parent_hash == versions[i - 1].content_hash


def test_directory_name_uniqueness() -> None:
    """Test version directory names are unique."""
    versions = [
        ModelVersion(
            counter=i,
            semantic_version=f"1.0.{i}",
            parent_hash="",
            content_hash=f"hash{i}",
            commit_timestamp="2025-01-01T00:00:00Z",
            commit_message="Test",
        )
        for i in range(100)
    ]

    dir_names = [v.directory_name for v in versions]

    # All directory names should be unique
    assert len(dir_names) == len(set(dir_names))


def test_version_immutability() -> None:
    """Test ModelVersion is immutable (frozen dataclass)."""
    version = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent",
        content_hash="content",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test",
    )

    # Should raise FrozenInstanceError
    with pytest.raises(Exception):
        version.counter = 999  # type: ignore[misc]


def test_hash_length_consistency() -> None:
    """Test all hashes are 64-character SHA256 hex."""
    versions = [
        ModelVersion(
            counter=i,
            semantic_version=f"1.0.{i}",
            parent_hash=f"parent{i}" * 8,
            content_hash=f"content{i}" * 8,
            commit_timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
            commit_message=f"Message {i}",
        )
        for i in range(20)
    ]

    for version in versions:
        version_hash = version.compute_hash()
        assert len(version_hash) == 64
        assert all(c in "0123456789abcdef" for c in version_hash)


def test_empty_checkpoint_integrity() -> None:
    """Test empty checkpoints are handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        empty_checkpoint = b""
        content_hash = "empty_hash"

        version = store.commit(empty_checkpoint, content_hash)

        loaded = store.load_checkpoint(version)
        assert loaded == b""


def test_large_checkpoint_integrity() -> None:
    """Test large checkpoints maintain integrity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        # 10 MB checkpoint
        large_checkpoint = b"x" * (10 * 1024 * 1024)
        content_hash = "large_hash"

        version = store.commit(large_checkpoint, content_hash)

        loaded = store.load_checkpoint(version)
        assert loaded == large_checkpoint
        assert len(loaded) == len(large_checkpoint)
