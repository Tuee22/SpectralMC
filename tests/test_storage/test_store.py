# tests/test_storage/test_store.py
"""Tests for blockchain model store."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spectralmc.storage.store import BlockchainModelStore
from spectralmc.storage.chain import ModelVersion
from spectralmc.storage.errors import VersionNotFoundError


@pytest.fixture
def temp_storage() -> Path:
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_storage: Path) -> BlockchainModelStore:
    """Create BlockchainModelStore instance."""
    return BlockchainModelStore(str(temp_storage))


def test_store_initialization(store: BlockchainModelStore) -> None:
    """Test store initializes correctly."""
    assert store.storage_path.exists()
    assert store.versions_path.exists()
    assert store.chain_file.parent.exists()


def test_get_head_empty(store: BlockchainModelStore) -> None:
    """Test get_head returns None when no commits exist."""
    head = store.get_head()
    assert head is None


def test_commit_genesis(store: BlockchainModelStore) -> None:
    """Test genesis commit creation."""
    checkpoint_data = b"test checkpoint data"
    content_hash = "abc123def456"

    version = store.commit(checkpoint_data, content_hash, "Initial commit")

    assert version.counter == 0
    assert version.semantic_version == "1.0.0"
    assert version.parent_hash == ""
    assert version.content_hash == content_hash
    assert version.commit_message == "Initial commit"


def test_commit_incremental(store: BlockchainModelStore) -> None:
    """Test incremental commits."""
    # First commit
    checkpoint1 = b"checkpoint 1"
    hash1 = "hash1"
    version1 = store.commit(checkpoint1, hash1, "First")

    # Second commit
    checkpoint2 = b"checkpoint 2"
    hash2 = "hash2"
    version2 = store.commit(checkpoint2, hash2, "Second")

    assert version2.counter == 1
    assert version2.semantic_version == "1.0.1"
    assert version2.parent_hash == hash1
    assert version2.content_hash == hash2
    assert version2.commit_message == "Second"


def test_get_head_after_commit(store: BlockchainModelStore) -> None:
    """Test get_head returns latest commit."""
    checkpoint1 = b"checkpoint 1"
    version1 = store.commit(checkpoint1, "hash1", "First")

    head = store.get_head()
    assert head is not None
    assert head.counter == version1.counter
    assert head.semantic_version == version1.semantic_version


def test_commit_chain(store: BlockchainModelStore) -> None:
    """Test multiple commits form proper chain."""
    versions = []
    for i in range(5):
        checkpoint = f"checkpoint {i}".encode()
        content_hash = f"hash{i}"
        message = f"Commit {i}"
        version = store.commit(checkpoint, content_hash, message)
        versions.append(version)

    # Verify chain properties
    for i, version in enumerate(versions):
        assert version.counter == i
        if i == 0:
            assert version.parent_hash == ""
        else:
            assert version.parent_hash == versions[i - 1].content_hash

    # Verify HEAD points to latest
    head = store.get_head()
    assert head is not None
    assert head.counter == 4


def test_get_version_by_id(store: BlockchainModelStore) -> None:
    """Test retrieving version by ID."""
    checkpoint = b"test checkpoint"
    content_hash = "abcdef123456"
    version = store.commit(checkpoint, content_hash, "Test")

    retrieved = store.get_version("v0000000000")
    assert retrieved.counter == version.counter
    assert retrieved.content_hash == version.content_hash


def test_get_version_not_found(store: BlockchainModelStore) -> None:
    """Test VersionNotFoundError for non-existent version."""
    with pytest.raises(VersionNotFoundError) as exc_info:
        store.get_version("v9999999999")

    assert "v9999999999" in str(exc_info.value)


def test_load_checkpoint(store: BlockchainModelStore) -> None:
    """Test loading checkpoint data."""
    checkpoint_data = b"test checkpoint binary data"
    content_hash = "checksum123"
    version = store.commit(checkpoint_data, content_hash)

    loaded_data = store.load_checkpoint(version)
    assert loaded_data == checkpoint_data


def test_checkpoint_persistence(store: BlockchainModelStore) -> None:
    """Test checkpoint data persists across store instances."""
    checkpoint_data = b"persistent data"
    content_hash = "persist123"
    version = store.commit(checkpoint_data, content_hash)

    # Create new store instance pointing to same directory
    new_store = BlockchainModelStore(str(store.storage_path))

    # Retrieve from new instance
    head = new_store.get_head()
    assert head is not None
    assert head.content_hash == content_hash

    loaded_data = new_store.load_checkpoint(head)
    assert loaded_data == checkpoint_data


def test_version_directory_structure(store: BlockchainModelStore) -> None:
    """Test version directory naming and structure."""
    checkpoint = b"test"
    content_hash = "abcd1234"
    version = store.commit(checkpoint, content_hash)

    version_dir = store.versions_path / version.directory_name
    assert version_dir.exists()
    assert (version_dir / "checkpoint.pb").exists()
    assert (version_dir / "metadata.json").exists()


def test_semantic_versioning_progression(store: BlockchainModelStore) -> None:
    """Test semantic version increments correctly."""
    versions = []
    for i in range(10):
        checkpoint = f"checkpoint {i}".encode()
        version = store.commit(checkpoint, f"hash{i}")
        versions.append(version)

    # All should be patch increments (1.0.0 -> 1.0.1 -> ... -> 1.0.9)
    assert versions[0].semantic_version == "1.0.0"
    assert versions[1].semantic_version == "1.0.1"
    assert versions[5].semantic_version == "1.0.5"
    assert versions[9].semantic_version == "1.0.9"
