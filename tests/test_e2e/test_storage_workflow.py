# tests/test_e2e/test_storage_workflow.py
"""End-to-end tests for storage workflows."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spectralmc.storage.store import BlockchainModelStore
from spectralmc.storage.chain import bump_semantic_version


def test_simple_commit_workflow() -> None:
    """Test basic commit workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        # Initial state: no HEAD
        assert store.get_head() is None

        # First commit (genesis)
        checkpoint1 = b"checkpoint 1"
        version1 = store.commit(checkpoint1, "hash1", "Genesis")

        assert version1.counter == 0
        assert version1.semantic_version == "1.0.0"
        assert version1.parent_hash == ""

        # Second commit
        checkpoint2 = b"checkpoint 2"
        version2 = store.commit(checkpoint2, "hash2", "Second commit")

        assert version2.counter == 1
        assert version2.semantic_version == "1.0.1"
        assert version2.parent_hash == "hash1"

        # Verify HEAD
        head = store.get_head()
        assert head is not None
        assert head.counter == 1
        assert head.content_hash == "hash2"


def test_multiple_commits_workflow() -> None:
    """Test workflow with multiple commits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        num_commits = 20
        versions = []

        for i in range(num_commits):
            checkpoint = f"checkpoint {i}".encode()
            content_hash = f"hash{i}"
            message = f"Commit {i}"
            version = store.commit(checkpoint, content_hash, message)
            versions.append(version)

        # Verify all versions are retrievable
        for version in versions:
            retrieved = store.get_version(version.version_id)
            assert retrieved.counter == version.counter
            assert retrieved.content_hash == version.content_hash

        # Verify HEAD points to latest
        head = store.get_head()
        assert head is not None
        assert head.counter == num_commits - 1


def test_load_checkpoint_workflow() -> None:
    """Test loading checkpoints after commits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        checkpoints = [f"checkpoint {i}" * 100 for i in range(10)]
        versions = []

        # Commit all checkpoints
        for i, checkpoint_str in enumerate(checkpoints):
            checkpoint = checkpoint_str.encode()
            version = store.commit(checkpoint, f"hash{i}")
            versions.append(version)

        # Load and verify each checkpoint
        for i, version in enumerate(versions):
            loaded = store.load_checkpoint(version)
            expected = checkpoints[i].encode()
            assert loaded == expected


def test_persistence_across_instances() -> None:
    """Test data persists across store instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First instance: create commits
        store1 = BlockchainModelStore(tmpdir)

        versions_created = []
        for i in range(5):
            checkpoint = f"checkpoint {i}".encode()
            version = store1.commit(checkpoint, f"hash{i}")
            versions_created.append(version)

        head1 = store1.get_head()
        assert head1 is not None

        # Second instance: verify data
        store2 = BlockchainModelStore(tmpdir)

        head2 = store2.get_head()
        assert head2 is not None
        assert head2.counter == head1.counter
        assert head2.content_hash == head1.content_hash

        # Verify all versions are still there
        for version in versions_created:
            retrieved = store2.get_version(version.version_id)
            assert retrieved.counter == version.counter

        # Add more commits from second instance
        version_new = store2.commit(b"new checkpoint", "hash_new")
        assert version_new.counter == 5


def test_version_retrieval_workflow() -> None:
    """Test retrieving specific versions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        # Create 10 versions
        for i in range(10):
            store.commit(f"checkpoint {i}".encode(), f"hash{i}")

        # Retrieve specific versions
        v0 = store.get_version("v0000000000")
        assert v0.counter == 0

        v5 = store.get_version("v0000000005")
        assert v5.counter == 5

        v9 = store.get_version("v0000000009")
        assert v9.counter == 9


def test_semantic_version_bumping_workflow() -> None:
    """Test semantic version progression."""
    current = "1.0.0"

    # Patch bumps
    for i in range(1, 10):
        current = bump_semantic_version(current, "patch")
        assert current == f"1.0.{i}"

    # Minor bump
    current = bump_semantic_version(current, "minor")
    assert current == "1.1.0"

    # More patch bumps
    for i in range(1, 5):
        current = bump_semantic_version(current, "patch")
        assert current == f"1.1.{i}"

    # Major bump
    current = bump_semantic_version(current, "major")
    assert current == "2.0.0"


def test_empty_store_workflow() -> None:
    """Test operations on empty store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BlockchainModelStore(tmpdir)

        # Empty store has no HEAD
        assert store.get_head() is None

        # Create genesis
        version = store.commit(b"first", "hash1")
        assert version.counter == 0

        # Now HEAD exists
        head = store.get_head()
        assert head is not None
        assert head.counter == 0
