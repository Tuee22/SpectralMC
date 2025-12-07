# tests/test_storage/test_chain.py
"""Tests for blockchain chain primitives."""

from __future__ import annotations

import torch

from spectralmc.storage.chain import (
    ModelVersion,
    bump_semantic_version,
    create_genesis_version,
)

assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"


def test_model_version_properties() -> None:
    """Test ModelVersion property methods."""
    version = ModelVersion(
        counter=42,
        semantic_version="1.2.3",
        parent_hash="abc123",
        content_hash="def456",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test commit",
    )

    assert version.version_id == "v0000000042"
    assert version.directory_name.startswith("v0000000042_1.2.3_def456")


def test_compute_hash_deterministic() -> None:
    """Test that hash computation is deterministic."""
    version = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="",
        content_hash="abc123",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Genesis",
    )

    hash1 = version.compute_hash()
    hash2 = version.compute_hash()

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest


def test_bump_semantic_version_patch() -> None:
    """Test patch version bump."""
    assert bump_semantic_version("1.2.3", "patch") == "1.2.4"


def test_bump_semantic_version_minor() -> None:
    """Test minor version bump."""
    assert bump_semantic_version("1.2.3", "minor") == "1.3.0"


def test_bump_semantic_version_major() -> None:
    """Test major version bump."""
    assert bump_semantic_version("1.2.3", "major") == "2.0.0"


def test_create_genesis_version() -> None:
    """Test genesis version creation."""
    genesis = create_genesis_version("abc123")

    assert genesis.counter == 0
    assert genesis.semantic_version == "1.0.0"
    assert genesis.parent_hash == ""
    assert genesis.content_hash == "abc123"
    assert genesis.commit_message == "Genesis version"
