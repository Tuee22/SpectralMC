# src/spectralmc/storage/chain.py
"""Blockchain primitives for model versioning."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelVersion:
    """
    Immutable representation of a versioned model in the blockchain.
    
    Each version forms a node in a Merkle chain where content_hash
    is the SHA256 of the serialized checkpoint.
    """
    counter: int
    semantic_version: str  # "MAJOR.MINOR.PATCH"
    parent_hash: str  # SHA256 of parent version ("" for v0)
    content_hash: str  # SHA256 of checkpoint.pb
    commit_timestamp: str  # ISO 8601 UTC
    commit_message: str
    
    @property
    def version_id(self) -> str:
        """Human-readable version identifier."""
        return f"v{self.counter:010d}"
    
    @property
    def directory_name(self) -> str:
        """S3 directory name for this version."""
        return f"{self.version_id}_{self.semantic_version}_{self.content_hash[:8]}"
    
    def compute_hash(self) -> str:
        """
        Compute SHA256 hash of this version for chain integrity.
        
        Returns:
            64-character hex string (SHA256 digest)
        """
        data = (
            f"{self.counter}|"
            f"{self.semantic_version}|"
            f"{self.parent_hash}|"
            f"{self.content_hash}|"
            f"{self.commit_timestamp}|"
            f"{self.commit_message}"
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


def bump_semantic_version(
    current: str,
    change_type: str = "patch"
) -> str:
    """
    Bump semantic version according to change type.
    
    Args:
        current: Current version in "MAJOR.MINOR.PATCH" format
        change_type: One of "major", "minor", "patch"
        
    Returns:
        New semantic version string
    """
    major, minor, patch = map(int, current.split("."))
    
    if change_type == "major":
        return f"{major + 1}.0.0"
    elif change_type == "minor":
        return f"{major}.{minor + 1}.0"
    else:  # patch
        return f"{major}.{minor}.{patch + 1}"


def create_genesis_version(content_hash: str, message: str = "Genesis version") -> ModelVersion:
    """
    Create the genesis version (v0) of the blockchain.
    
    Args:
        content_hash: SHA256 hash of the genesis checkpoint
        
    Returns:
        ModelVersion for v0
    """
    return ModelVersion(
        counter=0,
        semantic_version="1.0.0",
        parent_hash="",  # No parent for genesis
        content_hash=content_hash,
        commit_timestamp=datetime.now(timezone.utc).isoformat(),
        commit_message=message,
    )


__all__ = [
    "ModelVersion",
    "bump_semantic_version",
    "create_genesis_version",
]
