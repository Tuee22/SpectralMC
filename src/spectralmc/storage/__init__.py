# src/spectralmc/storage/__init__.py
"""
Blockchain model storage with versioning and integrity guarantees.

This module provides a simplified demonstration of blockchain-based model
versioning. Production implementation would include:
- Async S3 operations with aioboto3
- 10-step atomic commit with CAS
- TensorBoard integration
- Garbage collection
- Inference client with pinned/tracking modes
"""

from __future__ import annotations

from .chain import ModelVersion, bump_semantic_version, create_genesis_version
from .errors import (
    StorageError,
    CommitError,
    NotFastForwardError,
    ConflictError,
    ChecksumError,
    VersionNotFoundError,
    ChainCorruptionError,
)
from .store import AsyncBlockchainModelStore, retry_on_throttle

# Backward compatibility alias (deprecated - use AsyncBlockchainModelStore)
BlockchainModelStore = AsyncBlockchainModelStore
from .checkpoint import (
    create_checkpoint_from_snapshot,
    commit_snapshot,
    load_snapshot_from_checkpoint,
)

__all__ = [
    # Chain primitives
    "ModelVersion",
    "bump_semantic_version",
    "create_genesis_version",
    # Errors
    "StorageError",
    "CommitError",
    "NotFastForwardError",
    "ConflictError",
    "ChecksumError",
    "VersionNotFoundError",
    "ChainCorruptionError",
    # Store
    "AsyncBlockchainModelStore",
    "BlockchainModelStore",  # Backward compat alias
    "retry_on_throttle",
    # Checkpoint utilities
    "create_checkpoint_from_snapshot",
    "commit_snapshot",
    "load_snapshot_from_checkpoint",
]
