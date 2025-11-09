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

# IMPORTANT: Import torch façade first to set deterministic flags
import spectralmc.models.torch  # noqa: F401

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
from .inference import InferenceClient, InferenceMode, PinnedMode, TrackingMode
from .verification import (
    verify_chain,
    verify_chain_detailed,
    find_corruption,
    CorruptionReport,
)
from .gc import GarbageCollector, RetentionPolicy, GCReport, run_gc
from .tensorboard_writer import TensorBoardWriter, log_blockchain_to_tensorboard

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
    # Inference
    "InferenceClient",
    "InferenceMode",
    "PinnedMode",
    "TrackingMode",
    # Verification
    "verify_chain",
    "verify_chain_detailed",
    "find_corruption",
    "CorruptionReport",
    # Garbage Collection
    "GarbageCollector",
    "RetentionPolicy",
    "GCReport",
    "run_gc",
    # TensorBoard
    "TensorBoardWriter",
    "log_blockchain_to_tensorboard",
    # Checkpoint utilities
    "create_checkpoint_from_snapshot",
    "commit_snapshot",
    "load_snapshot_from_checkpoint",
]
