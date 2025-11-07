# src/spectralmc/storage/errors.py
"""Exception hierarchy for blockchain model storage."""

from __future__ import annotations


class StorageError(Exception):
    """Base exception for all storage-related errors."""
    pass


class CommitError(StorageError):
    """Failed to commit a new model version."""
    pass


class NotFastForwardError(CommitError):
    """Attempted to commit with a stale parent hash."""
    def __init__(self, expected_parent: str, actual_head: str) -> None:
        self.expected_parent = expected_parent
        self.actual_head = actual_head
        super().__init__(
            f"Not a fast-forward: expected parent {expected_parent[:8]}, "
            f"but current HEAD is {actual_head[:8]}"
        )


class ConflictError(CommitError):
    """Multiple concurrent commits attempted."""
    pass


class ChecksumError(StorageError):
    """Checksum verification failed."""
    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Checksum mismatch: expected {expected[:8]}, got {actual[:8]}"
        )


class VersionNotFoundError(StorageError):
    """Requested model version not found."""
    def __init__(self, version: str) -> None:
        self.version = version
        super().__init__(f"Version not found: {version}")


class ChainCorruptionError(StorageError):
    """Blockchain integrity violation detected."""
    pass


__all__ = [
    "StorageError",
    "CommitError",
    "NotFastForwardError",
    "ConflictError",
    "ChecksumError",
    "VersionNotFoundError",
    "ChainCorruptionError",
]
