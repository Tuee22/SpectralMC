"""
Storage Effect ADTs for S3 and blockchain operations.

This module defines frozen dataclasses representing all storage-related side effects,
enabling exhaustive pattern matching and type-safe effect composition.

Type Safety:
    - All effect types are frozen dataclasses (immutable)
    - Literal discriminators enable exhaustive pattern matching
    - Union types define closed sets of effect variants

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - blockchain_storage.md - Storage architecture
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ReadObject:
    """Request to read an object from S3 storage.

    Attributes:
        kind: Discriminator for pattern matching. Always "ReadObject".
        bucket: S3 bucket name.
        key: Object key within the bucket.
        output_id: Identifier for storing read content bytes in registry.
    """

    kind: Literal["ReadObject"] = "ReadObject"
    bucket: str = ""
    key: str = ""
    output_id: str = "content"


@dataclass(frozen=True)
class WriteObject:
    """Request to write an object to S3 storage.

    Attributes:
        kind: Discriminator for pattern matching. Always "WriteObject".
        bucket: S3 bucket name.
        key: Object key within the bucket.
        content_hash: SHA256 hash of the content for verification.
    """

    kind: Literal["WriteObject"] = "WriteObject"
    bucket: str = ""
    key: str = ""
    content_hash: str = ""


@dataclass(frozen=True)
class CommitVersion:
    """Request to commit a new model version to blockchain storage.

    Attributes:
        kind: Discriminator for pattern matching. Always "CommitVersion".
        parent_counter: Counter of the parent version (None for genesis).
        checkpoint_hash: SHA256 hash of the checkpoint content.
        message: Commit message describing the version.
    """

    kind: Literal["CommitVersion"] = "CommitVersion"
    parent_counter: int | None = None
    checkpoint_hash: str = ""
    message: str = ""


# Storage Effect Union
StorageEffect = ReadObject | WriteObject | CommitVersion
