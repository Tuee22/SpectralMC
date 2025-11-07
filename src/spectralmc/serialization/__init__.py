# src/spectralmc/serialization/__init__.py
"""
Pydantic ↔ Protobuf serialization layer for SpectralMC blockchain storage.

Provides type-safe bidirectional converters between Pydantic models (runtime
validation) and Protobuf messages (wire format).
"""

from __future__ import annotations

import hashlib
from typing import Protocol, TypeVar, Generic
from google.protobuf import message as _pb_message

# Type variables for generic Protocol
TPydantic = TypeVar("TPydantic")
TProto = TypeVar("TProto", bound=_pb_message.Message)


class ProtoSerializable(Protocol, Generic[TPydantic, TProto]):
    """
    Protocol for types that can convert between Pydantic and Protobuf.

    Implementers must provide static methods for bidirectional conversion.
    """

    @staticmethod
    def to_proto(pydantic_model: TPydantic) -> TProto:
        """Convert Pydantic model to Protobuf message."""
        ...

    @staticmethod
    def from_proto(proto_message: TProto) -> TPydantic:
        """Convert Protobuf message to Pydantic model."""
        ...


def compute_sha256(data: bytes) -> str:
    """
    Compute SHA256 hex digest of binary data.

    Args:
        data: Binary data to hash

    Returns:
        64-character hex string (SHA256 digest)
    """
    return hashlib.sha256(data).hexdigest()


def verify_checksum(data: bytes, expected_hash: str) -> bool:
    """
    Verify SHA256 checksum of binary data.

    Args:
        data: Binary data to verify
        expected_hash: Expected SHA256 hex digest

    Returns:
        True if checksums match, False otherwise
    """
    return compute_sha256(data) == expected_hash


__all__ = [
    "ProtoSerializable",
    "compute_sha256",
    "verify_checksum",
]
