"""ADTs for storage layer failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ValidationError:
    """Validation error for storage operations."""

    field: str
    value: int | str
    message: str
    kind: Literal["ValidationError"] = "ValidationError"


@dataclass(frozen=True)
class StartError:
    """Error starting InferenceClient."""

    message: str
    underlying_error: Exception | None = None
    kind: Literal["StartError"] = "StartError"


@dataclass(frozen=True)
class VersionNotFoundError:
    """Requested version not found in store."""

    counter: int
    available_versions: list[int]
    kind: Literal["VersionNotFoundError"] = "VersionNotFoundError"


@dataclass(frozen=True)
class VerificationError:
    """Blockchain verification error."""

    version_counter: int
    message: str
    missing_artifact: str | None = None
    kind: Literal["VerificationError"] = "VerificationError"


@dataclass(frozen=True)
class GCError:
    """Garbage collection error."""

    message: str
    protected_count: int
    minimum_required: int
    kind: Literal["GCError"] = "GCError"


# Union type for all storage errors
StorageError = ValidationError | StartError | VersionNotFoundError | VerificationError | GCError
