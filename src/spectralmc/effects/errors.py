"""
Effect error types for SpectralMC.

This module defines frozen dataclasses representing all effect-related errors,
enabling exhaustive pattern matching and type-safe error handling.

Type Safety:
    - All error types are frozen dataclasses (immutable)
    - Literal discriminators enable exhaustive pattern matching
    - Union types define closed sets of error variants

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - s3_errors.py - Similar error ADT patterns
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GPUError:
    """Error from GPU effect execution.

    Attributes:
        kind: Discriminator for pattern matching. Always "GPUError".
        message: Human-readable error description.
        cuda_error_code: Optional CUDA error code if applicable.
    """

    kind: Literal["GPUError"] = "GPUError"
    message: str = ""
    cuda_error_code: int | None = None


@dataclass(frozen=True)
class TrainingError:
    """Error from training effect execution.

    Attributes:
        kind: Discriminator for pattern matching. Always "TrainingError".
        message: Human-readable error description.
        step: Training step at which error occurred.
    """

    kind: Literal["TrainingError"] = "TrainingError"
    message: str = ""
    step: int | None = None


@dataclass(frozen=True)
class MonteCarloError:
    """Error from Monte Carlo effect execution.

    Attributes:
        kind: Discriminator for pattern matching. Always "MonteCarloError".
        message: Human-readable error description.
    """

    kind: Literal["MonteCarloError"] = "MonteCarloError"
    message: str = ""


@dataclass(frozen=True)
class StorageError:
    """Error from storage effect execution.

    Attributes:
        kind: Discriminator for pattern matching. Always "StorageError".
        message: Human-readable error description.
        bucket: S3 bucket involved in error.
        key: Object key involved in error.
    """

    kind: Literal["StorageError"] = "StorageError"
    message: str = ""
    bucket: str = ""
    key: str = ""


@dataclass(frozen=True)
class RNGError:
    """Error from RNG effect execution.

    Attributes:
        kind: Discriminator for pattern matching. Always "RNGError".
        message: Human-readable error description.
        rng_type: Type of RNG that caused the error.
    """

    kind: Literal["RNGError"] = "RNGError"
    message: str = ""
    rng_type: str = ""


@dataclass(frozen=True)
class MetadataError:
    """Error from metadata effect execution.

    Attributes:
        kind: Discriminator for pattern matching. Always "MetadataError".
        message: Human-readable error description.
        key: Metadata key that caused the error.
    """

    kind: Literal["MetadataError"] = "MetadataError"
    message: str = ""
    key: str = ""


@dataclass(frozen=True)
class LoggingError:
    """Error from logging effect execution.

    Attributes:
        kind: Discriminator for pattern matching. Always "LoggingError".
        message: Human-readable error description.
        logger_name: Logger where the error occurred.
    """

    kind: Literal["LoggingError"] = "LoggingError"
    message: str = ""
    logger_name: str = ""


# Master EffectError Union - enables exhaustive pattern matching
EffectError = (
    GPUError
    | TrainingError
    | MonteCarloError
    | StorageError
    | RNGError
    | MetadataError
    | LoggingError
)
