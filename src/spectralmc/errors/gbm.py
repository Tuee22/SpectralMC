"""Error ADTs for GBM simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import ValidationError

from spectralmc.errors.async_normals import (
    InvalidDType,
    InvalidShape,
    QueueBusy,
    QueueEmpty,
    SeedOutOfRange,
)
from spectralmc.result import Result

NormGeneratorError = InvalidShape | InvalidDType | QueueBusy | QueueEmpty | SeedOutOfRange


@dataclass(frozen=True)
class InvalidSimulationParams:
    """Game parameters violate domain constraints (Pydantic failure)."""

    error: ValidationError
    kind: Literal["InvalidSimulationParams"] = "InvalidSimulationParams"


@dataclass(frozen=True)
class GPUMemoryLimitExceeded:
    """GPU memory limit exceeded for simulation parameters."""

    total_paths: int
    max_paths: int
    network_size: int
    batches_per_mc_run: int
    kind: Literal["GPUMemoryLimitExceeded"] = "GPUMemoryLimitExceeded"


@dataclass(frozen=True)
class InvalidBlackScholesConfig:
    """Black-Scholes configuration validation failed."""

    error: ValidationError
    kind: Literal["InvalidBlackScholesConfig"] = "InvalidBlackScholesConfig"


GBMConfigError = InvalidSimulationParams | GPUMemoryLimitExceeded | InvalidBlackScholesConfig

T = TypeVar("T")
GBMConfigResult = Result[T, GBMConfigError]


@dataclass(frozen=True)
class NormalsUnavailable:
    """Normals generator not available (construction failed)."""

    error: NormGeneratorError
    kind: Literal["NormalsUnavailable"] = "NormalsUnavailable"


@dataclass(frozen=True)
class NormalsGenerationFailed:
    """Normals generation failed during simulation."""

    error: NormGeneratorError
    kind: Literal["NormalsGenerationFailed"] = "NormalsGenerationFailed"
