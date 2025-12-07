"""Trainer error ADTs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from spectralmc.errors.gbm import NormalsGenerationFailed, NormalsUnavailable
from spectralmc.errors.sampler import (
    DimensionMismatch,
    InvalidBounds,
    NegativeSamples,
    SamplerValidationFailed,
)


@dataclass(frozen=True)
class SamplerInitFailed:
    """Trainer failed to initialize the Sobol sampler."""

    error: SamplerError
    kind: Literal["SamplerInitFailed"] = "SamplerInitFailed"


@dataclass(frozen=True)
class InvalidTrainerConfig:
    """Invalid trainer configuration (e.g., missing blockchain_store)."""

    message: str
    kind: Literal["InvalidTrainerConfig"] = "InvalidTrainerConfig"


@dataclass(frozen=True)
class OptimizerStateSerializationFailed:
    """Serialization failure for optimizer snapshots."""

    message: str
    kind: Literal["OptimizerStateSerializationFailed"] = "OptimizerStateSerializationFailed"


@dataclass(frozen=True)
class PredictionFailed:
    """Inference request failed within the trainer."""

    message: str
    kind: Literal["PredictionFailed"] = "PredictionFailed"


SamplerError = DimensionMismatch | InvalidBounds | NegativeSamples | SamplerValidationFailed
TrainerError = (
    SamplerInitFailed
    | NormalsUnavailable
    | NormalsGenerationFailed
    | InvalidTrainerConfig
    | OptimizerStateSerializationFailed
    | PredictionFailed
)
