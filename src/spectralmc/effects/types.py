"""
Effect ADT - Algebraic Data Types for all side effects in SpectralMC.

This module re-exports all effect types and defines the master Effect union,
enabling exhaustive pattern matching and type-safe effect composition.

Type Safety:
    - All effect types are frozen dataclasses (immutable)
    - Literal discriminators enable exhaustive pattern matching
    - Union types define closed sets of effect variants
    - __post_init__ validation prevents illegal state construction

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

# Re-export all effect types for convenient single-import access
from spectralmc.effects.gpu import (
    DLPackTransfer,
    GPUEffect,
    KernelLaunch,
    StreamSync,
    TensorTransfer,
)
from spectralmc.effects.logging import LoggingEffect, LogMessage
from spectralmc.effects.metadata import (
    MetadataEffect,
    ReadMetadata,
    UpdateMetadata,
)
from spectralmc.effects.montecarlo import (
    ComputeFFT,
    GenerateNormals,
    MonteCarloEffect,
    SimulatePaths,
)
from spectralmc.effects.rng import (
    CaptureRNGState,
    RestoreRNGState,
    RNGEffect,
)
from spectralmc.effects.storage import (
    CommitVersion,
    ReadObject,
    StorageEffect,
    WriteObject,
)
from spectralmc.effects.training import (
    BackwardPass,
    ComputeLoss,
    ForwardPass,
    LogMetrics,
    OptimizerStep,
    TrainingEffect,
)


# Master Effect Union - enables exhaustive pattern matching across all effect types
Effect = (
    GPUEffect
    | TrainingEffect
    | MonteCarloEffect
    | StorageEffect
    | RNGEffect
    | MetadataEffect
    | LoggingEffect
)

__all__ = [
    # Master union
    "Effect",
    # GPU effects
    "GPUEffect",
    "TensorTransfer",
    "StreamSync",
    "KernelLaunch",
    "DLPackTransfer",
    # Training effects
    "TrainingEffect",
    "ForwardPass",
    "BackwardPass",
    "OptimizerStep",
    "ComputeLoss",
    "LogMetrics",
    "LoggingEffect",
    "LogMessage",
    # Monte Carlo effects
    "MonteCarloEffect",
    "GenerateNormals",
    "SimulatePaths",
    "ComputeFFT",
    # Storage effects
    "StorageEffect",
    "ReadObject",
    "WriteObject",
    "CommitVersion",
    # RNG effects
    "RNGEffect",
    "CaptureRNGState",
    "RestoreRNGState",
    # Metadata effects
    "MetadataEffect",
    "ReadMetadata",
    "UpdateMetadata",
]
