"""
Effect system for SpectralMC - separates effect description from execution.

This module provides the Effect Interpreter pattern where all side effects
(GPU operations, storage I/O, RNG state) are modeled as pure, immutable ADT types
that are interpreted by a single execution layer.

Core Principle: Pure code describes WHAT to do; the interpreter decides HOW and WHEN.

Benefits:
    - Testability: Pure effect descriptions can be tested without GPU hardware
    - Reproducibility: Effects capture complete execution context for deterministic replay
    - Composability: Effects combine without coupling to execution details
    - Type Safety: Invalid effect combinations prevented at compile time via ADTs

Example:
    >>> from spectralmc.effects import (
    ...     Effect,
    ...     GenerateNormals,
    ...     SimulatePaths,
    ...     StreamSync,
    ...     sequence_effects,
    ... )
    >>>
    >>> # Build effect description (pure, no side effects)
    >>> effects = sequence_effects(
    ...     GenerateNormals(rows=1024, cols=252, seed=42),
    ...     SimulatePaths(spot=100.0, vol=0.2),
    ...     StreamSync(stream_type="cupy"),
    ... )
    >>>
    >>> # Execute via interpreter (all side effects happen here)
    >>> result = await interpreter.interpret_sequence(effects)

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - reproducibility_proofs.md - Determinism guarantees
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

# Composition utilities
from spectralmc.effects.composition import (
    EffectParallel,
    EffectSequence,
    map_effect,
    parallel_effects,
    sequence_effects,
)

# Error types
from spectralmc.effects.errors import (
    EffectError,
    GPUError,
    LoggingError,
    MetadataError,
    MonteCarloError,
    RNGError,
    StorageError,
    TrainingError,
)

# Effect types
from spectralmc.effects.types import (
    BackwardPass,
    CaptureRNGState,
    CommitVersion,
    ComputeFFT,
    ComputeLoss,
    DLPackTransfer,
    Effect,
    ForwardPass,
    GenerateNormals,
    GPUEffect,
    KernelLaunch,
    LogMessage,
    LogMetrics,
    LoggingEffect,
    MetadataEffect,
    MonteCarloEffect,
    OptimizerStep,
    ReadMetadata,
    ReadObject,
    RestoreRNGState,
    RNGEffect,
    SimulatePaths,
    StorageEffect,
    StreamSync,
    TensorTransfer,
    TrainingEffect,
    UpdateMetadata,
    WriteObject,
)

# Interpreters
from spectralmc.effects.interpreter import (
    EffectInterpreter,
    GPUInterpreter,
    LoggingInterpreter,
    MetadataInterpreter,
    MonteCarloInterpreter,
    RNGInterpreter,
    SpectralMCInterpreter,
    StorageInterpreter,
    TrainingInterpreter,
    assert_never,
)

# Mock interpreter for testing
from spectralmc.effects.mock import MockInterpreter

# Shared registry for data flow
from spectralmc.effects.registry import (
    RegistryError,
    RegistryKeyNotFound,
    RegistryTypeMismatch,
    SharedRegistry,
)


__all__ = [
    # Master effect union
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
    # Composition
    "EffectSequence",
    "EffectParallel",
    "sequence_effects",
    "parallel_effects",
    "map_effect",
    # Interpreters
    "EffectInterpreter",
    "SpectralMCInterpreter",
    "GPUInterpreter",
    "TrainingInterpreter",
    "MonteCarloInterpreter",
    "StorageInterpreter",
    "RNGInterpreter",
    "MetadataInterpreter",
    "LoggingInterpreter",
    "MockInterpreter",
    "assert_never",
    # Errors
    "EffectError",
    "GPUError",
    "TrainingError",
    "MonteCarloError",
    "StorageError",
    "RNGError",
    "MetadataError",
    "LoggingError",
    # Registry
    "SharedRegistry",
    "RegistryError",
    "RegistryKeyNotFound",
    "RegistryTypeMismatch",
]
