"""
RNG Effect ADTs for random state capture and restoration.

This module defines frozen dataclasses representing all RNG-related side effects,
enabling exhaustive pattern matching and type-safe effect composition.

Type Safety:
    - All effect types are frozen dataclasses (immutable)
    - Literal discriminators enable exhaustive pattern matching
    - Union types define closed sets of effect variants

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - reproducibility_proofs.md - Determinism guarantees
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CaptureRNGState:
    """Request to capture current RNG state.

    Attributes:
        kind: Discriminator for pattern matching. Always "CaptureRNGState".
        rng_type: Which RNG state to capture.
        output_id: Identifier for storing captured state bytes in registry.
    """

    kind: Literal["CaptureRNGState"] = "CaptureRNGState"
    rng_type: Literal["torch_cpu", "torch_cuda", "cupy", "numpy"] = "torch_cpu"
    output_id: str = "rng_state"


@dataclass(frozen=True)
class RestoreRNGState:
    """Request to restore previously captured RNG state.

    Attributes:
        kind: Discriminator for pattern matching. Always "RestoreRNGState".
        rng_type: Which RNG state to restore.
        state_bytes: The captured state as bytes.
    """

    kind: Literal["RestoreRNGState"] = "RestoreRNGState"
    rng_type: Literal["torch_cpu", "torch_cuda", "cupy", "numpy"] = "torch_cpu"
    state_bytes: bytes = b""


# RNG Effect Union
RNGEffect = CaptureRNGState | RestoreRNGState
