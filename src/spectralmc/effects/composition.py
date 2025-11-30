"""
Effect composition utilities for sequencing and parallel execution.

This module provides pure combinators for composing effects without
executing them. Composition is purely structural - actual execution
happens only in the interpreter.

Type Safety:
    - All composition types are frozen dataclasses (immutable)
    - Generic types preserve result types through composition
    - Callable types enable custom result transformations

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from spectralmc.effects.types import Effect


T = TypeVar("T")


@dataclass(frozen=True)
class EffectSequence(Generic[T]):
    """Sequence of effects to execute in order.

    Attributes:
        effects: Tuple of effects to execute sequentially.
        continuation: Function to combine results into final value.

    Example:
        >>> training_step = EffectSequence(
        ...     effects=(
        ...         ForwardPass(model_id="cvnn", input_tensor_id="batch"),
        ...         BackwardPass(loss_tensor_id="loss"),
        ...         OptimizerStep(optimizer_id="adam"),
        ...         StreamSync(stream_type="torch"),
        ...     ),
        ...     continuation=lambda results: results[-1],
        ... )
    """

    effects: tuple[Effect, ...]
    continuation: Callable[[list[object]], T]


@dataclass(frozen=True)
class EffectParallel(Generic[T]):
    """Parallel effects to execute concurrently.

    Only use for effects that are truly independent with no ordering
    requirements. The interpreter may execute these in any order or
    concurrently.

    Attributes:
        effects: Tuple of independent effects.
        combiner: Function to combine all results.

    Example:
        >>> # Upload multiple artifacts in parallel
        >>> upload_artifacts = EffectParallel(
        ...     effects=(
        ...         WriteObject(bucket="models", key="v1/checkpoint.pb"),
        ...         WriteObject(bucket="models", key="v1/metadata.json"),
        ...         WriteObject(bucket="models", key="v1/content_hash.txt"),
        ...     ),
        ...     combiner=lambda results: all(r is not None for r in results),
        ... )
    """

    effects: tuple[Effect, ...]
    combiner: Callable[[list[object]], T]


def sequence_effects(*effects: Effect) -> EffectSequence[list[object]]:
    """Compose effects to execute sequentially.

    Returns:
        EffectSequence that executes effects in order and returns all results.

    Example:
        >>> seq = sequence_effects(
        ...     GenerateNormals(rows=1024, cols=252, seed=42),
        ...     SimulatePaths(spot=100.0, vol=0.2),
        ...     StreamSync(stream_type="cupy"),
        ... )
    """
    return EffectSequence(effects=effects, continuation=lambda x: x)


def parallel_effects(*effects: Effect) -> EffectParallel[list[object]]:
    """Compose effects to execute in parallel.

    Returns:
        EffectParallel that executes effects concurrently and returns all results.

    Example:
        >>> par = parallel_effects(
        ...     WriteObject(bucket="models", key="v1/a.pb"),
        ...     WriteObject(bucket="models", key="v1/b.pb"),
        ... )
    """
    return EffectParallel(effects=effects, combiner=lambda x: x)


def map_effect(effect: Effect, f: Callable[[object], T]) -> EffectSequence[T]:
    """Map a function over an effect's result.

    This is the functor operation for effects.

    Args:
        effect: The effect to map over.
        f: Function to apply to the effect's result.

    Returns:
        EffectSequence that executes the effect and applies f to its result.

    Example:
        >>> mapped = map_effect(
        ...     ReadObject(bucket="config", key="settings.json"),
        ...     lambda data: json.loads(data),
        ... )
    """
    return EffectSequence(effects=(effect,), continuation=lambda results: f(results[0]))
