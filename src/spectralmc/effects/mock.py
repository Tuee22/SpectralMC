"""
Mock interpreter for testing effect-producing code without side effects.

This module provides MockInterpreter which records effects without executing them,
enabling unit testing of pure effect-producing code without GPU hardware or network.

Example:
    >>> mock = MockInterpreter()
    >>> mock.mock_results[ForwardPass] = Success({"loss_tensor_id": "loss_001"})
    >>>
    >>> # Run your code that produces effects
    >>> result = await my_training_function(mock)
    >>>
    >>> # Verify correct effect sequence
    >>> assert len(mock.recorded_effects) == 4
    >>> assert isinstance(mock.recorded_effects[0], ForwardPass)

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - testing_requirements.md - Testing patterns
"""

from __future__ import annotations

from spectralmc.effects.errors import EffectError
from spectralmc.effects.types import Effect
from spectralmc.result import Result, Success


class MockInterpreter:
    """Test interpreter that records effects without execution.

    Use this for unit testing effect-producing code without side effects.
    Effects are recorded in order and can be inspected after the test.

    Attributes:
        recorded_effects: List of all effects that were interpreted.
        mock_results: Dict mapping effect types to predetermined results.

    Example:
        >>> mock = MockInterpreter()
        >>> mock.mock_results[ForwardPass] = Success({"loss": 0.1})
        >>>
        >>> # Execute code that produces effects
        >>> await execute_training_step(mock)
        >>>
        >>> # Verify effect sequence
        >>> mock.assert_effect_sequence([ForwardPass, BackwardPass, OptimizerStep])
    """

    def __init__(self) -> None:
        """Initialize mock interpreter with empty state."""
        self.recorded_effects: list[Effect] = []
        self.mock_results: dict[type[Effect], Result[object, EffectError]] = {}

    async def interpret(self, effect: Effect) -> Result[object, EffectError]:
        """Record effect and return mock result.

        Args:
            effect: The effect to record.

        Returns:
            Mock result if configured, otherwise Success(None).
        """
        self.recorded_effects.append(effect)
        effect_type = type(effect)
        return self.mock_results.get(effect_type, Success(None))

    def assert_effect_sequence(self, expected: list[type[Effect]]) -> None:
        """Assert that recorded effects match expected sequence.

        Args:
            expected: List of expected effect types in order.

        Raises:
            AssertionError: If recorded effects don't match expected.
        """
        actual = [type(e) for e in self.recorded_effects]
        match actual == expected:
            case True:
                return
            case False:
                raise AssertionError(f"Expected effect sequence {expected}, got {actual}")

    def assert_effect_count(self, count: int) -> None:
        """Assert the number of recorded effects.

        Args:
            count: Expected number of effects.

        Raises:
            AssertionError: If count doesn't match.
        """
        actual = len(self.recorded_effects)
        match actual == count:
            case True:
                return
            case False:
                raise AssertionError(f"Expected {count} effects, got {actual}")

    def assert_contains_effect(self, effect_type: type[Effect]) -> None:
        """Assert that at least one effect of given type was recorded.

        Args:
            effect_type: The effect type to look for.

        Raises:
            AssertionError: If no effect of given type was recorded.
        """
        first_match = next((e for e in self.recorded_effects if isinstance(e, effect_type)), None)
        match first_match:
            case None:
                raise AssertionError(f"No effect of type {effect_type.__name__} recorded")
            case _:
                return

    def get_effects_of_type(self, effect_type: type[Effect]) -> list[Effect]:
        """Get all recorded effects of a specific type.

        Args:
            effect_type: The effect type to filter by.

        Returns:
            List of effects matching the given type.
        """
        return [e for e in self.recorded_effects if isinstance(e, effect_type)]

    def clear(self) -> None:
        """Clear all recorded effects and mock results."""
        self.recorded_effects.clear()
        self.mock_results.clear()

    def reset_effects(self) -> None:
        """Clear recorded effects but keep mock results."""
        self.recorded_effects.clear()
