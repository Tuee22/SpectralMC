"""
Tests for MockInterpreter.

Verifies effect recording, mock results, and assertion helpers.
"""

from __future__ import annotations

import pytest


from spectralmc.effects import (
    BackwardPass,
    ForwardPass,
    GenerateNormals,
    MockInterpreter,
    OptimizerStep,
    StreamSync,
    TensorTransfer,
)
from spectralmc.effects.errors import TrainingError
from spectralmc.result import Failure, Success


class TestMockInterpreterRecording:
    """Tests for effect recording functionality."""

    @pytest.mark.asyncio
    async def test_records_single_effect(self) -> None:
        """MockInterpreter records a single effect."""
        mock = MockInterpreter()
        effect = StreamSync(stream_type="torch")

        await mock.interpret(effect)

        assert len(mock.recorded_effects) == 1
        assert mock.recorded_effects[0] == effect

    @pytest.mark.asyncio
    async def test_records_multiple_effects_in_order(self) -> None:
        """MockInterpreter records multiple effects in order."""
        mock = MockInterpreter()
        effect1 = ForwardPass(model_id="m", input_tensor_id="i")
        effect2 = BackwardPass(loss_tensor_id="l")
        effect3 = OptimizerStep(optimizer_id="o")

        await mock.interpret(effect1)
        await mock.interpret(effect2)
        await mock.interpret(effect3)

        assert len(mock.recorded_effects) == 3
        assert mock.recorded_effects[0] == effect1
        assert mock.recorded_effects[1] == effect2
        assert mock.recorded_effects[2] == effect3

    @pytest.mark.asyncio
    async def test_records_different_effect_types(self) -> None:
        """MockInterpreter records effects of different types."""
        mock = MockInterpreter()
        await mock.interpret(TensorTransfer(tensor_id="x"))
        await mock.interpret(StreamSync())
        await mock.interpret(ForwardPass(model_id="m"))
        await mock.interpret(GenerateNormals(rows=10, cols=10))

        assert len(mock.recorded_effects) == 4


class TestMockInterpreterResults:
    """Tests for mock result configuration."""

    @pytest.mark.asyncio
    async def test_returns_success_none_by_default(self) -> None:
        """MockInterpreter returns Success(None) by default."""
        mock = MockInterpreter()
        effect = StreamSync()

        result = await mock.interpret(effect)

        match result:
            case Success(value):
                assert value is None
            case Failure(_):
                pytest.fail("Expected Success")

    @pytest.mark.asyncio
    async def test_returns_configured_success_result(self) -> None:
        """MockInterpreter returns configured Success result."""
        mock = MockInterpreter()
        mock.mock_results[ForwardPass] = Success({"output_tensor_id": "out"})

        result = await mock.interpret(ForwardPass(model_id="m"))

        match result:
            case Success(value):
                assert value == {"output_tensor_id": "out"}
            case Failure(_):
                pytest.fail("Expected Success")

    @pytest.mark.asyncio
    async def test_returns_configured_failure_result(self) -> None:
        """MockInterpreter returns configured Failure result."""
        mock = MockInterpreter()
        expected_error = TrainingError(message="Model not found")
        mock.mock_results[ForwardPass] = Failure(expected_error)

        result = await mock.interpret(ForwardPass(model_id="missing"))

        match result:
            case Failure(error):
                assert error == expected_error
            case Success(_):
                pytest.fail("Expected Failure")

    @pytest.mark.asyncio
    async def test_different_results_for_different_types(self) -> None:
        """MockInterpreter returns different results for different effect types."""
        mock = MockInterpreter()
        mock.mock_results[ForwardPass] = Success({"stage": "forward"})
        mock.mock_results[BackwardPass] = Success({"stage": "backward"})

        forward_result = await mock.interpret(ForwardPass(model_id="m"))
        backward_result = await mock.interpret(BackwardPass(loss_tensor_id="l"))

        match forward_result:
            case Success(value):
                assert value == {"stage": "forward"}
            case _:
                pytest.fail("Expected Success for ForwardPass")

        match backward_result:
            case Success(value):
                assert value == {"stage": "backward"}
            case _:
                pytest.fail("Expected Success for BackwardPass")


class TestMockInterpreterAssertions:
    """Tests for assertion helper methods."""

    @pytest.mark.asyncio
    async def test_assert_effect_sequence_passes(self) -> None:
        """assert_effect_sequence passes when sequence matches."""
        mock = MockInterpreter()
        await mock.interpret(ForwardPass(model_id="m"))
        await mock.interpret(BackwardPass(loss_tensor_id="l"))
        await mock.interpret(OptimizerStep(optimizer_id="o"))

        # Should not raise
        mock.assert_effect_sequence([ForwardPass, BackwardPass, OptimizerStep])

    @pytest.mark.asyncio
    async def test_assert_effect_sequence_fails_on_mismatch(self) -> None:
        """assert_effect_sequence fails when sequence doesn't match."""
        mock = MockInterpreter()
        await mock.interpret(ForwardPass(model_id="m"))
        await mock.interpret(BackwardPass(loss_tensor_id="l"))

        with pytest.raises(AssertionError, match="Expected effect sequence"):
            mock.assert_effect_sequence([BackwardPass, ForwardPass])

    @pytest.mark.asyncio
    async def test_assert_effect_count_passes(self) -> None:
        """assert_effect_count passes when count matches."""
        mock = MockInterpreter()
        await mock.interpret(StreamSync())
        await mock.interpret(StreamSync())
        await mock.interpret(StreamSync())

        # Should not raise
        mock.assert_effect_count(3)

    @pytest.mark.asyncio
    async def test_assert_effect_count_fails_on_mismatch(self) -> None:
        """assert_effect_count fails when count doesn't match."""
        mock = MockInterpreter()
        await mock.interpret(StreamSync())
        await mock.interpret(StreamSync())

        with pytest.raises(AssertionError, match="Expected 5 effects, got 2"):
            mock.assert_effect_count(5)

    @pytest.mark.asyncio
    async def test_assert_contains_effect_passes(self) -> None:
        """assert_contains_effect passes when effect type is present."""
        mock = MockInterpreter()
        await mock.interpret(ForwardPass(model_id="m"))
        await mock.interpret(BackwardPass(loss_tensor_id="l"))

        # Should not raise
        mock.assert_contains_effect(BackwardPass)

    @pytest.mark.asyncio
    async def test_assert_contains_effect_fails_when_missing(self) -> None:
        """assert_contains_effect fails when effect type is not present."""
        mock = MockInterpreter()
        await mock.interpret(ForwardPass(model_id="m"))

        with pytest.raises(AssertionError, match="No effect of type OptimizerStep"):
            mock.assert_contains_effect(OptimizerStep)


class TestMockInterpreterFiltering:
    """Tests for effect filtering methods."""

    @pytest.mark.asyncio
    async def test_get_effects_of_type(self) -> None:
        """get_effects_of_type returns only effects of specified type."""
        mock = MockInterpreter()
        await mock.interpret(StreamSync(stream_type="torch"))
        await mock.interpret(ForwardPass(model_id="m"))
        await mock.interpret(StreamSync(stream_type="cupy"))
        await mock.interpret(BackwardPass(loss_tensor_id="l"))
        await mock.interpret(StreamSync(stream_type="numba"))

        sync_effects = mock.get_effects_of_type(StreamSync)

        assert len(sync_effects) == 3
        for effect in sync_effects:
            assert isinstance(effect, StreamSync)

    @pytest.mark.asyncio
    async def test_get_effects_of_type_empty(self) -> None:
        """get_effects_of_type returns empty list when no matching effects."""
        mock = MockInterpreter()
        await mock.interpret(ForwardPass(model_id="m"))

        sync_effects = mock.get_effects_of_type(StreamSync)

        assert len(sync_effects) == 0


class TestMockInterpreterClearing:
    """Tests for clearing functionality."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_state(self) -> None:
        """clear removes all recorded effects and mock results."""
        mock = MockInterpreter()
        mock.mock_results[ForwardPass] = Success({"test": True})
        await mock.interpret(ForwardPass(model_id="m"))

        mock.clear()

        assert len(mock.recorded_effects) == 0
        assert len(mock.mock_results) == 0

    @pytest.mark.asyncio
    async def test_reset_effects_keeps_mock_results(self) -> None:
        """reset_effects clears effects but keeps mock results."""
        mock = MockInterpreter()
        mock.mock_results[ForwardPass] = Success({"test": True})
        await mock.interpret(ForwardPass(model_id="m"))

        mock.reset_effects()

        assert len(mock.recorded_effects) == 0
        assert ForwardPass in mock.mock_results
