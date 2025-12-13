"""
Tests for Effect composition utilities.

Verifies EffectSequence, EffectParallel, and composition helpers.
"""

from __future__ import annotations

from spectralmc.effects import (
    BackwardPass,
    ForwardPass,
    OptimizerStep,
    StreamSync,
    TensorTransfer,
)
from spectralmc.effects.composition import (
    EffectParallel,
    EffectSequence,
    map_effect,
    parallel_effects,
    sequence_effects,
)


class TestEffectSequence:
    """Tests for EffectSequence composition."""

    def test_sequence_effects_creates_sequence(self) -> None:
        """sequence_effects creates an EffectSequence."""
        seq = sequence_effects(
            ForwardPass(model_id="m"),
            BackwardPass(loss_tensor_id="l"),
            OptimizerStep(optimizer_id="o"),
        )

        assert isinstance(seq, EffectSequence)
        assert len(seq.effects) == 3

    def test_sequence_effects_preserves_order(self) -> None:
        """sequence_effects preserves effect order."""
        effect1 = ForwardPass(model_id="m")
        effect2 = BackwardPass(loss_tensor_id="l")
        effect3 = OptimizerStep(optimizer_id="o")

        seq = sequence_effects(effect1, effect2, effect3)

        assert seq.effects[0] == effect1
        assert seq.effects[1] == effect2
        assert seq.effects[2] == effect3

    def test_sequence_continuation_identity(self) -> None:
        """sequence_effects continuation returns results as-is."""
        seq = sequence_effects(StreamSync(), StreamSync())
        results: list[object] = [{"a": 1}, {"b": 2}]

        combined = seq.continuation(results)

        assert combined == results

    def test_effect_sequence_frozen(self) -> None:
        """EffectSequence is immutable."""
        seq = sequence_effects(StreamSync())

        # Tuple is immutable, so we can't directly test frozen,
        # but we can verify the structure
        assert isinstance(seq.effects, tuple)


class TestEffectParallel:
    """Tests for EffectParallel composition."""

    def test_parallel_effects_creates_parallel(self) -> None:
        """parallel_effects creates an EffectParallel."""
        par = parallel_effects(
            StreamSync(stream_type="torch"),
            StreamSync(stream_type="cupy"),
            StreamSync(stream_type="numba"),
        )

        assert isinstance(par, EffectParallel)
        assert len(par.effects) == 3

    def test_parallel_effects_preserves_effects(self) -> None:
        """parallel_effects preserves all effects."""
        effect1 = TensorTransfer(tensor_id="a")
        effect2 = TensorTransfer(tensor_id="b")

        par = parallel_effects(effect1, effect2)

        assert effect1 in par.effects
        assert effect2 in par.effects

    def test_parallel_combiner_identity(self) -> None:
        """parallel_effects combiner returns results as-is."""
        par = parallel_effects(StreamSync(), StreamSync())
        results: list[object] = [None, None]

        combined = par.combiner(results)

        assert combined == results


class TestMapEffect:
    """Tests for map_effect functor operation."""

    def test_map_effect_creates_sequence(self) -> None:
        """map_effect creates an EffectSequence with one effect."""
        effect = ForwardPass(model_id="m")
        mapped = map_effect(effect, lambda x: {"transformed": x})

        assert isinstance(mapped, EffectSequence)
        assert len(mapped.effects) == 1
        assert mapped.effects[0] == effect

    def test_map_effect_applies_function(self) -> None:
        """map_effect continuation applies the mapping function."""
        effect = ForwardPass(model_id="m")
        mapped = map_effect(effect, lambda x: f"result: {x}")

        result = mapped.continuation([{"output": "tensor"}])

        assert result == "result: {'output': 'tensor'}"

    def test_map_effect_extracts_first_result(self) -> None:
        """map_effect continuation uses first result from list."""

        def double_value(x: object) -> object:
            if isinstance(x, int):
                return x * 2
            return x

        effect = StreamSync()
        mapped = map_effect(effect, double_value)

        result = mapped.continuation([42])

        assert result == 84


class TestCompositionIntegration:
    """Integration tests for effect composition."""

    def test_training_step_sequence(self) -> None:
        """Complete training step can be composed as sequence."""
        training_step = sequence_effects(
            ForwardPass(model_id="cvnn", input_tensor_id="batch"),
            BackwardPass(loss_tensor_id="loss"),
            OptimizerStep(optimizer_id="adam"),
            StreamSync(stream_type="torch"),
        )

        assert len(training_step.effects) == 4
        assert isinstance(training_step.effects[0], ForwardPass)
        assert isinstance(training_step.effects[1], BackwardPass)
        assert isinstance(training_step.effects[2], OptimizerStep)
        assert isinstance(training_step.effects[3], StreamSync)

    def test_parallel_stream_sync(self) -> None:
        """Multiple stream syncs can be composed in parallel."""
        sync_all = parallel_effects(
            StreamSync(stream_type="torch"),
            StreamSync(stream_type="cupy"),
            StreamSync(stream_type="numba"),
        )

        assert len(sync_all.effects) == 3
        stream_types = [e.stream_type for e in sync_all.effects if isinstance(e, StreamSync)]
        assert "torch" in stream_types
        assert "cupy" in stream_types
        assert "numba" in stream_types
