# src/spectralmc/serialization/training.py
"""
Converters for training configuration.

NOTE: Full implementation requires tensor converters.
"""

from __future__ import annotations

from spectralmc.gbm_trainer import TrainingConfig
from spectralmc.proto import training_pb2


class TrainingConfigConverter:
    """Convert TrainingConfig."""

    @staticmethod
    def to_proto(config: TrainingConfig) -> training_pb2.TrainingConfigProto:
        """Convert to proto."""
        proto = training_pb2.TrainingConfigProto()
        proto.num_batches = config.num_batches
        proto.batch_size = config.batch_size
        proto.learning_rate = config.learning_rate
        return proto

    @staticmethod
    def from_proto(proto: training_pb2.TrainingConfigProto) -> TrainingConfig:
        """Convert from proto."""
        return TrainingConfig(
            num_batches=proto.num_batches,
            batch_size=proto.batch_size,
            learning_rate=proto.learning_rate,
        )


__all__ = ["TrainingConfigConverter"]
