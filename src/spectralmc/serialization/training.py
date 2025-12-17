# src/spectralmc/serialization/training.py
"""
Converters for training configuration.

NOTE: Full implementation requires tensor converters.
"""

from __future__ import annotations

from spectralmc.gbm_trainer import TrainingConfig, build_training_config
from spectralmc.errors.trainer import InvalidTrainingConfig
from spectralmc.result import Failure, Result, Success
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
    def from_proto(
        proto: training_pb2.TrainingConfigProto,
    ) -> Result[TrainingConfig, InvalidTrainingConfig]:
        """Convert from proto."""
        match build_training_config(
            num_batches=proto.num_batches,
            batch_size=proto.batch_size,
            learning_rate=proto.learning_rate,
        ):
            case Failure(err):
                return Failure(err)
            case Success(cfg):
                return Success(cfg)


__all__ = ["TrainingConfigConverter"]
