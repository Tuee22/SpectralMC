# src/spectralmc/serialization/models.py
"""Converters for CVNN model configuration."""

from __future__ import annotations

from spectralmc.cvnn_factory import (
    ActivationCfg,
    ActivationKind,
    CVNNConfig,
    ExplicitWidth,
    LinearCfg,
    PreserveWidth,
    WidthSpec,
)
from spectralmc.proto import models_pb2

from .common import DTypeConverter


class WidthSpecConverter:
    """Convert WidthSpec ADT."""

    @staticmethod
    def to_proto(width: WidthSpec) -> models_pb2.WidthSpecProto:
        """Convert to proto."""
        proto = models_pb2.WidthSpecProto()
        if isinstance(width, PreserveWidth):
            proto.preserve.CopyFrom(models_pb2.PreserveWidthProto())
        elif isinstance(width, ExplicitWidth):
            proto.explicit.value = width.value
        return proto

    @staticmethod
    def from_proto(proto: models_pb2.WidthSpecProto) -> WidthSpec:
        """Convert from proto."""
        which = proto.WhichOneof("spec")
        if which == "preserve":
            return PreserveWidth()
        if which == "explicit":
            return ExplicitWidth(value=proto.explicit.value)
        raise ValueError(f"Unknown WidthSpec variant: {which}")


class ActivationCfgConverter:
    """Convert ActivationCfg."""

    @staticmethod
    def to_proto(cfg: ActivationCfg) -> models_pb2.ActivationCfgProto:
        """Convert to proto."""
        proto = models_pb2.ActivationCfgProto()
        kind_mapping = {
            ActivationKind.MOD_RELU: models_pb2.ACTIVATION_KIND_MOD_RELU,
            ActivationKind.Z_RELU: models_pb2.ACTIVATION_KIND_Z_RELU,
        }
        proto.kind = kind_mapping[cfg.kind]
        proto.bias = 0.0  # Not currently used in Pydantic model
        return proto

    @staticmethod
    def from_proto(proto: models_pb2.ActivationCfgProto) -> ActivationCfg:
        """Convert from proto."""
        kind_mapping = {
            models_pb2.ACTIVATION_KIND_MOD_RELU: ActivationKind.MOD_RELU,
            models_pb2.ACTIVATION_KIND_Z_RELU: ActivationKind.Z_RELU,
        }
        return ActivationCfg(kind=kind_mapping[proto.kind])


class LinearCfgConverter:
    """Convert LinearCfg."""

    @staticmethod
    def to_proto(cfg: LinearCfg) -> models_pb2.LinearCfgProto:
        """Convert to proto."""
        proto = models_pb2.LinearCfgProto()
        proto.width.CopyFrom(WidthSpecConverter.to_proto(cfg.width))
        proto.bias = cfg.bias
        if cfg.activation:
            proto.activation.CopyFrom(ActivationCfgConverter.to_proto(cfg.activation))
        return proto

    @staticmethod
    def from_proto(proto: models_pb2.LinearCfgProto) -> LinearCfg:
        """Convert from proto."""
        activation = None
        if proto.HasField("activation"):
            activation = ActivationCfgConverter.from_proto(proto.activation)
        return LinearCfg(
            width=WidthSpecConverter.from_proto(proto.width),
            bias=proto.bias,
            activation=activation,
        )


class CVNNConfigConverter:
    """Convert CVNNConfig."""

    @staticmethod
    def to_proto(config: CVNNConfig) -> models_pb2.CVNNConfigProto:
        """Convert to proto."""
        proto = models_pb2.CVNNConfigProto()
        proto.dtype = DTypeConverter.to_proto(config.dtype)
        # Note: LayerCfg is complex (recursive), simplified for now
        proto.seed = config.seed
        return proto

    @staticmethod
    def from_proto(proto: models_pb2.CVNNConfigProto) -> CVNNConfig:
        """Convert from proto."""
        return CVNNConfig(
            dtype=DTypeConverter.from_proto(proto.dtype),
            layers=[],  # Simplified
            seed=proto.seed,
        )


__all__ = [
    "WidthSpecConverter",
    "ActivationCfgConverter",
    "LinearCfgConverter",
    "CVNNConfigConverter",
]
