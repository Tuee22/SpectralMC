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
from spectralmc.errors.serialization import (
    InvalidWidthSpecProto,
    SerializationResult,
    UnknownActivationKind,
    ValidationFailed,
)
from spectralmc.proto import models_pb2
from spectralmc.result import Failure, Success
from spectralmc.validation import validate_model

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
    def from_proto(proto: models_pb2.WidthSpecProto) -> SerializationResult[WidthSpec]:
        """Convert from proto."""
        which = proto.WhichOneof("spec")
        if which == "preserve":
            return Success(PreserveWidth())
        if which == "explicit":
            return Success(ExplicitWidth(value=proto.explicit.value))
        return Failure(InvalidWidthSpecProto(variant=which))


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
    def from_proto(proto: models_pb2.ActivationCfgProto) -> SerializationResult[ActivationCfg]:
        """Convert from proto."""
        kind_mapping = {
            models_pb2.ACTIVATION_KIND_MOD_RELU: ActivationKind.MOD_RELU,
            models_pb2.ACTIVATION_KIND_Z_RELU: ActivationKind.Z_RELU,
        }
        kind = kind_mapping.get(proto.kind)
        if kind is None:
            return Failure(UnknownActivationKind(value=proto.kind))
        result = validate_model(ActivationCfg, kind=kind)
        match result:
            case Failure(error):
                return Failure(ValidationFailed(error=error))
            case Success(cfg):
                return Success(cfg)


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
    def from_proto(proto: models_pb2.LinearCfgProto) -> SerializationResult[LinearCfg]:
        """Convert from proto."""
        width_result = WidthSpecConverter.from_proto(proto.width)
        match width_result:
            case Failure(error):
                return Failure(error)
            case Success(width):
                pass

        activation: ActivationCfg | None = None
        if proto.HasField("activation"):
            activation_result = ActivationCfgConverter.from_proto(proto.activation)
            match activation_result:
                case Failure(error):
                    return Failure(error)
                case Success(act):
                    activation = act

        cfg_result = validate_model(
            LinearCfg,
            width=width,
            bias=proto.bias,
            activation=activation,
        )
        match cfg_result:
            case Failure(val_err):
                return Failure(ValidationFailed(error=val_err))
            case Success(cfg):
                return Success(cfg)


class CVNNConfigConverter:
    """Convert CVNNConfig."""

    @staticmethod
    def to_proto(config: CVNNConfig) -> SerializationResult[models_pb2.CVNNConfigProto]:
        """Convert to proto."""
        proto = models_pb2.CVNNConfigProto()
        match DTypeConverter.to_proto(config.dtype):
            case Failure(error):
                return Failure(error)
            case Success(dtype_proto):
                proto.dtype = dtype_proto
        # Note: LayerCfg is complex (recursive), simplified for now
        proto.seed = config.seed
        return Success(proto)

    @staticmethod
    def from_proto(proto: models_pb2.CVNNConfigProto) -> SerializationResult[CVNNConfig]:
        """Convert from proto."""
        match DTypeConverter.from_proto(proto.dtype):
            case Failure(error):
                return Failure(error)
            case Success(dtype):
                pass

        config_result = validate_model(
            CVNNConfig,
            dtype=dtype,
            layers=[],
            seed=proto.seed,
        )
        match config_result:
            case Failure(validation_error):
                return Failure(ValidationFailed(error=validation_error))
            case Success(cfg):
                return Success(cfg)


__all__ = [
    "WidthSpecConverter",
    "ActivationCfgConverter",
    "LinearCfgConverter",
    "CVNNConfigConverter",
]
