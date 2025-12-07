"""SpectralMC error ADTs."""

from spectralmc.errors.sampler import (
    DimensionMismatch,
    InvalidBounds,
    NegativeSamples,
    SamplerValidationFailed,
)
from spectralmc.errors.async_normals import (
    InvalidDType,
    InvalidShape,
    QueueBusy,
    QueueEmpty,
    SeedOutOfRange,
)
from spectralmc.errors.trainer import (
    OptimizerStateSerializationFailed,
    PredictionFailed,
    SamplerError,
    SamplerInitFailed,
)
from spectralmc.errors.gbm import NormalsGenerationFailed, NormalsUnavailable, NormGeneratorError
from spectralmc.errors.cvnn_factory import (
    CVNNFactoryError,
    CVNNFactoryResult,
    ModelOnWrongDevice,
    SerializationDeviceMismatch,
    UnhandledConfigNode,
)

__all__ = [
    "DimensionMismatch",
    "InvalidBounds",
    "NegativeSamples",
    "SamplerValidationFailed",
    "SamplerError",
    "SamplerInitFailed",
    "InvalidDType",
    "InvalidShape",
    "QueueBusy",
    "QueueEmpty",
    "SeedOutOfRange",
    "NormalsGenerationFailed",
    "NormalsUnavailable",
    "NormGeneratorError",
    "CVNNFactoryError",
    "CVNNFactoryResult",
    "ModelOnWrongDevice",
    "SerializationDeviceMismatch",
    "OptimizerStateSerializationFailed",
    "PredictionFailed",
    "UnhandledConfigNode",
]
