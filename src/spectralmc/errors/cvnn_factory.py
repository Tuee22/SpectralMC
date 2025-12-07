"""Error ADTs for the CVNN factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar

from spectralmc.errors.torch_facade import TorchFacadeError
from spectralmc.result import Result


@dataclass(frozen=True)
class UnhandledConfigNode:
    """CVNN config contains an unsupported or unhandled node."""

    node: str
    kind: Literal["UnhandledConfigNode"] = "UnhandledConfigNode"


@dataclass(frozen=True)
class ModelOnWrongDevice:
    """The CVNN must be on CPU before serialisation or loading."""

    device: str
    kind: Literal["ModelOnWrongDevice"] = "ModelOnWrongDevice"


@dataclass(frozen=True)
class SerializationDeviceMismatch:
    """Serialisation expected tensors on CPU but received another device or dtype."""

    message: str
    kind: Literal["SerializationDeviceMismatch"] = "SerializationDeviceMismatch"


CVNNFactoryError = (
    UnhandledConfigNode | ModelOnWrongDevice | SerializationDeviceMismatch | TorchFacadeError
)

T = TypeVar("T")
CVNNFactoryResult = Result[T, CVNNFactoryError]

__all__ = [
    "CVNNFactoryError",
    "CVNNFactoryResult",
    "ModelOnWrongDevice",
    "SerializationDeviceMismatch",
    "UnhandledConfigNode",
]
