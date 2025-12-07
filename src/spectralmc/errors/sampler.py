"""Error ADTs for Sobol sampler construction and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError


@dataclass(frozen=True)
class DimensionMismatch:
    """Dimension keys do not match Pydantic model fields."""

    kind: Literal["DimensionMismatch"] = "DimensionMismatch"
    expected_fields: tuple[str, ...] = ()
    provided_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class InvalidBounds:
    """Bounds are invalid for at least one dimension."""

    message: str
    kind: Literal["InvalidBounds"] = "InvalidBounds"


@dataclass(frozen=True)
class NegativeSamples:
    """Requested a negative number of Sobol samples."""

    n_samples: int
    kind: Literal["NegativeSamples"] = "NegativeSamples"


@dataclass(frozen=True)
class SamplerValidationFailed:
    """Pydantic validation failed when constructing sampler inputs."""

    error: ValidationError
    kind: Literal["SamplerValidationFailed"] = "SamplerValidationFailed"
