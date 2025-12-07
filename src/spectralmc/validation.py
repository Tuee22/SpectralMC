"""Helper utilities for pure, Result-based Pydantic validation."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from spectralmc.result import Failure, Result, Success


TModel = TypeVar("TModel", bound=BaseModel)

__all__: list[str] = ["validate_model"]


def validate_model(model_cls: type[TModel], **data: object) -> Result[TModel, ValidationError]:
    """
    Construct a Pydantic model and surface validation issues as a Result.

    Pydantic still performs validation internally (and will raise), but callers
    remain pure by catching that exception at the boundary and returning a
    Failure. This keeps SpectralMC code expression-oriented while retaining
    Pydantic's rich error messages.
    """
    try:
        return Success(model_cls(**data))
    except ValidationError as exc:
        return Failure(exc)
