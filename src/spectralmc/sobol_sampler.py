# src/spectralmc/sobol_sampler.py
"""
sobol_sampler.py
================
Quasi-random sampling based on the Sobol low-discrepancy sequence, with
automatic Pydantic validation and **strict static typing**.

Why this module is *float64-only*
---------------------------------
* **SciPy's implementation** of Sobol (`scipy.stats.qmc.Sobol`) always
  produces `numpy.float64` values.
* **Python's default floating-point type** is already a C `double`
  (`float64`), so no conversion is needed when passing values to user
  code.
* **Pydantic models accept Python floats** as-is and do their own
  validation; they do not benefit from a separate “precision” knob.

Adding a precision flag would therefore provide no additional
functionality while complicating both the public surface and the test
suite.  Every numerical array in the implementation is explicitly typed
as `np.float64` to make this design decision obvious.

Public API
----------
* :class:`BoundSpec` - inclusive lower/upper bound pair for one axis.
* :class:`SobolSampler` - generic sampler parameterised by a Pydantic
  model.

Both classes pass **``mypy --strict``** without ignores, casts, or Any.
"""

from __future__ import annotations

from typing import Annotated, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats.qmc import Sobol

from spectralmc.errors.sampler import (
    DimensionMismatch,
    InvalidBounds,
    NegativeSamples,
    SamplerValidationFailed,
)
from spectralmc.result import Failure, Result, Success
from spectralmc.validation import validate_model


__all__: list[str] = ["BoundSpec", "SobolConfig", "SobolSampler"]

# --------------------------------------------------------------------------- #
# Type helpers                                                                #
# --------------------------------------------------------------------------- #

PointT = TypeVar("PointT", bound=BaseModel)


class SobolConfig(BaseModel):
    """Configuration for Sobol sampler with validated parameters.

    Attributes
    ----------
    seed
        Non-negative seed for deterministic reproducibility.
    skip
        Non-negative number of initial points to discard (default: 0).

    Notes
    -----
    Validation is declarative via Pydantic Field constraints.
    """

    seed: Annotated[int, Field(ge=0, description="Non-negative seed for Sobol engine")]
    skip: Annotated[int, Field(ge=0, description="Number of initial points to skip")] = 0

    model_config = ConfigDict(frozen=True, extra="forbid")


class BoundSpec(BaseModel):
    """Inclusive numeric bounds for a single coordinate axis.

    Attributes
    ----------
    lower
        Inclusive lower bound.
    upper
        Inclusive upper bound.

    Raises
    ------
    ValueError
        If ``lower`` ≥ ``upper`` after validation.
    """

    lower: float
    upper: float

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ------------------------------------------------------------------ #
    # Validators                                                         #
    # ------------------------------------------------------------------ #

    @model_validator(mode="after")
    def _validate(self) -> BoundSpec:
        """Ensure the lower bound is strictly less than the upper bound."""
        if self.lower >= self.upper:
            raise ValueError("`lower` must be strictly less than `upper`.")
        return self


class SobolSampler(Generic[PointT]):
    """Generate Sobol points and validate them via a Pydantic model."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        fields: list[str],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        model: type[PointT],
        sampler: Sobol,
    ) -> None:
        self._fields = fields
        self._lower = lower
        self._upper = upper
        self._model = model
        self._sampler = sampler

    @classmethod
    def create(
        cls,
        pydantic_class: type[PointT],
        dimensions: dict[str, BoundSpec],
        *,
        config: SobolConfig,
    ) -> Result[SobolSampler[PointT], DimensionMismatch | InvalidBounds]:
        """Pure factory for SobolSampler."""

        fields: list[str] = list(pydantic_class.model_fields)
        if set(fields) != set(dimensions.keys()):
            return Failure(
                DimensionMismatch(
                    expected_fields=tuple(fields),
                    provided_fields=tuple(dimensions.keys()),
                )
            )

        try:
            lower = np.array([dimensions[f].lower for f in fields], dtype=np.float64)
            upper = np.array([dimensions[f].upper for f in fields], dtype=np.float64)
            sampler = Sobol(d=len(fields), scramble=True, seed=config.seed)
            if config.skip:
                sampler.fast_forward(config.skip)
        except Exception as exc:  # pragma: no cover - SciPy-specific edge
            return Failure(InvalidBounds(message=str(exc)))

        return Success(
            cls(fields=fields, lower=lower, upper=upper, model=pydantic_class, sampler=sampler)
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _construct(self, row: NDArray[np.float64]) -> Result[PointT, SamplerValidationFailed]:
        """Create a validated model instance from one coordinate row."""
        data = {name: float(row[i]) for i, name in enumerate(self._fields)}
        match validate_model(self._model, **data):
            case Success(model):
                return Success(model)
            case Failure(error):
                return Failure(SamplerValidationFailed(error=error))

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def sample(
        self, n_samples: int
    ) -> Result[list[PointT], NegativeSamples | SamplerValidationFailed]:
        """Return a list of Sobol points."""
        if n_samples < 0:
            return Failure(NegativeSamples(n_samples=n_samples))
        if n_samples == 0:
            return Success([])

        raw: NDArray[np.float64] = self._sampler.random(n_samples)
        scaled: NDArray[np.float64] = self._lower + (self._upper - self._lower) * raw

        results: list[PointT] = []
        for row in scaled:
            match self._construct(row):
                case Success(model):
                    results.append(model)
                case Failure(error):
                    return Failure(error)
        return Success(results)
