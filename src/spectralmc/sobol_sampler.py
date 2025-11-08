# src/spectralmc/sobol_sampler.py
"""
sobol_sampler.py
================
Quasi‑random sampling based on the Sobol low‑discrepancy sequence, with
automatic Pydantic validation and **strict static typing**.

Why this module is *float64‑only*
---------------------------------
* **SciPy’s implementation** of Sobol (`scipy.stats.qmc.Sobol`) always
  produces `numpy.float64` values.
* **Python’s default floating‑point type** is already a C `double`
  (`float64`), so no conversion is needed when passing values to user
  code.
* **Pydantic models accept Python floats** as‑is and do their own
  validation; they do not benefit from a separate “precision” knob.

Adding a precision flag would therefore provide no additional
functionality while complicating both the public surface and the test
suite.  Every numerical array in the implementation is explicitly typed
as `np.float64` to make this design decision obvious.

Public API
----------
* :class:`BoundSpec` – inclusive lower/upper bound pair for one axis.
* :class:`SobolSampler` – generic sampler parameterised by a Pydantic
  model.

Both classes pass **``mypy --strict``** without ignores, casts, or Any.
"""

from __future__ import annotations

from typing import Annotated, Dict, Generic, List, Type, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats.qmc import Sobol

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
    skip: Annotated[
        int, Field(ge=0, description="Number of initial points to skip")
    ] = 0

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
        If ``lower`` ≥ ``upper`` after validation.
    """

    lower: float
    upper: float

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ------------------------------------------------------------------ #
    # Validators                                                         #
    # ------------------------------------------------------------------ #

    @model_validator(mode="after")
    def _validate(self) -> "BoundSpec":  # noqa: D401
        """Ensure the lower bound is strictly less than the upper bound."""
        if self.lower >= self.upper:
            raise ValueError("`lower` must be strictly less than `upper`.")
        return self


class SobolSampler(Generic[PointT]):
    """Generate Sobol points and validate them via a Pydantic model.

    Parameters
    ----------
    pydantic_class
        The Pydantic model that defines the *schema* of each sample.
    dimensions
        Mapping ``field_name → BoundSpec``.  The keys **must** match the
        field names in ``pydantic_class``; order is irrelevant.
    config
        :class:`SobolConfig` instance containing seed and skip parameters.
        Validation is performed by Pydantic at construction time.

    Notes
    -----
    * All internal computations are in **float64** – the native precision
      of both SciPy’s Sobol engine and Python floats.
    * The class holds no mutable state aside from the SciPy engine; if you
      need checkpoints, serialise ``skip`` and reconstruct later.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        pydantic_class: Type[PointT],
        dimensions: Dict[str, BoundSpec],
        *,
        config: SobolConfig,
    ) -> None:

        # Field names in deterministic order
        self._fields: List[str] = list(pydantic_class.model_fields)
        if set(self._fields) != set(dimensions.keys()):
            raise ValueError("dimension keys do not match model fields")

        # Vectorised bounds
        self._lower: NDArray[np.float64] = np.array(
            [dimensions[f].lower for f in self._fields], dtype=np.float64
        )
        self._upper: NDArray[np.float64] = np.array(
            [dimensions[f].upper for f in self._fields], dtype=np.float64
        )

        self._model: Type[PointT] = pydantic_class
        self._sampler: Sobol = Sobol(
            d=len(self._fields), scramble=True, seed=config.seed
        )
        if config.skip:
            self._sampler.fast_forward(config.skip)

    # ------------------------------------------------------------------ #
    # Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _construct(self, row: NDArray[np.float64]) -> PointT:
        """Create a validated model instance from one coordinate row."""
        data = {name: float(row[i]) for i, name in enumerate(self._fields)}
        return self._model(**data)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def sample(self, n_samples: int) -> List[PointT]:
        """Return a list of Sobol points.

        Args:
            n_samples: Non‑negative number of points.  ``0`` returns ``[]``.

        Returns
        -------
        list[PointT]
            Instances of ``pydantic_class`` in generation order.

        Raises
        ------
        ValueError
            If ``n_samples`` is negative.
        """
        if n_samples < 0:
            raise ValueError("`n_samples` must be non‑negative")
        if n_samples == 0:
            return []

        raw: NDArray[np.float64] = self._sampler.random(n_samples)
        scaled: NDArray[np.float64] = self._lower + (self._upper - self._lower) * raw
        return [self._construct(row) for row in scaled]
