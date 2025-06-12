"""
Sobol-based quasi-random sampling with automatic Pydantic validation.
Clean under ``mypy --strict`` without any ignores or casts.
"""

from __future__ import annotations

from typing import Dict, Generic, List, Type, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, model_validator
from scipy.stats.qmc import Sobol

# --------------------------------------------------------------------------- #
# Type helpers                                                                #
# --------------------------------------------------------------------------- #

PointT = TypeVar("PointT", bound=BaseModel)


class BoundSpec(BaseModel):
    """Simple lower/upper bound pair defining one dimension."""

    lower: float
    upper: float

    @model_validator(mode="after")
    def _check_bounds(self) -> "BoundSpec":
        if self.lower >= self.upper:
            raise ValueError("`lower` must be strictly less than `upper`.")
        return self


class SobolSampler(Generic[PointT]):
    """Draw Sobol points and validate them via a Pydantic model ``PointT``."""

    def __init__(
        self,
        pydantic_class: Type[PointT],
        dimensions: Dict[str, BoundSpec],
        *,
        seed: int,
        skip: int = 0,
    ) -> None:
        self._dimension_names = list(pydantic_class.model_fields)

        if set(self._dimension_names) != set(dimensions.keys()):
            raise ValueError(
                "Dimension keys do not match model fields.\n"
                f" → model fields   : {self._dimension_names}\n"
                f" → dimension keys : {dimensions.keys()}"
            )
        if skip < 0:
            raise ValueError("`skip` must be a non-negative integer")
        if seed < 0:
            raise ValueError("`seed` must be a non-negative integer or None")

        self._pydantic_class: Type[PointT] = pydantic_class

        self._lower_bounds = np.array([dimensions[d].lower for d in self._dimension_names])
        self._upper_bounds = np.array([dimensions[d].upper for d in self._dimension_names])

        self._d = len(self._dimension_names)
        self._sampler: Sobol = Sobol(d=self._d, scramble=True, seed=seed)
        if skip:
            self._sampler.fast_forward(skip)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _create_instance(self, row: NDArray[np.float64]) -> PointT:
        data = {name: float(row[i]) for i, name in enumerate(self._dimension_names)}
        return self._pydantic_class(**data)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def sample(self, n_samples: int) -> List[PointT]:
        """Return ``n_samples`` validated Sobol points."""
        samples: NDArray[np.float64] = self._sampler.random(n_samples)
        scaled = (
            self._lower_bounds + (self._upper_bounds - self._lower_bounds) * samples
        )
        return [self._create_instance(row) for row in scaled]
