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
        skip: int = 0,
        seed: int | None = None,
    ) -> None:
        model_fields = set(pydantic_class.model_fields)
        dim_names = set(dimensions)

        if model_fields != dim_names:
            raise ValueError(
                "Dimension keys do not match model fields.\n"
                f" → model fields   : {sorted(model_fields)}\n"
                f" → dimension keys : {sorted(dim_names)}"
            )
        if skip < 0:
            raise ValueError("`skip` must be a non-negative integer")
        if seed is not None and seed < 0:
            raise ValueError("`seed` must be a non-negative integer or None")

        self._pydantic_class: Type[PointT] = pydantic_class
        self._dimension_names = list(dimensions)

        self._lower_bounds = np.array([v.lower for v in dimensions.values()])
        self._upper_bounds = np.array([v.upper for v in dimensions.values()])

        self._d = len(dimensions)
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
