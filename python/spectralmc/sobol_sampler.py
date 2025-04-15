# python/sobol_sampler.py

"""
Generates Sobol samples with optional pydantic validation.
Useful for param sweeps or advanced sampling in Monte Carlo or Bayesian contexts.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ValidationError, model_validator
from typing import (
    Type,
    Dict,
    List,
)
from numpy.typing import NDArray

# If your environment lacks typed stubs for "scipy.stats.qmc", you can do:
from scipy.stats.qmc import Sobol  # type: ignore[import-untyped]


class BoundSpec(BaseModel):
    lower: float
    upper: float

    @model_validator(mode="after")
    def check_bounds(self) -> BoundSpec:
        """
        Validate that 'lower' is strictly less than 'upper' after the fields are set.
        Pydantic 2.x uses instance-based validators with @model_validator.
        """
        if self.lower >= self.upper:
            raise ValueError("The 'lower' value must be strictly less than 'upper'.")
        return self


class SobolSampler:
    """
    A SobolSampler that draws quasi-random samples with optional pydantic checks.

    NOTE: This version will raise a ValidationError if any generated sample
    is invalid, rather than silently skipping it.
    """

    def __init__(
        self,
        pydantic_class: Type[BaseModel],
        dimensions: Dict[str, BoundSpec],
    ) -> None:
        self.pydantic_class = pydantic_class
        self.dimensions = dimensions
        self.dimension_names = list(dimensions.keys())

        # Convert BoundSpec fields into NumPy arrays for vectorized scaling
        self.lower_bounds = np.array(
            [v.lower for v in dimensions.values()], dtype=float
        )
        self.upper_bounds = np.array(
            [v.upper for v in dimensions.values()], dtype=float
        )

        # Number of dimensions
        self.d = len(dimensions)

        # Create a Sobol generator
        self.sampler = Sobol(d=self.d, scramble=True)

    def _create_instance(self, row: NDArray[np.float64]) -> BaseModel:
        """
        Build the Pydantic model instance from a single row.
        If validation fails, a ValidationError is raised immediately.
        """
        data = {self.dimension_names[i]: row[i] for i in range(self.d)}
        # No try/except: invalid data => immediate ValidationError
        return self.pydantic_class(**data)

    def sample(self, n_samples: int) -> List[BaseModel]:
        """
        Generate Sobol samples and return a list of validated model instances.
        If any sample fails validation, a ValidationError is raised for that sample.
        """
        # Generate Sobol samples in [0, 1]
        samples = self.sampler.random(n_samples)

        # Scale from [0, 1] to [lower, upper]
        scaled = self.lower_bounds + (self.upper_bounds - self.lower_bounds) * samples

        # Attempt to create a model instance for every row;
        # if any row is invalid, this will raise ValidationError immediately.
        return [self._create_instance(row) for row in scaled]
