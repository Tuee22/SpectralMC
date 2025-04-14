# python/spectralmc/sobol_sampler.py

"""
Generates Sobol samples with optional pydantic validation.
Useful for param sweeps or advanced sampling in Monte Carlo or Bayesian contexts.
"""

from __future__ import annotations

import numpy as np
from scipy.stats.qmc import Sobol  # type: ignore[import-untyped]
from pydantic import BaseModel, ValidationError
from typing import Type, List, Dict


class SobolSampler:
    """
    A SobolSampler that draws quasi-random samples with optional pydantic checks.
    """

    def __init__(
        self, pydantic_class: Type[BaseModel], dimensions: Dict[str, List[float]]
    ) -> None:
        self.pydantic_class = pydantic_class
        self.dimensions = dimensions
        self.dimension_names = list(dimensions.keys())
        self.lower_bounds = np.array([v[0] for v in dimensions.values()])
        self.upper_bounds = np.array([v[1] for v in dimensions.values()])
        self.d = len(dimensions)
        self.sampler = Sobol(d=self.d, scramble=True)

    def sample(self, n_samples: int) -> List[BaseModel]:
        samples = self.sampler.random(n_samples)
        scaled = self.lower_bounds + (self.upper_bounds - self.lower_bounds) * samples
        results: List[BaseModel] = []
        for row in scaled:
            data = {self.dimension_names[i]: row[i] for i in range(self.d)}
            try:
                instance = self.pydantic_class(**data)
                results.append(instance)
            except ValidationError as e:
                print("Validation error for:", data, e)
        return results
