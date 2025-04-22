# python/sobol_sampler.py

"""
Sobol‑based quasi‑random sampling with automatic Pydantic validation
===================================================================

This helper lets you describe a *d*‑dimensional hyper‑cube with **`BoundSpec`**
bounds, supply a **Pydantic model that represents *one point* inside that
space**, and obtain validated Sobol samples as strongly‑typed model instances.

Typical use‑cases include parameter sweeps, Bayesian optimisation, or other
Monte‑Carlo scenarios where low‑discrepancy sequences outperform plain IID
sampling.

-------------------------------------------------------------------
Quick example
-------------------------------------------------------------------

```python
from pydantic import BaseModel
from spectralmc.sobol_sampler import BoundSpec, SobolSampler, PointT

class MyPoint(BaseModel):
    x: float
    y: float

sampler: SobolSampler[PointT] = SobolSampler(
    MyPoint,
    {
        "x": BoundSpec(lower=0.0, upper=1.0),
        "y": BoundSpec(lower=-1.0, upper=1.0),
    },
    skip=32,  # burn‑in
    seed=42   # seed for scrambling
)

points = sampler.sample(8)
print(points[0].x, points[0].y)
```
"""

from __future__ import annotations

from typing import Dict, Generic, List, Type, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError, model_validator

from scipy.stats.qmc import Sobol  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# type helpers
# ---------------------------------------------------------------------------

#: **PointT** represents *one* Pydantic model instance that encodes the
#: coordinates of a single point in the Sobol‑sampled hyper‑cube.  Any class
#: derived from ``pydantic.BaseModel`` can play this role, provided its fields
#: match the keys given in ``dimensions``.
PointT = TypeVar("PointT", bound=BaseModel)


class BoundSpec(BaseModel):
    """
    Simple lower/upper bound pair used to define one dimension of the cube.
    Validation guarantees ``lower < upper``.
    """

    lower: float
    upper: float

    @model_validator(mode="after")
    def check_bounds(self) -> "BoundSpec":
        """Ensure lower < upper."""
        if self.lower >= self.upper:
            raise ValueError("The 'lower' value must be strictly less than 'upper'.")
        return self


class SobolSampler(Generic[PointT]):
    """
    Draw quasi‑random Sobol samples and validate them with a user‑supplied
    Pydantic model *PointT*.

    Parameters
    ----------
    pydantic_class :
        Model describing one sample point.
    dimensions :
        Mapping *field name → BoundSpec*.
    skip :
        **Burn‑in** — how many Sobol points to skip from the beginning of the
        sequence (``Sobol.fast_forward(skip)``).  Must be ≥ 0.
    seed :
        Seed used for scrambling the Sobol sequence. Must be a non-negative int or None.
    """

    def __init__(
        self,
        pydantic_class: Type[PointT],
        dimensions: Dict[str, BoundSpec],
        *,
        skip: int = 0,
        seed: int,
    ) -> None:
        model_fields: set[str] = set(pydantic_class.model_fields)
        dim_names: set[str] = set(dimensions)

        if model_fields != dim_names:
            raise ValueError(
                "Dimension keys do not match model fields.\n"
                f" → model fields   : {sorted(model_fields)}\n"
                f" → dimension keys : {sorted(dim_names)}"
            )

        if skip < 0:
            raise ValueError("`skip` must be a non‑negative integer")

        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("`seed` must be a non-negative integer or None")

        self._pydantic_class: Type[PointT] = pydantic_class
        self._dimensions = dimensions
        self._dimension_names = list(dimensions.keys())

        self._lower_bounds = np.array(
            [v.lower for v in dimensions.values()], dtype=float
        )
        self._upper_bounds = np.array(
            [v.upper for v in dimensions.values()], dtype=float
        )

        self._d = len(dimensions)

        self._sampler = Sobol(d=self._d, scramble=True, seed=seed)

        if skip:
            self._sampler.fast_forward(skip)

    def _create_instance(self, row: NDArray[np.float64]) -> PointT:
        """Instantiate PointT from a single NumPy row."""
        data = {self._dimension_names[i]: row[i] for i in range(self._d)}
        return self._pydantic_class(**data)

    def sample(self, n_samples: int) -> List[PointT]:
        """
        Generate *n_samples* Sobol points and return them as validated PointT
        instances.

        Raises
        ------
        ValidationError
            If any sample violates the model constraints.
        """
        samples = self._sampler.random(n_samples)

        scaled = (
            self._lower_bounds + (self._upper_bounds - self._lower_bounds) * samples
        )

        return [self._create_instance(row) for row in scaled]
