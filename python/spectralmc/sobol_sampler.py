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
)

points = sampler.sample(8)
print(points[0].x, points[0].y)
```
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Generic, List, Type, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError, model_validator

# If your environment lacks typed stubs for "scipy.stats.qmc", you can do:
from scipy.stats.qmc import Sobol  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# type helpers
# ---------------------------------------------------------------------------

#: **PointT** represents *one* Pydantic model instance that encodes the
#: coordinates of a single point in the Sobol‑sampled hyper‑cube.  Any class
#: derived from ``pydantic.BaseModel`` can play this role, provided its fields
#: match the keys given in ``dimensions``.
PointT = TypeVar("PointT", bound=BaseModel)

# ---------------------------------------------------------------------------
# pydantic bound specification
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# main sampler
# ---------------------------------------------------------------------------


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

    Any `ValidationError` raised by the model bubbles up immediately (no silent
    skipping).
    """

    # ---------------------------------------------------------------------
    # construction
    # ---------------------------------------------------------------------

    def __init__(
        self,
        pydantic_class: Type[PointT],
        dimensions: Dict[str, BoundSpec],
        *,
        skip: int = 0,
    ) -> None:
        # ------------------------------------------------------------------
        # 0) Ensure every dimension name matches a field on the Pydantic model
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 1) Store references and pre‑compute helpers
        # ------------------------------------------------------------------
        self._pydantic_class: Type[PointT] = pydantic_class
        self._dimensions = dimensions
        self._dimension_names = list(dimensions.keys())

        # Vectorised lower / upper bounds
        self._lower_bounds = np.array(
            [v.lower for v in dimensions.values()], dtype=float
        )
        self._upper_bounds = np.array(
            [v.upper for v in dimensions.values()], dtype=float
        )

        # Dimensionality
        self._d = len(dimensions)

        # Sobol generator
        self._sampler = Sobol(d=self._d, scramble=True)

        # Burn‑in / fast‑forward
        if skip:
            self._sampler.fast_forward(skip)

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------

    def _create_instance(self, row: NDArray[np.float64]) -> PointT:
        """Instantiate PointT from a single NumPy row."""
        data = {self._dimension_names[i]: row[i] for i in range(self._d)}
        return self._pydantic_class(**data)

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------

    def sample(self, n_samples: int) -> List[PointT]:
        """
        Generate *n_samples* Sobol points and return them as validated PointT
        instances.

        Raises
        ------
        ValidationError
            If any sample violates the model constraints.
        """
        # Raw Sobol points in [0,1]
        samples = self._sampler.random(n_samples)

        # Scale to [lower, upper]
        scaled = (
            self._lower_bounds + (self._upper_bounds - self._lower_bounds) * samples
        )

        return [self._create_instance(row) for row in scaled]


# =============================================================================
# self‑contained sanity checks – run with ``python -m sobol_sampler``
# =============================================================================


def _run_unit_tests() -> None:
    """Very small, zero‑dependency test‑suite exercising key paths."""
    print("Running SobolSampler self‑tests …")

    # 1) BoundSpec happy‑path
    BoundSpec(lower=0.0, upper=1.0)

    # 2) BoundSpec invalid bounds
    try:
        BoundSpec(lower=1.0, upper=1.0)
        raise AssertionError("Expected ValidationError for equal bounds")
    except ValidationError:
        pass

    # 3) SobolSampler basic sampling + attribute access + skip
    class Point(BaseModel):
        x: float
        y: float

    dims = {
        "x": BoundSpec(lower=0.0, upper=1.0),
        "y": BoundSpec(lower=-1.0, upper=1.0),
    }

    sampler_noskip: SobolSampler[Point] = SobolSampler(Point, dims)
    sampler_skip: SobolSampler[Point] = SobolSampler(Point, dims, skip=8)

    pts_noskip: List[Point] = sampler_noskip.sample(4)
    pts_skip: List[Point] = sampler_skip.sample(4)

    assert (pts_noskip[0].x, pts_noskip[0].y) != (
        pts_skip[0].x,
        pts_skip[0].y,
    ), "Skipping had no effect on the generated sequence"

    # 4) Ensure ValidationError propagates
    class AlwaysFail(BaseModel):
        z: float

        @model_validator(mode="after")
        def _fail(self) -> "AlwaysFail":  # noqa: D401
            raise ValueError("Forced failure")

    failing_sampler: SobolSampler[AlwaysFail] = SobolSampler(
        AlwaysFail, {"z": BoundSpec(lower=0.0, upper=1.0)}
    )
    try:
        failing_sampler.sample(1)
        raise AssertionError("Expected ValidationError but none was raised")
    except ValidationError:
        pass

    print("✓ All SobolSampler unit‑tests passed.")


def main() -> None:
    """Entry‑point for `python -m sobol_sampler`."""
    _run_unit_tests()


if __name__ == "__main__":
    # Supports both `python sobol_sampler.py` and `python -m sobol_sampler`.
    if Path(sys.argv[0]).stem in {"sobol_sampler", "__main__"}:
        main()
