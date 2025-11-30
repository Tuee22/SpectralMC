# tests/test_sobol_sampler.py
"""
test_sobol_sampler.py
=====================
Strictly-typed pytest suite for
:class:`spectralmc.sobol_sampler.SobolSampler`.

The sampler is float64-only, so the tests do **not** parametrize over
precision.  Focus areas:

1. **Skip reproducibility** - `skip=n` must produce the same tail as
   manually discarding ``n`` samples.
2. **Bound adherence** in 1- and 3-D cases.
3. **Constructor validation** for dimension mismatch, negative params,
   and invalid :class:`BoundSpec`.
4. **Propagation** of user-defined validator failures.
5. A quick **smoke test** exercising nominal two-dimensional usage.

The file passes **``mypy --strict``** with no ignores.
"""

from __future__ import annotations

import math
from typing import Sequence

import pytest
from pydantic import BaseModel, ValidationError, model_validator

from spectralmc.sobol_sampler import BoundSpec, SobolConfig, SobolSampler


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

_SEED: int = 42
_TOL: float = 1e-12
_N_SAMPLES_BOUND_CHECK: int = 1_024  # 2**10

_SKIP_REPRO: tuple[tuple[int, int], ...] = (
    (0, 1),
    (8, 8),
    (16, 16),
)

# --------------------------------------------------------------------------- #
# Pydantic models                                                             #
# --------------------------------------------------------------------------- #


class Point(BaseModel):
    """Two-dimensional point."""

    x: float
    y: float


class OneDim(BaseModel):
    """Single-axis model used for 1-D bound checks."""

    x: float


class ThreeDim(BaseModel):
    """Three-axis model used for 3-D bound checks."""

    x: float
    y: float
    z: float


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #


def _pairs(pts: Sequence[Point]) -> list[tuple[float, float]]:
    """Convert a sequence of :class:`Point` to ``[(x, y), …]`` tuples."""
    return [(p.x, p.y) for p in pts]


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_skip, n_check", _SKIP_REPRO)
def test_skip_repro(n_skip: int, n_check: int) -> None:
    """`skip` must yield a deterministic sequence offset."""
    dims = {
        "x": BoundSpec(lower=0.0, upper=1.0),
        "y": BoundSpec(lower=-1.0, upper=1.0),
    }

    manual = SobolSampler(Point, dims, config=SobolConfig(seed=_SEED))
    _ = manual.sample(n_skip)  # burn-in
    expected = manual.sample(n_check)

    fast = SobolSampler(Point, dims, config=SobolConfig(seed=_SEED, skip=n_skip))
    got = fast.sample(n_check)

    assert _pairs(expected) == _pairs(got)


_BOUND_CASES: tuple[tuple[dict[str, BoundSpec], type[BaseModel]], ...] = (
    ({"x": BoundSpec(lower=0.0, upper=1.0)}, OneDim),
    (
        {
            "x": BoundSpec(lower=-3.14, upper=3.14),
            "y": BoundSpec(lower=-1.0, upper=1.0),
            "z": BoundSpec(lower=10.0, upper=20.0),
        },
        ThreeDim,
    ),
)


@pytest.mark.parametrize(("dims", "model_cls"), _BOUND_CASES)
def test_bounds(dims: dict[str, BoundSpec], model_cls: type[BaseModel]) -> None:
    """Every generated coordinate must lie within its declared bounds."""
    sampler = SobolSampler(model_cls, dims, config=SobolConfig(seed=_SEED))
    pts = sampler.sample(_N_SAMPLES_BOUND_CHECK)

    for obj in pts:
        for axis, b in dims.items():
            val = float(getattr(obj, axis))
            assert b.lower - _TOL <= val <= b.upper + _TOL


def test_boundspec_validation() -> None:
    """`BoundSpec` must enforce `lower < upper`."""
    with pytest.raises(ValidationError):
        BoundSpec(lower=1.0, upper=1.0)


def test_dim_mismatch() -> None:
    """Constructor should fail when field names differ from dim keys."""

    class XY(BaseModel):
        x: float
        y: float

    dims = {"x": BoundSpec(lower=0.0, upper=1.0)}
    with pytest.raises(ValueError, match="dimension keys do not match"):
        SobolSampler(XY, dims, config=SobolConfig(seed=_SEED))


@pytest.mark.parametrize("param", ["skip", "seed"])
def test_negative_args(param: str) -> None:
    """Negative `skip` or `seed` must raise *ValidationError* via Pydantic."""
    dims = {"x": BoundSpec(lower=0.0, upper=1.0)}
    if param == "skip":
        with pytest.raises(ValidationError):
            SobolSampler(Point, dims, config=SobolConfig(seed=_SEED, skip=-1))
    else:
        with pytest.raises(ValidationError):
            SobolSampler(Point, dims, config=SobolConfig(seed=-1))


def test_validator_bubbles() -> None:
    """Exceptions inside user validators must propagate unchanged."""

    class AlwaysFail(BaseModel):
        z: float

        @model_validator(mode="after")
        def _fail(self) -> AlwaysFail:
            raise ValueError("forced failure")

    dims = {"z": BoundSpec(lower=0.0, upper=1.0)}
    sampler = SobolSampler(AlwaysFail, dims, config=SobolConfig(seed=_SEED))
    with pytest.raises(ValidationError, match="forced failure"):
        sampler.sample(1)


def test_smoke_two_dim() -> None:
    """End-to-end sanity check sampling four 2-D points."""
    dims = {
        "x": BoundSpec(lower=0.0, upper=1.0),
        "y": BoundSpec(lower=-1.0, upper=1.0),
    }
    sampler = SobolSampler(Point, dims, config=SobolConfig(seed=_SEED))
    pts = sampler.sample(4)

    pairs = _pairs(pts)
    assert len(set(pairs)) == 4  # uniqueness
    xs, ys = zip(*pairs)
    assert not math.isclose(min(xs), max(xs))
    assert not math.isclose(min(ys), max(ys))
