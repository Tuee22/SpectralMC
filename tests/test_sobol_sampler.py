"""Strictly-typed pytest suite for :class:`spectralmc.sobol_sampler.SobolSampler`.

Features
~~~~~~~~
* Deterministic reproducibility checks via the ``skip`` parameter.
* Bound-adherence tests across 1-D and 3-D spaces.
* Constructor validation for mismatched dimensions, negative parameters and
  invalid *BoundSpec* pairs.
* Smoke test covering a typical two-dimensional sampling scenario.

Design constraints
~~~~~~~~~~~~~~~~~~
* Passes ``mypy --strict`` assuming third-party stubs are present.
* Uses power-of-two sample counts to silence SciPy Sobol balance warnings.
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence, Tuple

import pytest
from pydantic import BaseModel, ValidationError, model_validator

from spectralmc.sobol_sampler import BoundSpec, SobolSampler

# --------------------------------------------------------------------------- #
# Global constants                                                             #
# --------------------------------------------------------------------------- #

_SEED: int = 42
_TOLERANCE: float = 1e-12
_N_SAMPLES_BOUND_CHECK: int = 1_024  # 2**10

# (skip, n_check) pairs — all powers of two except the zero-skip baseline.
_SKIP_REPRO_CASES: Tuple[Tuple[int, int], ...] = (
    (0, 1),  # 1  == 2**0
    (8, 8),  # 8  == 2**3
    (16, 16),  # 16 == 2**4
)

# --------------------------------------------------------------------------- #
# Pydantic models used in tests                                                #
# --------------------------------------------------------------------------- #


class Point(BaseModel):
    """Two-dimensional point with *x* and *y* coordinates."""

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
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #


def _as_tuples(points: Sequence[Point]) -> List[Tuple[float, float]]:
    """Return ``[(x, y), …]`` tuples for a sequence of *Point* instances."""

    return [(p.x, p.y) for p in points]


# --------------------------------------------------------------------------- #
# Reproducibility via *skip*                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_skip, n_check", _SKIP_REPRO_CASES)
def test_skip_reproducibility(n_skip: int, n_check: int) -> None:
    """Ensure that the *skip* argument yields a deterministic sequence offset.

    A sampler constructed with ``skip == n`` must produce the same *next*
    ``n_check`` points as another sampler that draws and discards the first
    ``n`` points manually.
    """

    dims: dict[str, BoundSpec] = {
        "x": BoundSpec(lower=0.0, upper=1.0),
        "y": BoundSpec(lower=-1.0, upper=1.0),
    }

    sampler_manual = SobolSampler(Point, dims, seed=_SEED)
    _ = sampler_manual.sample(n_skip)  # Burn-in.
    tail_manual: List[Point] = sampler_manual.sample(n_check)

    sampler_skip = SobolSampler(Point, dims, seed=_SEED, skip=n_skip)
    tail_skip: List[Point] = sampler_skip.sample(n_check)

    assert _as_tuples(tail_manual) == _as_tuples(tail_skip)


# --------------------------------------------------------------------------- #
# Bound adherence                                                              #
# --------------------------------------------------------------------------- #


_BOUNDS_FIXTURES: Tuple[Tuple[dict[str, BoundSpec], type[BaseModel]], ...] = (
    (
        {"x": BoundSpec(lower=0.0, upper=1.0)},
        OneDim,
    ),
    (
        {
            "x": BoundSpec(lower=-3.14, upper=3.14),
            "y": BoundSpec(lower=-1.0, upper=1.0),
            "z": BoundSpec(lower=10.0, upper=20.0),
        },
        ThreeDim,
    ),
)


@pytest.mark.parametrize(("dims", "model_cls"), _BOUNDS_FIXTURES)
def test_samples_respect_bounds(
    dims: dict[str, BoundSpec], model_cls: type[BaseModel]
) -> None:
    """All generated coordinates must lie inside their declared bounds."""

    sampler: SobolSampler[BaseModel] = SobolSampler(model_cls, dims, seed=_SEED)
    points: List[BaseModel] = sampler.sample(_N_SAMPLES_BOUND_CHECK)

    for instance in points:
        for axis, bound in dims.items():
            value: float = getattr(instance, axis)
            assert (
                bound.lower - _TOLERANCE <= value <= bound.upper + _TOLERANCE
            ), f"{axis}={value} outside [{bound.lower}, {bound.upper}]"


# --------------------------------------------------------------------------- #
# Constructor validation                                                       #
# --------------------------------------------------------------------------- #


def test_invalid_boundspec_raises() -> None:
    """*BoundSpec* must enforce ``lower < upper`` at model-validation time."""

    with pytest.raises(ValidationError):
        BoundSpec(lower=1.0, upper=1.0)


def test_dimension_mismatch_error() -> None:
    """Sampler constructor should fail when field names disagree with *dims*."""

    class XY(BaseModel):
        x: float
        y: float

    dims: dict[str, BoundSpec] = {"x": BoundSpec(lower=0.0, upper=1.0)}
    pattern = re.compile(r"dimension keys do not match model fields", re.IGNORECASE)
    with pytest.raises(ValueError, match=pattern):
        SobolSampler(XY, dims, seed=_SEED)


@pytest.mark.parametrize("param", ["skip", "seed"])
def test_negative_parameters_disallowed(param: str) -> None:
    """Negative values for ``skip`` and ``seed`` must raise *ValueError*."""

    dims: dict[str, BoundSpec] = {"x": BoundSpec(lower=0.0, upper=1.0)}

    if param == "skip":
        with pytest.raises(ValueError):
            SobolSampler(Point, dims, seed=_SEED, skip=-1)
    else:  # param == "seed"
        with pytest.raises(ValueError):
            SobolSampler(Point, dims, seed=-1)


# --------------------------------------------------------------------------- #
# Validation propagation                                                       #
# --------------------------------------------------------------------------- #


def test_validation_error_bubbles() -> None:
    """Errors raised inside user-defined validators must propagate unchanged."""

    class AlwaysFail(BaseModel):
        z: float

        @model_validator(mode="after")
        def _fail(self) -> "AlwaysFail":  # noqa: D401 – intentional failure
            raise ValueError("forced failure")

    dims: dict[str, BoundSpec] = {"z": BoundSpec(lower=0.0, upper=1.0)}
    sampler: SobolSampler[AlwaysFail] = SobolSampler(AlwaysFail, dims, seed=_SEED)
    with pytest.raises(ValidationError, match="forced failure"):
        sampler.sample(1)


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #


def test_typical_usage_smoke() -> None:
    """End-to-end sanity check sampling four 2-D points."""

    dims: dict[str, BoundSpec] = {
        "x": BoundSpec(lower=0.0, upper=1.0),
        "y": BoundSpec(lower=-1.0, upper=1.0),
    }

    sampler: SobolSampler[Point] = SobolSampler(Point, dims, seed=_SEED)
    pts: List[Point] = sampler.sample(4)  # 4 == 2**2

    tuples: List[Tuple[float, float]] = _as_tuples(pts)
    assert len(set(tuples)) == 4  # No duplicates in first four draws.

    xs, ys = zip(*tuples)
    assert not math.isclose(min(xs), max(xs)), "x-coordinates are constant"
    assert not math.isclose(min(ys), max(ys)), "y-coordinates are constant"
