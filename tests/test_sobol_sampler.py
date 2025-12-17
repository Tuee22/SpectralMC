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
from typing import Sequence, cast

import pytest

from pydantic import BaseModel, model_validator

from spectralmc.errors.sampler import BoundSpecInvalid, DimensionMismatch, SamplerValidationFailed
from spectralmc.result import Failure, Success
from spectralmc.sobol_sampler import (
    BoundSpec,
    DomainBounds,
    SobolConfig,
    SobolSampler,
    build_bound_spec,
    build_domain_bounds,
    build_sobol_config,
)


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


def _make_domain_bounds(
    model_cls: type[BaseModel], dims: dict[str, BoundSpec]
) -> DomainBounds[BaseModel]:
    match build_domain_bounds(model_cls, dims):
        case Success(domain_bounds):
            return domain_bounds
        case Failure(err):
            pytest.fail(f"failed to build domain bounds: {err}")


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_skip, n_check", _SKIP_REPRO)
def test_skip_repro(n_skip: int, n_check: int) -> None:
    """`skip` must yield a deterministic sequence offset."""
    dims = _make_domain_bounds(
        Point,
        {
            "x": build_bound_spec(0.0, 1.0).unwrap(),
            "y": build_bound_spec(-1.0, 1.0).unwrap(),
        },
    )

    manual = SobolSampler.create(Point, dims, config=SobolConfig(seed=_SEED))
    match manual:
        case Success(m):
            burn = m.sample(n_skip)
            match burn:
                case Failure(burn_err):
                    pytest.fail(f"burn-in failed: {burn_err}")
            expected_result = m.sample(n_check)
            match expected_result:
                case Success(expected):
                    pass
                case Failure(expected_err):
                    pytest.fail(f"expected sample failed: {expected_err}")
        case Failure(manual_err):
            pytest.fail(f"manual sampler failed: {manual_err}")

    fast = SobolSampler.create(Point, dims, config=SobolConfig(seed=_SEED, skip=n_skip))
    match fast:
        case Success(fast_sampler):
            got_result = fast_sampler.sample(n_check)
            match got_result:
                case Success(got):
                    pass
                case Failure(got_err):
                    pytest.fail(f"fast sample failed: {got_err}")
        case Failure(fast_err):
            pytest.fail(f"fast sampler failed: {fast_err}")

    assert _pairs(cast(Sequence[Point], expected)) == _pairs(cast(Sequence[Point], got))


_BOUND_CASES: tuple[tuple[DomainBounds[BaseModel], type[BaseModel]], ...] = (
    (_make_domain_bounds(OneDim, {"x": build_bound_spec(0.0, 1.0).unwrap()}), OneDim),
    (
        _make_domain_bounds(
            ThreeDim,
            {
                "x": build_bound_spec(-3.14, 3.14).unwrap(),
                "y": build_bound_spec(-1.0, 1.0).unwrap(),
                "z": build_bound_spec(10.0, 20.0).unwrap(),
            },
        ),
        ThreeDim,
    ),
)


@pytest.mark.parametrize(("dims", "model_cls"), _BOUND_CASES)
def test_bounds(dims: DomainBounds[BaseModel], model_cls: type[BaseModel]) -> None:
    """Every generated coordinate must lie within its declared bounds."""
    sampler = SobolSampler.create(model_cls, dims, config=SobolConfig(seed=_SEED))
    match sampler:
        case Success(s):
            pts_result = s.sample(_N_SAMPLES_BOUND_CHECK)
            match pts_result:
                case Success(pts):
                    pass
                case Failure(err):
                    pytest.fail(f"sampling failed: {err}")
        case Failure(err):
            pytest.fail(f"sampler creation failed: {err}")

    for obj in pts:
        for axis, b in dims.items():
            val = float(getattr(obj, axis))
            assert b.lower - _TOL <= val <= b.upper + _TOL


def test_boundspec_validation() -> None:
    """`BoundSpec` must enforce `lower < upper`."""
    result = build_bound_spec(lower=1.0, upper=1.0)
    match result:
        case Failure(BoundSpecInvalid(lower=low, upper=up)):
            assert low == 1.0
            assert up == 1.0
        case Success(_):
            pytest.fail("Expected BoundSpecInvalid failure for equal bounds")


def test_dim_mismatch() -> None:
    """Constructor should fail when field names differ from dim keys."""

    class XY(BaseModel):
        x: float
        y: float

    dims = {"x": build_bound_spec(0.0, 1.0).unwrap()}
    match build_domain_bounds(XY, dims):
        case Failure(err):
            assert isinstance(err, DimensionMismatch)
        case Success(_):
            pytest.fail("expected DimensionMismatch error")


@pytest.mark.parametrize("param", ["skip", "seed"])
def test_negative_args(param: str) -> None:
    """Negative `skip` or `seed` must yield Failure via builder."""
    if param == "skip":
        result = build_sobol_config(seed=_SEED, skip=-1)
    else:
        result = build_sobol_config(seed=-1)
    assert isinstance(result, Failure)


def test_validator_bubbles() -> None:
    """Exceptions inside user validators must propagate unchanged."""

    class AlwaysFail(BaseModel):
        z: float

        @model_validator(mode="after")
        def _fail(self) -> AlwaysFail:
            raise ValueError("forced failure")

    dims = _make_domain_bounds(AlwaysFail, {"z": build_bound_spec(0.0, 1.0).unwrap()})
    sampler = SobolSampler.create(AlwaysFail, dims, config=SobolConfig(seed=_SEED))
    match sampler:
        case Success(s):
            result = s.sample(1)
            match result:
                case Failure(err):
                    assert isinstance(err, SamplerValidationFailed)
                    assert "forced failure" in str(err.error)
                case Success(_):
                    pytest.fail("expected validation failure")
        case Failure(err):
            pytest.fail(f"sampler creation failed: {err}")


def test_smoke_two_dim() -> None:
    """End-to-end sanity check sampling four 2-D points."""
    dims = _make_domain_bounds(
        Point,
        {
            "x": build_bound_spec(0.0, 1.0).unwrap(),
            "y": build_bound_spec(-1.0, 1.0).unwrap(),
        },
    )
    sampler = SobolSampler.create(Point, dims, config=SobolConfig(seed=_SEED))
    match sampler:
        case Success(s):
            pts_result = s.sample(4)
            match pts_result:
                case Success(pts):
                    pass
                case Failure(err):
                    pytest.fail(f"sampling failed: {err}")
        case Failure(err):
            pytest.fail(f"sampler creation failed: {err}")

    pairs = _pairs(cast(Sequence[Point], pts))
    assert len(set(pairs)) == 4  # uniqueness
    xs, ys = zip(*pairs)
    assert not math.isclose(min(xs), max(xs))
    assert not math.isclose(min(ys), max(ys))
