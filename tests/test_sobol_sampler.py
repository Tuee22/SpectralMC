from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError, model_validator

from spectralmc.sobol_sampler import BoundSpec, SobolSampler

SEED = 42


class Point(BaseModel):
    x: float
    y: float


def test_skip_changes_sequence() -> None:
    dims = {
        "x": BoundSpec(lower=0.0, upper=1.0),
        "y": BoundSpec(lower=-1.0, upper=1.0),
    }
    p0 = SobolSampler(Point, dims, seed=SEED).sample(1)[0]
    p8 = SobolSampler(Point, dims, skip=8, seed=SEED).sample(1)[0]
    assert (p0.x, p0.y) != (p8.x, p8.y)


def test_validation_error_bubbles() -> None:
    class AlwaysFail(BaseModel):
        z: float

        @model_validator(mode="after")
        def _fail(self) -> "AlwaysFail":  # noqa: D401
            raise ValueError("forced failure")

    dims = {"z": BoundSpec(lower=0.0, upper=1.0)}
    sampler = SobolSampler(AlwaysFail, dims, seed=SEED)
    with pytest.raises(ValidationError):
        sampler.sample(1)
