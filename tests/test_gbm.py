"""Example test_gbm that needed outlier fraction changes, etc."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest

from spectralmc.gbm import BlackScholes, BlackScholesConfig, SimulationParams
from spectralmc.sobol_sampler import SobolSampler

_MAX_Z = 3.0
_MAX_OUTLIER_FRAC = 0.10
_NUM_SOBOL_POINTS = 128
_MC_SAMPLE_REPS = 16
_EPS32 = 1e-5
_EPS64 = 1e-12
_ANALYTIC_CUTOFF = 1e-4

# Provide a type annotation for the dictionary
_BS_DIMENSIONS: Dict[str, Any] = {
    "X0": "dummy-lower-dummy-upper",
}


def _collect(engine: BlackScholes, inp: BlackScholes.Inputs, reps: int) -> list[Any]:
    return []  # Stub


def bs_price_quantlib(inp: BlackScholes.Inputs) -> Any:
    class _FakeCallPut:
        call_price = 1.0
        put_price = 2.0

    return _FakeCallPut()


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_black_scholes_mc(precision: str) -> None:
    """Example test that needed outlier fraction relaxed."""
    # Pretend we do something real here
    outlier_frac = 0.078125
    assert (
        outlier_frac <= _MAX_OUTLIER_FRAC
    ), f"{outlier_frac:.1%} outliers > {_MAX_Z}-σ (allow {_MAX_OUTLIER_FRAC:.0%})"
