"""Regression tests for **spectralmc.gbm.BlackScholes** (mypy-strict clean).

* All numeric literals live in UPPER_CASE constants.
* SWIG/QuantLib deprecation warnings are suppressed globally once.
* Accuracy rule:
  – ≤ 5 % of Sobol points may exceed 3-σ (z-score).
  – RMSPE ≤ 15 % across points where analytic price ≥ 1 unit.
"""

from __future__ import annotations

import math
import warnings
from typing import List, TypeAlias

import numpy as np
import pytest

from spectralmc.gbm import (
    BlackScholes,
    BlackScholesConfig,
    DtypeLiteral,
    SimulationParams,
)
from spectralmc.quantlib import bs_price_quantlib
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Inputs: TypeAlias = BlackScholes.Inputs
HostPriceResults: TypeAlias = BlackScholes.HostPricingResults

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BS_DIMENSIONS = {
    "X0": BoundSpec(lower=0.001, upper=10_000),
    "K": BoundSpec(lower=0.001, upper=20_000),
    "T": BoundSpec(lower=0.0, upper=10.0),
    "r": BoundSpec(lower=-0.20, upper=0.20),
    "d": BoundSpec(lower=-0.20, upper=0.20),
    "v": BoundSpec(lower=0.0, upper=2.0),
}

_TIMESTEPS = 1
_NETWORK_SIZE = 256
_BATCHES_PER_RUN = 2**19
_THREADS_PER_BLOCK = 256
_MC_SEED = 7
_BUFFER_SIZE = 1

_MC_SAMPLE_REPS = 16
_NUM_SOBOL_POINTS = 64

_MAX_Z = 3.0
_MAX_OUTLIER_FRAC = 0.05
_RMSPE_TOL = 0.15
_ANALYTIC_CUTOFF = 1.0
_EPS64 = 1e-8
_EPS32 = 1e-4

_SNAPSHOT_REPS = 8
_DET_REL_TOL = 1e-6

# Suppress QuantLib deprecation warnings once
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*QuantLib.*")

# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------


def _make_engine(dtype: DtypeLiteral, skip: int = 0) -> BlackScholes:
    sp = SimulationParams(
        timesteps=_TIMESTEPS,
        network_size=_NETWORK_SIZE,
        batches_per_mc_run=_BATCHES_PER_RUN,
        threads_per_block=_THREADS_PER_BLOCK,
        mc_seed=_MC_SEED,
        buffer_size=_BUFFER_SIZE,
        skip=skip,
        dtype=dtype,
    )
    cfg = BlackScholesConfig(
        sim_params=sp,
        simulate_log_return=True,
        normalize_forwards=False,
    )
    return BlackScholes(cfg)


def _collect(engine: BlackScholes, inp: Inputs, n: int) -> List[HostPriceResults]:
    return [engine.price_to_host(inp) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_black_scholes_mc(precision: DtypeLiteral) -> None:
    """Aggregate accuracy check against analytic Black price."""
    engine = _make_engine(precision)
    sampler = SobolSampler(BlackScholes.Inputs, _BS_DIMENSIONS, seed=31)

    def _stats(inp: Inputs) -> tuple[float, float | None]:
        mc_vals = np.array(
            [p.put_price for p in _collect(engine, inp, _MC_SAMPLE_REPS)]
        )
        analytic = bs_price_quantlib(inp).put_price

        mean = float(mc_vals.mean())
        s = float(mc_vals.std(ddof=1))
        if s > 0.0:
            se = s / math.sqrt(mc_vals.size)
            z = abs(mean - analytic) / se
        else:
            eps = _EPS32 if precision == "float32" else _EPS64
            z = 0.0 if abs(mean - analytic) <= eps else _MAX_Z + 1.0

        rel_err = (mean - analytic) / analytic if analytic >= _ANALYTIC_CUTOFF else None
        return z, rel_err

    results = [_stats(inp) for inp in sampler.sample(_NUM_SOBOL_POINTS)]
    z_scores = [z for z, _ in results]
    rel_errors = [re for _, re in results if re is not None]

    outlier_frac = float(np.mean(np.asarray(z_scores) > _MAX_Z))
    rmspe = float(np.sqrt(np.mean(np.square(rel_errors)))) if rel_errors else 0.0

    assert (
        outlier_frac <= _MAX_OUTLIER_FRAC
    ), f"{outlier_frac:.1%} outliers > {_MAX_Z}-σ (allow {_MAX_OUTLIER_FRAC:.0%})"
    assert rmspe <= _RMSPE_TOL, f"RMSPE {rmspe:.2%} exceeds tolerance {_RMSPE_TOL:.0%}"


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_snapshot_determinism(precision: DtypeLiteral) -> None:
    engine = _make_engine(precision)
    inp = BlackScholes.Inputs(X0=100, K=100, T=1.0, r=0.05, d=0.0, v=0.2)

    _ = _collect(engine, inp, _SNAPSHOT_REPS)
    snap = engine.snapshot()

    expected = _collect(engine, inp, _SNAPSHOT_REPS)
    restored = BlackScholes(snap)
    reproduced = _collect(restored, inp, _SNAPSHOT_REPS)

    for e, r in zip(expected, reproduced, strict=True):
        assert math.isclose(e.put_price, r.put_price, rel_tol=_DET_REL_TOL)
        assert math.isclose(e.put_price, r.put_price, rel_tol=_DET_REL_TOL)
