# tests/test_gbm.py
"""Regression tests for **spectralmc.gbm.BlackScholes**.

* Strictly typed – `mypy --strict` clean.
* All numeric literals live in UPPER_CASE constants.
* SWIG/QuantLib `DeprecationWarning`s are suppressed once at import time.
* Accuracy rule:
  - ≤ 5 % of Sobol points may exceed 3-σ (z-score).
  - RMSPE ≤ 15 % across points where analytic price ≥ 1 unit.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, List, Literal, TypeAlias

warnings.filterwarnings(
    "ignore",
    message=r"builtin type .* has no __module__ attribute",
    category=DeprecationWarning,
)

import numpy as np
import numpy.typing as npt
import pytest
import QuantLib as ql  # type: ignore[import-untyped]

from spectralmc.gbm import (
    BlackScholes,
    BlackScholesConfig,
    SimulationParams,
    DtypeLiteral,
)
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Inputs: TypeAlias = BlackScholes.Inputs
HostPriceResults: TypeAlias = BlackScholes.HostPricingResults

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BS_DIMENSIONS: dict[str, BoundSpec] = {
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

# ---------------------------------------------------------------------------
# Analytic Black helper
# ---------------------------------------------------------------------------


def bs_price_quantlib(inp: Inputs) -> HostPriceResults:
    """Analytic Black price via QuantLib."""
    std = inp.v * math.sqrt(inp.T)
    df = math.exp(-inp.r * inp.T)
    fwd = inp.X0 * math.exp((inp.r - inp.d) * inp.T)

    put = ql.blackFormula(ql.Option.Put, inp.K, fwd, std, df)
    call = ql.blackFormula(ql.Option.Call, inp.K, fwd, std, df)

    put_intr = df * max(inp.K - fwd, 0.0)
    call_intr = df * max(fwd - inp.K, 0.0)

    return HostPriceResults(
        put_price_intrinsic=put_intr,
        call_price_intrinsic=call_intr,
        underlying=fwd,
        put_convexity=put - put_intr,
        call_convexity=call - call_intr,
        put_price=put,
        call_price=call,
    )


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
    """Aggregate accuracy check with comprehensions (no for-loops)."""
    engine = _make_engine(precision)
    sampler = SobolSampler(BlackScholes.Inputs, _BS_DIMENSIONS, seed=31)

    def _stats(inp: Inputs) -> tuple[float, float | None]:
        mc_vals = np.array(
            [p.call_price for p in _collect(engine, inp, _MC_SAMPLE_REPS)]
        )
        analytic = bs_price_quantlib(inp).call_price

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

    assert outlier_frac <= _MAX_OUTLIER_FRAC, (
        f"{outlier_frac:.1%} outliers > {_MAX_Z}-σ " f"(allow {_MAX_OUTLIER_FRAC:.0%})"
    )
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
        assert math.isclose(e.call_price, r.call_price, rel_tol=_DET_REL_TOL)
        assert math.isclose(e.put_price, r.put_price, rel_tol=_DET_REL_TOL)
