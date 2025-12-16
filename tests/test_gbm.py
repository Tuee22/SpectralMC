# tests/test_gbm.py
"""
Regression tests for :pymod:`spectralmc.gbm`.

Each constant lives in UPPER-CASE to keep *mypy --strict* and the
project's flake-strict config happy.
"""

from __future__ import annotations

import math
import warnings
from typing import TypeAlias

import numpy as np
import pytest


from spectralmc.effects import ForwardNormalization, PathScheme
from spectralmc.gbm import (
    BlackScholes,
    ThreadsPerBlock,
)
from spectralmc.models.numerical import Precision
from spectralmc.quantlib import bs_price_quantlib
from spectralmc.sobol_sampler import SobolConfig, SobolSampler
from spectralmc.testing import default_domain_bounds
from tests.helpers import expect_success, make_black_scholes_config, make_simulation_params


# ─────────────────────────────── type aliases ───────────────────────────────
Inputs: TypeAlias = BlackScholes.Inputs
HostPriceResults: TypeAlias = BlackScholes.HostPricingResults

# ──────────────────────────────── constants ─────────────────────────────────
_BS_DIMENSIONS = default_domain_bounds(
    x0=(0.001, 10_000),
    k=(0.001, 20_000),
    t=(0.0, 10.0),
    r=(-0.20, 0.20),
    d=(-0.20, 0.20),
    v=(0.0, 2.0),
)

_TIMESTEPS = 1
_NETWORK_SIZE = 256
_BATCHES_PER_RUN = (
    2**15
)  # Reduced workload (now 32,768 batches) to keep test runtime under the 60s timeout
_THREADS_PER_BLOCK: ThreadsPerBlock = 256
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

# QuantLib deprecation warnings are noisy on recent releases - silence once.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*QuantLib.*")


# ─────────────────────── helper constructors / utilities ────────────────────
def _make_engine(precision: Precision, *, skip: int = 0) -> BlackScholes:
    """Factory for a deterministic engine instance."""
    sim_params = make_simulation_params(
        timesteps=_TIMESTEPS,
        network_size=_NETWORK_SIZE,
        batches_per_mc_run=_BATCHES_PER_RUN,
        threads_per_block=_THREADS_PER_BLOCK,
        mc_seed=_MC_SEED,
        buffer_size=_BUFFER_SIZE,
        skip=skip,
        dtype=precision,
    )

    cfg = make_black_scholes_config(
        sim_params=sim_params,
        path_scheme=PathScheme.LOG_EULER,
        normalization=ForwardNormalization.RAW,
    )
    return BlackScholes(cfg)


def _collect(engine: BlackScholes, inp: Inputs, n: int) -> list[HostPriceResults]:
    """Return *n* independent Monte-Carlo prices."""
    return [expect_success(engine.price_to_host(inp)) for _ in range(n)]


# ───────────────────────────────── tests ────────────────────────────────────
@pytest.mark.parametrize("precision", [Precision.float32, Precision.float64])
def test_black_scholes_mc(precision: Precision) -> None:
    """Aggregate MC accuracy against analytic Black-Scholes."""
    engine = _make_engine(precision)
    sampler_result = SobolSampler.create(
        pydantic_class=BlackScholes.Inputs,
        dimensions=_BS_DIMENSIONS,
        config=SobolConfig(seed=31, skip=0),
    )

    def _stats(inp: Inputs) -> tuple[float, float | None]:
        mc_vals = np.array([p.put_price for p in _collect(engine, inp, _MC_SAMPLE_REPS)])
        analytic = bs_price_quantlib(inp).put_price

        mean = float(mc_vals.mean())
        s = float(mc_vals.std(ddof=1))
        if s > 0.0:
            se = s / math.sqrt(mc_vals.size)
            z = abs(mean - analytic) / se
        else:
            eps = _EPS32 if precision is Precision.float32 else _EPS64
            z = 0.0 if abs(mean - analytic) <= eps else _MAX_Z + 1.0

        rel_err = (mean - analytic) / analytic if analytic >= _ANALYTIC_CUTOFF else None
        return z, rel_err

    sampler = expect_success(sampler_result)
    samples = expect_success(sampler.sample(_NUM_SOBOL_POINTS))
    results = [_stats(inp) for inp in samples]
    z_scores = [z for z, _ in results]
    rel_errors = [re for _, re in results if re is not None]

    outlier_frac = float(np.mean(np.asarray(z_scores) > _MAX_Z))
    rmspe = float(np.sqrt(np.mean(np.square(rel_errors)))) if rel_errors else 0.0

    assert outlier_frac <= _MAX_OUTLIER_FRAC
    assert rmspe <= _RMSPE_TOL


@pytest.mark.parametrize("precision", [Precision.float64, Precision.float32])
def test_snapshot_determinism(precision: Precision) -> None:
    """Snapshot → restore must preserve RNG state exactly."""
    engine = _make_engine(precision)
    inp = BlackScholes.Inputs(X0=100, K=100, T=1.0, r=0.05, d=0.0, v=0.2)

    _ = _collect(engine, inp, _SNAPSHOT_REPS)
    snap = expect_success(engine.snapshot())
    expected = _collect(engine, inp, _SNAPSHOT_REPS)
    restored = BlackScholes(snap)
    reproduced = _collect(restored, inp, _SNAPSHOT_REPS)

    for e, r in zip(expected, reproduced, strict=True):
        assert math.isclose(e.put_price, r.put_price, rel_tol=_DET_REL_TOL)
        assert math.isclose(e.call_price, r.call_price, rel_tol=_DET_REL_TOL)
