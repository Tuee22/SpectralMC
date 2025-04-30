"""tests/test_bs_mc.py – GPU MC vs. QuantLib Black‐Scholes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple, TypeAlias, cast
from typing_extensions import Literal
from pydantic import BaseModel
from operator import attrgetter

import numpy as np
import pytest
from scipy.stats import t # type: ignore[import-untyped]
import pandas as pd

import cupy as cp  # type: ignore[import-untyped]
import QuantLib as ql  # type: ignore[import-untyped]

from spectralmc.gbm import BlackScholes, SimulationParams, DtypeLiteral
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

# ─────────────────────────────────── constants ──────────────────────────────────
_BS_DIMENSIONS: Final[Dict[str, BoundSpec]] = {
    "X0": BoundSpec(lower=0.001, upper=10_000),
    "K": BoundSpec(lower=0.001, upper=20_000),
    "T": BoundSpec(lower=0.0, upper=10.0),
    "r": BoundSpec(lower=-0.20, upper=0.20),
    "d": BoundSpec(lower=-0.20, upper=0.20),
    "v": BoundSpec(lower=0.0, upper=2.0),
}

_TIMESTEPS: Final[int] = 1_024
_NETWORK_SIZE: Final[int] = 256
_BATCHES_PER_RUN: Final[int] = 1_024
_THREADS_PER_BLOCK: Final[int] = 256
_MC_SEED: Final[int] = 42
_BUFFER_SIZE: Final[int] = 1

_SOBOL_N: Final[int] = 128
_REPS_COLD: Final[int] = 64
_REPS_HQ: Final[int] = 512
_EPSILON: Final[float] = 1e-6
_P_THRESHOLD: Final[float] = 0.05


_ARTEFACT_DIR: Final[Path] = Path(__file__).with_suffix("").parent / ".failed_artifacts"
_ARTEFACT_DIR.mkdir(exist_ok=True)


# ───────────────────────────── analytic Black helper ────────────────────────────
def bs_price_quantlib(inp: BlackScholes.Inputs) -> BlackScholes.HostPricingResults:
    """Closed-form Black price via QuantLib.blackFormula."""
    std = inp.v * math.sqrt(inp.T)
    df = math.exp(-inp.r * inp.T)
    fwd = inp.X0 * math.exp((inp.r - inp.d) * inp.T)

    call = ql.blackFormula(ql.Option.Call, inp.K, fwd, std, df)
    put = ql.blackFormula(ql.Option.Put, inp.K, fwd, std, df)

    call_intrinsic = df * max(fwd - inp.K, 0.0)
    put_intrinsic = df * max(inp.K - fwd, 0.0)

    return BlackScholes.HostPricingResults(
        call_price_intrinsic=call_intrinsic,
        put_price_intrinsic=put_intrinsic,
        underlying=fwd,
        call_convexity=call - call_intrinsic,
        put_convexity=put - put_intrinsic,
        call_price=call,
        put_price=put,
    )


def two_tailed_t_pvalue(sample: np.ndarray, mu0: float = 0.0) -> float:
    """
    Return the two-tailed p-value for a one-sample Student’s t-test.

    Parameters
    ----------
    sample : np.ndarray
        1-D array of observations (numeric).
    mu0 : float, optional (default = 0.0)
        The population mean under the null hypothesis H₀.

    Returns
    -------
    float
        Two-tailed p-value for testing H₀ : mean(sample) == mu0.
    """
    # Ensure a NumPy array of floats
    x = np.asarray(sample, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("At least two observations are required for a t-test.")

    # Sample statistics
    x_bar = x.mean()
    s = x.std(ddof=1)  # unbiased (n-1) denominator
    se = s / np.sqrt(n)  # standard error of the mean

    # t statistic and degrees of freedom
    t_stat = (x_bar - mu0) / se
    df = n - 1

    # Two-tailed p-value: 2 * P(T > |t_stat|)
    p_val = 2.0 * t.sf(abs(t_stat), df)

    return p_val


def SamplingResults(BaseModel):
    """results container for _mc_sample function below"""
    input: Inputs
    sample: List[BlackScholes.HostPricingResults]
    analytic_price: BlackScholes.HostPricingResults
    p_value: float


# ───────────────────────────── Monte-Carlo helpers ─────────────────────────────
def _mc_sample(engine: BlackScholes, input: Inputs, n: int) -> SamplingResults:
    """uses the MC engine to sample n MC prices, and performs a student T
    test to see if they are within tolerance"""
    sample = [engine.price_to_host(input) for _ in range(n)]
    analytic_price = bs_price_quantlib(input)

    # perform parity check
    parity_check = np.array([s.put_convexity - s.call_convexity for s in sample])
    assert (
        np.max(np.abs(parity_check)) < _EPSILON
    ), f"Error: parity check failed with input {input}"

    # compute sampling errors wrt analytic price
    sample_errors = (
        np.array([s.call_convexity for s in sample]) - analytic_price.call_convexity
    )

    # perform student-t test
    p_value = two_tailed_t_pvalue(sample_errors)

    return SamplingResults(
        input=input, sample=sample, analytic_price=analytic_price, p_value=p_value
    )


# ──────────────────────────────── PyTest entry ────────────────────────────────
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_black_scholes_mc(precision: DtypeLiteral) -> None:
    """GPU Monte-Carlo pricer must match QuantLib within 3 σ & 1 % RMSE."""
    sp = SimulationParams(
        timesteps=_TIMESTEPS,
        network_size=_NETWORK_SIZE,
        batches_per_mc_run=_BATCHES_PER_RUN,
        threads_per_block=_THREADS_PER_BLOCK,
        mc_seed=_MC_SEED,
        buffer_size=_BUFFER_SIZE,
        dtype=precision,
    )
    engine = BlackScholes(sp)
    sampler = SobolSampler(BlackScholes.Inputs, _BS_DIMENSIONS, seed=43)
    points = sampler.sample(_SOBOL_N)

    cold_sampling_results: List[SamplingResults] = [
        _mc_sample(enging=engine, input=p, n=_REPS_COLD) for p in points
    ]

    hq_sampling_results: List[SamplingResults] = [
        _mc_sample(engine=engine, input=r.input, n=_REPS_HQ)
        for r in cold_sampling_results
        if r.p_value < _P_THRESHOLD
    ]

    sorted_failures: List[SamplingResults] = sorted(
        (r for r in hq_sampling_results if r.p_value < _P_THRESHOLD),
        key=attrgetter("p_value"),
        reverse=True,
    )

    if failures:
        # construct pandas df showing results
        input = pd.DataFrame([f.input.dump_model() for f in sorted_failures])
        analytic_price = pd.DataFrame(
            [f.analytic_price.dump_model() for f in sorted_failures], index=input
        )
        p_value = pd.DataFrame(
            [f.p_value.dump_model() for f in sorted_failures], index=input
        )
        df = pd.concat(
            {"analytic_price": analytic_price, "p_value": p_value}, axis="columns"
        )

        artefact = _ARTEFACT_DIR / f"bs_mc_failure_{precision}.parquet"
        df.assign(dtype=precision).to_parquet(artefact)
        pytest.fail("; ".join([*failures, f"artefact → {artefact}"]))
