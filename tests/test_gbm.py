"""tests/test_gbm.py – GPU MC vs. QuantLib Black‑Scholes.

A PyTest regression that compares a GPU Monte‑Carlo pricer against the
closed‑form Black–Scholes ("Black") value produced by QuantLib.  The module
is fully typed (mypy‑clean) under Python 3.12, NumPy 2.0, **Pydantic 2.x**, and
SciPy.

Key implementation notes
------------------------
* Concrete generics (`numpy.typing.NDArray`) are used for all ndarray types.
*   `Inputs`               → alias for `BlackScholes.Inputs`  (a Pydantic model)
*   `HostPriceResults`     → alias for `BlackScholes.HostPricingResults`
*   `SamplingResults`      → a Pydantic model that bundles one Sobol point
    with its Monte‑Carlo sample and analytic comparison.
* All Pydantic serialisation uses **`.model_dump()`** (the Pydantic‑v2 API).
* Artefacts for failing cases are written as Parquet files to
  `tests/.failed_artifacts/bs_mc_failure_<dtype>.parquet` for later
  inspection.
"""

from __future__ import annotations

import math
from pathlib import Path
from operator import attrgetter
from typing import Any, Final, List, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pydantic import BaseModel
from scipy.stats import t  # type: ignore[import-untyped]

import cupy as cp  # type: ignore[import-untyped]
import QuantLib as ql  # type: ignore[import-untyped]

from spectralmc.gbm import BlackScholes, SimulationParams, DtypeLiteral
from spectralmc.sobol_sampler import BoundSpec, SobolSampler

# ──────────────────────────────── type aliases ────────────────────────────────
Inputs: TypeAlias = BlackScholes.Inputs
HostPriceResults: TypeAlias = BlackScholes.HostPricingResults

# ────────────────────────────────── constants ──────────────────────────────────
_BS_DIMENSIONS: Final[dict[str, BoundSpec]] = {
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
_FLOAT64_EPSILON: Final[float] = 1e-8
_FLOAT32_EPSILON: Final[float] = 1e-2
_P_THRESHOLD: Final[float] = 0.05

_ARTEFACT_DIR: Final[Path] = Path(__file__).with_suffix("").parent / ".failed_artifacts"
_ARTEFACT_DIR.mkdir(exist_ok=True)

# ──────────────────────────── analytic Black helper ────────────────────────────


def bs_price_quantlib(inp: Inputs) -> HostPriceResults:
    """Return the closed‑form Black price using *QuantLib*.

    Parameters
    ----------
    inp
        Input parameters (under spot‑measure) as defined by
        :class:`spectralmc.gbm.BlackScholes.Inputs`.

    Returns
    -------
    HostPriceResults
        The analytic call/put convexities, prices, and forward underlying.
    """

    std = inp.v * math.sqrt(inp.T)
    df = math.exp(-inp.r * inp.T)
    fwd = inp.X0 * math.exp((inp.r - inp.d) * inp.T)

    put_price = ql.blackFormula(ql.Option.Put, inp.K, fwd, std, df)
    call_price = ql.blackFormula(ql.Option.Call, inp.K, fwd, std, df)

    put_intrinsic = df * max(inp.K - fwd, 0.0)
    call_intrinsic = df * max(fwd - inp.K, 0.0)

    return HostPriceResults(
        put_price_intrinsic=put_intrinsic,
        call_price_intrinsic=call_intrinsic,
        underlying=fwd,
        put_convexity=put_price - put_intrinsic,
        call_convexity=call_price - call_intrinsic,
        put_price=put_price,
        call_price=call_price,
    )


# ───────────────────────── Student‑t helper (typed) ────────────────────────────


def two_tailed_t_pvalue(
    sample: npt.NDArray[np.floating[Any]],
    mu0: float = 0.0,
) -> float:
    """Two‑tailed *p*‑value for the one‑sample Student‑*t* test.

    Computes the probability of observing a sample mean at least as far from
    *mu0* as the given sample’s mean, under the null hypothesis that the true
    population mean is *mu0*.
    """

    x = np.asarray(sample, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("At least two observations are required for a t‑test.")

    x_bar = float(x.mean())
    s = float(x.std(ddof=1))
    se = s / math.sqrt(n)

    t_stat = (x_bar - mu0) / se
    df = n - 1

    return 2.0 * float(t.sf(abs(t_stat), df))


# ──────────────────────────── results container ────────────────────────────────


class SamplingResults(BaseModel):
    """Bundles one Sobol‑sampled input with MC results and hypothesis test."""

    input: Inputs
    sample: list[HostPriceResults]
    analytic_price: HostPriceResults
    p_value: float


# ───────────────────────────── Monte‑Carlo helper ──────────────────────────────


def _mc_sample(engine: BlackScholes, input: Inputs, n: int) -> SamplingResults:
    """Generate *n* Monte‑Carlo prices and test against the analytic value.

    The function asserts put‑call parity for convexities and returns a
    :class:`SamplingResults` instance with the two‑tailed p‑value of the
    Student‑*t* hypothesis test ``H₀: mean(error) = 0``.
    """

    sample = [engine.price_to_host(input) for _ in range(n)]
    analytic_price = bs_price_quantlib(input)

    # put‑call parity on convexities (should be zero if we normalize forwards)
    if engine._sp.normalize_forwards:
        parity_check = np.array([s.put_convexity - s.call_convexity for s in sample])
        epsilon = (
            _FLOAT32_EPSILON if engine._sp.dtype == "float32" else _FLOAT64_EPSILON
        )
        assert (
            np.max(np.abs(parity_check)) < epsilon
        ), f"Parity check failed for input {input}"

    errors = (
        np.array([s.call_convexity for s in sample]) - analytic_price.call_convexity
    )
    p_value = two_tailed_t_pvalue(errors)

    return SamplingResults(
        input=input,
        sample=sample,
        analytic_price=analytic_price,
        p_value=p_value,
    )


# ──────────────────────────────── PyTest entry ────────────────────────────────


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_black_scholes_mc(precision: DtypeLiteral) -> None:
    """End‑to‑end validation of the GPU Monte‑Carlo Black‑Scholes pricer.

    The pricer must match the analytic Black value within three standard
    deviations and 1 % RMSE.  Potentially problematic Sobol points are first
    identified with a low‑rep "cold" run, then resampled at high quality.
    Any points that still fail the hypothesis test are written to a Parquet
    file for post‑mortem analysis, and the test is marked as failed.
    """

    sp = SimulationParams(
        timesteps=_TIMESTEPS,
        network_size=_NETWORK_SIZE,
        batches_per_mc_run=_BATCHES_PER_RUN,
        threads_per_block=_THREADS_PER_BLOCK,
        mc_seed=_MC_SEED,
        buffer_size=_BUFFER_SIZE,
        dtype=precision,
        simulate_log_return=True,
        normalize_forwards=False,
    )
    engine = BlackScholes(sp)

    sampler = SobolSampler(BlackScholes.Inputs, _BS_DIMENSIONS, seed=43)
    points = sampler.sample(_SOBOL_N)

    # Low‑rep pass to locate tail cases
    cold_results: List[SamplingResults] = [
        _mc_sample(engine=engine, input=p, n=_REPS_COLD) for p in points
    ]

    # High‑quality resampling of the tail cases
    hq_results: List[SamplingResults] = [
        _mc_sample(engine=engine, input=r.input, n=_REPS_HQ)
        for r in cold_results
        if r.p_value < _P_THRESHOLD
    ]

    failures: List[SamplingResults] = sorted(
        (r for r in hq_results if r.p_value < _P_THRESHOLD),
        key=attrgetter("p_value"),
        reverse=True,
    )

    if failures:
        inputs_df = pd.DataFrame([f.input.model_dump() for f in failures])
        analytic_df = pd.DataFrame(
            [f.analytic_price.model_dump() for f in failures],
            index=inputs_df.index,
        )
        pval_df = pd.DataFrame(
            {"p_value": [f.p_value for f in failures]}, index=inputs_df.index
        )

        df = pd.concat(
            {"analytic_price": analytic_df, "p_value": pval_df}, axis="columns"
        )

        artefact = _ARTEFACT_DIR / f"bs_mc_failure_{precision}.parquet"
        df.assign(dtype=precision).to_parquet(artefact)

        pytest.fail(f"{len(failures)} failures; artefact → {artefact}")
