"""tests/test_bs_mc.py – GPU MC vs. QuantLib Black‐Scholes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple, TypeAlias, cast

import numpy as np
import pandas as pd
import pytest
from typing_extensions import Literal

import cupy as cp  # type: ignore[import-untyped]
import QuantLib as ql  # type: ignore[import-untyped]

from spectralmc.gbm import BlackScholes, SimulationParams
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
_REPS_COLD: Final[int] = 3
_REPS_HQ: Final[int] = 7

_K_SIGMA: Final[float] = 3.0
_RMSE_TOL: Final[float] = 1.0e-2

_ARTEFACT_DIR: Final[Path] = Path(__file__).with_suffix("").parent / ".failed_artifacts"
_ARTEFACT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────── typing help ────────────────────────────────
StatT: TypeAlias = Tuple[float, float]
PriceT: TypeAlias = Tuple[StatT, StatT]


# ───────────────────────────── analytic Black helper ────────────────────────────
def bs_price_quantlib(inp: BlackScholes.Inputs) -> BlackScholes.HostPricingResults:
    """Closed-form Black price via QuantLib.blackFormula."""
    std = inp.v * math.sqrt(inp.T)
    disc = math.exp(-inp.r * inp.T)
    fwd = inp.X0 * math.exp((inp.r - inp.d) * inp.T)

    call = ql.blackFormula(ql.Option.Call, inp.K, fwd, std, disc)
    put = ql.blackFormula(ql.Option.Put, inp.K, fwd, std, disc)

    cint = disc * max(fwd - inp.K, 0.0)
    pint = disc * max(inp.K - fwd, 0.0)

    return BlackScholes.HostPricingResults(
        call_price_intrinsic=cint,
        put_price_intrinsic=pint,
        underlying=fwd,
        call_convexity=call - cint,
        put_convexity=put - pint,
        call_price=call,
        put_price=put,
    )


# ───────────────────────────── Monte-Carlo helpers ─────────────────────────────
def _mc_stats(engine: BlackScholes, inp: BlackScholes.Inputs, reps: int) -> PriceT:
    sample = np.array(
        [
            (host.call_price, host.put_price)
            for host in (engine.price_to_host(inp) for _ in range(reps))
        ],
        dtype=float,
    )
    means = sample.mean(0)
    stderrs = np.maximum(
        sample.std(0, ddof=1) / math.sqrt(reps),
        1e-8 * np.maximum(1.0, np.abs(means)),
    )
    return (
        (float(means[0]), float(stderrs[0])),
        (float(means[1]), float(stderrs[1])),
    )


# ───────────────────────────── Data-frame pipeline ─────────────────────────────
def _cold_pass(points: List[BlackScholes.Inputs], eng: BlackScholes) -> pd.DataFrame:
    def _row(idx_pt: Tuple[int, BlackScholes.Inputs]) -> Dict[str, float]:
        idx, p = idx_pt
        (mc_c, se_c), (mc_p, se_p) = _mc_stats(eng, p, _REPS_COLD)
        ana = bs_price_quantlib(p)
        return {
            "idx": idx,
            **p.model_dump(),
            "mc_call_mean": mc_c,
            "mc_call_se": se_c,
            "mc_put_mean": mc_p,
            "mc_put_se": se_p,
            "ana_call": ana.call_price,
            "ana_put": ana.put_price,
        }

    df = pd.DataFrame([_row(t) for t in enumerate(points)]).set_index("idx")
    df["abs_err_call"] = (df["mc_call_mean"] - df["ana_call"]).abs()
    df["abs_err_put"] = (df["mc_put_mean"] - df["ana_put"]).abs()
    df["hq_needed"] = (df["abs_err_call"] > _K_SIGMA * df["mc_call_se"]) | (
        df["abs_err_put"] > _K_SIGMA * df["mc_put_se"]
    )
    df["hq_failed"] = False
    return df


def _hq_upgrade(
    df: pd.DataFrame,
    pts: List[BlackScholes.Inputs],
    eng: BlackScholes,
) -> pd.DataFrame:
    def _upgrade(
        row: pd.Series[Any],
    ) -> pd.Series[Any]:  # 👈 Series[Any] => no mypy error
        if not bool(row["hq_needed"]):
            return row
        idx = cast(int, row.name)
        (mc_c, se_c), (mc_p, se_p) = _mc_stats(eng, pts[idx], _REPS_HQ)
        ana_c, ana_p = float(row["ana_call"]), float(row["ana_put"])
        within = (
            abs(mc_c - ana_c) <= _K_SIGMA * se_c
            and abs(mc_p - ana_p) <= _K_SIGMA * se_p
        )
        upd = row.copy()
        upd.loc[
            [
                "mc_call_mean",
                "mc_call_se",
                "mc_put_mean",
                "mc_put_se",
                "abs_err_call",
                "abs_err_put",
                "hq_needed",
                "hq_failed",
            ]
        ] = [
            mc_c,
            se_c,
            mc_p,
            se_p,
            abs(mc_c - ana_c),
            abs(mc_p - ana_p),
            False,
            not within,
        ]
        return upd

    return cast(pd.DataFrame, df.apply(_upgrade, axis=1))


# ──────────────────────────────── PyTest entry ────────────────────────────────
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_black_scholes_mc(precision: str) -> None:
    """GPU Monte-Carlo pricer must match QuantLib within 3 σ & 1 % RMSE."""
    sp = SimulationParams(
        timesteps=_TIMESTEPS,
        network_size=_NETWORK_SIZE,
        batches_per_mc_run=_BATCHES_PER_RUN,
        threads_per_block=_THREADS_PER_BLOCK,
        mc_seed=_MC_SEED,
        buffer_size=_BUFFER_SIZE,
        dtype=cast(Literal["float32", "float64"], precision),
    )
    engine = BlackScholes(sp)
    sampler = SobolSampler(BlackScholes.Inputs, _BS_DIMENSIONS, seed=43)
    points = sampler.sample(_SOBOL_N)

    df = _hq_upgrade(_cold_pass(points, engine), points, engine)

    failures: List[str] = []
    if df["hq_failed"].any():
        failures.append(f"{df['hq_failed'].sum()} >3σ post-HQ")

    rel_call = (df["mc_call_mean"] - df["ana_call"]).abs() / df["ana_call"].abs()
    rel_put = (df["mc_put_mean"] - df["ana_put"]).abs() / df["ana_put"].abs()
    rms_c, rms_p = np.sqrt((rel_call**2).mean()), np.sqrt((rel_put**2).mean())
    if rms_c >= _RMSE_TOL:
        failures.append(f"Call RMSE {rms_c:.3%} ≥ 1 %")
    if rms_p >= _RMSE_TOL:
        failures.append(f"Put RMSE {rms_p:.3%} ≥ 1 %")

    if failures:
        artefact = _ARTEFACT_DIR / f"bs_mc_failure_{precision}.parquet"
        df.assign(dtype=precision).to_parquet(artefact)
        pytest.fail("; ".join([*failures, f"artefact → {artefact}"]))

    print(df.head())
