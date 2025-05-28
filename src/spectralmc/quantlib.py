"""QuantLib analytic Black pricing helpers (mypy-strict clean)."""

from __future__ import annotations

import math

import QuantLib as ql  # runtime dependency; we supply stubs for typing

from spectralmc.gbm import BlackScholes

# Public re-export for type annotations
Inputs = BlackScholes.Inputs
HostPriceResults = BlackScholes.HostPricingResults

__all__ = ["bs_price_quantlib"]


def bs_price_quantlib(inp: Inputs) -> HostPriceResults:  # noqa: D401
    """Return analytic Black price using QuantLib."""
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
