"""Stubbed subset of QuantLib used by spectralmc.test modules."""

from __future__ import annotations

class Option:
    Put: int
    Call: int

# Black formula; returns a price in same currency units as inputs

def blackFormula(
    optionType: int, strike: float, forward: float, stdDev: float, discount: float
) -> float: ...
