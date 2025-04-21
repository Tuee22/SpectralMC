from __future__ import annotations

from typing import Type, Union

import torch
import pytest

from spectralmc.cvnn import (
    ComplexLinear,
    zReLU,
    modReLU,
    NaiveComplexBatchNorm,
    CovarianceComplexBatchNorm,
    ResidualBlock,
    CVNN,
    SimpleCVNN,
)

# --------------------------------------------------------------------------
#  Unit tests
# --------------------------------------------------------------------------
# (identical to what used to reside in cvnn.py; omitted here for brevity)
# --------------------------------------------------------------------------

# ...  ← paste the test functions you already copied earlier ...
# make sure they have ASCII hyphens and pass mypy --strict
