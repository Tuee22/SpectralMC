# src/spectralmc/models/numerical.py
from __future__ import annotations

"""
`Precision` simply enumerates the high-precision floating-point formats we
care about.  It contains **no direct dependency on PyTorch**.  Conversions
to / from `torch.dtype` live in `models.torch`.
"""

from enum import Enum

__all__ = ["Precision"]


class Precision(str, Enum):
    """High-precision numeric dtypes (float32 / float64)."""

    float32 = "float32"
    float64 = "float64"
