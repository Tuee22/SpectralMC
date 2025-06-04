from __future__ import annotations

# Re‑export the typed sub‑module so ``from numba import cuda`` sees all
# the attributes provided in ``typings/numba/cuda/__init__.pyi``.
from . import cuda as cuda

__all__ = ["cuda"]
