"""
spectralmc.async_normals
========================
Asynchronous generation of standard-normal matrices on the GPU
with latency hiding and deterministic checkpointing.

"""

from __future__ import annotations

from itertools import cycle
from time import time
from typing import List, Literal, Optional, Union

import cupy as cp  # needs the stubs in typings/cupy
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ConcurrentNormGeneratorConfig", "ConcurrentNormGenerator"]

# --------------------------------------------------------------------------- #
# Type aliases                                                                #
# --------------------------------------------------------------------------- #

# FloatStr = Literal["float32", "float64"]
# FloatScalar = Union[np.float32, np.float64]
# _InputDType = Union[FloatStr, FloatScalar]

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

_SEED_LIMIT: int = 1_000_000_000  # exclusive upper bound for CuPy seeds


# --------------------------------------------------------------------------- #
# Configuration model                                                         #
# --------------------------------------------------------------------------- #


class ConcurrentNormGeneratorConfig(BaseModel):
    """Frozen checkpoint capturing the *global* random state."""

    rows: int = Field(..., gt=0, description="Matrix height (>0)")
    cols: int = Field(..., gt=0, description="Matrix width (>0)")
    seed: int = Field(..., gt=0, description="Base seed for NumPy RNG (>0)")
    dtype: FloatStr = Field(..., description="Either 'float32' or 'float64'")
    skips: int = Field(0, ge=0, description="Matrices already delivered (>=0)")

    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _normalize_dtype(dtype: _InputDType) -> cp.dtype:
    """Validate and convert *dtype* to ``cp.dtype`` (float32 or float64)."""
    cupy_dtype = cp.dtype(dtype)
    if cupy_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be either 'float32' or 'float64'")
    return cupy_dtype


# --------------------------------------------------------------------------- #
# Internal generator (not exported)                                           #
# --------------------------------------------------------------------------- #


class _NormGenerator:
    """Generate a stream of standard-normal matrices on a private CUDA stream."""

    def __init__(self, rows: int, cols: int, *, dtype: _InputDType) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        self._rows: int = rows
        self._cols: int = cols
        self._dtype: cp.dtype = _normalize_dtype(dtype)

        self._stream: cp.cuda.Stream = cp.cuda.Stream()
        self._generated: Optional[cp.ndarray] = None
        self._event: Optional[cp.cuda.Event] = None
        self._sync_time: float = 0.0

    # ---------------- asynchronous pipeline --------------------------- #

    def enqueue(self, seed: int) -> None:
        """Launch the kernel that fills the *next* matrix (non-blocking)."""
        if self._generated is not None:
            raise RuntimeError("Previous matrix not yet consumed")
        if seed <= 0:
            raise ValueError("seed must be positive")

        self._event = cp.cuda.Event(disable_timing=True)
        with self._stream:
            rng = cp.random.default_rng(seed)
            self._generated = rng.standard_normal(
                (self._rows, self._cols), dtype=self._dtype
            )
            self._event.record()

    def get_matrix(self, next_seed: int) -> cp.ndarray:
        """Synchronise, return ready matrix, queue another with *next_seed*."""
        if self._generated is None:
            raise RuntimeError("No matrix available; call enqueue() first")

        t0 = time()
        self._stream.synchronize()
        self._sync_time += time() - t0

        ready = self._generated
        self._generated = None
        self.enqueue(next_seed)
        return ready

    # ---------------- diagnostics ------------------------------------- #

    def get_time_spent_synchronizing(self) -> float:
        return self._sync_time

    def is_ready(self) -> bool:  # noqa: D401
        if self._event is None:
            return False
        status = int(cp.cuda.runtime.eventQuery(self._event.ptr))
        return status == 0  # 0 == cudaSuccess

    @property
    def dtype(self) -> cp.dtype:  # noqa: D401
        return self._dtype


# --------------------------------------------------------------------------- #
# Public concurrent generator                                                 #
# --------------------------------------------------------------------------- #


class ConcurrentNormGenerator:
    """Latency-hiding pool of :class:`_NormGenerator` objects."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, buffer_size: int, config: ConcurrentNormGeneratorConfig):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        self._rows = config.rows
        self._cols = config.cols
        self._dtype = _normalize_dtype(config.dtype)
        self._base_seed = config.seed
        self._served = config.skips

        # Master NumPy RNG advanced to current position
        self._np_rng = np.random.default_rng(self._base_seed)
        if self._served:
            self._np_rng.integers(0, _SEED_LIMIT, size=self._served)

        # Helper to build & prime a stream-bound generator
        def _make() -> _NormGenerator:
            dtype_str: FloatStr = "float32" if self._dtype == cp.float32 else "float64"
            seed = int(self._np_rng.integers(0, _SEED_LIMIT))
            gen = _NormGenerator(self._rows, self._cols, dtype=dtype_str)
            gen.enqueue(seed)
            return gen

        self._pool: List[_NormGenerator] = [_make() for _ in range(buffer_size)]
        self._it = cycle(self._pool)

        self._idle_accum: float = 0.0
        self._idle_start: Optional[float] = None
        self._update_idle_state()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _update_idle_state(self) -> None:
        all_ready = all(gen.is_ready() for gen in self._pool)
        now = time()
        if all_ready:
            if self._idle_start is None:
                self._idle_start = now
        else:
            if self._idle_start is not None:
                self._idle_accum += now - self._idle_start
                self._idle_start = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_matrix(self) -> cp.ndarray:
        gen = next(self._it)
        next_seed = int(self._np_rng.integers(0, _SEED_LIMIT))
        mat = gen.get_matrix(next_seed)
        self._served += 1
        self._update_idle_state()
        return mat

    def snapshot(self) -> ConcurrentNormGeneratorConfig:
        return ConcurrentNormGeneratorConfig(
            rows=self._rows,
            cols=self._cols,
            seed=self._base_seed,
            dtype="float32" if self._dtype == cp.float32 else "float64",
            skips=self._served,
        )

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def get_time_spent_synchronizing(self) -> float:
        return sum(gen.get_time_spent_synchronizing() for gen in self._pool)

    def get_idle_time(self) -> float:
        self._update_idle_state()
        if self._idle_start is not None:
            return self._idle_accum + (time() - self._idle_start)
        return self._idle_accum

    @property
    def dtype(self) -> cp.dtype:  # noqa: D401
        return self._dtype
