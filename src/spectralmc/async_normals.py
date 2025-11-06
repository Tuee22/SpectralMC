# src/spectralmc/async_normals.py
"""
spectralmc.async_normals
========================
Latency‑hiding generation of standard‑normal matrices that live on the
GPU, with deterministic checkpoint/restore and *zero* static‑typing
compromises.

Design overview
---------------
1. **Snapshot fidelity** – you can pause, serialise the state, and later
   resume (even with a different buffer size) without breaking the random
   sequence.
2. **Latency hiding** – a configurable pool of CUDA streams means the host
   seldom blocks waiting for kernels to finish.
3. **Strict typing** – the module passes `mypy --strict` without ignores,
   casts, or Any.  Precision is serialised via
   `spectralmc.models.numerical.Precision`; internally we only keep the
   raw `cp.dtype`.

Typical usage
-------------
Example::

    from spectralmc.async_normals import (
        BufferConfig,
        ConcurrentNormGenerator,
        ConcurrentNormGeneratorConfig,
    )
    from spectralmc.models.numerical import Precision

    cfg = ConcurrentNormGeneratorConfig(
        rows=1024,
        cols=512,
        seed=123,
        dtype=Precision.float32,
    )
    buffer = BufferConfig.create(size=4, matrix_rows=cfg.rows, matrix_cols=cfg.cols)
    gen = ConcurrentNormGenerator(buffer=buffer, config=cfg)

    gpu_matrix = gen.get_matrix()   # cupy.ndarray on the device
    checkpoint = gen.snapshot()     # serialisable pydantic model
"""

from __future__ import annotations

from itertools import cycle
from time import time
from typing import Annotated, List, Optional

import cupy as cp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from spectralmc.models.numerical import Precision

__all__: list[str] = [
    "BufferConfig",
    "ConcurrentNormGeneratorConfig",
    "ConcurrentNormGenerator",
]

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

_SEED_LIMIT: int = 1_000_000_000  # exclusive upper bound for CuPy RNG seeds

# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #


def _validate_cupy_dtype(dtype: cp.dtype) -> cp.dtype:
    """Return *dtype* if it is float32 or float64, otherwise raise."""
    if dtype not in (cp.dtype(cp.float32), cp.dtype(cp.float64)):
        raise ValueError("dtype must be cp.float32 or cp.float64")
    return dtype


# --------------------------------------------------------------------------- #
# Pydantic configuration models                                               #
# --------------------------------------------------------------------------- #


class BufferConfig(BaseModel):
    """Buffer size configuration with validation against matrix dimensions.

    Attributes
    ----------
    size
        Number of concurrent workers (must be positive and ≤ matrix elements).

    Notes
    -----
    Use the `create` class method to construct with cross-field validation.
    """

    size: Annotated[int, Field(gt=0, description="Number of concurrent worker streams")]

    model_config = ConfigDict(frozen=True)

    @classmethod
    def create(cls, size: int, matrix_rows: int, matrix_cols: int) -> "BufferConfig":
        """Construct BufferConfig with validation against matrix dimensions.

        Parameters
        ----------
        size
            Desired buffer size (number of concurrent workers).
        matrix_rows
            Matrix height.
        matrix_cols
            Matrix width.

        Returns
        -------
        BufferConfig
            Validated configuration instance.

        Raises
        ------
        ValueError
            If size exceeds total matrix elements (wastes memory).
        """
        max_size = matrix_rows * matrix_cols
        if size > max_size:
            raise ValueError(
                f"buffer_size ({size}) exceeds matrix elements ({max_size}). "
                f"Cannot allocate more buffers than matrix elements."
            )
        return cls(size=size)


class ConcurrentNormGeneratorConfig(BaseModel):
    """Immutable serialisable snapshot of the global RNG state.

    Attributes
    ----------
    rows
        Matrix height (> 0).
    cols
        Matrix width (> 0).
    seed
        Base seed for the global NumPy RNG (> 0).  Each worker draws its own
        CuPy seed from this generator.
    dtype
        Requested numeric precision as `Precision.float32` or
        `Precision.float64`.
    skips
        How many matrices have already been produced (≥ 0).  Used to advance
        the NumPy RNG when restoring from a checkpoint.
    """

    rows: int = Field(..., gt=0)
    cols: int = Field(..., gt=0)
    seed: int = Field(..., gt=0)
    dtype: Precision
    skips: int = Field(0, ge=0)

    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- #
# Internal single‑stream generator                                            #
# --------------------------------------------------------------------------- #


class _NormGenerator:
    """Generate standard‑normal matrices on a dedicated CUDA stream."""

    def __init__(self, rows: int, cols: int, *, dtype: cp.dtype) -> None:
        """Validate shapes, store dtype, and allocate the CUDA stream."""
        if min(rows, cols) <= 0:
            raise ValueError("rows and cols must both be positive")
        self._rows: int = rows
        self._cols: int = cols
        self._dtype: cp.dtype = _validate_cupy_dtype(dtype)

        self._stream: cp.cuda.Stream = cp.cuda.Stream()
        self._generated: Optional[cp.ndarray] = None
        self._event: Optional[cp.cuda.Event] = None
        self._sync_time: float = 0.0

    # ---------------- asynchronous pipeline --------------------------- #

    def enqueue(self, seed: int) -> None:
        """Launch a kernel that fills the next matrix (non‑blocking)."""
        if self._generated is not None:
            raise RuntimeError("previous matrix not yet consumed")
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
        """Synchronise, return the ready matrix, then queue another."""
        if self._generated is None:  # pragma: no cover
            raise RuntimeError("no matrix enqueued – call enqueue() first")

        t0 = time()
        self._stream.synchronize()
        self._sync_time += time() - t0

        ready, self._generated = self._generated, None
        self.enqueue(next_seed)
        return ready

    # ---------------- diagnostics ------------------------------------- #

    def get_time_spent_synchronizing(self) -> float:
        """Total host‑side synchronisation latency (seconds)."""
        return self._sync_time

    def is_ready(self) -> bool:
        """True iff the current matrix has finished on the GPU."""
        return (
            self._event is not None
            and int(cp.cuda.runtime.eventQuery(self._event.ptr)) == 0
        )

    # ---------------- read‑only props --------------------------------- #

    @property
    def dtype(self) -> cp.dtype:
        """CuPy dtype produced by this worker."""
        return self._dtype


# --------------------------------------------------------------------------- #
# Public concurrent generator                                                 #
# --------------------------------------------------------------------------- #


class ConcurrentNormGenerator:
    """Latency‑hiding pool of `_NormGenerator` workers."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self, buffer: BufferConfig, config: ConcurrentNormGeneratorConfig
    ) -> None:

        # Static parameters
        self._rows: int = config.rows
        self._cols: int = config.cols
        self._dtype: cp.dtype = config.dtype.to_cupy()

        # RNG state
        self._base_seed: int = config.seed
        self._served: int = config.skips
        self._np_rng = np.random.default_rng(self._base_seed)
        if self._served:
            self._np_rng.integers(0, _SEED_LIMIT, size=self._served)  # fast‑forward

        # Build worker pool
        def _make() -> _NormGenerator:
            seed = int(self._np_rng.integers(0, _SEED_LIMIT))
            gen = _NormGenerator(self._rows, self._cols, dtype=self._dtype)
            gen.enqueue(seed)
            return gen

        self._pool: List[_NormGenerator] = [_make() for _ in range(buffer.size)]
        self._it = cycle(self._pool)

        # Idle‑time diagnostics
        self._idle_accum: float = 0.0
        self._idle_start: Optional[float] = None
        self._update_idle_state()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _update_idle_state(self) -> None:
        """Accumulate time spent with *all* workers simultaneously ready."""
        all_ready = all(gen.is_ready() for gen in self._pool)
        now = time()
        if all_ready and self._idle_start is None:
            self._idle_start = now
        elif not all_ready and self._idle_start is not None:
            self._idle_accum += now - self._idle_start
            self._idle_start = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def get_matrix(self) -> cp.ndarray:
        """Return the next matrix; enqueue another in its place."""
        gen = next(self._it)
        next_seed = int(self._np_rng.integers(0, _SEED_LIMIT))
        mat = gen.get_matrix(next_seed)
        self._served += 1
        self._update_idle_state()
        return mat

    def snapshot(self) -> ConcurrentNormGeneratorConfig:
        """Produce a deterministic checkpoint of the current state."""
        return ConcurrentNormGeneratorConfig(
            rows=self._rows,
            cols=self._cols,
            seed=self._base_seed,
            dtype=Precision.from_cupy(self._dtype),
            skips=self._served,
        )

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def get_time_spent_synchronizing(self) -> float:
        """Aggregate host‑side synchronisation latency (seconds)."""
        return sum(gen.get_time_spent_synchronizing() for gen in self._pool)

    def get_idle_time(self) -> float:
        """Total time (seconds) during which *every* worker was ready."""
        self._update_idle_state()
        return (
            self._idle_accum + (time() - self._idle_start)
            if self._idle_start is not None
            else self._idle_accum
        )

    # ---------------- read‑only props --------------------------------- #

    @property
    def dtype(self) -> cp.dtype:
        """CuPy dtype produced by the generator."""
        return self._dtype
