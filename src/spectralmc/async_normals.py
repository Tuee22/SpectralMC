# src/spectralmc/async_normals.py
"""
Asynchronous generation of GPU‑resident standard‑normal matrices
with latency hiding and deterministic checkpointing.

Key features
------------
* **Precision‑safe typing** – all dtype handling goes through
  :class:`spectralmc.models.numerical.Precision`.
* **Private CUDA streams** per worker to overlap host/device work.
* **Deterministic snapshots** – the generator can be paused and later
  restored (even with a different buffer size) and will continue the
  random sequence exactly.
* **Zero ignores** under *mypy --strict*.

Public API
----------
* :class:`ConcurrentNormGeneratorConfig` – immutable snapshot model.
* :class:`ConcurrentNormGenerator`   – high‑level generator pool.

The companion test‑suite lives in *tests/test_async_normals.py*.
"""

from __future__ import annotations

from itertools import cycle
from time import time
from typing import List, Optional

import cupy as cp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from spectralmc.models.numerical import Precision

__all__: list[str] = [
    "ConcurrentNormGeneratorConfig",
    "ConcurrentNormGenerator",
]

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

_SEED_LIMIT: int = 1_000_000_000  # exclusive upper bound for CuPy seeds

# --------------------------------------------------------------------------- #
# Pydantic configuration model                                                #
# --------------------------------------------------------------------------- #


class ConcurrentNormGeneratorConfig(BaseModel):
    """
    Immutable checkpoint capturing the *global* RNG state.

    Attributes
    ----------
    rows
        Matrix height (> 0).
    cols
        Matrix width (> 0).
    seed
        Base seed for the *global* NumPy `Generator` (> 0).
    dtype
        Desired numeric precision (``Precision.float32`` or
        ``Precision.float64``).
    skips
        How many matrices have already been delivered (≥ 0).
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
    """
    Generate a stream of standard‑normal matrices on a **private** CUDA stream.

    Notes
    -----
    *Construction* is cheap; the expensive work happens in :meth:`enqueue`.
    * All kernels are launched on the dedicated stream, allowing the host to
      enqueue work for *other* streams while the GPU is busy.
    * Synchronisation latency is tracked for diagnostics.
    """

    def __init__(self, rows: int, cols: int, *, precision: Precision) -> None:
        if min(rows, cols) <= 0:
            raise ValueError("`rows` and `cols` must both be positive.")
        self._rows: int = rows
        self._cols: int = cols
        self._dtype: cp.dtype = precision.to_cupy()
        self._stream: cp.cuda.Stream = cp.cuda.Stream()
        self._generated: Optional[cp.ndarray] = None
        self._event: Optional[cp.cuda.Event] = None
        self._sync_time: float = 0.0

    # ---------------- asynchronous pipeline --------------------------- #

    def enqueue(self, seed: int) -> None:
        """Launch a kernel that fills the *next* matrix (non‑blocking)."""
        if self._generated is not None:
            raise RuntimeError("previous matrix not yet consumed")
        if seed <= 0:
            raise ValueError("`seed` must be positive")

        self._event = cp.cuda.Event(disable_timing=True)
        with self._stream:
            rng = cp.random.default_rng(seed)
            self._generated = rng.standard_normal(
                (self._rows, self._cols), dtype=self._dtype
            )
            self._event.record()

    def get_matrix(self, next_seed: int) -> cp.ndarray:
        """
        Synchronise, return the ready matrix, then queue the next one.

        Parameters
        ----------
        next_seed
            Seed for the *subsequent* matrix.
        """
        if self._generated is None:  # pragma: no cover
            raise RuntimeError("no matrix available – call `enqueue()` first")

        t0 = time()
        self._stream.synchronize()
        self._sync_time += time() - t0

        ready, self._generated = self._generated, None
        self.enqueue(next_seed)
        return ready

    # ---------------- diagnostics ------------------------------------- #

    def get_time_spent_synchronizing(self) -> float:  # noqa: D401
        """Total host‑side time (seconds) spent waiting on this stream."""
        return self._sync_time

    def is_ready(self) -> bool:  # noqa: D401
        """Whether the currently enqueued matrix has finished."""
        return (
            self._event is not None
            and int(cp.cuda.runtime.eventQuery(self._event.ptr)) == 0
        )

    @property
    def dtype(self) -> cp.dtype:  # noqa: D401
        """CuPy dtype of generated matrices."""
        return self._dtype


# --------------------------------------------------------------------------- #
# Public concurrent generator                                                 #
# --------------------------------------------------------------------------- #


class ConcurrentNormGenerator:
    """
    Pool of :class:`_NormGenerator` objects that hides kernel latency.

    Matrices are fetched via :meth:`get_matrix`; the pool cycles through
    its workers so at least one matrix is almost always *ready*.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        buffer_size: int,
        config: ConcurrentNormGeneratorConfig,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("`buffer_size` must be positive")

        self._rows: int = config.rows
        self._cols: int = config.cols
        self._precision: Precision = config.dtype
        self._dtype: cp.dtype = self._precision.to_cupy()
        self._base_seed: int = config.seed
        self._served: int = config.skips

        # NumPy RNG advanced to the current position
        self._np_rng = np.random.default_rng(self._base_seed)
        if self._served:
            self._np_rng.integers(0, _SEED_LIMIT, size=self._served)

        # Build and prime the pool
        def _make() -> _NormGenerator:
            seed = int(self._np_rng.integers(0, _SEED_LIMIT))
            gen = _NormGenerator(self._rows, self._cols, precision=self._precision)
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
        """
        Track how long *all* workers have simultaneously been ready.

        The accumulator only increases while the entire pool is idle.
        """
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
        """Return the next matrix, queuing another in its place."""
        gen = next(self._it)
        next_seed = int(self._np_rng.integers(0, _SEED_LIMIT))
        mat = gen.get_matrix(next_seed)
        self._served += 1
        self._update_idle_state()
        return mat

    def snapshot(self) -> ConcurrentNormGeneratorConfig:
        """Return a *deterministic* checkpoint of the current global state."""
        return ConcurrentNormGeneratorConfig(
            rows=self._rows,
            cols=self._cols,
            seed=self._base_seed,
            dtype=self._precision,
            skips=self._served,
        )

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def get_time_spent_synchronizing(self) -> float:
        """Aggregate host‑side synchronisation time across the entire pool."""
        return sum(gen.get_time_spent_synchronizing() for gen in self._pool)

    def get_idle_time(self) -> float:
        """Total time (seconds) during which *every* worker was ready."""
        self._update_idle_state()
        return (
            self._idle_accum + (time() - self._idle_start)
            if self._idle_start is not None
            else self._idle_accum
        )

    @property
    def dtype(self) -> cp.dtype:  # noqa: D401
        """CuPy dtype produced by the generator."""
        return self._dtype
