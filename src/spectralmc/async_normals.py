# src/spectralmc/async_normals.py
"""
Asynchronous generation of GPU‑resident standard‑normal matrices
with latency hiding and deterministic checkpointing.

Key points
----------
* **Serialization = `Precision`**, internal state = *raw* ``cp.dtype``.
  The public checkpoint model stores :class:`~spectralmc.models.numerical.Precision`;
  once deserialized, we immediately convert to the corresponding CuPy dtype
  and keep *only that* in memory.  Snapshots convert back via
  :meth:`Precision.from_cupy`.
* **Private CUDA streams** per worker so host computation can overlap
  GPU kernels.
* Passes **mypy --strict** without ignores or `Any`.

Public API
----------
* :class:`ConcurrentNormGeneratorConfig`
* :class:`ConcurrentNormGenerator`
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
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _validate_cupy_dtype(dtype: cp.dtype) -> cp.dtype:
    """
    Ensure *dtype* is either ``cp.float32`` or ``cp.float64``.

    Returns
    -------
    cp.dtype
        The same dtype (identity) if valid.

    Raises
    ------
    ValueError
        If *dtype* is not one of the supported formats.
    """
    if dtype not in (cp.dtype(cp.float32), cp.dtype(cp.float64)):
        raise ValueError("dtype must be cp.float32 or cp.float64")
    return dtype


# --------------------------------------------------------------------------- #
# Pydantic configuration model                                                #
# --------------------------------------------------------------------------- #


class ConcurrentNormGeneratorConfig(BaseModel):
    """
    Immutable checkpoint capturing the *global* RNG state.

    Attributes
    ----------
    rows, cols
        Matrix dimensions (> 0).
    seed
        Base seed for the *global* NumPy RNG (> 0).
    dtype
        Precision of generated matrices (serialised as
        :class:`~spectralmc.models.numerical.Precision`).
    skips
        Number of matrices already delivered (≥ 0).
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
    Generate a stream of standard‑normal matrices on a dedicated CUDA stream.

    The constructor only validates parameters and allocates the stream;
    the first random kernel is dispatched via :meth:`enqueue`.
    """

    def __init__(self, rows: int, cols: int, *, dtype: cp.dtype) -> None:
        if min(rows, cols) <= 0:
            raise ValueError("`rows` and `cols` must both be positive")
        self._rows: int = rows
        self._cols: int = cols
        self._dtype: cp.dtype = _validate_cupy_dtype(dtype)

        self._stream: cp.cuda.Stream = cp.cuda.Stream()
        self._generated: Optional[cp.ndarray] = None
        self._event: Optional[cp.cuda.Event] = None
        self._sync_time: float = 0.0

    # ---------------- asynchronous pipeline --------------------------- #

    def enqueue(self, seed: int) -> None:
        """Launch the kernel that will fill the *next* matrix (non‑blocking)."""
        if self._generated is not None:
            raise RuntimeError("previous matrix not yet consumed")
        if seed <= 0:
            raise ValueError("`seed` must be positive")

        self._event = cp.cuda.Event(disable_timing=True)
        with self._stream:
            rng = cp.random.default_rng(seed)
            self._generated = rng.standard_normal(
                (self._rows, self._cols),
                dtype=self._dtype,
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
            raise RuntimeError("no matrix enqueued – call `enqueue()` first")

        t0 = time()
        self._stream.synchronize()
        self._sync_time += time() - t0

        ready, self._generated = self._generated, None
        self.enqueue(next_seed)
        return ready

    # ---------------- diagnostics ------------------------------------- #

    def get_time_spent_synchronizing(self) -> float:  # noqa: D401
        """Total host‑side synchronisation time (seconds)."""
        return self._sync_time

    def is_ready(self) -> bool:  # noqa: D401
        """Return ``True`` iff the current matrix has finished on the GPU."""
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
    its workers so that at least one matrix is *usually* ready.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, buffer_size: int, config: ConcurrentNormGeneratorConfig) -> None:
        if buffer_size <= 0:
            raise ValueError("`buffer_size` must be positive")

        self._rows: int = config.rows
        self._cols: int = config.cols
        self._dtype: cp.dtype = config.dtype.to_cupy()
        self._base_seed: int = config.seed
        self._served: int = config.skips

        # Master NumPy RNG advanced to the current position
        self._np_rng = np.random.default_rng(self._base_seed)
        if self._served:
            self._np_rng.integers(0, _SEED_LIMIT, size=self._served)

        # Build and prime the pool
        def _make() -> _NormGenerator:
            seed = int(self._np_rng.integers(0, _SEED_LIMIT))
            gen = _NormGenerator(self._rows, self._cols, dtype=self._dtype)
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
        Track how long *all* workers have been simultaneously ready.

        The accumulator only increases while the entire pool sits idle.
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
        """Return the next matrix, immediately enqueuing a replacement."""
        gen = next(self._it)
        next_seed = int(self._np_rng.integers(0, _SEED_LIMIT))
        mat = gen.get_matrix(next_seed)
        self._served += 1
        self._update_idle_state()
        return mat

    def snapshot(self) -> ConcurrentNormGeneratorConfig:
        """Return a deterministic checkpoint of the current global state."""
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
        """Aggregate host‑side synchronisation time across the pool."""
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
