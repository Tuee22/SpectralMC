"""
Asynchronous normal generation with CuPy for GPU‑based Monte‑Carlo.

"""

from __future__ import annotations

from itertools import cycle
from time import time
from typing import Iterator, List, Literal, Optional, Union

import cupy as cp  # type: ignore[import-untyped]
import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatStr = Literal["float32", "float64"]
FloatScalar = Union[np.float32, np.float64]
_InputDType = Union[FloatStr, FloatScalar]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _normalize_dtype(dtype: _InputDType) -> cp.dtype:
    """Convert *dtype* to ``cp.dtype`` and validate it is float32 or float64."""
    cupy_dtype = cp.dtype(dtype)
    if cupy_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be either 'float32' or 'float64'")
    return cupy_dtype


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------


class NormGenerator:
    """Generate a matrix of standard normals on a private CuPy stream."""

    def __init__(
        self,
        rows: int,
        cols: int,
        seed: int,
        *,
        dtype: _InputDType = "float32",
    ) -> None:
        if rows <= 0 or cols <= 0 or seed <= 0:
            raise ValueError("rows, cols, and seed must be positive")

        self._rows = rows
        self._cols = cols
        self._dtype = _normalize_dtype(dtype)

        self._stream = cp.cuda.Stream()
        self._rng = cp.random.default_rng(seed)
        self._generated: Optional[cp.ndarray] = None
        self._event: Optional[cp.cuda.Event] = None
        self._sync_time = 0.0
        self._enqueue()

    # ---------------- iteration protocol ---------------------------------

    def __iter__(self) -> Iterator[cp.ndarray]:
        while True:
            yield self.get_matrix()

    # ---------------- internals ------------------------------------------

    def _enqueue(self) -> None:
        if self._generated is not None:
            raise RuntimeError("Previous matrix not yet consumed")
        self._event = cp.cuda.Event(disable_timing=True)
        with self._stream:
            self._generated = self._rng.standard_normal(
                (self._rows, self._cols), dtype=self._dtype
            )
            self._event.record()  # event marks end of kernel launch

    # ---------------- public API -----------------------------------------

    def get_matrix(self) -> cp.ndarray:
        if self._generated is None:
            raise RuntimeError("No matrix available")
        t0 = time()
        self._stream.synchronize()
        self._sync_time += time() - t0

        ready = self._generated
        self._generated = None
        self._enqueue()
        return ready

    def get_time_spent_synchronizing(self) -> float:
        return self._sync_time

    # -------------- readiness check (for idle‑time) ----------------------

    def is_ready(self) -> bool:
        """Return *True* iff the CUDA event has completed."""
        if self._event is None:
            return False
        # 0 == cudaSuccess  → event finished
        status: int = cp.cuda.runtime.eventQuery(self._event.ptr)
        return status == 0

    # --------------------------------------------------------------------

    @property
    def dtype(self) -> cp.dtype:  # noqa: D401
        """Precision of the generated normals."""
        return self._dtype


class ConcurrentNormGenerator:
    """Round‑robin pool of *NormGenerator* objects to hide latency.

    Added functionality: :meth:`get_idle_time` reports the cumulative seconds
    during which *all* generators were simultaneously finished (idle).
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        seed: int,
        buffer_size: int,
        *,
        skip_steps: int = 0,
        dtype: _InputDType = "float32",
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        np_rng = np.random.default_rng(seed)
        for _ in range(skip_steps):
            _ = np_rng.integers(low=0, high=10**9)

        norm_dtype = _normalize_dtype(dtype)

        def _make() -> NormGenerator:
            child_seed = int(np_rng.integers(low=0, high=10**9))
            return NormGenerator(rows, cols, child_seed, dtype=norm_dtype)

        self._pool: List[NormGenerator] = [_make() for _ in range(buffer_size)]
        self._it = cycle(self._pool)

        # idle‑time bookkeeping
        self._idle_accum: float = 0.0
        self._idle_start: Optional[float] = None
        self._update_idle_state()  # evaluate initial status

    # ---------------- iteration protocol ---------------------------------

    def __iter__(self) -> Iterator[cp.ndarray]:
        while True:
            yield self.get_matrix()

    # ---------------- internal ------------------------------------------

    def _update_idle_state(self) -> None:
        """Transition idle/busy states and accumulate idle seconds."""
        all_ready = all(gen.is_ready() for gen in self._pool)
        now = time()

        if all_ready:
            if self._idle_start is None:
                self._idle_start = now  # entered idle state
        else:
            if self._idle_start is not None:
                self._idle_accum += now - self._idle_start
                self._idle_start = None  # exited idle state

    # ---------------- public API -----------------------------------------

    def get_matrix(self) -> cp.ndarray:
        gen = next(self._it)
        mat = gen.get_matrix()
        # generation for *gen* is now queued; readiness landscape changed
        self._update_idle_state()
        return mat

    def get_time_spent_synchronizing(self) -> float:
        return sum(g.get_time_spent_synchronizing() for g in self._pool)

    def get_idle_time(self) -> float:
        """Total seconds so far where **all** generators were idle."""
        self._update_idle_state()
        if self._idle_start is not None:
            return self._idle_accum + (time() - self._idle_start)
        return self._idle_accum

    # --------------------------------------------------------------------

    @property
    def dtype(self) -> cp.dtype:  # noqa: D401
        return self._pool[0].dtype
