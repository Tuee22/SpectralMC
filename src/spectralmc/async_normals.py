"""
Asynchronous normal generation with CuPy.

* `NormGenerator`  – one private stream, generates a matrix in the
  background.
* `ConcurrentNormGenerator` – a round-robin pool of generators to hide
  RNG latency.

Now supports **float32 / float64** via the `dtype` argument.
"""

from __future__ import annotations

import cupy as cp  # type: ignore[import-untyped]
import numpy as np
from itertools import cycle
from time import time
from typing import Iterator, List, Optional


class NormGenerator:
    """
    Generates a matrix of standard normals asynchronously on its own stream.
    """

    def __init__(self, rows: int, cols: int, seed: int, dtype: cp.dtype) -> None:
        assert rows > 0 and cols > 0 and seed > 0
        self._rows = rows
        self._cols = cols
        self._dtype = dtype
        self._stream = cp.cuda.Stream()
        self._rng = cp.random.default_rng(seed)
        self._generated: Optional[cp.ndarray] = None
        self._wait_time = 0.0
        self._kickoff()

    # ------------------------------------------------------------------

    def _kickoff(self) -> None:
        if self._generated is not None:  # pragma: no cover
            raise RuntimeError("Previous matrix not retrieved yet.")
        with self._stream:
            self._generated = self._rng.standard_normal(
                (self._rows, self._cols), dtype=self._dtype
            )

    def __iter__(self) -> Iterator[cp.ndarray]:
        while True:
            yield self.get_matrix()

    def get_matrix(self) -> cp.ndarray:
        if self._generated is None:  # pragma: no cover
            raise RuntimeError("No matrix available.")
        t0 = time()
        self._stream.synchronize()
        self._wait_time += time() - t0
        ret = self._generated
        self._generated = None
        self._kickoff()
        return ret

    def get_time_spent_synchronizing(self) -> float:
        return self._wait_time


class ConcurrentNormGenerator:
    """
    Round-robin pool of :class:`NormGenerator` instances.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        seed: int,
        buffer_size: int,
        dtype: cp.dtype,
        skip_steps: int = 0,
    ) -> None:
        assert buffer_size > 0
        rng = np.random.default_rng(seed)
        for _ in range(skip_steps):
            _ = rng.integers(0, 2**31)

        def make_gen() -> NormGenerator:
            child_seed = int(rng.integers(0, 2**31))
            return NormGenerator(rows, cols, child_seed, dtype)

        self._pool: List[NormGenerator] = [make_gen() for _ in range(buffer_size)]
        self._iter = cycle(self._pool)

    # ------------------------------------------------------------------

    def get_matrix(self) -> cp.ndarray:
        gen = next(self._iter)
        return gen.get_matrix()

    def get_time_spent_synchronizing(self) -> float:
        return sum(g.get_time_spent_synchronizing() for g in self._pool)

    def __iter__(self) -> Iterator[cp.ndarray]:
        while True:
            yield self.get_matrix()
