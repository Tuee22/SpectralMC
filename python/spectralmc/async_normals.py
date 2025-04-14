# python/spectralmc/async_normals.py

"""
Asynchronous normal generation with CuPy, to reduce waiting times in GPU-based
Monte Carlo. A NormGenerator uses a dedicated CuPy stream. A ConcurrentNormGenerator
manages multiple generators in round-robin.
"""

from __future__ import annotations

import cupy as cp  # type: ignore[import-untyped]
import numpy as np
from time import time
from itertools import cycle
from typing import Iterator, Optional, List


class NormGenerator:
    """
    Generates a rows x cols matrix of standard normals asynchronously.
    """

    def __init__(self, rows: int, cols: int, seed: int) -> None:
        assert rows > 0 and cols > 0 and seed > 0
        self._rows = rows
        self._cols = cols
        self._stream = cp.cuda.Stream()
        self._cp_random_gen = cp.random.default_rng(seed)
        self._generated_array: Optional[cp.ndarray] = None
        self._time_spent_synchronizing = 0.0
        self._create_new_matrix()

    def __iter__(self) -> Iterator[cp.ndarray]:
        while True:
            yield self.get_matrix()

    def _create_new_matrix(self) -> None:
        if self._generated_array is not None:
            raise RuntimeError("Previous matrix not retrieved yet.")
        with self._stream:
            self._generated_array = self._cp_random_gen.standard_normal(
                (self._rows, self._cols)
            )

    def get_time_spent_synchronizing(self) -> float:
        return self._time_spent_synchronizing

    def get_matrix(self) -> cp.ndarray:
        if self._generated_array is None:
            raise RuntimeError("No matrix available.")
        start = time()
        self._stream.synchronize()
        end = time()
        self._time_spent_synchronizing += end - start

        ret_mat = self._generated_array
        self._generated_array = None
        self._create_new_matrix()
        return ret_mat


class ConcurrentNormGenerator:
    """
    Maintains a pool of NormGenerator objects for concurrency.
    """

    def __init__(
        self, rows: int, cols: int, seed: int, buffer_size: int, skip_steps: int = 0
    ) -> None:
        assert buffer_size > 0
        self._np_rng = np.random.default_rng(seed)
        for _ in range(skip_steps):
            _ = self._np_rng.integers(low=0, high=10**9)

        def create_gen() -> NormGenerator:
            new_seed = int(self._np_rng.integers(low=0, high=10**9))
            return NormGenerator(rows, cols, new_seed)

        self._cache: List[NormGenerator] = [create_gen() for _ in range(buffer_size)]
        self._cache_iterator = cycle(self._cache)

    def __iter__(self) -> Iterator[cp.ndarray]:
        return self._generator()

    def _generator(self) -> Iterator[cp.ndarray]:
        while True:
            yield self.get_matrix()

    def get_time_spent_synchronizing(self) -> float:
        return sum(ng.get_time_spent_synchronizing() for ng in self._cache)

    def get_matrix(self) -> cp.ndarray:
        gen = next(self._cache_iterator)
        return gen.get_matrix()
