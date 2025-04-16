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
    Generates a matrix of standard normals asynchronously using a dedicated CuPy stream.

    Explanation:
    -----------
    - The constructor spins up a CuPy RNG (random number generator) and immediately
      allocates a matrix of random normals on a private stream (`self._stream`).
    - When `get_matrix()` is called, this class forces a synchronization on its
      internal stream to ensure the previous batch of normals is fully generated.
    - The returned CuPy array is the one that was just synchronized, and a *new*
      generation of normals is immediately kicked off in the background for the
      next call, using the same private stream.

    Why Asynchronous?
    -----------------
    - By generating randoms in a separate stream, user code can proceed with other
      GPU tasks on different streams without blocking. Only when the user *needs*
      the final random values do we synchronize (`self._stream.synchronize()`).
    - This is especially useful in scenarios where random generation can be
      "hidden" behind other compute, effectively overlapping normal generation
      with other GPU tasks.
    """

    def __init__(self, rows: int, cols: int, seed: int) -> None:
        """
        :param rows: Number of rows in the random matrix
        :param cols: Number of columns in the random matrix
        :param seed: Seed to initialize the random generator

        Explanation:
        ------------
        - We assert that rows, cols, and seed are > 0, since we expect valid
          dimensions and a non-zero seed.
        - We create a dedicated CuPy stream for this generator to run on.
        - We initialize the CuPy random generator with the given seed.
        - We immediately call _create_new_matrix() so the first batch of
          normals starts generating in the background right away.
        """
        assert rows > 0 and cols > 0 and seed > 0
        self._rows = rows
        self._cols = cols
        self._stream = cp.cuda.Stream()
        self._cp_random_gen = cp.random.default_rng(seed)
        self._generated_array: Optional[cp.ndarray] = None
        self._time_spent_synchronizing = 0.0
        self._create_new_matrix()

    def __iter__(self) -> Iterator[cp.ndarray]:
        """
        Provides an iterator interface to continually fetch new random matrices.

        Explanation:
        ------------
        - We repeatedly yield the next available random matrix via get_matrix().
        - This makes the NormGenerator objects usable in loops like:
              for mat in norm_generator:
                  ... do something with mat ...
        """
        while True:
            yield self.get_matrix()

    def _create_new_matrix(self) -> None:
        """
        Private helper to allocate a new batch of random normals on this generator's stream.

        Explanation:
        ------------
        - If we haven't yet "consumed" the previously generated array (i.e.,
          it's still in self._generated_array), we raise an error to avoid overwriting it.
        - In the 'with self._stream:' context, we issue a command to produce
          standard normals of shape (rows, cols).
        - Actual generation is queued on this generator's private CuPy stream,
          so it can proceed asynchronously until we explicitly sync or read it.
        """
        if self._generated_array is not None:
            raise RuntimeError("Previous matrix not retrieved yet.")
        with self._stream:
            self._generated_array = self._cp_random_gen.standard_normal(
                (self._rows, self._cols)
            )

    def get_time_spent_synchronizing(self) -> float:
        """
        :return: The total time (seconds) this generator has spent in stream synchronizations.

        Explanation:
        ------------
        - Each call to get_matrix() forces a stream synchronization to ensure
          the random data is ready. We accumulate the time of these sync calls
          in _time_spent_synchronizing.
        - This value can be helpful in performance diagnostics.
        """
        return self._time_spent_synchronizing

    def get_matrix(self) -> cp.ndarray:
        """
        Retrieves the latest batch of randoms, synchronizing on the private stream first.

        Explanation:
        ------------
        - We ensure there's a generated array to return. If not, we raise an error.
        - We then synchronize the stream to guarantee the random data is available.
          This is the only blocking step in this class:
            self._stream.synchronize()
        - After retrieving and returning the ready random matrix,
          we immediately create a new matrix asynchronously for future use.
        - The synergy: while the user is consuming or processing the returned array,
          the next batch of randoms begins generating in the background.
        """
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
    Maintains multiple NormGenerator objects to provide concurrency in random generation.

    Explanation:
    -----------
    - We hold a 'buffer' of NormGenerator instances, each with its own CuPy stream.
    - When the user needs a new batch of random numbers, we rotate to the next generator
      in a round-robin fashion. Meanwhile, the others could be busy generating
      or be idle waiting for usage.
    - This means the user rarely waits for random generation (assuming enough
      parallelism and buffer_size). If the user repeatedly calls get_matrix(), each
      call obtains randoms from a generator that likely already finished generating.
      In effect, we hide the random generation cost by splitting it across multiple
      streams in advance.

    Usage Notes:
    -----------
    - buffer_size controls how many generators are in the pool.
      More generators => more concurrency (up to GPU resource limits),
      but also more memory usage and overhead. Typically, a value of 2-4
      is enough to cover typical random generation latencies.
    """

    def __init__(
        self, rows: int, cols: int, seed: int, buffer_size: int, skip_steps: int = 0
    ) -> None:
        """
        :param rows: Number of rows in each random matrix
        :param cols: Number of columns in each random matrix
        :param seed: Base seed for reproducibility
        :param buffer_size: How many NormGenerator objects to spawn
        :param skip_steps: How many times to skip ahead in the RNG before creating the generators

        Explanation:
        ------------
        - We first create an np.random.default_rng() with 'seed'. We optionally
          skip ahead in this RNG's integer draws (skip_steps times) to decorrelate
          subsequent seeds if desired.
        - Each NormGenerator in the cache is given its own unique seed, which we
          obtain via self._np_rng.integers(). This ensures that each generator
          in the pool has a distinct random seed, so they're not all producing
          the same random sequence in parallel.
        - The resulting self._cache is a list of NormGenerator instances, each
          fully asynchronous. We cycle through them using itertools.cycle()
          for round-robin distribution.
        """
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
        """
        Provides an infinite iterator over random matrices, round-robin style.

        Explanation:
        ------------
        - We delegate to self._generator(), which yields consecutive
          random matrices from the pool.
        - This is convenient if you want to do something like:
               for rand_mat in concurrent_gen:
                   ... do stuff ...
          and never run out of random draws.
        """
        return self._generator()

    def _generator(self) -> Iterator[cp.ndarray]:
        """
        Internal generator function yielding random matrices in a loop.

        Explanation:
        ------------
        - This method calls get_matrix() indefinitely, providing a continuous
          stream of random matrices from the internal NormGenerator pool.
        """
        while True:
            yield self.get_matrix()

    def get_time_spent_synchronizing(self) -> float:
        """
        :return: The total time (seconds) spent synchronizing across *all* NormGenerators.

        Explanation:
        ------------
        - Each NormGenerator logs its own synchronization time. We sum them
          here to get a global measure of how long the entire system has spent
          waiting on GPU random generation.
        - Useful for diagnosing concurrency gains or overhead.
        """
        return sum(ng.get_time_spent_synchronizing() for ng in self._cache)

    def get_matrix(self) -> cp.ndarray:
        """
        Fetches a new random matrix from the next NormGenerator in round-robin order.

        Explanation:
        ------------
        - We do 'gen = next(self._cache_iterator)' to pick the next NormGenerator
          in the cycle.
        - That generator may already have completed (or be nearly complete)
          generating its random numbers on its private stream. If so, the
          synchronization overhead in gen.get_matrix() is minimal, effectively
          hiding random generation latency from the user.
        - Right after returning the matrix, that generator spawns another batch
          of randoms asynchronously, so it will be ready again by the time this
          cycle comes back around to it.
        """
        gen = next(self._cache_iterator)
        return gen.get_matrix()


# -------------------------------------------------------------------------
# Simple Usage Examples / Tests
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: Basic NormGenerator usage
    print("Test 1: NormGenerator")
    rows, cols, seed = 10, 5, 123
    norm_gen = NormGenerator(rows, cols, seed)
    mat1 = norm_gen.get_matrix()
    print(f"Shape of first matrix: {mat1.shape}")
    print(f"Time spent synchronizing so far: {norm_gen.get_time_spent_synchronizing():.6f} s\n")

    # Test 2: Obtain another matrix and verify shape again
    mat2 = norm_gen.get_matrix()
    print(f"Shape of second matrix: {mat2.shape}")
    print(f"Time spent synchronizing now: {norm_gen.get_time_spent_synchronizing():.6f} s\n")

    # Test 3: ConcurrentNormGenerator usage
    print("Test 3: ConcurrentNormGenerator")
    c_rows, c_cols, c_seed, c_buffer_size = 10, 5, 999, 2
    cng = ConcurrentNormGenerator(c_rows, c_cols, c_seed, c_buffer_size)

    # Grab a few matrices in sequence
    c_mat1 = cng.get_matrix()
    c_mat2 = cng.get_matrix()
    c_mat3 = cng.get_matrix()

    print(f"Shapes of concurrent matrices: {c_mat1.shape}, {c_mat2.shape}, {c_mat3.shape}")
    sync_time = cng.get_time_spent_synchronizing()
    print(f"Total time spent synchronizing across all generators: {sync_time:.6f} s")

    # Optional: Show that they are indeed different
    # (Not strictly guaranteed for small draws, but typically different seeds => different data)
    print("First matrix stats:", float(cp.mean(c_mat1)), float(cp.std(c_mat1)))
    print("Second matrix stats:", float(cp.mean(c_mat2)), float(cp.std(c_mat2)))
    print("Third matrix stats:", float(cp.mean(c_mat3)), float(cp.std(c_mat3)))