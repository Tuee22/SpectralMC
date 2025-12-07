# src/spectralmc/async_normals.py
"""
spectralmc.async_normals
========================
Latency-hiding generation of standard-normal matrices that live on the
GPU, with deterministic checkpoint/restore and *zero* static-typing
compromises.

Design overview
---------------
1. **Snapshot fidelity** - you can pause, serialise the state, and later
   resume (even with a different buffer size) without breaking the random
   sequence.
2. **Latency hiding** - a configurable pool of CUDA streams means the host
   seldom blocks waiting for kernels to finish.
3. **Strict typing** - the module passes `mypy --strict` without ignores,
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

    cfg_result = ConcurrentNormGeneratorConfig.create(
        rows=1024,
        cols=512,
        seed=123,
        dtype=Precision.float32,
    )
    buffer_result = BufferConfig.create(size=4, matrix_rows=1024, matrix_cols=512)

    gen_result = cfg_result.and_then(lambda cfg: ConcurrentNormGenerator.create(buffer_result, cfg))

    match gen_result:
        case Success(gen):
            match gen.get_matrix():
                case Success(gpu_matrix):
                    checkpoint = Success(gen.snapshot())     # serialisable pydantic model
                case Failure(error):
                    checkpoint = Failure(error)
        case Failure(error):
            checkpoint = Failure(error)
"""

from __future__ import annotations

from itertools import cycle
from time import time
from typing import Annotated

import cupy as cp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spectralmc.effects import (
    CaptureRNGState,
    EffectSequence,
    GenerateNormals,
    StreamSync,
    sequence_effects,
)
from spectralmc.errors import InvalidDType, InvalidShape, QueueBusy, QueueEmpty, SeedOutOfRange
from spectralmc.models.numerical import Precision
from spectralmc.result import Failure, Result, Success
from spectralmc.validation import validate_model


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


def _validate_cupy_dtype(dtype: cp.dtype) -> Result[cp.dtype, InvalidDType]:
    """Return *dtype* if it is float32 or float64, otherwise Failure."""
    return (
        Success(dtype)
        if dtype in (cp.dtype(cp.float32), cp.dtype(cp.float64))
        else Failure(InvalidDType(requested=str(dtype)))
    )


# --------------------------------------------------------------------------- #
# Pydantic configuration models                                               #
# --------------------------------------------------------------------------- #


class BufferConfig(BaseModel):
    """Buffer size configuration with validation against matrix dimensions."""

    size: Annotated[int, Field(gt=0, description="Number of concurrent worker streams")]

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def create(
        cls, size: int, matrix_rows: int, matrix_cols: int
    ) -> Result[BufferConfig, InvalidShape]:
        max_size = matrix_rows * matrix_cols
        return (
            Failure(
                InvalidShape(
                    rows=matrix_rows,
                    cols=matrix_cols,
                )
            )
            if size > max_size or min(matrix_rows, matrix_cols) <= 0
            else Success(cls(size=size))
        )


class ConcurrentNormGeneratorConfig(BaseModel):
    """Immutable serialisable snapshot of the global RNG state.

    Attributes
    ----------
    rows
        Matrix height (> 0).
    cols
        Matrix width (> 0).
    seed
        Base seed for the global NumPy RNG (> 0).  Each worker draws its own
        CuPy seed from this generator.
    dtype
        Requested numeric precision as `Precision.float32` or
        `Precision.float64`.
    skips
        How many matrices have already been produced (≥ 0).  Used to advance
        the NumPy RNG when restoring from a checkpoint.
    """

    rows: int = Field(..., gt=0)
    cols: int = Field(..., gt=0)
    seed: int = Field(..., gt=0)
    dtype: Precision
    skips: int = Field(0, ge=0)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def create(
        cls, *, rows: int, cols: int, seed: int, dtype: Precision, skips: int = 0
    ) -> Result["ConcurrentNormGeneratorConfig", ValidationError]:
        return validate_model(cls, rows=rows, cols=cols, seed=seed, dtype=dtype, skips=skips)


# --------------------------------------------------------------------------- #
# Internal single-stream generator                                            #
# --------------------------------------------------------------------------- #


class _NormGenerator:
    """Generate standard-normal matrices on a dedicated CUDA stream."""

    def __init__(self, rows: int, cols: int, *, dtype: cp.dtype) -> None:
        self._rows: int = rows
        self._cols: int = cols
        self._dtype: cp.dtype = dtype

        self._stream: cp.cuda.Stream = cp.cuda.Stream()
        self._generated: cp.ndarray | None = None
        self._event: cp.cuda.Event | None = None
        self._sync_time: float = 0.0

    @classmethod
    def create(
        cls, rows: int, cols: int, *, dtype: cp.dtype
    ) -> Result["_NormGenerator", InvalidShape | InvalidDType]:
        if min(rows, cols) <= 0:
            return Failure(InvalidShape(rows=rows, cols=cols))
        match _validate_cupy_dtype(dtype):
            case Failure(error):
                return Failure(error)
            case Success(validated):
                return Success(cls(rows, cols, dtype=validated))

    # ---------------- asynchronous pipeline --------------------------- #

    def enqueue(self, seed: int) -> Result[None, QueueBusy | SeedOutOfRange]:
        """Launch a kernel that fills the next matrix (non-blocking)."""
        if self._generated is not None:
            return Failure(QueueBusy())
        if seed <= 0 or seed >= _SEED_LIMIT:
            return Failure(SeedOutOfRange(seed=seed))

        self._event = cp.cuda.Event(disable_timing=True)
        with self._stream:
            rng = cp.random.default_rng(seed)
            self._generated = rng.standard_normal((self._rows, self._cols), dtype=self._dtype)
            self._event.record()
        return Success(None)

    def get_matrix(
        self, next_seed: int
    ) -> Result[cp.ndarray, QueueEmpty | QueueBusy | SeedOutOfRange]:
        """Synchronise, return the ready matrix, then queue another."""
        if self._generated is None:  # pragma: no cover
            return Failure(QueueEmpty())

        t0 = time()
        self._stream.synchronize()
        self._sync_time += time() - t0

        ready, self._generated = self._generated, None
        enqueue_result = self.enqueue(next_seed)
        match enqueue_result:
            case Success(_):
                return Success(ready)
            case Failure(error):
                return Failure(error)

    # ---------------- diagnostics ------------------------------------- #

    def get_time_spent_synchronizing(self) -> float:
        """Total host-side synchronisation latency (seconds)."""
        return self._sync_time

    def is_ready(self) -> bool:
        """True iff the current matrix has finished on the GPU."""
        return self._event is not None and int(cp.cuda.runtime.eventQuery(self._event.ptr)) == 0

    # ---------------- read-only props --------------------------------- #

    @property
    def dtype(self) -> cp.dtype:
        """CuPy dtype produced by this worker."""
        return self._dtype


# --------------------------------------------------------------------------- #
# Public concurrent generator                                                 #
# --------------------------------------------------------------------------- #


class ConcurrentNormGenerator:
    """Latency-hiding pool of `_NormGenerator` workers."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        pool: list[_NormGenerator],
        rows: int,
        cols: int,
        dtype: cp.dtype,
        base_seed: int,
        served: int,
        np_rng: np.random.Generator,
    ) -> None:

        self._rows = rows
        self._cols = cols
        self._dtype = dtype
        self._base_seed = base_seed
        self._served = served
        self._np_rng = np_rng

        self._pool = pool
        self._it = cycle(self._pool)

        # Idle-time diagnostics
        self._idle_accum: float = 0.0
        self._idle_start: float | None = None
        self._update_idle_state()

    @classmethod
    def create(
        cls,
        buffer_result: Result[BufferConfig, InvalidShape],
        config: ConcurrentNormGeneratorConfig,
    ) -> Result[
        "ConcurrentNormGenerator", InvalidShape | InvalidDType | QueueBusy | SeedOutOfRange
    ]:

        match buffer_result:
            case Failure(error):
                return Failure(error)
            case Success(buffer):
                pass

        rows = config.rows
        cols = config.cols
        dtype = config.dtype.to_cupy()

        base_seed = config.seed
        served = config.skips
        np_rng = np.random.default_rng(base_seed)
        if served:
            np_rng.integers(0, _SEED_LIMIT, size=served)  # fast-forward

        def _make() -> (
            Result[_NormGenerator, InvalidShape | InvalidDType | QueueBusy | SeedOutOfRange]
        ):
            seed = int(np_rng.integers(0, _SEED_LIMIT))
            gen_result = _NormGenerator.create(rows, cols, dtype=dtype)
            match gen_result:
                case Success(gen):
                    enqueue_result = gen.enqueue(seed)
                    return enqueue_result if isinstance(enqueue_result, Failure) else Success(gen)
                case Failure(error):
                    return Failure(error)

        pool: list[_NormGenerator] = []
        for _ in range(buffer.size):
            gen_result = _make()
            match gen_result:
                case Success(gen):
                    pool.append(gen)
                case Failure(error):
                    return Failure(error)

        return Success(
            cls(
                pool=pool,
                rows=rows,
                cols=cols,
                dtype=dtype,
                base_seed=base_seed,
                served=served,
                np_rng=np_rng,
            )
        )

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

    def get_matrix(self) -> Result[cp.ndarray, QueueEmpty | QueueBusy | SeedOutOfRange]:
        """Return the next matrix; enqueue another in its place."""
        gen = next(self._it)
        next_seed = int(self._np_rng.integers(0, _SEED_LIMIT))
        match gen.get_matrix(next_seed):
            case Success(mat):
                self._served += 1
                self._update_idle_state()
                return Success(mat)
            case Failure(error):
                return Failure(error)

    def snapshot(self) -> ConcurrentNormGeneratorConfig:
        """Produce a deterministic checkpoint of the current state."""
        return ConcurrentNormGeneratorConfig(
            rows=self._rows,
            cols=self._cols,
            seed=self._base_seed,
            dtype=Precision.from_cupy(self._dtype),
            skips=self._served,
        )

    def build_generation_effects(self) -> EffectSequence[list[object]]:
        """Build pure effect sequence describing normal matrix generation.

        This method produces an immutable effect description that can be:
        - Inspected and tested without GPU hardware
        - Serialized for reproducibility tracking
        - Composed with other effects in larger workflows

        The actual execution happens when the interpreter processes these effects.

        Returns:
            EffectSequence describing: RNG capture → normal generation → stream sync.

        Example:
            >>> effects = gen.build_generation_effects()
            >>> # Pure description - no side effects yet
            >>> result = await interpreter.interpret_sequence(effects)
        """
        return sequence_effects(
            CaptureRNGState(rng_type="numpy"),
            GenerateNormals(
                rows=self._rows,
                cols=self._cols,
                seed=self._base_seed,
                skip=self._served,
            ),
            StreamSync(stream_type="cupy"),
        )

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def get_time_spent_synchronizing(self) -> float:
        """Aggregate host-side synchronisation latency (seconds)."""
        return sum(gen.get_time_spent_synchronizing() for gen in self._pool)

    def get_idle_time(self) -> float:
        """Total time (seconds) during which *every* worker was ready."""
        self._update_idle_state()
        return (
            self._idle_accum + (time() - self._idle_start)
            if self._idle_start is not None
            else self._idle_accum
        )

    # ---------------- read-only props --------------------------------- #

    @property
    def dtype(self) -> cp.dtype:
        """CuPy dtype produced by the generator."""
        return self._dtype
