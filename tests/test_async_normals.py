# tests/test_async_normals.py
"""
tests.test_async_normals
========================
End‑to‑end tests for `spectralmc.async_normals` with **zero ignores**.
The CuPy stub package under *stubs/* provides every symbol touched,
so the suite type‑checks under `mypy --strict`.

Test matrix
-----------
* _NormGenerator – enqueue/get, dtype, synchronisation metrics.
* Concurrent generator – checkpoint reproducibility, idle‑time tracking.
* Validation (CPU‑only) – Pydantic schema errors for bad payloads.
"""

from __future__ import annotations

from typing import List

import cupy as cp
import pytest
from pydantic import ValidationError

from spectralmc import async_normals
from spectralmc.async_normals import BufferConfig
from spectralmc.models.numerical import Precision

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=(Precision.float32, Precision.float64), ids=("f32", "f64"))
def precision(request: pytest.FixtureRequest) -> Precision:
    """Precision value under test (ensures mypy sees `Precision`)."""
    param = request.param
    assert isinstance(param, Precision)
    return param


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #


def _collect(
    gen: async_normals.ConcurrentNormGenerator,
    n: int,
) -> List[cp.ndarray]:
    """Collect *n* matrices from a generator."""
    return [gen.get_matrix() for _ in range(n)]


# --------------------------------------------------------------------------- #
# _NormGenerator tests                                                        #
# --------------------------------------------------------------------------- #


def test_private_norm_generator(precision: Precision) -> None:
    """Smoke‑test the private single‑stream generator."""
    rows, cols = 4, 6
    gen = async_normals._NormGenerator(rows, cols, dtype=precision.to_cupy())
    gen.enqueue(123)

    first = gen.get_matrix(456)
    second = gen.get_matrix(789)

    assert first.shape == (rows, cols)
    assert first.dtype == precision.to_cupy()
    assert not cp.allclose(first, second)

    before = gen.get_time_spent_synchronizing()
    _ = gen.get_matrix(111)
    after = gen.get_time_spent_synchronizing()
    assert after >= before

    cp.cuda.Device().synchronize()
    assert gen.is_ready() is True


# --------------------------------------------------------------------------- #
# ConcurrentNormGenerator tests                                               #
# --------------------------------------------------------------------------- #


def test_checkpoint_reproducibility(precision: Precision) -> None:
    """Checkpoint, restore, and ensure the random stream continues."""
    rows, cols, buffer = 3, 5, 3
    cfg0 = async_normals.ConcurrentNormGeneratorConfig(
        rows=rows,
        cols=cols,
        seed=42,
        dtype=precision,
        skips=0,
    )
    gen0 = async_normals.ConcurrentNormGenerator(
        BufferConfig.create(buffer, rows, cols), cfg0
    )

    initial = _collect(gen0, 10)
    snap = gen0.snapshot()

    expected = _collect(gen0, 6)
    assert len(initial) == 10

    gen_same = async_normals.ConcurrentNormGenerator(
        BufferConfig.create(buffer, snap.rows, snap.cols), snap
    )
    got_same = _collect(gen_same, 6)
    assert all(cp.allclose(e, g) for e, g in zip(expected, got_same, strict=True))

    gen_diff = async_normals.ConcurrentNormGenerator(
        BufferConfig.create(buffer + 2, snap.rows, snap.cols), snap
    )
    got_diff = _collect(gen_diff, 6)
    assert all(cp.allclose(e, g) for e, g in zip(expected, got_diff, strict=True))


def test_diagnostics(precision: Precision) -> None:
    """Synchronisation and idle time metrics must be monotonic."""
    cfg = async_normals.ConcurrentNormGeneratorConfig(
        rows=2,
        cols=2,
        seed=7,
        dtype=precision,
        skips=0,
    )
    gen = async_normals.ConcurrentNormGenerator(
        BufferConfig.create(2, cfg.rows, cfg.cols), cfg
    )

    t0 = gen.get_time_spent_synchronizing()
    _ = gen.get_matrix()
    t1 = gen.get_time_spent_synchronizing()
    assert t1 >= t0

    cp.cuda.Device().synchronize()
    idle_before = gen.get_idle_time()
    _ = gen.get_matrix()
    cp.cuda.Device().synchronize()
    idle_after = gen.get_idle_time()
    assert idle_after >= idle_before

    assert gen.dtype == precision.to_cupy()


# --------------------------------------------------------------------------- #
# Validation tests (CPU‑only)                                                 #
# --------------------------------------------------------------------------- #


def test_norm_config_validation() -> None:
    """Ensure invalid payloads are rejected by the Pydantic schema."""
    with pytest.raises(ValidationError):
        async_normals.ConcurrentNormGeneratorConfig.model_validate(
            {"rows": 0, "cols": 2, "seed": 1, "dtype": Precision.float32, "skips": 0}
        )

    with pytest.raises(ValidationError):
        async_normals.ConcurrentNormGeneratorConfig.model_validate(
            {"rows": 1, "cols": 2, "seed": 1, "dtype": "float16", "skips": 0}
        )
