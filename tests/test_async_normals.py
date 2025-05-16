# tests/test_async_normals.py
"""End‑to‑end tests for **spectralmc.async_normals**.

* No ``cast`` or ``type: ignore`` lines (except one for the CuPy import stub).
* Compatible with ``mypy --strict``.
* Fails outright if CuPy is not installed (as requested).
"""

from __future__ import annotations

from typing import List, Literal

import cupy as cp  # type: ignore[import-untyped]
import pytest

from spectralmc import async_normals

DType = Literal["float32", "float64"]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["float32", "float64"], ids=["f32", "f64"])
def dtype_str(request: pytest.FixtureRequest) -> DType:
    """Return the dtype literal currently under test."""

    if request.param == "float32":
        return "float32"
    return "float64"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _collect(gen: async_normals.ConcurrentNormGenerator, n: int) -> List[cp.ndarray]:
    return [gen.get_matrix() for _ in range(n)]


# ---------------------------------------------------------------------------
# _NormGenerator tests
# ---------------------------------------------------------------------------


def test_private_norm_generator(dtype_str: DType) -> None:  # noqa: D103
    rows, cols = 4, 6
    gen = async_normals._NormGenerator(rows, cols, dtype=dtype_str)
    gen.enqueue(123)

    first = gen.get_matrix(456)
    second = gen.get_matrix(789)

    assert first.shape == (rows, cols)
    assert first.dtype == cp.dtype(dtype_str)
    assert not cp.allclose(first, second)

    before = gen.get_time_spent_synchronizing()
    _ = gen.get_matrix(111)
    after = gen.get_time_spent_synchronizing()
    assert after >= before

    cp.cuda.Device().synchronize()
    assert gen.is_ready() is True


# ---------------------------------------------------------------------------
# ConcurrentNormGenerator: checkpointing & diagnostics
# ---------------------------------------------------------------------------


def test_checkpoint_reproducibility(dtype_str: DType) -> None:  # noqa: D103
    rows, cols, buffer = 3, 5, 3
    cfg0 = async_normals.NormGenConfig(
        rows=rows, cols=cols, seed=42, dtype=dtype_str, skips=0
    )
    gen0 = async_normals.ConcurrentNormGenerator(buffer, cfg0)

    # Produce an initial sequence
    initial = _collect(gen0, 10)
    snap = gen0.snapshot()

    # Expected continuation from original generator
    expected = _collect(gen0, 6)

    # Same buffer size restoration
    gen_same = async_normals.ConcurrentNormGenerator(buffer, snap)
    got_same = _collect(gen_same, 6)
    for exp, got in zip(expected, got_same, strict=True):
        assert cp.allclose(exp, got)

    # Different buffer size restoration
    gen_diff = async_normals.ConcurrentNormGenerator(buffer + 2, snap)
    got_diff = _collect(gen_diff, 6)
    for exp, got in zip(expected, got_diff, strict=True):
        assert cp.allclose(exp, got)


def test_diagnostics(dtype_str: DType) -> None:  # noqa: D103
    cfg = async_normals.NormGenConfig(rows=2, cols=2, seed=7, dtype=dtype_str, skips=0)
    gen = async_normals.ConcurrentNormGenerator(2, cfg)

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

    assert gen.dtype == cp.dtype(dtype_str)


# ---------------------------------------------------------------------------
# Validation tests (CPU‑only)
# ---------------------------------------------------------------------------


def test_norm_config_validation() -> None:  # noqa: D103
    with pytest.raises(ValueError):
        async_normals.NormGenConfig(rows=0, cols=2, seed=1, dtype="float32", skips=0)
    with pytest.raises(ValueError):
        async_normals.NormGenConfig(rows=1, cols=2, seed=1, dtype="float16", skips=0)  # type: ignore[arg-type]
