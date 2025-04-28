from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spectralmc import async_normals

if TYPE_CHECKING:
    import cupy as cp  # type: ignore[import-untyped]


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401 — runtime check only
    except ModuleNotFoundError:
        return False
    return True


# --------------------------------------------------------------------------- #
# NormGenerator                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_cupy(), reason="CuPy not installed")
def test_norm_generator_shapes_and_state() -> None:
    import cupy as cp  # runtime-only import

    gen = async_normals.NormGenerator(5, 7, seed=123, dtype=cp.float32)
    a: "cp.ndarray" = gen.get_matrix()
    b: "cp.ndarray" = gen.get_matrix()
    assert a.shape == (5, 7)
    assert b.shape == (5, 7)
    assert not cp.allclose(a, b)
    assert gen.get_time_spent_synchronizing() > 0.0


# --------------------------------------------------------------------------- #
# ConcurrentNormGenerator                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_cupy(), reason="CuPy not installed")
def test_concurrent_norm_generator_round_robin() -> None:
    import cupy as cp

    rows, cols = 4, 6
    cgen = async_normals.ConcurrentNormGenerator(
        rows, cols, seed=999, buffer_size=3, dtype=cp.float64
    )
    mats = [cgen.get_matrix() for _ in range(6)]
    for m in mats:
        assert m.shape == (rows, cols)
    assert cgen.get_time_spent_synchronizing() >= 0.0
