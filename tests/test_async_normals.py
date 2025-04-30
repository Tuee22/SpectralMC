"""
Asynchronous normal generation with CuPy for GPU‑based Monte‑Carlo.

Revision history
----------------
* **2025‑04‑28** – strict typings without ``Any``; only *float32*/*float64*.
* **2025‑04‑28** – added *idle‑time* tracking in ``ConcurrentNormGenerator``.
"""

from __future__ import annotations

"""Expanded unit‐test suite for *spectralmc.async_normals*.

This covers **every public method/property** of both `NormGenerator` and
`ConcurrentNormGenerator` using *both* `float32` and `float64` precisions.
"""

from typing import TYPE_CHECKING, Iterator

import pytest

from spectralmc import async_normals

if TYPE_CHECKING:  # pragma: no cover
    import cupy as cp  # type: ignore[import-untyped]


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401  (runtime check only)
    except ModuleNotFoundError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Helper fixtures                                                              #
# --------------------------------------------------------------------------- #

from typing import Literal, cast


@pytest.fixture(params=["float32", "float64"], ids=["f32", "f64"])
def dtype_str(
    request: pytest.FixtureRequest,
) -> Literal["float32", "float64"]:  # noqa: D401
    """Literal precision names so mypy knows the exact string value."""
    return cast(Literal["float32", "float64"], request.param)


# --------------------------------------------------------------------------- #
# NormGenerator                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_cupy(), reason="CuPy not installed")
def test_norm_generator_all_methods(dtype_str: str) -> None:  # noqa: D103
    import cupy as cp

    rows, cols = 5, 7
    gen = async_normals.NormGenerator(
        rows, cols, seed=42, dtype=cast(Literal["float32", "float64"], dtype_str)
    )

    # --- iterator protocol ------------------------------------------------
    it: Iterator["cp.ndarray"] = iter(gen)
    first = next(it)

    # --- direct call ------------------------------------------------------
    second = gen.get_matrix()

    # shapes and dtype -----------------------------------------------------
    assert first.shape == (rows, cols)
    assert second.shape == (rows, cols)
    assert first.dtype == cp.dtype(dtype_str)

    # new batch should differ statistically
    assert not cp.allclose(first, second)

    # dtype property -------------------------------------------------------
    assert gen.dtype == cp.dtype(dtype_str)

    # synchronisation bookkeeping -----------------------------------------
    pre_sync = gen.get_time_spent_synchronizing()
    _ = gen.get_matrix()
    post_sync = gen.get_time_spent_synchronizing()
    assert post_sync >= pre_sync

    # readiness check ------------------------------------------------------
    cp.cuda.Device().synchronize()  # ensure GPU work is done
    assert gen.is_ready() is True


@pytest.mark.skipif(not _has_cupy(), reason="CuPy not installed")
def test_norm_generator_invalid_dtype() -> None:  # noqa: D103
    with pytest.raises(ValueError):
        async_normals.NormGenerator(2, 2, seed=1, dtype="float16")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# ConcurrentNormGenerator                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _has_cupy(), reason="CuPy not installed")
def test_concurrent_norm_generator_all_methods(dtype_str: str) -> None:  # noqa: D103
    import cupy as cp

    rows, cols = 4, 6
    cgen = async_normals.ConcurrentNormGenerator(
        rows,
        cols,
        seed=999,
        buffer_size=3,
        dtype=cast(Literal["float32", "float64"], dtype_str),
    )

    # iterator protocol ----------------------------------------------------
    it: Iterator["cp.ndarray"] = iter(cgen)
    mats = [next(it) for _ in range(4)]  # round‑robin across generators

    # shapes & dtype -------------------------------------------------------
    for m in mats:
        assert m.shape == (rows, cols)
        assert m.dtype == cp.dtype(dtype_str)

    # direct get_matrix ----------------------------------------------------
    mats += [cgen.get_matrix() for _ in range(4)]

    # time bookkeeping -----------------------------------------------------
    assert cgen.get_time_spent_synchronizing() >= 0.0

    # idle‑time logic ------------------------------------------------------
    cp.cuda.Device().synchronize()  # wait for all in‑flight kernels
    idle_before = cgen.get_idle_time()
    assert idle_before >= 0.0

    # a new request should break the idle stretch, then resume -------------
    _ = cgen.get_matrix()
    cp.cuda.Device().synchronize()
    idle_after = cgen.get_idle_time()
    assert idle_after >= idle_before

    # dtype property -------------------------------------------------------
    assert cgen.dtype == cp.dtype(dtype_str)
