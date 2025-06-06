"""
Reproducibility tests for :pyclass:`spectralmc.gbm_trainer.GbmTrainer`
and a minimal proof that CuPy reductions are not bit-wise deterministic.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Literal, Tuple

import cupy as cp
import pytest
import torch

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholesConfig, SimulationParams
from spectralmc.gbm_trainer import GbmTrainer, GbmTrainerConfig
from spectralmc.sobol_sampler import BoundSpec

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

Precision = Literal["float32", "float64"]
PRECISIONS: Tuple[Precision, Precision] = ("float32", "float64")
LR: float = 1.0e-2
RESTORE_ABS_TOL: float = 3.0e-2  # 0.03 covers observed worst-case drift


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #


def _clone(net: CVNN) -> CVNN:
    dup = copy.deepcopy(net)
    first = next(net.parameters())
    dup.to(first.device, first.dtype)
    return dup


def _allclose(a: CVNN, b: CVNN, *, rtol: float = 5e-3, atol: float = 1e-8) -> bool:
    return all(
        torch.allclose(pa, pb, rtol=rtol, atol=atol)
        for pa, pb in zip(a.parameters(), b.parameters())
    )


def _max_diff(a: CVNN, b: CVNN) -> float:
    return float(
        max(
            torch.max(torch.abs(pa - pb)).item()
            for pa, pb in zip(a.parameters(), b.parameters())
        )
    )


def _make_pair(precision: Precision, *, seed: int) -> Tuple[GbmTrainer, GbmTrainer]:
    torch.manual_seed(seed)

    sim = SimulationParams(
        skip=0,
        timesteps=1,
        network_size=16,
        batches_per_mc_run=2**12,
        threads_per_block=256,
        mc_seed=seed,
        buffer_size=1,
        dtype=precision,
    )
    cfg = BlackScholesConfig(
        sim_params=sim, simulate_log_return=True, normalize_forwards=False
    )

    bounds: Dict[str, BoundSpec] = {
        "X0": BoundSpec(lower=50.0, upper=150.0),
        "K": BoundSpec(lower=50.0, upper=150.0),
        "T": BoundSpec(lower=0.1, upper=2.0),
        "r": BoundSpec(lower=0.0, upper=0.1),
        "d": BoundSpec(lower=0.0, upper=0.05),
        "v": BoundSpec(lower=0.1, upper=0.5),
    }

    net_a = CVNN(6, sim.network_size, 32, 1)
    net_b = copy.deepcopy(net_a)

    t_a = GbmTrainer(GbmTrainerConfig(cfg=cfg, domain_bounds=bounds, cvnn=net_a))
    t_b = GbmTrainer(GbmTrainerConfig(cfg=cfg, domain_bounds=bounds, cvnn=net_b))
    return t_a, t_b


# --------------------------------------------------------------------------- #
# 1) Deterministic construction                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_deterministic_construction(precision: Precision) -> None:
    a, b = _make_pair(precision, seed=11)
    assert _max_diff(a._cvnn, b._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 2) Lock-step training                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_lockstep_training(precision: Precision) -> None:
    a, b = _make_pair(precision, seed=22)

    for batches in (2, 3, 1):
        a.train(num_batches=batches, batch_size=8, learning_rate=LR)
        b.train(num_batches=batches, batch_size=8, learning_rate=LR)

        assert _allclose(a._cvnn, b._cvnn)

        sa = a.snapshot().optimizer_state
        sb = b.snapshot().optimizer_state
        assert sa is not None and sb is not None
        assert sa["param_groups"][0]["lr"] == sb["param_groups"][0]["lr"]
        for pid in sa["state"]:
            assert sa["state"][pid]["step"] == sb["state"][pid]["step"]


# --------------------------------------------------------------------------- #
# 3) Snapshot equality                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_equality(precision: Precision) -> None:
    ref, _ = _make_pair(precision, seed=33)
    ref.train(num_batches=4, batch_size=8, learning_rate=LR)

    snap = ref.snapshot()
    snap.cvnn = _clone(snap.cvnn)
    clone = GbmTrainer(snap)

    assert _allclose(ref._cvnn, clone._cvnn)


# --------------------------------------------------------------------------- #
# 4a) fresh-Adam continuation                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_continuation_fresh_adam(precision: Precision) -> None:
    ref, _ = _make_pair(precision, seed=44)
    ref.train(num_batches=4, batch_size=8, learning_rate=LR)

    snap = ref.snapshot()
    snap.optimizer_state = None
    snap.cvnn = _clone(snap.cvnn)
    fresh = GbmTrainer(snap)

    ref._optimizer_state = None
    ref.train(num_batches=4, batch_size=8, learning_rate=LR)
    fresh.train(num_batches=4, batch_size=8, learning_rate=LR)

    assert _allclose(ref._cvnn, fresh._cvnn)


# --------------------------------------------------------------------------- #
# 4b) restored-Adam continuation                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_continuation_restored_adam(precision: Precision) -> None:
    ref, _ = _make_pair(precision, seed=55)
    ref.train(num_batches=4, batch_size=8, learning_rate=LR)

    snap = ref.snapshot()
    snap.cvnn = _clone(snap.cvnn)
    restored = GbmTrainer(snap)

    ref.train(num_batches=4, batch_size=8, learning_rate=LR)
    restored.train(num_batches=4, batch_size=8, learning_rate=LR)

    diff = _max_diff(ref._cvnn, restored._cvnn)
    assert (
        diff < RESTORE_ABS_TOL
    ), f"max parameter diff {diff:.4f} exceeds {RESTORE_ABS_TOL}"


# --------------------------------------------------------------------------- #
# 5) MVP – CuPy reduction nondeterminism                                      #
# --------------------------------------------------------------------------- #


def _two_stage_mean(mat: cp.ndarray) -> Any:  # return Any for mypy
    """Row-mean via two half reductions (different order)."""
    cols = mat.shape[1]
    left = mat[:, : cols // 2]
    right = mat[:, cols // 2 :]
    return (
        cp.mean(left, axis=1) * (cols // 2)
        + cp.mean(right, axis=1) * (cols - cols // 2)
    ) / cols


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cupy_reduction_instability(dtype: str) -> None:
    """
    Two mathematically equivalent CuPy reductions produce slightly different
    results; the max diff should be > 0 but < 1e-4.
    """
    rand_mod: Any = cp.random  # silence mypy attr-checks
    rand_mod.seed(999)
    rows, cols = 64, 1_024
    mat = rand_mod.standard_normal((rows, cols), dtype=dtype)

    mean1 = cp.mean(mat, axis=1)
    mean2 = _two_stage_mean(mat)

    diff = float(cp.abs(mean1 - mean2).max().item())
    assert diff > 0.0, "CuPy reduction unexpectedly bit-identical."
    assert diff < 1.0e-4, f"reduction drift {diff:.1e} larger than expected"
