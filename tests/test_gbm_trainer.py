from __future__ import annotations

"""Reproducibility and serialization tests for :class:`spectralmc.gbm_trainer.GbmTrainer`.

Checks
------
1. Deterministic model construction
2. Lock-step training determinism
3. Snapshot / restore determinism
4. Snapshot without optimiser (acceptable drift)
5. Demonstration of reduction-order instability
6. Loss-less optimiser JSON ⇄ torch round-trip
"""

from typing import Dict, Literal, Tuple, TypeGuard, Any

import copy
import math

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import Tensor

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholesConfig, SimulationParams
from spectralmc.gbm_trainer import GbmTrainer, GbmTrainerConfig
from spectralmc.models.torch import AdamOptimizerState
from spectralmc.sobol_sampler import BoundSpec

# --------------------------------------------------------------------------- #
# Configuration & simple helpers                                              #
# --------------------------------------------------------------------------- #

Precision = Literal["float32", "float64"]
PRECISIONS: Tuple[Precision, Precision] = ("float32", "float64")

LEARNING_RATE: float = 1.0e-2
ABSOLUTE_TOLERANCE: float = 3.0e-2


def _clone_model(model: CVNN) -> CVNN:
    """Return a deep copy of *model* on the same device / dtype."""
    dup = copy.deepcopy(model)
    param = next(model.parameters())
    return dup.to(param.device, param.dtype)


def _models_close(
    a: CVNN, b: CVNN, *, rtol: float = 5.0e-3, atol: float = 1.0e-8
) -> bool:
    """Return ``True`` iff all parameters of *a* and *b* match within tolerance."""
    return all(
        torch.allclose(pa, pb, rtol=rtol, atol=atol)
        for pa, pb in zip(a.parameters(), b.parameters())
    )


def _max_param_diff(a: CVNN, b: CVNN) -> float:
    """Largest absolute element-wise difference across all parameters."""
    return max(
        (
            float(torch.max(torch.abs(pa - pb)))
            for pa, pb in zip(a.parameters(), b.parameters())
        ),
        default=0.0,
    )


def _tree_equal(x: object, y: object) -> bool:  # noqa: D401
    """Deep equality for nested containers that may contain tensors."""
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return bool(torch.equal(x, y))

    if isinstance(x, dict) and isinstance(y, dict):
        return (
            all(k in y and _tree_equal(v, y[k]) for k, v in x.items())
            and x.keys() == y.keys()
        )

    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(_tree_equal(a, b) for a, b in zip(x, y))

    return bool(x == y)


def _make_pair(precision: Precision, *, seed: int) -> Tuple[GbmTrainer, GbmTrainer]:
    """Create two identical :class:`GbmTrainer` instances for *precision*."""
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
        sim_params=sim,
        simulate_log_return=True,
        normalize_forwards=False,
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
# 1. Deterministic construction                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_deterministic_construction(precision: Precision) -> None:
    a, b = _make_pair(precision, seed=42)
    assert _max_param_diff(a._cvnn, b._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 2. Lock-step training determinism                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_lockstep_training(precision: Precision) -> None:
    a, b = _make_pair(precision, seed=43)
    for batches in (2, 3, 1):
        a.train(num_batches=batches, batch_size=8, learning_rate=LEARNING_RATE)
        b.train(num_batches=batches, batch_size=8, learning_rate=LEARNING_RATE)
        assert _models_close(a._cvnn, b._cvnn)


# --------------------------------------------------------------------------- #
# 3. Snapshot / restore determinism                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_cycle_deterministic(precision: Precision) -> None:
    a, _ = _make_pair(precision, seed=44)
    a.train(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)

    snap = a.snapshot()
    snap.cvnn = _clone_model(snap.cvnn)
    b = GbmTrainer(snap)

    a.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    b.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)

    assert _models_close(a._cvnn, b._cvnn)


# --------------------------------------------------------------------------- #
# 4. Snapshot without optimiser                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_restart_without_optimizer(precision: Precision) -> None:
    a, _ = _make_pair(precision, seed=45)
    a.train(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)

    snap = a.snapshot()
    snap.optimizer_state = None
    snap.cvnn = _clone_model(snap.cvnn)
    b = GbmTrainer(snap)

    a.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    b.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)

    diff = _max_param_diff(a._cvnn, b._cvnn)
    assert diff < ABSOLUTE_TOLERANCE, f"Drift {diff:.4f} exceeds threshold"


# --------------------------------------------------------------------------- #
# 5. Reduction-order instability demo                                         #
# --------------------------------------------------------------------------- #
# helpers.py  (or put this right above your function)


def _is_float64(arr: npt.NDArray[Any]) -> TypeGuard[npt.NDArray[np.float64]]:
    """Return True iff the ndarray’s dtype is exactly float64."""
    return bool(arr.dtype == np.float64)


def _two_stage_mean(mat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    cols = mat.shape[1]
    left = np.mean(mat[:, : cols // 2], axis=1).astype(np.float64)
    right = np.mean(mat[:, cols // 2 :], axis=1).astype(np.float64)
    result = (left * (cols // 2) + right * (cols - cols // 2)) / cols
    assert _is_float64(result)  # narrows `result` for mypy
    return result  # passes --strict


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numpy_reduction_instability(dtype: str) -> None:
    rng = np.random.default_rng(999)
    mat = rng.standard_normal((64, 1024)).astype(dtype)

    mean1 = np.mean(mat, axis=1)
    mean2 = _two_stage_mean(mat.astype(np.float64))

    diff = float(np.abs(mean1 - mean2).max())
    assert diff == 0.0


# --------------------------------------------------------------------------- #
# 6. Optimiser state round-trip                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_optimizer_serialization_roundtrip(precision: Precision) -> None:
    trainer, _ = _make_pair(precision, seed=50)
    trainer.train(num_batches=4, batch_size=8, learning_rate=LEARNING_RATE)
    snap = trainer.snapshot()

    opt_state = snap.optimizer_state
    assert opt_state is not None

    round_trip = AdamOptimizerState.model_validate(opt_state.model_dump(mode="python"))

    torch_state = opt_state.to_torch(device=trainer._device)
    reloaded_state = round_trip.to_torch(device=trainer._device)

    for key in ("state", "param_groups"):
        assert key in torch_state and key in reloaded_state
        assert _tree_equal(torch_state[key], reloaded_state[key])
