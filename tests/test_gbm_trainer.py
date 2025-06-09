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


def _clone_model(model: CVNN) -> CVNN:
    """Return a deep copy of *model* on the same device / dtype."""
    dup = copy.deepcopy(model)
    param = next(model.parameters())
    return dup.to(param.device, param.dtype)


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


def _make_gbm_trainer(precision: Precision, *, seed: int) -> GbmTrainer:
    """Create an instance via deterministic seeding"""
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

    net = CVNN(6, sim.network_size, 32, 1)

    return GbmTrainer(GbmTrainerConfig(cfg=cfg, domain_bounds=bounds, cvnn=net))


# --------------------------------------------------------------------------- #
# 1. Deterministic construction                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_deterministic_construction(precision: Precision) -> None:
    a = _make_gbm_trainer(precision, seed=42)
    b = _make_gbm_trainer(precision, seed=42)
    assert _max_param_diff(a._cvnn, b._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 2. Lock-step training determinism                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_lockstep_training(precision: Precision) -> None:
    a = _make_gbm_trainer(precision, seed=43)
    b = _make_gbm_trainer(precision, seed=43)
    for batches in (2, 3, 1):
        a.train(num_batches=batches, batch_size=8, learning_rate=LEARNING_RATE)
        b.train(num_batches=batches, batch_size=8, learning_rate=LEARNING_RATE)
        assert _max_param_diff(a._cvnn, b._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 3. Snapshot / restore determinism                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_cycle_deterministic(precision: Precision) -> None:
    a = _make_gbm_trainer(precision, seed=44)
    b = _make_gbm_trainer(precision, seed=44)
    a.train(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)

    snap = a.snapshot()
    snap.cvnn = _clone_model(snap.cvnn)
    b = GbmTrainer(snap)

    a.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    b.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)

    assert _max_param_diff(a._cvnn, b._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 4. Snapshot without optimiser                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_restart(precision: Precision) -> None:
    a = _make_gbm_trainer(precision, seed=45)
    a.train(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)

    # create snapshot
    snap = a.snapshot()

    # create a copy
    snap.cvnn = _clone_model(snap.cvnn)
    b = GbmTrainer(snap)

    a.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    b.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)

    assert _max_param_diff(a._cvnn, b._cvnn) == 0.0


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_restart_without_optimizer(precision: Precision) -> None:
    a = _make_gbm_trainer(precision, seed=45)
    a.train(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)

    # create snapshot
    snap = a.snapshot()

    # remove both optimizer state
    snap.optimizer_state = None
    a._optimizer_state = None

    # create
    snap.cvnn = _clone_model(snap.cvnn)
    b = GbmTrainer(snap)

    a.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    b.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)

    assert _max_param_diff(a._cvnn, b._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 5. Optimiser state round-trip                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_optimizer_serialization_roundtrip(precision: Precision) -> None:
    trainer = _make_gbm_trainer(precision, seed=50)
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
