from __future__ import annotations

"""Reproducibility and serialization tests for GbmTrainer.

This module tests:
- Deterministic initialization of models
- Lockstep training determinism
- Snapshot and restore equivalence
- Optimizer state round-trip serialization
- Demonstration of numerical nondeterminism
"""

import copy
import math
from typing import Dict, Literal, Tuple, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholesConfig, SimulationParams
from spectralmc.gbm_trainer import GbmTrainer, GbmTrainerConfig
from spectralmc.models.torch import AdamOptimizerState
from spectralmc.sobol_sampler import BoundSpec


Precision = Literal["float32", "float64"]
PRECISIONS: Tuple[Precision, Precision] = ("float32", "float64")
LEARNING_RATE: float = 1e-2
ABSOLUTE_TOLERANCE: float = 3e-2


def _clone_model(model: CVNN) -> CVNN:
    """Return a deep copy of a CVNN with the same device and dtype."""
    dup = copy.deepcopy(model)
    param = next(model.parameters())
    return dup.to(param.device, param.dtype)


def _models_close(a: CVNN, b: CVNN, rtol: float = 5e-3, atol: float = 1e-8) -> bool:
    """Check whether two CVNN models have approximately equal parameters."""
    return all(
        torch.allclose(pa, pb, rtol=rtol, atol=atol)
        for pa, pb in zip(a.parameters(), b.parameters())
    )


def _max_param_diff(a: CVNN, b: CVNN) -> float:
    """Compute the maximum absolute difference between CVNN model parameters."""
    return max(
        (
            torch.max(torch.abs(pa - pb)).item()
            for pa, pb in zip(a.parameters(), b.parameters())
        ),
        default=0.0,
    )


def _make_pair(precision: Precision, *, seed: int) -> Tuple[GbmTrainer, GbmTrainer]:
    """Create two identical GbmTrainer instances for reproducibility testing."""
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


@pytest.mark.parametrize("precision", PRECISIONS)
def test_deterministic_construction(precision: Precision) -> None:
    """Ensure two models initialized with the same seed are bit-identical."""
    a, b = _make_pair(precision, seed=42)
    assert _max_param_diff(a._cvnn, b._cvnn) == 0.0


@pytest.mark.parametrize("precision", PRECISIONS)
def test_lockstep_training(precision: Precision) -> None:
    """Verify that models trained step-by-step with same data remain identical."""
    a, b = _make_pair(precision, seed=43)
    for batches in (2, 3, 1):
        a.train(num_batches=batches, batch_size=8, learning_rate=LEARNING_RATE)
        b.train(num_batches=batches, batch_size=8, learning_rate=LEARNING_RATE)
        assert _models_close(a._cvnn, b._cvnn)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_cycle_deterministic(precision: Precision) -> None:
    """Ensure restored models from snapshot produce the same weights after further training."""
    a, _ = _make_pair(precision, seed=44)
    a.train(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)

    snap = a.snapshot()
    snap.cvnn = _clone_model(snap.cvnn)
    b = GbmTrainer(snap)

    a.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    b.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)

    assert _models_close(a._cvnn, b._cvnn)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_restart_without_optimizer(precision: Precision) -> None:
    """Check that reinitializing optimizer leads to drift, but within acceptable range."""
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


def _two_stage_mean(mat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Split and average matrix rows in two parts to show non-determinism."""
    cols = mat.shape[1]

    left_mean = np.mean(mat[:, : cols // 2], axis=1).astype(np.float64)
    right_mean = np.mean(mat[:, cols // 2 :], axis=1).astype(np.float64)

    weighted = (left_mean * (cols // 2) + right_mean * (cols - cols // 2)) / cols
    # Tell mypy explicitly what type this is
    return cast(npt.NDArray[np.float64], weighted)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numpy_reduction_instability(dtype: str) -> None:
    """Show that mathematically equivalent means via different order can diverge slightly."""
    rng = np.random.default_rng(999)
    mat = rng.standard_normal((64, 1024)).astype(dtype)
    mean1 = np.mean(mat, axis=1)
    mean2 = _two_stage_mean(mat.astype(np.float64))
    diff = float(np.abs(mean1 - mean2).max())
    assert diff > 0.0, "Unexpected bit-identical result in staged mean."
    assert diff < 1e-4, f"Excessive drift in mean: {diff:.4e}"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_optimizer_serialization_roundtrip(precision: Precision) -> None:
    """Ensure full optimizer state can be serialized to dict and restored exactly."""
    trainer, _ = _make_pair(precision, seed=50)
    trainer.train(num_batches=4, batch_size=8, learning_rate=LEARNING_RATE)
    snap = trainer.snapshot()

    opt_state = snap.optimizer_state
    assert opt_state is not None

    roundtrip = AdamOptimizerState.model_validate(opt_state.model_dump(mode="python"))
    reloaded = roundtrip.to_torch(device=trainer._cvnn.device)
    torch_state = opt_state.to_torch(device=trainer._cvnn.device)

    for k in ("state", "param_groups"):
        assert k in torch_state and k in reloaded
        assert torch_state[k] == reloaded[k]
