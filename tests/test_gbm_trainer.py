# tests/test_gbm_trainer.py
"""
Granular reproducibility tests for :pyclass:`spectralmc.gbm_trainer.GbmTrainer`.

The suite isolates *exactly* where reproducibility may break:

1.  **`test_snapshot_weights_equal`** – after an initial training phase
    a snapshot is taken; a *clone* constructed from that snapshot must have
    **identical parameters** to the reference trainer *before any extra work*.

2.  **`test_zero_lr_without_opt_state`** – continuing training *without* the
    optimiser state and with ``learning_rate = 0`` must keep parameters frozen
    on **both** trainers.

3.  **`test_zero_lr_with_opt_state`** – continuing training *with* optimiser
    state (thus restoring the original learning-rate) must yield identical
    parameters on both trainers when they perform the *same* extra work.

A helper ``_make_trainer`` builds a small configuration suitable for CI; the
Sobol sampler uses deterministic Sobol sequences, so both trainers see the same
data.
"""

from __future__ import annotations

import copy
from typing import Literal, Tuple

import pytest
import torch

from spectralmc.cvnn import CVNN
from spectralmc.gbm import BlackScholes, BlackScholesConfig, SimulationParams
from spectralmc.gbm_trainer import (
    GbmTrainer,
    GbmTrainerConfig,
)
from spectralmc.sobol_sampler import BoundSpec


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #


def _clone_model(model: CVNN) -> CVNN:
    """Deep-copy *model* onto the same device/dtype."""
    model_clone = copy.deepcopy(model)
    first_param = next(model.parameters())
    model_clone.to(first_param.device, first_param.dtype)
    return model_clone


def _weights_equal(lhs: CVNN, rhs: CVNN) -> bool:
    """Return **True** if all learnable parameters match bit-for-bit."""
    return all(torch.equal(a, b) for a, b in zip(lhs.parameters(), rhs.parameters()))


def _make_trainer(
    *,
    precision: Literal["float32", "float64"],
    seed: int = 123,
) -> Tuple[GbmTrainer, int]:
    """Construct a small trainer for fast tests.

    Returns
    -------
    Tuple[trainer, num_features]
        The trainer and the underlying network size (FFT length).
    """
    sim_params = SimulationParams(
        skip=0,
        timesteps=1,
        network_size=16,
        batches_per_mc_run=2**12,
        threads_per_block=256,
        mc_seed=seed,
        buffer_size=1,
        dtype=precision,
    )
    bs_cfg = BlackScholesConfig(
        sim_params=sim_params,
        simulate_log_return=True,
        normalize_forwards=False,
    )
    bounds = {
        "X0": BoundSpec(lower=50.0, upper=150.0),
        "K": BoundSpec(lower=50.0, upper=150.0),
        "T": BoundSpec(lower=0.1, upper=2.0),
        "r": BoundSpec(lower=0.0, upper=0.1),
        "d": BoundSpec(lower=0.0, upper=0.05),
        "v": BoundSpec(lower=0.1, upper=0.5),
    }
    cvnn = CVNN(
        input_features=6,
        output_features=sim_params.network_size,
        hidden_features=32,
        num_residual_blocks=1,
    )
    cfg = GbmTrainerConfig(cfg=bs_cfg, domain_bounds=bounds, cvnn=cvnn)
    return GbmTrainer(cfg), sim_params.network_size


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

PRECISIONS: Tuple[Literal["float32"], Literal["float64"]] = ("float32", "float64")


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_weights_equal(precision: Literal["float32", "float64"]) -> None:
    """Snapshot then reconstruct – parameters must match exactly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required.")

    trainer_ref, _ = _make_trainer(precision=precision)
    trainer_ref.train(num_batches=4, batch_size=8, learning_rate=1e-2)

    snapshot = trainer_ref.snapshot()
    snapshot.cvnn = _clone_model(snapshot.cvnn)  # independent copy

    trainer_clone = GbmTrainer(snapshot)

    assert _weights_equal(
        trainer_ref._cvnn, trainer_clone._cvnn  # noqa: SLF001
    ), "Weights differ immediately after snapshot reconstruction."


@pytest.mark.parametrize("precision", PRECISIONS)
def test_zero_lr_without_opt_state(precision: Literal["float32", "float64"]) -> None:
    """With LR=0 *and* optimiser state removed, weights must stay frozen."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required.")

    trainer_ref, _ = _make_trainer(precision=precision)
    trainer_ref.train(num_batches=4, batch_size=8, learning_rate=1e-2)

    snap_no_opt = trainer_ref.snapshot()
    snap_no_opt.optimizer_state = None
    snap_no_opt.cvnn = _clone_model(snap_no_opt.cvnn)

    trainer_a = GbmTrainer(snap_no_opt)

    # Freeze both trainers with LR = 0 and NO optimiser state
    trainer_ref._optimizer_state = None
    trainer_ref.train(
        num_batches=2, batch_size=8, learning_rate=0.0, optimizer_state=None
    )

    trainer_a.train(
        num_batches=2, batch_size=8, learning_rate=0.0, optimizer_state=None
    )

    assert _weights_equal(
        trainer_ref._cvnn, trainer_a._cvnn  # noqa: SLF001
    ), "Weights changed despite LR=0 and no optimiser state."


@pytest.mark.parametrize("precision", PRECISIONS)
def test_zero_lr_with_opt_state(precision: Literal["float32", "float64"]) -> None:
    """LR=0 but restoring optimiser state should keep both trainers in sync."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required.")

    trainer_ref, _ = _make_trainer(precision=precision)
    trainer_ref.train(num_batches=4, batch_size=8, learning_rate=1e-2)
    snapshot = trainer_ref.snapshot()
    snapshot.cvnn = _clone_model(snapshot.cvnn)

    trainer_b = GbmTrainer(snapshot)

    # Continue both trainers; optimiser state will restore LR=1e-2 internally
    trainer_ref.train(num_batches=2, batch_size=8, learning_rate=0.0)
    trainer_b.train(num_batches=2, batch_size=8, learning_rate=0.0)

    assert _weights_equal(
        trainer_ref._cvnn, trainer_b._cvnn  # noqa: SLF001
    ), "Weights diverged when optimiser state was restored."
