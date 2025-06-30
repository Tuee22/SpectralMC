# tests/test_gbm_trainer.py
"""Determinism & serialisation tests for :pymod:`spectralmc.gbm_trainer`.

The suite exercises

1. deterministic model construction;
2. lock‑step training determinism;
3. snapshot/restore reproducibility;
4. restart from a snapshot *without* optimiser state;
5. round‑trip JSON ↔ optimiser state; and
6. an end‑to‑end smoke test for :pyfunc:`GbmCVNNPricer.predict_price`.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Literal, Sequence, Tuple

import pytest
import torch
from torch import Tensor

from spectralmc.cvnn_factory import (
    ActivationCfg,
    ActivationKind,
    CVNNConfig,
    LinearCfg,
    build_model,
)
from spectralmc.gbm import BlackScholes, BlackScholesConfig, SimulationParams
from spectralmc.gbm_trainer import ComplexValuedModel, GbmCVNNPricer, GbmCVNNPricerConfig
from spectralmc.models.torch import AdamOptimizerState
from spectralmc.sobol_sampler import BoundSpec

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

Precision = Literal["float32", "float64"]
PRECISIONS: Tuple[Precision, Precision] = ("float32", "float64")
LEARNING_RATE: float = 1.0e-2


# --------------------------------------------------------------------------- #
# Local helpers                                                               #
# --------------------------------------------------------------------------- #


def _clone_model(model: ComplexValuedModel) -> ComplexValuedModel:
    """Deep‑copy *model* onto the same device and dtype."""
    duplicate = copy.deepcopy(model)
    first_param = next(iter(model.parameters()))
    duplicate.to(first_param.device, first_param.dtype)
    return duplicate


def _max_param_diff(a: ComplexValuedModel, b: ComplexValuedModel) -> float:
    """Return the L∞‑norm of the parameter difference."""
    return max(
        (
            float(torch.abs(param_a - param_b).max().item())
            for param_a, param_b in zip(a.parameters(), b.parameters())
        ),
        default=0.0,
    )


def _tree_equal(x: object, y: object) -> bool:  # noqa: D401
    """Deep structural equality for (potentially nested) containers."""
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return bool(torch.equal(x, y))

    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_tree_equal(v, y[k]) for k, v in x.items())

    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(_tree_equal(a, b) for a, b in zip(x, y))

    return bool(x == y)


def _make_cvnn(
    n_inputs: int, n_outputs: int, *, seed: int, hidden: int = 32
) -> ComplexValuedModel:
    """Create a small CVNN via :pyfunc:`spectralmc.cvnn_factory.build_model`."""
    cfg = CVNNConfig(
        layers=[
            LinearCfg(
                width=hidden,
                activation=ActivationCfg(kind=ActivationKind.MOD_RELU),
            ),
            LinearCfg(width=n_outputs),
        ],
        seed=seed,
    )
    return build_model(n_inputs=n_inputs, n_outputs=n_outputs, cfg=cfg)


def _make_gbm_trainer(precision: Precision, *, seed: int) -> GbmCVNNPricer:
    """Deterministically construct a :class:`GbmCVNNPricer` instance."""
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

    net = _make_cvnn(6, sim.network_size, seed=seed)
    return GbmCVNNPricer(GbmCVNNPricerConfig(cfg=cfg, domain_bounds=bounds, cvnn=net))


# --------------------------------------------------------------------------- #
# 1. Deterministic construction                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_deterministic_construction(precision: Precision) -> None:
    first = _make_gbm_trainer(precision, seed=42)
    second = _make_gbm_trainer(precision, seed=42)
    assert _max_param_diff(first._cvnn, second._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 2. Lock‑step training determinism                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_lockstep_training(precision: Precision) -> None:
    first = _make_gbm_trainer(precision, seed=43)
    second = _make_gbm_trainer(precision, seed=43)

    for batches in (2, 3, 1):
        first.train(num_batches=batches, batch_size=8, learning_rate=LEARNING_RATE)
        second.train(num_batches=batches, batch_size=8, learning_rate=LEARNING_RATE)
        assert _max_param_diff(first._cvnn, second._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 3. Snapshot / restore determinism                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_cycle_deterministic(precision: Precision) -> None:
    original = _make_gbm_trainer(precision, seed=44)
    clone = _make_gbm_trainer(precision, seed=44)

    original.train(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)

    snapshot = original.snapshot()
    snapshot = snapshot.model_copy(update={"cvnn": _clone_model(snapshot.cvnn)})
    clone = GbmCVNNPricer(snapshot)

    original.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    clone.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)

    assert _max_param_diff(original._cvnn, clone._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 4. Snapshot restart *without* optimiser                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_restart_without_optimizer(precision: Precision) -> None:
    trainer = _make_gbm_trainer(precision, seed=45)
    trainer.train(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)

    snap = trainer.snapshot().model_copy(update={"optimizer_state": None})
    trainer._optimizer_state = None  # reset live instance as well

    snap = snap.model_copy(update={"cvnn": _clone_model(snap.cvnn)})
    restarted = GbmCVNNPricer(snap)

    trainer.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    restarted.train(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)

    assert _max_param_diff(trainer._cvnn, restarted._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 5. Optimiser state round‑trip                                               #
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


# --------------------------------------------------------------------------- #
# 6. Smoke test for predict_price                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_predict_price_smoke(precision: Precision) -> None:
    trainer = _make_gbm_trainer(precision, seed=60)
    trainer.train(num_batches=1, batch_size=4, learning_rate=LEARNING_RATE)

    contracts: Sequence[BlackScholes.Inputs] = [
        BlackScholes.Inputs(X0=100.0, K=100.0, T=1.0, r=0.05, d=0.02, v=0.20),
        BlackScholes.Inputs(X0=120.0, K=110.0, T=0.5, r=0.03, d=0.01, v=0.25),
    ]

    results = trainer.predict_price(contracts)

    assert len(results) == len(contracts)
    for res in results:
        for value in res.model_dump(mode="python").values():
            assert isinstance(value, float) and math.isfinite(value)
