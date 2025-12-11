# tests/test_gbm_trainer.py
"""
Determinism & serialisation test-suite for
:class:`spectralmc.gbm_trainer.GbmCVNNPricer`.

The suite covers:

1. deterministic network construction,
2. lock-step training determinism,
3. snapshot/restore reproducibility,
4. restart from a snapshot *without* optimiser state,
5. SafeTensor ⇄ JSON round-trip for the optimiser, and
6. an end-to-end smoke test for
   :pyfunc:`spectralmc.gbm_trainer.GbmCVNNPricer.predict_price`.

All tests pass under **mypy --strict** with zero ignores.
"""

from __future__ import annotations

import copy
import math
from typing import Sequence, Union

import pytest
import torch
from torch import Tensor

from spectralmc.cvnn_factory import (
    ActivationCfg,
    ActivationKind,
    CVNNConfig,
    ExplicitWidth,
    LinearCfg,
    build_model,
)
from spectralmc.gbm import BlackScholes
from spectralmc.gbm_trainer import (
    ComplexValuedModel,
    GbmCVNNPricer,
    GbmCVNNPricerConfig,
    TrainingConfig,
)
from spectralmc.models.numerical import Precision
from spectralmc.models.torch import AdamOptimizerState
from spectralmc.models.torch import DType as TorchDTypeEnum
from spectralmc.sobol_sampler import BoundSpec, build_bound_spec
from tests.helpers import expect_success, make_black_scholes_config, make_simulation_params


# Module-level GPU requirement - test file fails immediately without GPU
assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"


# --------------------------------------------------------------------------- #
# Global constants                                                            #
# --------------------------------------------------------------------------- #

PRECISIONS: tuple[Precision, Precision] = (Precision.float32, Precision.float64)
LEARNING_RATE: float = 1.0e-2


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #


def _clone_model(model: ComplexValuedModel) -> ComplexValuedModel:
    """Deep copy *model* onto the same device/dtype."""
    dup = copy.deepcopy(model)
    first = next(iter(model.parameters()))
    dup.to(first.device, first.dtype)
    return dup


def _max_param_diff(a: ComplexValuedModel, b: ComplexValuedModel) -> float:
    """Return the L∞-norm between two models' parameters."""
    return max(
        (
            float(torch.abs(pa - pb).max().item())
            for pa, pb in zip(a.parameters(), b.parameters(), strict=True)
        ),
        default=0.0,
    )


def _tree_equal(x: object, y: object) -> bool:
    """Deep equality for arbitrarily nested containers."""
    if isinstance(x, Tensor) and isinstance(y, Tensor):
        return bool(torch.equal(x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_tree_equal(v, y[k]) for k, v in x.items())
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(_tree_equal(a, b) for a, b in zip(x, y))
    return x == y


def _make_cvnn(
    n_inputs: int,
    n_outputs: int,
    *,
    seed: int,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> ComplexValuedModel:
    """Factory wrapper around :pyfunc:`spectralmc.cvnn_factory.build_model`."""
    enum_dtype = expect_success(TorchDTypeEnum.from_torch(dtype))
    cfg = CVNNConfig(
        dtype=enum_dtype,
        layers=[
            LinearCfg(
                width=ExplicitWidth(value=32),
                activation=ActivationCfg(kind=ActivationKind.MOD_RELU),
            ),
            LinearCfg(width=ExplicitWidth(value=n_outputs)),
        ],
        seed=seed,
    )
    return expect_success(build_model(n_inputs=n_inputs, n_outputs=n_outputs, cfg=cfg)).to(
        device, dtype
    )


def _make_gbm_trainer(precision: Precision, *, seed: int) -> GbmCVNNPricer:
    """Deterministically construct a :class:`GbmCVNNPricer`."""
    device = torch.device("cuda:0")
    torch_dtype = torch.float32 if precision is Precision.float32 else torch.float64
    torch.manual_seed(seed)

    sim = make_simulation_params(
        skip=0,
        timesteps=1,
        network_size=16,
        batches_per_mc_run=2**12,
        threads_per_block=256,
        mc_seed=seed,
        buffer_size=1,
        dtype=precision,
    )

    cfg = make_black_scholes_config(
        sim_params=sim,
        simulate_log_return=True,
        normalize_forwards=False,
    )

    bounds: dict[str, BoundSpec] = {
        "X0": build_bound_spec(50.0, 150.0).unwrap(),
        "K": build_bound_spec(50.0, 150.0).unwrap(),
        "T": build_bound_spec(0.1, 2.0).unwrap(),
        "r": build_bound_spec(0.0, 0.1).unwrap(),
        "d": build_bound_spec(0.0, 0.05).unwrap(),
        "v": build_bound_spec(0.1, 0.5).unwrap(),
    }

    net = _make_cvnn(6, sim.network_size, seed=seed, device=device, dtype=torch_dtype)
    return expect_success(
        GbmCVNNPricer.create(GbmCVNNPricerConfig(cfg=cfg, domain_bounds=bounds, cvnn=net))
    )


# --------------------------------------------------------------------------- #
# 1. Deterministic construction                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_deterministic_construction(precision: Precision) -> None:
    first = _make_gbm_trainer(precision, seed=42)
    second = _make_gbm_trainer(precision, seed=42)
    assert _max_param_diff(first._cvnn, second._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 2. Lock-step training determinism                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_lockstep_training(precision: Precision) -> None:
    first = _make_gbm_trainer(precision, seed=43)
    second = _make_gbm_trainer(precision, seed=43)

    for batches in (2, 3, 1):
        cfg = TrainingConfig(num_batches=batches, batch_size=8, learning_rate=LEARNING_RATE)
        first.train(cfg)
        second.train(cfg)
        assert _max_param_diff(first._cvnn, second._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 3. Snapshot / restore determinism                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_cycle_deterministic(precision: Precision) -> None:
    trainer = _make_gbm_trainer(precision, seed=44)
    trainer.train(TrainingConfig(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE))

    snap = expect_success(trainer.snapshot()).model_copy(
        update={"cvnn": _clone_model(trainer._cvnn)}
    )
    clone = expect_success(GbmCVNNPricer.create(snap))

    cfg = TrainingConfig(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    trainer.train(cfg)
    clone.train(cfg)

    assert _max_param_diff(trainer._cvnn, clone._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 4. Snapshot restart *without* optimiser                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_restart_without_optimizer(precision: Precision) -> None:
    trainer = _make_gbm_trainer(precision, seed=45)
    trainer.train(TrainingConfig(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE))

    snap = expect_success(trainer.snapshot()).model_copy(
        update={"optimizer_state": None, "cvnn": _clone_model(trainer._cvnn)}
    )
    restarted = expect_success(GbmCVNNPricer.create(snap))

    # Create a fresh trainer without optimizer state for comparison
    # (same model state as restarted, but created fresh)
    trainer_without_opt = _make_gbm_trainer(precision, seed=45)
    trainer_without_opt.train(
        TrainingConfig(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)
    )
    # Create fresh trainer from same snapshot (no optimizer state)
    snap_for_comparison = expect_success(trainer_without_opt.snapshot()).model_copy(
        update={"optimizer_state": None, "cvnn": _clone_model(trainer_without_opt._cvnn)}
    )
    trainer_without_opt = expect_success(GbmCVNNPricer.create(snap_for_comparison))

    cfg = TrainingConfig(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    trainer_without_opt.train(cfg)
    restarted.train(cfg)

    assert _max_param_diff(trainer_without_opt._cvnn, restarted._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 5. Optimiser state round-trip                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_optimizer_serialization_roundtrip(precision: Precision) -> None:
    trainer = _make_gbm_trainer(precision, seed=50)
    trainer.train(TrainingConfig(num_batches=4, batch_size=8, learning_rate=LEARNING_RATE))
    snap = expect_success(trainer.snapshot())

    opt_state = snap.optimizer_state
    assert opt_state is not None, "Optimizer state should be preserved in snapshot after training"
    assert isinstance(opt_state, AdamOptimizerState), "Optimizer state should be AdamOptimizerState"
    # Validate structure - param_states dict should have entries for trained parameters
    assert len(opt_state.param_states) > 0, "Optimizer state should track parameters"
    assert len(opt_state.param_groups) > 0, "Optimizer state should have parameter groups"

    round_trip = AdamOptimizerState.model_validate(opt_state.model_dump(mode="python"))
    torch_state = expect_success(opt_state.to_torch())
    reloaded_state = expect_success(round_trip.to_torch())

    for key in ("state", "param_groups"):
        assert key in torch_state and key in reloaded_state
        assert _tree_equal(torch_state[key], reloaded_state[key])


# --------------------------------------------------------------------------- #
# 6. Smoke test for predict_price                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_predict_price_smoke(precision: Precision) -> None:
    trainer = _make_gbm_trainer(precision, seed=60)
    trainer.train(TrainingConfig(num_batches=1, batch_size=4, learning_rate=LEARNING_RATE))

    contracts: Sequence[BlackScholes.Inputs] = [
        BlackScholes.Inputs(X0=100.0, K=100.0, T=1.0, r=0.05, d=0.02, v=0.20),
        BlackScholes.Inputs(X0=120.0, K=110.0, T=0.5, r=0.03, d=0.01, v=0.25),
    ]
    results = expect_success(trainer.predict_price(contracts))

    assert len(results) == len(contracts)
    for res in results:
        for val in res.model_dump(mode="python").values():
            assert isinstance(val, float) and math.isfinite(val)
