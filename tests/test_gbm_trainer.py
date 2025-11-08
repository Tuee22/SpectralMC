# tests/test_gbm_trainer.py
"""
Determinism & serialisation test‑suite for
:class:`spectralmc.gbm_trainer.GbmCVNNPricer`.

The suite covers:

1. deterministic network construction,
2. lock‑step training determinism,
3. snapshot/restore reproducibility,
4. restart from a snapshot *without* optimiser state,
5. SafeTensor ⇄ JSON round‑trip for the optimiser, and
6. an end‑to‑end smoke test for
   :pyfunc:`spectralmc.gbm_trainer.GbmCVNNPricer.predict_price`.

All tests pass under **mypy --strict** with zero ignores.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Sequence, Tuple, Union

import pytest

from spectralmc.models.numerical import Precision
from spectralmc.models.torch import DType as TorchDTypeEnum
from spectralmc.cvnn_factory import (
    ActivationCfg,
    ActivationKind,
    CVNNConfig,
    ExplicitWidth,
    LinearCfg,
    build_model,
)
from spectralmc.gbm import BlackScholes, BlackScholesConfig, SimulationParams
from spectralmc.gbm_trainer import (
    ComplexValuedModel,
    GbmCVNNPricer,
    GbmCVNNPricerConfig,
    TrainingConfig,
)
from spectralmc.models.torch import AdamOptimizerState
from spectralmc.sobol_sampler import BoundSpec

import torch
from torch import Tensor

# --------------------------------------------------------------------------- #
# Patch forward reference in SimulationParams                                 #
# --------------------------------------------------------------------------- #
from typing import Literal as _Lit

import spectralmc.gbm as _gbm_mod

setattr(_gbm_mod, "DtypeLiteral", _Lit["float32", "float64"])
SimulationParams.model_rebuild()

# --------------------------------------------------------------------------- #
# Global constants                                                            #
# --------------------------------------------------------------------------- #

PRECISIONS: Tuple[Precision, Precision] = (Precision.float32, Precision.float64)
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
    """Return the L∞‑norm between two models’ parameters."""
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
    enum_dtype = TorchDTypeEnum.from_torch(dtype)
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
    return build_model(n_inputs=n_inputs, n_outputs=n_outputs, cfg=cfg).to(
        device, dtype
    )


def _make_gbm_trainer(precision: Precision, *, seed: int) -> GbmCVNNPricer:
    """Deterministically construct a :class:`GbmCVNNPricer`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float32 if precision is Precision.float32 else torch.float64
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

    net = _make_cvnn(6, sim.network_size, seed=seed, device=device, dtype=torch_dtype)
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
        cfg = TrainingConfig(
            num_batches=batches, batch_size=8, learning_rate=LEARNING_RATE
        )
        first.train(cfg)
        second.train(cfg)
        assert _max_param_diff(first._cvnn, second._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 3. Snapshot / restore determinism                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_cycle_deterministic(precision: Precision) -> None:
    trainer = _make_gbm_trainer(precision, seed=44)
    trainer.train(
        TrainingConfig(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)
    )

    snap = trainer.snapshot().model_copy(update={"cvnn": _clone_model(trainer._cvnn)})
    clone = GbmCVNNPricer(snap)

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
    trainer.train(
        TrainingConfig(num_batches=3, batch_size=8, learning_rate=LEARNING_RATE)
    )

    snap = trainer.snapshot().model_copy(
        update={"optimizer_state": None, "cvnn": _clone_model(trainer._cvnn)}
    )
    restarted = GbmCVNNPricer(snap)

    # Reset original trainer's optimizer state to match restarted trainer
    trainer._optimizer_state = None

    cfg = TrainingConfig(num_batches=2, batch_size=8, learning_rate=LEARNING_RATE)
    trainer.train(cfg)
    restarted.train(cfg)

    assert _max_param_diff(trainer._cvnn, restarted._cvnn) == 0.0


# --------------------------------------------------------------------------- #
# 5. Optimiser state round‑trip                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_snapshot_optimizer_serialization_roundtrip(precision: Precision) -> None:
    trainer = _make_gbm_trainer(precision, seed=50)
    trainer.train(
        TrainingConfig(num_batches=4, batch_size=8, learning_rate=LEARNING_RATE)
    )
    snap = trainer.snapshot()

    opt_state = snap.optimizer_state
    assert opt_state is not None

    round_trip = AdamOptimizerState.model_validate(opt_state.model_dump(mode="python"))
    torch_state = opt_state.to_torch()
    reloaded_state = round_trip.to_torch()

    for key in ("state", "param_groups"):
        assert key in torch_state and key in reloaded_state
        assert _tree_equal(torch_state[key], reloaded_state[key])


# --------------------------------------------------------------------------- #
# 6. Smoke test for predict_price                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", PRECISIONS)
def test_predict_price_smoke(precision: Precision) -> None:
    trainer = _make_gbm_trainer(precision, seed=60)
    trainer.train(
        TrainingConfig(num_batches=1, batch_size=4, learning_rate=LEARNING_RATE)
    )

    contracts: Sequence[BlackScholes.Inputs] = [
        BlackScholes.Inputs(X0=100.0, K=100.0, T=1.0, r=0.05, d=0.02, v=0.20),
        BlackScholes.Inputs(X0=120.0, K=110.0, T=0.5, r=0.03, d=0.01, v=0.25),
    ]
    results = trainer.predict_price(contracts)

    assert len(results) == len(contracts)
    for res in results:
        for val in res.model_dump(mode="python").values():
            assert isinstance(val, float) and math.isfinite(val)
