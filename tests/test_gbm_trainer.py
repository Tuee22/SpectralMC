# tests/spectralmc/test_gbm_trainer.py
"""
Integration test for gbm_trainer.py.

* Builds a large‑scale SimulationParams instance.
* Trains for a limited number of steps on GPU.
* Runs a minimal prediction sanity‑check.
"""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from spectralmc.gbm import BlackScholes, BlackScholesConfig, SimulationParams
from spectralmc.sobol_sampler import BoundSpec
from spectralmc.cvnn import CVNN
from spectralmc.gbm_trainer import GbmTrainer


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_trainer_integration(precision: Literal["float32", "float64"]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Black‑Scholes Monte‑Carlo engine.")

    # ------------------------------------------------------------------ #
    # 1) Simulation parameters & engine config                           #
    # ------------------------------------------------------------------ #
    sim_params = SimulationParams(
        skip=0,
        timesteps=1,
        network_size=64,
        batches_per_mc_run=2**20,  # 64 × 2²⁰ total paths
        threads_per_block=256,
        mc_seed=123,
        buffer_size=1,
        dtype=precision,
    )

    bs_cfg = BlackScholesConfig(
        sim_params=sim_params,
        simulate_log_return=True,
        normalize_forwards=False,
    )

    # ------------------------------------------------------------------ #
    # 2) Sobol domain bounds                                             #
    # ------------------------------------------------------------------ #
    domain_example = {
        "X0": BoundSpec(lower=50.0, upper=150.0),
        "K": BoundSpec(lower=50.0, upper=150.0),
        "T": BoundSpec(lower=0.1, upper=2.0),
        "r": BoundSpec(lower=0.0, upper=0.1),
        "d": BoundSpec(lower=0.0, upper=0.05),
        "v": BoundSpec(lower=0.1, upper=0.5),
    }

    # ------------------------------------------------------------------ #
    # 3) CVNN model & trainer                                            #
    # ------------------------------------------------------------------ #
    net = CVNN(
        input_features=6,
        output_features=sim_params.network_size,
        hidden_features=128,
        num_residual_blocks=1,
    )

    trainer = GbmTrainer(
        cfg=bs_cfg,
        domain_bounds=domain_example,
        cvnn=net,
    )

    # Train for a handful of steps to catch regressions -----------------
    trainer.train(num_batches=64, batch_size=64, learning_rate=1e-2)

    # ------------------------------------------------------------------ #
    # 4) Simple prediction smoke‑test                                    #
    # ------------------------------------------------------------------ #
    input_list = [
        BlackScholes.Inputs(X0=100.0, K=100.0, T=1.0, r=0.02, d=0.01, v=0.2),
        BlackScholes.Inputs(X0=110.0, K=105.0, T=0.5, r=0.03, d=0.00, v=0.3),
    ]
    results = trainer.predict_price(input_list)
    assert len(results) == 2

    for idx, res in enumerate(results):
        assert res.put_price >= 0.0, "Predicted put price must be non‑negative."
        print(f"[{precision}] contract {idx}: {res}")
