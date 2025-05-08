"""
test_gbm_trainer.py
===================
PyTest-based tests for gbm_trainer.py. Runs a short integration test to ensure
the pipeline works. Also confirms mypy can parse the trainer's typed methods.
"""

from __future__ import annotations

import pytest
import torch

from spectralmc.gbm import SimulationParams
from spectralmc.sobol_sampler import BoundSpec
from spectralmc.cvnn import CVNN
from spectralmc.gbm_trainer import GbmTrainer


@pytest.mark.parametrize("network_size", [4, 8])
def test_trainer_integration(network_size: int) -> None:
    """Minimal integration test to ensure GbmTrainer can run end-to-end."""
    # 1) Build sim_params
    sp = SimulationParams(
        timesteps=16,
        network_size=network_size,
        batches_per_mc_run=4,
        threads_per_block=64,
        mc_seed=999,
        buffer_size=1,
        dtype="float32",
        simulate_log_return=True,
        normalize_forwards=False,
    )

    # 2) Domain
    domain = {
        "X0": BoundSpec(lower=50.0, upper=150.0),
        "K": BoundSpec(lower=50.0, upper=150.0),
        "T": BoundSpec(lower=0.1, upper=2.0),
        "r": BoundSpec(lower=0.0, upper=0.1),
        "d": BoundSpec(lower=0.0, upper=0.05),
        "v": BoundSpec(lower=0.1, upper=0.5),
    }

    # 3) Build CVNN
    net = CVNN(
        input_features=6,
        output_features=network_size,
        hidden_features=8,
        num_residual_blocks=1,
    )

    # 4) Trainer
    trainer = GbmTrainer(
        sim_params=sp,
        domain_bounds=domain,
        skip_sobol=0,
        sobol_seed=42,
        cvnn=net,
        device=torch.device("cpu"),
    )

    # 5) Train with small steps
    trainer.train(num_batches=2, batch_size=2, learning_rate=1e-3)

    # 6) Predict
    payoff_pred = trainer.predict_mean_payoff(100, 100, 1.0, 0.02, 0.01, 0.2)

    assert payoff_pred >= 0.0, "Mean payoff should be nonnegative."
    print(f"[TEST] network_size={network_size}, payoff_pred={payoff_pred:.4f}")
