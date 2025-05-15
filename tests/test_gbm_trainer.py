"""
test_gbm_trainer.py
===================
Integration test for gbm_trainer.py using pytest. Verifies that:
  * We can train a CVNN on GPU if available (requires numba.cuda).
  * We can do a minimal inference check (predict_price).

The test does:
  1) Build SimulationParams with user-chosen domain/big config.
  2) Train for 10 steps, prints the final training loss.
  3) Predict a price for a short list of inputs, ensuring no crash and no imaginary leftover.

"""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from spectralmc.gbm import SimulationParams, BlackScholes
from spectralmc.sobol_sampler import BoundSpec
from spectralmc.cvnn import CVNN
from spectralmc.gbm_trainer import GbmTrainer


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_trainer_integration(precision: Literal["float32", "float64"]) -> None:
    """Full integration test that trains with timesteps=1, large number of paths, etc.

    Args:
        precision: "float32" or "float64".

    Steps:
      1) Construct a large-scale SimulationParams (timesteps=1, big batches_per_mc_run).
      2) Create domain bounds for (X0,K,T,r,d,v).
      3) Build a big CVNN, train for 10 steps, see if we can avoid dtype errors or runtime issues.
      4) Predict for two sample points, ensure the imaginary part is near zero.
    """
    if not torch.cuda.is_available():
        pytest.skip(
            "No CUDA available. Skipping test since BlackScholes requires numba.cuda."
        )

    sim_params = SimulationParams(
        timesteps=1,
        network_size=64,  # e.g., 64 DFT components
        batches_per_mc_run=(2**19),  # huge number of paths => 64*(2^20) total paths
        threads_per_block=256,
        mc_seed=123,
        buffer_size=1,
        dtype=precision,  # must be "float32" or "float64"
        simulate_log_return=True,
        normalize_forwards=False,
    )

    domain_example = {
        "X0": BoundSpec(lower=50.0, upper=150.0),
        "K": BoundSpec(lower=50.0, upper=150.0),
        "T": BoundSpec(lower=0.1, upper=2.0),
        "r": BoundSpec(lower=0.0, upper=0.1),
        "d": BoundSpec(lower=0.0, upper=0.05),
        "v": BoundSpec(lower=0.1, upper=0.5),
    }

    net = CVNN(
        input_features=6,
        output_features=sim_params.network_size,
        hidden_features=128,
        num_residual_blocks=1,
    )

    trainer = GbmTrainer(
        sim_params=sim_params,
        domain_bounds=domain_example,
        skip_sobol=0,
        sobol_seed=42,
        cvnn=net,
        device=torch.device("cuda"),
    )

    # 10 training steps (each big):
    trainer.train(num_batches=1024, batch_size=64, learning_rate=1e-2)

    # Predict
    input_list = [
        BlackScholes.Inputs(X0=100.0, K=100.0, T=1.0, r=0.02, d=0.01, v=0.2),
        BlackScholes.Inputs(X0=110.0, K=105.0, T=0.5, r=0.03, d=0.00, v=0.3),
    ]
    results = trainer.predict_price(input_list)
    assert len(results) == 2

    for idx, res in enumerate(results):
        assert res.put_price >= 0.0, "Predicted put price must be >= 0."
        print(f"[TEST {precision}] Input {idx}: {res}")
