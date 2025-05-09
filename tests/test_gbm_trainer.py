"""
test_gbm_trainer.py
===================
Uses pytest to confirm that gbm_trainer.py integrates with:
  * SobolSampler
  * BlackScholes
  * CVNN

We do a short run to ensure training does not crash, and do a minimal
inference check. The type-checking is verified via mypy --strict.
"""

from __future__ import annotations

import pytest
import torch

from spectralmc.gbm import SimulationParams, BlackScholes
from spectralmc.sobol_sampler import BoundSpec
from spectralmc.cvnn import CVNN
from spectralmc.gbm_trainer import GbmTrainer


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_trainer_integration(precision: str) -> None:
    """
    End-to-end integration test with small config:
      1. Construct SimulationParams using either float32 or float64.
      2. Create domain bounds for (X0,K,T,r,d,v).
      3. Build a minimal CVNN (hidden_features=8, say).
      4. Train for a few steps, ensuring GPU usage if available.
      5. Predict a price for a simple input or two, verifying no crash.

    If device is CPU-only, we skip the actual training to avoid errors (since numba.cuda is needed).
    """

    # If no CUDA is available, skip test or we risk a crash
    if not torch.cuda.is_available():
        pytest.skip(
            "GPU not available, skipping training test (numba.cuda requirement)."
        )

    sim_params = SimulationParams(
        timesteps=1,
        network_size=128,
        batches_per_mc_run=2**21,
        threads_per_block=256,
        mc_seed=123,
        buffer_size=4,
        dtype=precision,
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

    # Basic CVNN
    net = CVNN(
        input_features=6,
        output_features=sim_params.network_size,
        hidden_features=8,
        num_residual_blocks=1,
    )

    trainer = GbmTrainer(
        sim_params=sim_params,
        domain_bounds=domain_example,
        skip_sobol=0,
        sobol_seed=42,
        cvnn=net,
        device=torch.device("cuda"),  # Force GPU usage
    )

    # Train briefly
    trainer.train(num_batches=512, batch_size=128, learning_rate=1e-3)

    # Predict
    # Construct a list of BlackScholes.Inputs for inference
    input_list = [
        BlackScholes.Inputs(X0=100.0, K=100.0, T=1.0, r=0.02, d=0.01, v=0.2),
        BlackScholes.Inputs(X0=110.0, K=105.0, T=0.5, r=0.03, d=0.00, v=0.3),
    ]
    results = trainer.predict_price(input_list)

    assert len(results) == 2
    for hpr in results:
        # put_price should be >= 0
        assert hpr.put_price >= 0.0
        # check imaginary parted was near zero in ifft => no error thrown
        # other fields are plausible
        print(f"[TEST] HostPricingResults: {hpr}")
