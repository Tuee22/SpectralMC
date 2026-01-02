"""Guided full-stack test for GbmCVNNPricer.

Reading this test in order walks through the major subsystems:
- config building (Sobol sampler + GBM + domain bounds)
- GPU-only training loop (Sobol → GBM CUDA kernel → cuFFT → CVNN optimizer)
- blockchain checkpoint commit/reload
- deterministic inference from the reloaded snapshot
"""

from __future__ import annotations

import math

import pytest
import torch

from spectralmc.gbm import BlackScholes
from spectralmc.gbm_trainer import (
    ComplexValuedModel,
    GbmCVNNPricer,
    GbmCVNNPricerConfig,
)
from spectralmc.models.numerical import Precision
from spectralmc.storage import (
    AsyncBlockchainModelStore,
    commit_snapshot,
    load_snapshot_from_checkpoint,
)
from tests.helpers import (
    expect_success,
    make_black_scholes_config,
    make_domain_bounds,
    make_gbm_cvnn_config,
    make_simulation_params,
    make_test_cvnn,
    make_training_config,
    seed_all_rngs,
)

DOMAIN_BOUNDS = make_domain_bounds()
SIM_PARAMS = make_simulation_params(
    timesteps=16,
    network_size=128,
    batches_per_mc_run=4,
    threads_per_block=256,
    mc_seed=7,
    buffer_size=512,
    skip=0,
    dtype=Precision.float32,
)
BS_CONFIG = make_black_scholes_config(sim_params=SIM_PARAMS)


def _make_pricer_config(model: ComplexValuedModel, *, global_step: int = 0) -> GbmCVNNPricerConfig:
    """Standardized pricer config for this guided walkthrough."""
    return make_gbm_cvnn_config(
        model,
        global_step=global_step,
        sim_params=SIM_PARAMS,
        bs_config=BS_CONFIG,
        domain_bounds=DOMAIN_BOUNDS,
    )


@pytest.mark.asyncio
async def test_full_stack_cvnn_pricer_workflow(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Walk the entire pipeline: config → training → checkpoint → reload → inference."""
    seed_all_rngs(123)

    # Build deterministic CVNN and pricer config (touches model factory + config builders)
    model = make_test_cvnn(
        n_inputs=6,
        n_outputs=SIM_PARAMS.network_size,
        seed=123,
        dtype=torch.float32,
    )
    config = _make_pricer_config(model)
    pricer = expect_success(GbmCVNNPricer.create(config))

    training_config = make_training_config(num_batches=4, batch_size=8, learning_rate=1.0e-2)

    # Train through Sobol sampling → GBM CUDA kernel → cuFFT → CVNN backward pass
    train_result = expect_success(pricer.train(training_config))
    assert train_result.total_batches == training_config.num_batches

    # Persist checkpoint to blockchain storage (covers serialization + chain metadata)
    snapshot = train_result.updated_config
    version = await commit_snapshot(
        async_store,
        snapshot,
        "full-stack demo checkpoint",
    )
    assert version.counter == 0
    assert version.commit_message == "full-stack demo checkpoint"

    # Reload snapshot from blockchain into a fresh CVNN template
    cvnn_template = make_test_cvnn(
        n_inputs=6,
        n_outputs=SIM_PARAMS.network_size,
        seed=999,  # Different seed to prove weights come from checkpoint
        dtype=torch.float32,
    )
    reloaded_snapshot = expect_success(
        await load_snapshot_from_checkpoint(
            async_store,
            version,
            cvnn_template,
            train_result.updated_config,
        )
    )

    # Run inference end-to-end on the reloaded pricer
    loaded_pricer = expect_success(GbmCVNNPricer.create(reloaded_snapshot))
    contracts = [
        BlackScholes.Inputs(X0=100.0, K=95.0, T=0.5, r=0.03, d=0.01, v=0.25),
        BlackScholes.Inputs(X0=120.0, K=105.0, T=1.0, r=0.02, d=0.00, v=0.30),
    ]
    results = expect_success(loaded_pricer.predict_price(contracts))

    assert len(results) == len(contracts)
    for result in results:
        values = result.model_dump(mode="python").values()
        assert all(math.isfinite(val) for val in values)
