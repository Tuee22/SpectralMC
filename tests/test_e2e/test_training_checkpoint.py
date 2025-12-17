# tests/test_e2e/test_training_checkpoint.py
"""End-to-end tests for training with blockchain checkpoint storage."""

from __future__ import annotations

import pytest

from spectralmc.gbm import BlackScholesConfig
from spectralmc.models.numerical import Precision
from spectralmc.models.torch import AdamOptimizerState, AdamParamGroup, AdamParamState
from spectralmc.storage import (
    AsyncBlockchainModelStore,
    commit_snapshot,
    load_snapshot_from_checkpoint,
)
import torch
from tests.helpers import (
    expect_success,
    make_black_scholes_config,
    make_domain_bounds,
    make_gbm_cvnn_config,
    make_simulation_params,
    seed_all_rngs,
)

DOMAIN_BOUNDS = make_domain_bounds()


def _make_checkpoint_black_scholes_config() -> BlackScholesConfig:
    """Create the shared Black-Scholes config used across these tests."""
    sim_params = make_simulation_params(
        timesteps=100,
        network_size=1024,
        batches_per_mc_run=8,
        threads_per_block=256,
        mc_seed=42,
        buffer_size=10000,
        skip=0,
        dtype=Precision.float32,
    )

    return make_black_scholes_config(sim_params=sim_params)


@pytest.mark.asyncio
async def test_checkpoint_simple_model(async_store: AsyncBlockchainModelStore) -> None:
    """Test creating and loading a checkpoint with a simple model."""
    seed_all_rngs(42)
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 5),
    )

    # Create optimizer state
    param_state_0 = expect_success(
        AdamParamState.from_torch(
            {
                "step": 10,
                "exp_avg": torch.randn(10, 5),
                "exp_avg_sq": torch.randn(10, 5),
            }
        )
    )

    param_states = {0: param_state_0}

    param_groups = (
        AdamParamGroup(
            params=[0],
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        ),
    )

    optimizer_state = AdamOptimizerState(param_states=param_states, param_groups=param_groups)

    # Create a minimal GbmCVNNPricerConfig
    # Note: This is a simplified test, real usage would have proper BlackScholes config
    bs_config = _make_checkpoint_black_scholes_config()

    snapshot = make_gbm_cvnn_config(
        model,
        global_step=100,
        bs_config=bs_config,
        domain_bounds=DOMAIN_BOUNDS,
        optimizer_state=optimizer_state,
    )

    # Commit snapshot (async)
    version = await commit_snapshot(async_store, snapshot, "Test checkpoint")

    assert version.counter == 0
    assert version.semantic_version == "1.0.0"
    assert version.commit_message == "Test checkpoint"

    # Verify checkpoint was stored
    head_result = await async_store.get_head()
    head = expect_success(head_result)
    assert head.counter == version.counter

    # Load checkpoint back
    new_model = torch.nn.Sequential(
        torch.nn.Linear(5, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 5),
    )

    loaded_snapshot = expect_success(
        await load_snapshot_from_checkpoint(async_store, version, new_model, snapshot)
    )

    # Verify model parameters are identical
    original_state = model.state_dict()
    loaded_state = loaded_snapshot.cvnn.state_dict()

    assert set(original_state.keys()) == set(loaded_state.keys())
    for key in original_state:
        assert torch.allclose(
            original_state[key], loaded_state[key], rtol=1e-6, atol=1e-9
        ), f"Parameter {key} mismatch"

    # Verify optimizer state
    assert loaded_snapshot.optimizer_state is not None, "Loaded optimizer state should be preserved"
    assert isinstance(
        loaded_snapshot.optimizer_state, type(optimizer_state)
    ), "Optimizer state type should match"
    assert len(loaded_snapshot.optimizer_state.param_states) == len(
        optimizer_state.param_states
    ), "Optimizer param states count should match"

    # Verify global step and RNG state
    assert loaded_snapshot.global_step == 100
    assert loaded_snapshot.torch_cpu_rng_state == snapshot.torch_cpu_rng_state


@pytest.mark.asyncio
async def test_checkpoint_multiple_commits(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test multiple sequential commits create proper chain."""
    seed_all_rngs(42)
    model = torch.nn.Linear(10, 10)

    bs_config = _make_checkpoint_black_scholes_config()

    versions = []

    # Create 5 checkpoints
    for i in range(5):
        # Modify model slightly
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.1)

        snapshot = make_gbm_cvnn_config(
            model,
            global_step=i * 100,
            bs_config=bs_config,
            domain_bounds=DOMAIN_BOUNDS,
            sobol_skip=0,
        )

        version = await commit_snapshot(async_store, snapshot, f"Epoch {i}")
        versions.append(version)

    # Verify chain structure
    for i, version in enumerate(versions):
        assert version.counter == i
        assert version.semantic_version == f"1.0.{i}"

        if i == 0:
            assert version.parent_hash == ""
        else:
            assert version.parent_hash == versions[i - 1].content_hash

    # Verify HEAD
    head_result = await async_store.get_head()
    head = expect_success(head_result)
    assert head.counter == 4
    assert head.commit_message == "Epoch 4"


@pytest.mark.asyncio
async def test_checkpoint_content_hash_integrity(
    async_store: AsyncBlockchainModelStore,
) -> None:
    """Test that different checkpoints have different content hashes."""
    seed_all_rngs(42)
    model1 = torch.nn.Linear(5, 5)
    model2 = torch.nn.Linear(5, 5)

    # Initialize with different values
    seed_all_rngs(42)
    model1.weight.data = torch.randn(5, 5)

    seed_all_rngs(123)
    model2.weight.data = torch.randn(5, 5)

    bs_config = _make_checkpoint_black_scholes_config()

    snapshot1 = make_gbm_cvnn_config(
        model1,
        global_step=0,
        bs_config=bs_config,
        domain_bounds=DOMAIN_BOUNDS,
        sobol_skip=0,
    )

    snapshot2 = make_gbm_cvnn_config(
        model2,
        global_step=0,
        bs_config=bs_config,
        domain_bounds=DOMAIN_BOUNDS,
        sobol_skip=0,
    )

    version1 = await commit_snapshot(async_store, snapshot1, "Model 1")
    version2 = await commit_snapshot(async_store, snapshot2, "Model 2")

    # Different models should have different content hashes
    assert version1.content_hash != version2.content_hash
