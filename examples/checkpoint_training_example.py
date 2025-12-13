#!/usr/bin/env python3
"""
Example: Training with blockchain checkpoint storage.

Demonstrates the complete workflow:
1. Train a model
2. Take a snapshot
3. Commit to blockchain storage
4. Load checkpoint back
5. Verify reproducibility
"""

from __future__ import annotations

import asyncio
import torch
from spectralmc.runtime import get_torch_handle
from spectralmc.models.torch import AdamOptimizerState, AdamParamState, AdamParamGroup
from spectralmc.gbm import build_black_scholes_config, build_simulation_params
from spectralmc.gbm_trainer import GbmCVNNPricerConfig
from spectralmc.models.numerical import Precision
from spectralmc.result import Success, Failure
from spectralmc.storage import (
    AsyncBlockchainModelStore,
    commit_snapshot,
    load_snapshot_from_checkpoint,
)


get_torch_handle()


async def main() -> None:
    """Demonstrate checkpoint training workflow."""
    print("=== Training with Blockchain Checkpoint Storage ===")

    print("\n1. Creating Model and Training State")

    # Create a simple model (in practice, this would be a CVNN)
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10),
    )

    # Simulate training by modifying weights
    print("   Simulating training...")
    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.randn_like(param)

    # Create optimizer state (in practice, from actual Adam optimizer)
    param_states = {
        0: AdamParamState.from_torch(
            {
                "step": 100,
                "exp_avg": torch.randn(20, 10),
                "exp_avg_sq": torch.randn(20, 10),
            }
        ),
    }

    param_groups = [
        AdamParamGroup(
            params=[0],
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
    ]

    optimizer_state = AdamOptimizerState(
        param_states=param_states, param_groups=param_groups
    )

    # Create simulation config (required for GbmCVNNPricerConfig)
    match build_simulation_params(
        timesteps=100,
        network_size=1024,
        batches_per_mc_run=8,
        threads_per_block=256,
        mc_seed=42,
        buffer_size=10000,
        skip=0,
        dtype=Precision.float32,
    ):
        case Failure(err):
            print(f"   ERROR: SimulationParams failed: {err}")
            return
        case Success(sim_params):
            pass

    match build_black_scholes_config(
        sim_params=sim_params,
        simulate_log_return=True,
        normalize_forwards=True,
    ):
        case Failure(err):
            print(f"   ERROR: BlackScholesConfig failed: {err}")
            return
        case Success(bs_config):
            pass

    # Capture RNG state for reproducibility
    cpu_rng_state = torch.get_rng_state().numpy().tobytes()

    # Create snapshot (in practice, from GbmCVNNPricer.snapshot())
    snapshot = GbmCVNNPricerConfig(
        cfg=bs_config,
        domain_bounds={},
        cvnn=model,
        optimizer_state=optimizer_state,
        global_step=100,
        sobol_skip=0,
        torch_cpu_rng_state=cpu_rng_state,
        torch_cuda_rng_states=[],
    )

    print(
        "   Model created with {} parameters".format(
            sum(p.numel() for p in model.parameters())
        )
    )

    print("\n2. Committing to Blockchain Storage")

    # Create async blockchain store (uses MinIO/S3)
    async with AsyncBlockchainModelStore("opt-models") as store:
        print(f"   Storage bucket: opt-models")

        # Commit snapshot
        version = await commit_snapshot(
            store, snapshot, message="Training checkpoint at step 100"
        )

        print(f"   ✓ Committed: {version.version_id}")
        print(f"   Semantic version: {version.semantic_version}")
        print(f"   Content hash: {version.content_hash[:16]}...")
        print(f"   Message: {version.commit_message}")

        print("\n3. Verifying Blockchain Integrity")

        head_result = await store.get_head()
        match head_result:
            case Success(head):
                print(f"   HEAD points to: {head.version_id}")
                print(f"   Parent hash: {head.parent_hash or '(genesis)'}")

                # Compute version hash for integrity
                version_hash = version.compute_hash()
                print(f"   Version hash: {version_hash[:16]}...")

                print("\n4. Simulating More Training")

                # Save original state for later verification
                original_state_dict = {
                    k: v.clone() for k, v in model.state_dict().items()
                }

                # Simulate more training
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(0.1)

                snapshot2 = GbmCVNNPricerConfig(
                    cfg=bs_config,
                    domain_bounds={},
                    cvnn=model,
                    optimizer_state=optimizer_state,
                    global_step=200,
                    sobol_skip=0,
                    torch_cpu_rng_state=torch.get_rng_state().numpy().tobytes(),
                    torch_cuda_rng_states=[],
                )

                version2 = await commit_snapshot(
                    store, snapshot2, "Training checkpoint at step 200"
                )
                print(f"   ✓ Committed: {version2.version_id}")
                print(f"   Semantic version: {version2.semantic_version}")

                print("\n5. Blockchain Chain Structure")
                print(f"   v1: {version.version_id} (parent: genesis)")
                print(
                    f"   v2: {version2.version_id} (parent: {version2.parent_hash[:8]}...)"
                )
                print(f"   Chain valid: {version2.parent_hash == version.content_hash}")

                print("\n6. Loading Previous Checkpoint")

                # Load first checkpoint back
                new_model = torch.nn.Sequential(
                    torch.nn.Linear(10, 20),
                    torch.nn.ReLU(),
                    torch.nn.Linear(20, 10),
                )

                loaded_snapshot = await load_snapshot_from_checkpoint(
                    store, version, new_model, snapshot
                )

                print(f"   Loaded version: {version.version_id}")
                print(f"   Global step: {loaded_snapshot.global_step}")

                # Verify parameters match the original checkpoint (before modification in step 4)
                loaded_state_dict = loaded_snapshot.cvnn.state_dict()

                all_match = all(
                    torch.allclose(
                        original_state_dict[key],
                        loaded_state_dict[key],
                        rtol=1e-6,
                        atol=1e-9,
                    )
                    for key in original_state_dict.keys()
                )

                print(f"   Parameters match original checkpoint: {all_match}")

                # Show that current model is different (modified in step 4)
                current_state_dict = model.state_dict()
                current_different = not all(
                    torch.allclose(
                        original_state_dict[key],
                        current_state_dict[key],
                        rtol=1e-6,
                        atol=1e-9,
                    )
                    for key in original_state_dict.keys()
                )
                print(f"   Current model differs from checkpoint: {current_different}")
            case Failure(error):
                print(f"   Error retrieving HEAD: {error}")
                print("Cannot verify blockchain integrity. Aborting.")
                return

    print("\n✓ Example complete!")


if __name__ == "__main__":
    asyncio.run(main())
