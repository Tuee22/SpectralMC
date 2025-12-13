# examples/training_with_blockchain_storage.py
"""
Complete example: Train GBM model with automatic blockchain commits.

This example demonstrates:
1. Training a GbmCVNNPricer with automatic blockchain commits
2. Both periodic commits (every N batches) and final commit after training
3. Loading the trained model from blockchain storage
4. Running inference with the loaded model

Usage:
    python examples/training_with_blockchain_storage.py
"""

from __future__ import annotations

import asyncio

import torch
from spectralmc.runtime import get_torch_handle

from spectralmc.gbm import BlackScholes, build_black_scholes_config, build_simulation_params
from spectralmc.gbm_trainer import (
    GbmCVNNPricerConfig,
    GbmCVNNPricer,
    FinalAndIntervalCommit,
    TrainingConfig,
)
from spectralmc.models.numerical import Precision
from spectralmc.models.torch import Device
from spectralmc.result import Success, Failure
from spectralmc.sobol_sampler import BoundSpec, build_bound_spec
from spectralmc.storage import (
    AsyncBlockchainModelStore,
    InferenceClient,
    TrackingMode,
    load_snapshot_from_checkpoint,
)

get_torch_handle()
nn = torch.nn
assert torch.cuda.is_available(), "CUDA required for SpectralMC blockchain training example"


def create_simple_cvnn(
    input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 100
) -> nn.Module:
    """Create a simple complex-valued neural network."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def create_training_config() -> GbmCVNNPricerConfig:
    """Create initial training configuration."""
    # Simulation parameters
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
            raise RuntimeError(f"Failed to build SimulationParams: {err}")
        case Success(sim_params):
            pass

    # Black-Scholes configuration
    match build_black_scholes_config(
        sim_params=sim_params,
        simulate_log_return=True,
        normalize_forwards=True,
    ):
        case Failure(err):
            raise RuntimeError(f"Failed to build BlackScholesConfig: {err}")
        case Success(bs_config):
            pass

    # Create model
    model = create_simple_cvnn().to(device=Device.cuda.to_torch())

    # Domain bounds for option parameters
    domain_bounds = {
        "X0": build_bound_spec(80.0, 120.0).unwrap(),
        "K": build_bound_spec(80.0, 120.0).unwrap(),
        "T": build_bound_spec(0.1, 2.0).unwrap(),
        "v": build_bound_spec(0.1, 0.5).unwrap(),
        "r": build_bound_spec(0.0, 0.1).unwrap(),
        "d": build_bound_spec(0.0, 0.05).unwrap(),
    }

    # Get RNG state for reproducibility
    cpu_rng_state = torch.get_rng_state().numpy().tobytes()
    cuda_rng_states = [state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()]

    return GbmCVNNPricerConfig(
        cfg=bs_config,
        domain_bounds=domain_bounds,
        cvnn=model,
        optimizer_state=None,
        global_step=0,
        sobol_skip=0,
        torch_cpu_rng_state=cpu_rng_state,
        torch_cuda_rng_states=cuda_rng_states,
    )


async def train_with_blockchain() -> None:
    """Train model with automatic blockchain commits."""
    print("=" * 80)
    print("Training with Blockchain Storage Integration")
    print("=" * 80)

    # Initialize blockchain storage
    bucket_name = "training-example-bucket"
    print(f"\n1. Initializing blockchain storage: {bucket_name}")

    async with AsyncBlockchainModelStore(bucket_name) as store:
        # Create initial configuration
        print("\n2. Creating training configuration...")
        config = create_training_config()

        # Initialize trainer
        print("\n3. Initializing trainer...")
        pricer = GbmCVNNPricer(config)

        # Training configuration
        training_config = TrainingConfig(
            num_batches=50,  # Small number for demo
            batch_size=32,
            learning_rate=0.001,
        )

        print(f"\n4. Starting training:")
        print(f"   - Batches: {training_config.num_batches}")
        print(f"   - Batch size: {training_config.batch_size}")
        print(f"   - Learning rate: {training_config.learning_rate}")
        print(f"   - Periodic commits: Every 10 batches")
        print(f"   - Final commit: Yes")

        # Train with blockchain integration
        pricer.train(
            training_config,
            blockchain_store=store,
            commit_plan=FinalAndIntervalCommit(
                interval=10,
                commit_message_template="Training checkpoint: step={step}, loss={loss:.6f}",
            ),
        )

        print("\n5. Training complete!")

        # List versions
        head_result = await store.get_head()
        match head_result:
            case Success(head):
                print(f"\n6. Blockchain status:")
                print(f"   - Total versions: {head.counter + 1}")
                print(f"   - Latest version: v{head.counter:010d}")
                print(f"   - Content hash: {head.content_hash[:16]}...")
                print(f"   - Commit message: {head.commit_message}")

                # Load model back from blockchain
                print("\n7. Loading model from blockchain storage...")
                model_template = create_simple_cvnn().to(device=Device.cuda.to_torch())
                config_template = create_training_config()

                loaded_snapshot = await load_snapshot_from_checkpoint(
                    store,
                    head,
                    model_template,
                    config_template,
                )

                print(f"   - Loaded global_step: {loaded_snapshot.global_step}")
                print(f"   - Loaded sobol_skip: {loaded_snapshot.sobol_skip}")

                # Run inference with loaded model
                print("\n8. Running inference with loaded model...")
                loaded_pricer = GbmCVNNPricer(loaded_snapshot)

                # Create sample input
                test_input = BlackScholes.Inputs(
                    X0=100.0,
                    K=100.0,
                    T=1.0,
                    v=0.2,
                    r=0.05,
                    d=0.0,
                )

                match loaded_pricer.predict_price([test_input]):
                    case Success(results):
                        print(f"   - Test input: spot=100, strike=100, T=1.0, vol=0.2")
                        print(f"   - Predicted call price: ${results[0].call_price:.4f}")
                        print(f"   - Predicted put price: ${results[0].put_price:.4f}")
                    case Failure(error):
                        print(f"   - Inference failed: {error}")
                        return
            case Failure(error):
                print(f"\nError retrieving HEAD pointer: {error}")
                print("Cannot proceed with model loading. Skipping inference demo.")
                return

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


async def demonstrate_inference_client() -> None:
    """Demonstrate InferenceClient with the trained model."""
    print("\n" + "=" * 80)
    print("Demonstrating InferenceClient (Tracking Mode)")
    print("=" * 80)

    bucket_name = "training-example-bucket"

    async with AsyncBlockchainModelStore(bucket_name) as store:
        # Create template
        model_template = create_simple_cvnn().to(device=Device.cuda.to_torch())
        config_template = create_training_config()

        # Use InferenceClient in tracking mode
        print("\n1. Starting InferenceClient (tracking mode)...")
        async with InferenceClient(
            mode=TrackingMode(),  # Track latest
            poll_interval=10.0,
            store=store,
            model_template=model_template,
            config_template=config_template,
        ) as client:
            snapshot = client.get_model()
            version = client.get_current_version()

            # Type narrowing for version (cannot be None after getting model)
            assert version is not None, "Version should exist after get_model()"

            print(f"   - Loaded version: v{version.counter:010d}")
            print(f"   - Global step: {snapshot.global_step}")

            # Run inference
            pricer = GbmCVNNPricer(snapshot)
            test_input = BlackScholes.Inputs(
                X0=105.0,
                K=100.0,
                T=0.5,
                v=0.25,
                r=0.03,
                d=0.01,
            )

            match pricer.predict_price([test_input]):
                case Success(results):
                    print(f"\n2. Inference results:")
                    print(f"   - Call price: ${results[0].call_price:.4f}")
                    print(f"   - Put price: ${results[0].put_price:.4f}")
                case Failure(error):
                    print(f"\n2. Inference failed: {error}")

    print("\n" + "=" * 80)
    print("InferenceClient example complete!")
    print("=" * 80)


async def main() -> None:
    """Run all examples."""
    # Run training example
    await train_with_blockchain()

    # Run inference client example
    await demonstrate_inference_client()


if __name__ == "__main__":
    asyncio.run(main())
