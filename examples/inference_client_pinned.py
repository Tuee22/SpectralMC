# examples/inference_client_pinned.py
"""
Example: InferenceClient with PinnedMode (production deployment).

This demonstrates loading a specific model version using PinnedMode and keeping
it locked for stable, reproducible predictions. Ideal for A/B testing or
production environments where you need version stability.
"""

import asyncio
import torch

from spectralmc.storage import AsyncBlockchainModelStore, InferenceClient, PinnedMode
from spectralmc.gbm_trainer import GbmCVNNPricerConfig
from spectralmc.gbm import BlackScholesConfig, SimulationParams
from spectralmc.models.numerical import Precision
from spectralmc.result import Success, Failure


async def main() -> None:
    print("=== InferenceClient Pinned Mode Example ===\n")

    # 1. Connect to storage
    print("1. Connecting to blockchain storage")
    bucket_name = "opt-models"

    async with AsyncBlockchainModelStore(bucket_name) as store:
        # 2. Check available versions
        print("\n2. Fetching available versions")
        head_result = await store.get_head()
        match head_result:
            case Success(head):
                print(f"   Latest version: {head.counter}")
                print(f"   Semantic version: {head.semantic_version}")
                print(f"   Content hash: {head.content_hash[:16]}...")
            case Failure(error):
                print(f"   ERROR: Failed to load model versions")
                print(f"   Details: {error}")
                print(f"   Please ensure blockchain storage is initialized.")
                return

        # 3. Create model template for loading
        print("\n3. Creating model template")
        # Note: This should match the architecture of stored models
        model_template = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5),
        )

        # 4. Create config template
        sim_params = SimulationParams(
            timesteps=100,
            network_size=1024,
            batches_per_mc_run=8,
            threads_per_block=256,
            mc_seed=42,
            buffer_size=10000,
            skip=0,
            dtype=Precision.float32,
        )

        bs_config = BlackScholesConfig(
            sim_params=sim_params,
            simulate_log_return=True,
            normalize_forwards=True,
        )

        config_template = GbmCVNNPricerConfig(
            cfg=bs_config,
            domain_bounds={},
            cvnn=model_template,
            optimizer_state=None,
            global_step=0,
            sobol_skip=0,
            torch_cpu_rng_state=torch.get_rng_state().numpy().tobytes(),
            torch_cuda_rng_states=[],
        )

        # 5. Create InferenceClient in PINNED mode
        # Pin to version 0 (genesis) for maximum stability
        pinned_version = 0
        print(f"\n4. Creating InferenceClient (PINNED to version {pinned_version})")
        print(f"   Mode: Pinned (never auto-updates)")

        client = InferenceClient(
            mode=PinnedMode(counter=pinned_version),  # Pin to specific version
            poll_interval=60.0,  # Ignored in pinned mode
            store=store,
            model_template=model_template,
            config_template=config_template,
        )

        # 6. Start client and run predictions
        async with client:
            print("\n5. Client started")

            # Get loaded version info
            loaded_version = client.get_current_version()
            assert loaded_version is not None
            print(f"   Loaded version: {loaded_version.counter}")
            print(f"   Content hash: {loaded_version.content_hash[:16]}...")
            print(f"   Commit message: {loaded_version.commit_message}")

            # Get model snapshot
            snapshot = client.get_model()
            print(f"   Global step: {snapshot.global_step}")

            # Run predictions
            print("\n6. Running predictions")
            model = snapshot.cvnn
            model.eval()

            with torch.no_grad():
                for i in range(3):
                    inputs = torch.randn(10, 5)
                    # Note: For actual ComplexValuedModel, use:
                    # real, imag = model(real_in, imag_in)
                    raw_outputs = model(inputs)  # type: ignore[call-arg]
                    # CVNN models return tuple (real, imag), extract first element
                    outputs = (
                        raw_outputs[0]
                        if isinstance(raw_outputs, tuple)
                        else raw_outputs
                    )
                    print(
                        f"   Batch {i+1}: input={inputs.shape}, output={outputs.shape}"
                    )

            print("\n7. Simulating long-running service")
            print(
                "   (In production, model remains pinned to version {})".format(
                    pinned_version
                )
            )
            print(
                "   Even if new versions are committed, this client stays on v{}.".format(
                    pinned_version
                )
            )

            # Wait a bit to simulate service running
            await asyncio.sleep(1.0)

    print("\n✓ Example complete!")
    print("\nKey points about PinnedMode:")
    print("  - Model version is locked at initialization using PinnedMode(counter=N)")
    print("  - Never auto-updates, even if new versions available")
    print("  - Ideal for production stability and A/B testing")
    print("  - Reproducible predictions for same input")


if __name__ == "__main__":
    asyncio.run(main())
