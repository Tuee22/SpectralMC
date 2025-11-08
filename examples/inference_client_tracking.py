# examples/inference_client_tracking.py
"""
Example: InferenceClient in tracking mode (development/continuous learning).

This demonstrates automatically updating to the latest model version as
training produces new checkpoints. Ideal for development, staging, or
continuous learning systems that need minimal staleness.
"""

import asyncio
import torch

from spectralmc.storage import AsyncBlockchainModelStore, InferenceClient
from spectralmc.gbm_trainer import GbmCVNNPricerConfig
from spectralmc.gbm import BlackScholesConfig, SimulationParams
from spectralmc.models.numerical import Precision


async def main() -> None:
    print("=== InferenceClient Tracking Mode Example ===\n")

    # 1. Connect to storage
    print("1. Connecting to blockchain storage")
    bucket_name = "opt-models"

    async with AsyncBlockchainModelStore(bucket_name) as store:
        # 2. Check initial state
        print("\n2. Checking current versions")
        head = await store.get_head()
        if head is None:
            print("   ERROR: No versions found in storage!")
            print("   Run checkpoint_training_example.py first to create versions.")
            return

        print(f"   Current HEAD: version {head.counter}")
        print(f"   Semantic version: {head.semantic_version}")

        # 3. Create model template
        print("\n3. Creating model template")
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

        # 5. Create InferenceClient in TRACKING mode
        print("\n4. Creating InferenceClient (TRACKING mode)")
        print("   Mode: Tracking (auto-updates to latest)")
        print("   Poll interval: 5 seconds")

        client = InferenceClient(
            version_counter=None,  # None = tracking mode
            poll_interval=5.0,  # Check for updates every 5 seconds
            store=store,
            model_template=model_template,
            config_template=config_template,
        )

        # 6. Start client
        async with client:
            print("\n5. Client started")

            # Get loaded version
            loaded_version = client.get_current_version()
            assert loaded_version is not None
            print(f"   Loaded latest version: {loaded_version.counter}")
            print(f"   Global step: {client.get_model().global_step}")

            # 7. Run predictions continuously
            print("\n6. Running predictions continuously")
            print("   (Client will auto-update if new versions are committed)")
            print("   Press Ctrl+C to stop\n")

            last_version = loaded_version.counter

            try:
                for i in range(20):  # Simulate 20 prediction batches
                    # Check if version changed
                    current_version = client.get_current_version()
                    assert current_version is not None

                    if current_version.counter != last_version:
                        print(
                            f"   🔄 AUTO-UPGRADED: v{last_version} → v{current_version.counter}"
                        )
                        print(
                            f"      New global step: {client.get_model().global_step}"
                        )
                        last_version = current_version.counter

                    # Run prediction
                    snapshot = client.get_model()
                    model = snapshot.cvnn
                    model.eval()

                    with torch.no_grad():
                        inputs = torch.randn(10, 5)
                        raw_outputs = model(inputs)  # type: ignore[call-arg]
                        # CVNN models return tuple (real, imag), extract first element
                        outputs = (
                            raw_outputs[0]
                            if isinstance(raw_outputs, tuple)
                            else raw_outputs
                        )

                    print(
                        f"   Batch {i+1:2d} (v{current_version.counter}): "
                        f"input={inputs.shape}, output={outputs.shape}"
                    )

                    # Wait before next batch
                    await asyncio.sleep(2.0)

            except KeyboardInterrupt:
                print("\n   Interrupted by user")

    print("\n✓ Example complete!")
    print("\nKey points about tracking mode:")
    print("  - Automatically loads latest version on start")
    print("  - Polls for updates every poll_interval seconds")
    print("  - Hot-swaps model atomically when new version available")
    print("  - Ideal for development and continuous learning systems")
    print("  - Minimizes staleness without manual intervention")


if __name__ == "__main__":
    asyncio.run(main())
