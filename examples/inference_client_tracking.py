# examples/inference_client_tracking.py
"""
Example: InferenceClient in tracking mode (development/continuous learning).

This demonstrates automatically updating to the latest model version as
training produces new checkpoints. Ideal for development, staging, or
continuous learning systems that need minimal staleness.
"""

import asyncio

import torch
from spectralmc.runtime import get_torch_handle

from spectralmc.storage import AsyncBlockchainModelStore, InferenceClient, TrackingMode
from spectralmc.testing import make_gbm_cvnn_config, make_test_simulation_params, seed_all_rngs
from spectralmc.result import Success, Failure

get_torch_handle()


async def main() -> None:
    print("=== InferenceClient Tracking Mode Example ===\n")

    # 1. Connect to storage
    print("1. Connecting to blockchain storage")
    bucket_name = "opt-models"

    async with AsyncBlockchainModelStore(bucket_name) as store:
        # 2. Check initial state
        print("\n2. Checking current versions")
        head_result = await store.get_head()
        seed_all_rngs(42)
        match head_result:
            case Success(head):
                print(f"   Current HEAD: version {head.counter}")
                print(f"   Semantic version: {head.semantic_version}")
            case Failure(error):
                print(f"   ERROR: Failed to load model versions")
                print(f"   Details: {error}")
                print(f"   Please ensure blockchain storage is initialized.")
                return

        # 3. Create model template
        print("\n3. Creating model template")
        model_template = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5),
        )

        # 4. Create config template
        sim_params = make_test_simulation_params()
        config_template = make_gbm_cvnn_config(
            model_template,
            global_step=0,
            sim_params=sim_params,
            domain_bounds={},
        )

        # 5. Create InferenceClient in TRACKING mode
        print("\n4. Creating InferenceClient (TRACKING mode)")
        print("   Mode: Tracking (auto-updates to latest)")
        print("   Poll interval: 5 seconds")

        client = InferenceClient(
            mode=TrackingMode(),  # TrackingMode = auto-updates to latest
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
                        raw_outputs = model(inputs)
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
