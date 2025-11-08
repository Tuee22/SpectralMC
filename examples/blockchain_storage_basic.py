#!/usr/bin/env python3
"""
Basic blockchain storage example.

Demonstrates:
- Creating a blockchain model store
- Committing checkpoints
- Retrieving versions
- Loading checkpoints
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from spectralmc.storage import AsyncBlockchainModelStore


async def main() -> None:
    """Run basic blockchain storage demo."""
    # Create temporary storage directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Storage directory: {tmpdir}")

        # Initialize store
        async with AsyncBlockchainModelStore(tmpdir) as store:
            print("\nAsyncBlockchainModelStore initialized")

            # Create genesis commit
            print("\n=== Genesis Commit ===")
            checkpoint1 = b"Model weights at epoch 0"
            content_hash1 = "abc123def456"
            version1 = await store.commit(checkpoint1, content_hash1, "Initial model")

            print(f"Version ID: {version1.version_id}")
            print(f"Semantic Version: {version1.semantic_version}")
            print(f"Counter: {version1.counter}")
            print(f"Parent Hash: {version1.parent_hash or '(none - genesis)'}")
            print(f"Content Hash: {version1.content_hash}")
            print(f"Directory: {version1.directory_name}")

            # Second commit
            print("\n=== Second Commit ===")
            checkpoint2 = b"Model weights at epoch 100"
            content_hash2 = "789ghi012jkl"
            version2 = await store.commit(
                checkpoint2, content_hash2, "After 100 epochs"
            )

            print(f"Version ID: {version2.version_id}")
            print(f"Semantic Version: {version2.semantic_version}")
            print(f"Counter: {version2.counter}")
            print(f"Parent Hash: {version2.parent_hash}")
            print(f"Content Hash: {version2.content_hash}")

            # Third commit
            print("\n=== Third Commit ===")
            checkpoint3 = b"Model weights at epoch 200"
            content_hash3 = "345mno678pqr"
            version3 = await store.commit(
                checkpoint3, content_hash3, "After 200 epochs"
            )

            print(f"Version ID: {version3.version_id}")
            print(f"Semantic Version: {version3.semantic_version}")
            print(f"Counter: {version3.counter}")

            # Retrieve HEAD
            print("\n=== HEAD Pointer ===")
            head = await store.get_head()
            if head:
                print(f"HEAD points to: {head.version_id}")
                print(f"Commit message: {head.commit_message}")

            # Retrieve specific version
            print("\n=== Retrieve Specific Version ===")
            retrieved = await store.get_version("v0000000001")
            print(f"Retrieved: {retrieved.version_id}")
            print(f"Message: {retrieved.commit_message}")

            # Load checkpoint data
            print("\n=== Load Checkpoint ===")
            loaded = await store.load_checkpoint(version2)
            print(f"Loaded {len(loaded)} bytes: {loaded.decode()}")

            # Verify chain integrity
            print("\n=== Chain Integrity ===")
            print(f"Genesis parent: {version1.parent_hash or '(empty)'}")
            print(f"v1 → v2 link: {version2.parent_hash == version1.content_hash}")
            print(f"v2 → v3 link: {version3.parent_hash == version2.content_hash}")

            print("\n✓ Demo complete")


if __name__ == "__main__":
    asyncio.run(main())
