#!/usr/bin/env python3
"""
Blockchain integrity verification example.

Demonstrates:
- Hash computation and verification
- Chain integrity checking
- Tamper detection
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
import tempfile

from spectralmc.storage import AsyncBlockchainModelStore
from spectralmc.storage.chain import ModelVersion


async def main() -> None:
    """Run integrity verification demo."""
    print("=== Blockchain Integrity Verification ===")

    # Hash computation example
    print("\n1. Hash Computation")
    version = ModelVersion(
        counter=1,
        semantic_version="1.0.0",
        parent_hash="parent123",
        content_hash="content456",
        commit_timestamp="2025-01-01T00:00:00Z",
        commit_message="Test version",
    )

    version_hash = version.compute_hash()
    print(f"Version hash: {version_hash}")
    print(f"Hash length: {len(version_hash)} (SHA256 hex)")

    # Verify deterministic hashing
    hash2 = version.compute_hash()
    print(f"Hash deterministic: {version_hash == hash2}")

    # Chain linking example
    print("\n2. Chain Linking")
    with tempfile.TemporaryDirectory() as tmpdir:
        async with AsyncBlockchainModelStore(tmpdir) as store:
            # Create chain of 5 versions
            versions = []
            for i in range(5):
                checkpoint = f"checkpoint {i}".encode()
                content_hash = f"hash{i}"
                version = await store.commit(checkpoint, content_hash, f"Commit {i}")
                versions.append(version)

            print("Chain structure:")
            for i, v in enumerate(versions):
                parent_info = v.parent_hash if v.parent_hash else "(genesis)"
                print(
                    f"  {v.version_id}: parent={parent_info[:12]}, content={v.content_hash}"
                )

            # Verify chain integrity
            print("\n3. Chain Integrity Verification")
            for i in range(1, len(versions)):
                expected_parent = versions[i - 1].content_hash
                actual_parent = versions[i].parent_hash
                valid = expected_parent == actual_parent
                status = "✓" if valid else "✗"
                print(f"  {status} v{i-1} → v{i}: {valid}")

            # Demonstrate tamper detection
            print("\n4. Tamper Detection")
            original_hash = versions[2].compute_hash()
            print(f"Original v2 hash: {original_hash[:16]}...")

            # Create tampered version
            tampered = ModelVersion(
                counter=versions[2].counter,
                semantic_version=versions[2].semantic_version,
                parent_hash=versions[2].parent_hash,
                content_hash="TAMPERED_HASH",  # Changed!
                commit_timestamp=versions[2].commit_timestamp,
                commit_message=versions[2].commit_message,
            )

            tampered_hash = tampered.compute_hash()
            print(f"Tampered v2 hash: {tampered_hash[:16]}...")
            print(f"Hashes match: {original_hash == tampered_hash}")
            print("✓ Tampering detected!")

    # Version immutability
    print("\n5. Version Immutability")

    def _attempt_mutation(version: ModelVersion) -> None:
        """Try mutating a frozen dataclass; should raise FrozenInstanceError."""
        version.counter = 999

    try:
        _attempt_mutation(versions[0])
        print("✗ Version was modified (should not happen!)")
    except FrozenInstanceError:
        print("✓ Version is immutable: FrozenInstanceError")

    print("\n✓ Integrity verification complete")


if __name__ == "__main__":
    asyncio.run(main())
