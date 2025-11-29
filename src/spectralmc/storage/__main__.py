# src/spectralmc/storage/__main__.py
"""CLI tool for blockchain storage operations.

Usage:
    python -m spectralmc.storage verify <bucket-name> [--detailed]
    python -m spectralmc.storage find-corruption <bucket-name>
    python -m spectralmc.storage list-versions <bucket-name>
    python -m spectralmc.storage inspect <bucket-name> <version-id>
    python -m spectralmc.storage gc-preview <bucket-name> <keep-versions> [--protect-tags TAGS]
    python -m spectralmc.storage gc-run <bucket-name> <keep-versions> [--protect-tags TAGS] [--yes]
    python -m spectralmc.storage tensorboard-log <bucket-name> [--log-dir DIR]

Examples:
    # Verify chain integrity
    python -m spectralmc.storage verify my-model-bucket

    # Find first corrupted version
    python -m spectralmc.storage find-corruption my-model-bucket

    # List all versions
    python -m spectralmc.storage list-versions my-model-bucket

    # Inspect specific version
    python -m spectralmc.storage inspect my-model-bucket v0000000042

    # Preview garbage collection (dry run)
    python -m spectralmc.storage gc-preview my-model-bucket 10

    # Run garbage collection (delete old versions, keep last 10)
    python -m spectralmc.storage gc-run my-model-bucket 10 --yes

    # Protect specific versions from deletion
    python -m spectralmc.storage gc-run my-model-bucket 5 --protect-tags 3,7,12

    # Log blockchain to TensorBoard
    python -m spectralmc.storage tensorboard-log my-model-bucket --log-dir runs/my_experiment
"""

from __future__ import annotations

# IMPORTANT: Import torch façade first to set deterministic flags
import spectralmc.models.torch  # noqa: F401

import asyncio
import sys
import argparse
import json
from typing import NoReturn, Never

from ..result import Result, Success, Failure
from .store import AsyncBlockchainModelStore
from .verification import verify_chain, verify_chain_detailed, find_corruption
from .gc import run_gc
from .tensorboard_writer import log_blockchain_to_tensorboard
from .errors import ChainCorruptionError, VersionNotFoundError, HeadNotFoundError, StorageError
from .s3_errors import (
    S3BucketNotFound,
    S3ObjectNotFound,
    S3AccessDenied,
    S3NetworkError,
    S3UnknownError,
)


def assert_never(value: Never) -> Never:
    """
    Type-safe exhaustiveness check for pattern matching.

    This function should never be reachable. If the type checker allows
    this function to be called, it means the match statement is not exhaustive.

    Args:
        value: Value that should be of type Never (unreachable)

    Raises:
        AssertionError: Always (this should never execute)
    """
    raise AssertionError(f"Unhandled case: {value!r}")


async def cmd_verify(bucket_name: str, detailed: bool = False) -> int:
    """
    Verify blockchain integrity.

    Uses functional programming with Result types and exhaustive pattern matching
    to handle all error cases explicitly. No unhandled exceptions.

    Args:
        bucket_name: S3 bucket name
        detailed: Show detailed corruption report

    Returns:
        Exit code:
            0: Chain is valid
            1: Chain is corrupted
            2: Operational error (S3, network, etc.)
    """
    async with AsyncBlockchainModelStore(bucket_name) as store:
        result = await verify_chain(store)

        match result:
            case Success(report) if report.is_valid and detailed:
                # Detailed output for valid chain
                print(
                    json.dumps(
                        {
                            "is_valid": True,
                            "corrupted_version": None,
                            "corruption_type": None,
                            "details": report.details,
                        },
                        indent=2,
                    )
                )
                return 0

            case Success(report) if report.is_valid:
                # Simple output for valid chain
                print(f"✓ Chain integrity verified for bucket: {bucket_name}")
                return 0

            case Success(report):
                # Chain is corrupted (is_valid = False)
                if detailed:
                    print(
                        json.dumps(
                            {
                                "is_valid": False,
                                "corrupted_version": (
                                    report.corrupted_version.counter
                                    if report.corrupted_version
                                    else None
                                ),
                                "corruption_type": report.corruption_type,
                                "details": report.details,
                            },
                            indent=2,
                        ),
                        file=sys.stderr,
                    )
                else:
                    print(f"✗ Chain corruption detected:", file=sys.stderr)
                    print(f"  Type: {report.corruption_type}", file=sys.stderr)
                    print(f"  Details: {report.details}", file=sys.stderr)
                return 1

            case Failure(S3BucketNotFound(bucket, msg)):
                print(f"✗ Error: Bucket not found: {bucket}", file=sys.stderr)
                print(f"  {msg}", file=sys.stderr)
                return 2

            case Failure(S3ObjectNotFound(bucket, key, msg)):
                print(f"✗ Error: Object not found in bucket {bucket}: {key}", file=sys.stderr)
                print(f"  {msg}", file=sys.stderr)
                return 2

            case Failure(S3AccessDenied(bucket, operation, msg)):
                print(f"✗ Error: Access denied to bucket: {bucket}", file=sys.stderr)
                print(f"  Operation: {operation}", file=sys.stderr)
                print(f"  {msg}", file=sys.stderr)
                return 2

            case Failure(S3NetworkError(msg, retry_count)):
                print(f"✗ Error: Network error:", file=sys.stderr)
                print(f"  {msg}", file=sys.stderr)
                print(f"  Retries attempted: {retry_count}", file=sys.stderr)
                return 2

            case Failure(S3UnknownError(code, msg)):
                print(f"✗ Error: S3 error ({code}):", file=sys.stderr)
                print(f"  {msg}", file=sys.stderr)
                return 2

            case _:
                # All S3OperationError cases covered above
                # This should be unreachable but mypy can't prove it for union types
                raise AssertionError(f"Unexpected result type: {result}")


async def cmd_find_corruption(bucket_name: str) -> int:
    """
    Find first corrupted version in chain.

    Args:
        bucket_name: S3 bucket name

    Returns:
        Exit code (0 = no corruption, 1 = corruption found, 2 = error)
    """
    try:
        async with AsyncBlockchainModelStore(bucket_name) as store:
            corrupted = await find_corruption(store)

            if corrupted is None:
                print(f"✓ No corruption found in bucket: {bucket_name}")
                return 0
            else:
                print(
                    json.dumps(
                        {
                            "corrupted": True,
                            "version_counter": corrupted.counter,
                            "version_id": corrupted.version_id,
                            "semantic_version": corrupted.semantic_version,
                            "content_hash": corrupted.content_hash,
                            "commit_timestamp": corrupted.commit_timestamp,
                        },
                        indent=2,
                    )
                )
                return 1

    except (ChainCorruptionError, VersionNotFoundError, StorageError, HeadNotFoundError) as e:
        print(f"✗ Error finding corruption: {e}", file=sys.stderr)
        return 2


async def cmd_list_versions(bucket_name: str) -> int:
    """
    List all versions in the blockchain.

    Args:
        bucket_name: S3 bucket name

    Returns:
        Exit code (0 = success, 2 = error)
    """
    try:
        async with AsyncBlockchainModelStore(bucket_name) as store:
            head_result = await store.get_head()

            match head_result:
                case Failure(_):
                    print(f"✓ No versions in bucket: {bucket_name}")
                    return 0
                case Success(head):
                    print(f"Versions in bucket '{bucket_name}':")
                    print(
                        f"{'Counter':<10} {'Version ID':<15} {'SemVer':<10} {'Content Hash':<16} {'Timestamp':<25}"
                    )
                    print("-" * 90)

            for counter in range(head.counter + 1):
                version_id = f"v{counter:010d}"
                version = await store.get_version(version_id)

                print(
                    f"{version.counter:<10} "
                    f"{version.version_id:<15} "
                    f"{version.semantic_version:<10} "
                    f"{version.content_hash[:12]}... "
                    f"{version.commit_timestamp[:25]:<25}"
                )

            print(f"\nTotal: {head.counter + 1} versions")
            return 0

    except (VersionNotFoundError, StorageError, HeadNotFoundError) as e:
        print(f"✗ Error listing versions: {e}", file=sys.stderr)
        return 2


async def cmd_inspect(bucket_name: str, version_id: str) -> int:
    """
    Inspect a specific version in detail.

    Args:
        bucket_name: S3 bucket name
        version_id: Version ID (e.g., "v0000000042")

    Returns:
        Exit code (0 = success, 2 = error)
    """
    try:
        async with AsyncBlockchainModelStore(bucket_name) as store:
            version = await store.get_version(version_id)

            print(
                json.dumps(
                    {
                        "counter": version.counter,
                        "version_id": version.version_id,
                        "semantic_version": version.semantic_version,
                        "parent_hash": version.parent_hash,
                        "content_hash": version.content_hash,
                        "commit_timestamp": version.commit_timestamp,
                        "commit_message": version.commit_message,
                        "directory_name": version.directory_name,
                    },
                    indent=2,
                )
            )

            return 0

    except (VersionNotFoundError, StorageError) as e:
        print(f"✗ Error inspecting version: {e}", file=sys.stderr)
        return 2


async def cmd_gc_preview(
    bucket_name: str, keep_versions: int, protect_tags: str = ""
) -> int:
    """
    Preview garbage collection (dry run).

    Args:
        bucket_name: S3 bucket name
        keep_versions: Number of recent versions to keep
        protect_tags: Comma-separated list of version counters to protect

    Returns:
        Exit code (0 = success, 2 = error)
    """
    try:
        async with AsyncBlockchainModelStore(bucket_name) as store:
            # Parse protected tags
            tags = []
            if protect_tags:
                tags = [int(t.strip()) for t in protect_tags.split(",")]

            # Run GC in dry-run mode
            report = await run_gc(
                store, keep_versions=keep_versions, protect_tags=tags, dry_run=True
            )

            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "deleted_versions": report.deleted_versions,
                        "protected_versions": report.protected_versions,
                        "bytes_freed": report.bytes_freed,
                        "mb_freed": round(report.bytes_freed / (1024 * 1024), 2),
                    },
                    indent=2,
                )
            )

            return 0

    except (VersionNotFoundError, StorageError, HeadNotFoundError) as e:
        print(f"✗ Error during GC preview: {e}", file=sys.stderr)
        return 2


async def cmd_gc_run(
    bucket_name: str, keep_versions: int, protect_tags: str = "", confirm: bool = False
) -> int:
    """
    Run garbage collection (delete old versions).

    Args:
        bucket_name: S3 bucket name
        keep_versions: Number of recent versions to keep
        protect_tags: Comma-separated list of version counters to protect
        confirm: If True, skip confirmation prompt

    Returns:
        Exit code (0 = success, 2 = error)
    """
    try:
        async with AsyncBlockchainModelStore(bucket_name) as store:
            # Parse protected tags
            tags = []
            if protect_tags:
                tags = [int(t.strip()) for t in protect_tags.split(",")]

            # First, preview what will be deleted
            if not confirm:
                preview_report = await run_gc(
                    store, keep_versions=keep_versions, protect_tags=tags, dry_run=True
                )

                print(f"Will delete {len(preview_report.deleted_versions)} versions:")
                print(f"  Versions to delete: {preview_report.deleted_versions}")
                print(f"  Versions to keep: {preview_report.protected_versions}")
                print(
                    f"  Space to free: {round(preview_report.bytes_freed / (1024 * 1024), 2)} MB"
                )
                print()

                response = input("Proceed with deletion? [y/N] ")
                if response.lower() != "y":
                    print("Aborted")
                    return 0

            # Actually run GC
            report = await run_gc(
                store, keep_versions=keep_versions, protect_tags=tags, dry_run=False
            )

            print(f"✓ Garbage collection completed:")
            print(
                f"  Deleted {len(report.deleted_versions)} versions: {report.deleted_versions}"
            )
            print(f"  Freed {round(report.bytes_freed / (1024 * 1024), 2)} MB")

            return 0

    except (VersionNotFoundError, StorageError, HeadNotFoundError) as e:
        print(f"✗ Error during GC: {e}", file=sys.stderr)
        return 2


async def cmd_tensorboard_log(
    bucket_name: str, log_dir: str = "runs/blockchain_models"
) -> int:
    """
    Log blockchain to TensorBoard.

    Args:
        bucket_name: S3 bucket name
        log_dir: TensorBoard log directory

    Returns:
        Exit code (0 = success, 2 = error)
    """
    try:
        async with AsyncBlockchainModelStore(bucket_name) as store:
            print(f"Logging blockchain to TensorBoard: {log_dir}")

            await log_blockchain_to_tensorboard(store, log_dir=log_dir)

            print(f"✓ Blockchain logged to TensorBoard")
            print(f"  View with: tensorboard --logdir={log_dir}")

            return 0

    except (VersionNotFoundError, StorageError, HeadNotFoundError, IOError, OSError) as e:
        print(f"✗ Error logging to TensorBoard: {e}", file=sys.stderr)
        return 2


def main() -> NoReturn:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SpectralMC blockchain storage CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify blockchain integrity")
    verify_parser.add_argument("bucket_name", help="S3 bucket name")
    verify_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed corruption report (JSON)"
    )

    # find-corruption command
    find_parser = subparsers.add_parser(
        "find-corruption", help="Find first corrupted version"
    )
    find_parser.add_argument("bucket_name", help="S3 bucket name")

    # list-versions command
    list_parser = subparsers.add_parser(
        "list-versions", help="List all versions in blockchain"
    )
    list_parser.add_argument("bucket_name", help="S3 bucket name")

    # inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a specific version")
    inspect_parser.add_argument("bucket_name", help="S3 bucket name")
    inspect_parser.add_argument("version_id", help="Version ID (e.g., v0000000042)")

    # gc-preview command
    gc_preview_parser = subparsers.add_parser(
        "gc-preview", help="Preview garbage collection (dry run)"
    )
    gc_preview_parser.add_argument("bucket_name", help="S3 bucket name")
    gc_preview_parser.add_argument(
        "keep_versions", type=int, help="Number of recent versions to keep"
    )
    gc_preview_parser.add_argument(
        "--protect-tags",
        default="",
        help="Comma-separated version counters to protect (e.g., '3,5,7')",
    )

    # gc-run command
    gc_run_parser = subparsers.add_parser(
        "gc-run", help="Run garbage collection (delete old versions)"
    )
    gc_run_parser.add_argument("bucket_name", help="S3 bucket name")
    gc_run_parser.add_argument(
        "keep_versions", type=int, help="Number of recent versions to keep"
    )
    gc_run_parser.add_argument(
        "--protect-tags",
        default="",
        help="Comma-separated version counters to protect (e.g., '3,5,7')",
    )
    gc_run_parser.add_argument(
        "--yes", action="store_true", dest="confirm", help="Skip confirmation prompt"
    )

    # tensorboard-log command
    tb_log_parser = subparsers.add_parser(
        "tensorboard-log", help="Log blockchain to TensorBoard"
    )
    tb_log_parser.add_argument("bucket_name", help="S3 bucket name")
    tb_log_parser.add_argument(
        "--log-dir",
        default="runs/blockchain_models",
        help="TensorBoard log directory (default: runs/blockchain_models)",
    )

    args = parser.parse_args()

    # Dispatch to command handler
    if args.command == "verify":
        exit_code = asyncio.run(cmd_verify(args.bucket_name, args.detailed))
    elif args.command == "find-corruption":
        exit_code = asyncio.run(cmd_find_corruption(args.bucket_name))
    elif args.command == "list-versions":
        exit_code = asyncio.run(cmd_list_versions(args.bucket_name))
    elif args.command == "inspect":
        exit_code = asyncio.run(cmd_inspect(args.bucket_name, args.version_id))
    elif args.command == "gc-preview":
        exit_code = asyncio.run(
            cmd_gc_preview(args.bucket_name, args.keep_versions, args.protect_tags)
        )
    elif args.command == "gc-run":
        exit_code = asyncio.run(
            cmd_gc_run(
                args.bucket_name, args.keep_versions, args.protect_tags, args.confirm
            )
        )
    elif args.command == "tensorboard-log":
        exit_code = asyncio.run(cmd_tensorboard_log(args.bucket_name, args.log_dir))
    else:
        parser.print_help()
        sys.exit(2)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
