# src/spectralmc/storage/verification.py
"""Chain verification utilities for detecting tampering and corruption."""

from __future__ import annotations

import logging
from typing import Optional, List
from dataclasses import dataclass

from botocore.exceptions import ClientError

from ..result import Result, Success, Failure
from .store import AsyncBlockchainModelStore
from .chain import ModelVersion
from .errors import ChainCorruptionError, HeadNotFoundError, VersionNotFoundError
from .s3_errors import S3OperationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorruptionReport:
    """Report of chain corruption detection.

    Attributes:
        is_valid: True if chain is intact
        corrupted_version: First corrupted version (None if valid)
        corruption_type: Type of corruption detected
        details: Human-readable description
    """

    is_valid: bool
    corrupted_version: Optional[ModelVersion]
    corruption_type: Optional[str]
    details: str




async def verify_chain_detailed(store: AsyncBlockchainModelStore) -> CorruptionReport:
    """
    Verify chain and return detailed report.

    Args:
        store: AsyncBlockchainModelStore instance

    Returns:
        CorruptionReport with validation results
    """
    # Fetch HEAD
    head_result = await store.get_head()

    match head_result:
        case Failure(_):
            # Empty chain is valid (no corruption)
            return CorruptionReport(
                is_valid=True,
                corrupted_version=None,
                corruption_type=None,
                details="Empty chain (no versions committed)",
            )
        case Success(head):
            pass  # Continue with verification

    # Fetch all versions from 0 to HEAD
    versions: List[ModelVersion] = []
    for counter in range(head.counter + 1):
        version_id = f"v{counter:010d}"
        try:
            version = await store.get_version(version_id)
            versions.append(version)
        except VersionNotFoundError as e:
            # Missing version in sequence
            return CorruptionReport(
                is_valid=False,
                corrupted_version=None,
                corruption_type="missing_version",
                details=f"Version {counter} missing from chain: {e}",
            )

    # Validate genesis block
    genesis = versions[0]
    if genesis.counter != 0:
        return CorruptionReport(
            is_valid=False,
            corrupted_version=genesis,
            corruption_type="invalid_genesis_counter",
            details=f"Genesis counter is {genesis.counter}, expected 0",
        )

    if genesis.parent_hash != "":
        return CorruptionReport(
            is_valid=False,
            corrupted_version=genesis,
            corruption_type="invalid_genesis_parent",
            details=f"Genesis parent_hash is '{genesis.parent_hash}', expected empty string",
        )

    if genesis.semantic_version != "1.0.0":
        return CorruptionReport(
            is_valid=False,
            corrupted_version=genesis,
            corruption_type="invalid_genesis_semver",
            details=f"Genesis semantic_version is '{genesis.semantic_version}', expected '1.0.0'",
        )

    # Validate chain links
    for i in range(1, len(versions)):
        current = versions[i]
        previous = versions[i - 1]

        # Check counter increments by exactly 1
        if current.counter != previous.counter + 1:
            return CorruptionReport(
                is_valid=False,
                corrupted_version=current,
                corruption_type="non_sequential_counter",
                details=(
                    f"Version {current.counter} follows {previous.counter}, "
                    f"expected {previous.counter + 1}"
                ),
            )

        # Check Merkle chain property (parent_hash matches previous content_hash)
        if current.parent_hash != previous.content_hash:
            return CorruptionReport(
                is_valid=False,
                corrupted_version=current,
                corruption_type="broken_merkle_chain",
                details=(
                    f"Version {current.counter} parent_hash '{current.parent_hash[:8]}...' "
                    f"does not match previous content_hash '{previous.content_hash[:8]}...'"
                ),
            )

        # Check semantic version progression (patch increment)
        expected_semver = f"1.0.{current.counter}"
        if current.semantic_version != expected_semver:
            return CorruptionReport(
                is_valid=False,
                corrupted_version=current,
                corruption_type="invalid_semver_progression",
                details=(
                    f"Version {current.counter} semantic_version is '{current.semantic_version}', "
                    f"expected '{expected_semver}'"
                ),
            )

    # All checks passed
    return CorruptionReport(
        is_valid=True,
        corrupted_version=None,
        corruption_type=None,
        details=f"Chain verified: {len(versions)} versions intact",
    )


async def verify_chain(
    store: AsyncBlockchainModelStore,
) -> Result[CorruptionReport, S3OperationError]:
    """
    Verify blockchain integrity with functional error handling.

    Validates:
    1. Genesis block (counter=0, empty parent_hash, semver="1.0.0")
    2. Merkle chain (parent_hash matches previous content_hash)
    3. Counter monotonicity (strictly increasing by 1)
    4. Semantic version progression (valid semver bumps)

    Returns Result types for all errors, enabling exhaustive pattern matching.

    Args:
        store: AsyncBlockchainModelStore instance

    Returns:
        Success(CorruptionReport) with validation results:
            - CorruptionReport.is_valid: True if chain intact
            - CorruptionReport.corrupted_version: First corrupted version (if any)
            - CorruptionReport.corruption_type: Type of corruption detected
            - CorruptionReport.details: Human-readable description

        Failure(S3OperationError) for S3 operational failures:
            - S3BucketNotFound: Bucket doesn't exist
            - S3AccessDenied: Permission denied
            - S3NetworkError: Network/timeout error
            - S3UnknownError: Other S3 errors

    Example:
        ```python
        async with AsyncBlockchainModelStore("bucket") as store:
            result = await verify_chain(store)
            match result:
                case Success(report) if report.is_valid:
                    print(f"Chain valid: {report.details}")
                case Success(report):
                    print(f"Corruption: {report.corruption_type}")
                case Failure(S3BucketNotFound(bucket=b)):
                    print(f"Bucket not found: {b}")
                case Failure(error):
                    print(f"S3 error: {error}")
        ```
    """
    from .s3_errors import S3OperationError

    # Fetch HEAD
    head_result = await store.get_head()

    match head_result:
        case Failure(error):
            # Check if it's HeadNotFoundError (empty chain - valid)
            if isinstance(error, HeadNotFoundError):
                return Success(
                    CorruptionReport(
                        is_valid=True,
                        corrupted_version=None,
                        corruption_type=None,
                        details="Empty chain (no versions committed)",
                    )
                )
            # S3OperationError propagates
            return Failure(error)

        case Success(head):
            pass  # Continue with verification

    # Call verify_chain_detailed which does the actual validation
    # Note: verify_chain_detailed can still raise VersionNotFoundError
    # from get_version(), but this is a data corruption issue, not an S3 error
    try:
        report = await verify_chain_detailed(store)
        return Success(report)
    except ClientError:
        # If we hit a ClientError during verification, it's an S3 error
        # This can happen if get_version() encounters S3 errors
        from .s3_errors import S3UnknownError

        return Failure(
            S3UnknownError(
                error_code="VerificationError",
                message="S3 error during chain verification",
            )
        )


async def find_corruption(store: AsyncBlockchainModelStore) -> Optional[ModelVersion]:
    """
    Find first corrupted version in chain.

    Args:
        store: AsyncBlockchainModelStore instance

    Returns:
        First corrupted ModelVersion, or None if chain intact

    Example:
        ```python
        corrupted = await find_corruption(store)
        if corrupted:
            print(f"Corruption at version {corrupted.counter}")
        ```
    """
    report = await verify_chain_detailed(store)
    return report.corrupted_version


async def verify_version_completeness(
    store: AsyncBlockchainModelStore, version: ModelVersion
) -> bool:
    """
    Verify that a version has all required artifacts.

    Checks for:
    - checkpoint.pb
    - metadata.json
    - content_hash.txt

    Args:
        store: AsyncBlockchainModelStore instance
        version: Version to check

    Returns:
        True if all artifacts exist

    Raises:
        ValueError: If any required artifact is missing
    """
    if store._s3_client is None:
        raise RuntimeError(
            "S3 client not initialized. Use 'async with' context manager."
        )

    version_dir = version.directory_name
    required_files = [
        f"versions/{version_dir}/checkpoint.pb",
        f"versions/{version_dir}/metadata.json",
        f"versions/{version_dir}/content_hash.txt",
    ]

    for key in required_files:
        try:
            await store._s3_client.head_object(Bucket=store.bucket_name, Key=key)
        except ClientError as e:
            raise ValueError(f"Version {version.counter} missing artifact: {key} - {e}")

    return True
