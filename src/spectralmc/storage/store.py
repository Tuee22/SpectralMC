# src/spectralmc/storage/store.py
"""
Production-grade blockchain model store with async S3 operations.

Implements:
- 10-step atomic commit with ETag-based Compare-And-Swap (CAS)
- Conflict resolution and automatic rollback
- Connection pooling and exponential backoff retry logic
- Audit logging for compliance and debugging
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from functools import wraps
from typing import Optional, Dict, Any, Callable, TypeVar, cast
from types import TracebackType

import aioboto3  # type: ignore[import-untyped]
from botocore.config import Config
from botocore.exceptions import ClientError

from .chain import ModelVersion, create_genesis_version, bump_semantic_version
from .errors import (
    NotFastForwardError,
    VersionNotFoundError,
    ConflictError,
    CommitError,
)

# Type variable for retry decorator
F = TypeVar("F", bound=Callable[..., Any])


def retry_on_throttle(
    max_retries: int = 5, base_delay: float = 0.1, max_delay: float = 5.0
) -> Callable[[F], F]:
    """
    Decorator to retry S3 operations on throttling errors.

    Uses exponential backoff: delay = min(base_delay * (2 ** attempt), max_delay)

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Decorated async function with retry logic
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except ClientError as e:
                    error_code = e.response["Error"]["Code"]

                    # Don't retry on precondition failures (conflicts)
                    if error_code in ("PreconditionFailed", "412"):
                        raise

                    # Retry on throttling and transient errors
                    if error_code in (
                        "SlowDown",
                        "RequestLimitExceeded",
                        "ServiceUnavailable",
                    ):
                        if attempt < max_retries:
                            delay = min(base_delay * (2**attempt), max_delay)
                            await asyncio.sleep(delay)
                            last_exception = e
                            continue

                    # Don't retry other errors
                    raise

            # Max retries exceeded
            if last_exception:
                raise last_exception

            raise RuntimeError("Unexpected retry logic failure")

        return cast(F, wrapper)

    return decorator


class AsyncBlockchainModelStore:
    """
    Production-grade blockchain model store with async S3 operations.

    Features:
    - Atomic commits with Compare-And-Swap (CAS) using S3 ETags
    - Automatic conflict detection and rollback
    - Connection pooling for high-throughput workloads
    - Exponential backoff retry logic for throttling
    - Append-only audit log for compliance

    Usage:
        async with AsyncBlockchainModelStore("opt-models") as store:
            version = await store.commit(checkpoint_data, content_hash, "Initial commit")
            loaded_data = await store.load_checkpoint(version)
    """

    def __init__(
        self,
        bucket_name: str = "opt-models",
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
    ) -> None:
        """
        Initialize async S3 store.

        Args:
            bucket_name: S3 bucket name (default: "opt-models")
            endpoint_url: S3 endpoint URL (reads from AWS_ENDPOINT_URL env if None)
            aws_access_key_id: AWS access key (reads from AWS_ACCESS_KEY_ID env if None)
            aws_secret_access_key: AWS secret key (reads from AWS_SECRET_ACCESS_KEY env if None)
            region_name: AWS region (default: "us-east-1")
        """
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url or os.environ.get("AWS_ENDPOINT_URL")
        self.aws_access_key_id = aws_access_key_id or os.environ.get(
            "AWS_ACCESS_KEY_ID"
        )
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        )
        self.region_name = region_name

        # Connection pooling configuration
        self.boto_config = Config(
            max_pool_connections=50,
            connect_timeout=5,
            read_timeout=60,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        # Session and client (initialized in __aenter__)
        self.session: Optional[aioboto3.Session] = None
        self._s3_client: Any = None  # aioboto3 client
        self._client_context: Any = None  # Context manager for client

    async def __aenter__(self) -> AsyncBlockchainModelStore:
        """Enter async context manager."""
        self.session = aioboto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

        self._client_context = self.session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            config=self.boto_config,
        )
        self._s3_client = await self._client_context.__aenter__()

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit async context manager."""
        if self._client_context:
            await self._client_context.__aexit__(exc_type, exc_val, exc_tb)
            self._s3_client = None
            self._client_context = None

    # -------------------------------------------------------------------------
    # Internal helper methods
    # -------------------------------------------------------------------------

    @retry_on_throttle(max_retries=5)
    async def _upload_bytes(self, key: str, data: bytes) -> None:
        """Upload bytes to S3 with retry logic."""
        await self._s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
        )

    @retry_on_throttle(max_retries=5)
    async def _upload_json(self, key: str, data: Dict[str, Any]) -> None:
        """Upload JSON to S3 with retry logic."""
        await self._s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(data, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    @retry_on_throttle(max_retries=5)
    async def _download_bytes(self, key: str) -> bytes:
        """Download bytes from S3 with retry logic."""
        response = await self._s3_client.get_object(
            Bucket=self.bucket_name,
            Key=key,
        )
        # Read the streaming body
        data = await response["Body"].read()
        return cast(bytes, data)

    @retry_on_throttle(max_retries=5)
    async def _download_json(self, key: str) -> Dict[str, Any]:
        """Download JSON from S3 with retry logic."""
        data = await self._download_bytes(key)
        return cast(Dict[str, Any], json.loads(data.decode("utf-8")))

    @retry_on_throttle(max_retries=5)
    async def _delete_object(self, key: str) -> None:
        """Delete object from S3 (used for rollback)."""
        try:
            await self._s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=key,
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code != "NoSuchKey":
                raise

    async def _rollback_artifacts(self, version_dir: str) -> None:
        """
        Delete uploaded artifacts for a failed commit.

        Args:
            version_dir: Version directory name (e.g., "v0000000042_1.0.42_abc123")
        """
        keys_to_delete = [
            f"versions/{version_dir}/checkpoint.pb",
            f"versions/{version_dir}/metadata.json",
            f"versions/{version_dir}/content_hash.txt",
        ]

        # Delete in parallel
        await asyncio.gather(
            *[self._delete_object(key) for key in keys_to_delete],
            return_exceptions=True,  # Don't fail if some deletes fail
        )

    @retry_on_throttle(max_retries=5)
    async def _append_audit_log(self, version: ModelVersion) -> None:
        """
        Append commit to audit log for compliance and debugging.

        Args:
            version: ModelVersion to log
        """
        log_entry = {
            "version_id": version.version_id,
            "counter": version.counter,
            "semantic_version": version.semantic_version,
            "parent_hash": version.parent_hash,
            "content_hash": version.content_hash,
            "commit_timestamp": version.commit_timestamp,
            "commit_message": version.commit_message,
        }

        # Append to audit log (JSONL format)
        log_line = json.dumps(log_entry) + "\n"

        # Note: S3 doesn't support append operations natively
        # In production, use Amazon Kinesis Data Firehose or similar
        # For now, we create timestamped log files
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        log_key = f"audit_log/{timestamp}_{version.version_id}.jsonl"

        await self._upload_bytes(log_key, log_line.encode("utf-8"))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @retry_on_throttle(max_retries=5)
    async def get_head(self) -> Optional[ModelVersion]:
        """
        Get current HEAD version.

        Returns:
            Current HEAD version, or None if no commits yet

        Raises:
            ClientError: If S3 operation fails
        """
        try:
            data = await self._download_json("chain.json")
            return ModelVersion(**data)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                return None
            raise

    async def commit(
        self, checkpoint_data: bytes, content_hash: str, message: str = ""
    ) -> ModelVersion:
        """
        Commit a new model version with atomic CAS.

        Implements 10-step atomic commit algorithm:
        1. Checkpoint data and hash already provided by caller
        2. Fetch current HEAD
        3. Build new ModelVersion metadata
        4. Upload artifacts in parallel (checkpoint.pb, metadata.json, content_hash.txt)
        5. Fetch chain.json + ETag
        6. Verify fast-forward (parent hash == current HEAD)
        7. CAS write: PUT chain.json with If-Match=<ETag>
        8. On conflict (412 Precondition Failed): Rollback + raise ConflictError
        9. On success: Append to audit log
        10. Return ModelVersion

        Args:
            checkpoint_data: Serialized checkpoint bytes
            content_hash: SHA256 hash of checkpoint
            message: Optional commit message

        Returns:
            New ModelVersion

        Raises:
            ConflictError: If concurrent commit detected
            CommitError: If commit fails for other reasons
        """
        # Step 2: Fetch current HEAD
        current_head = await self.get_head()

        # Step 3: Build ModelVersion metadata
        if current_head is None:
            # Genesis commit
            version = create_genesis_version(content_hash, message or "Genesis version")
        else:
            # Incremental commit
            new_counter = current_head.counter + 1
            new_semver = bump_semantic_version(current_head.semantic_version, "patch")

            version = ModelVersion(
                counter=new_counter,
                semantic_version=new_semver,
                parent_hash=current_head.content_hash,
                content_hash=content_hash,
                commit_timestamp=datetime.now(timezone.utc).isoformat(),
                commit_message=message,
            )

        version_dir = version.directory_name

        # Step 4: Upload artifacts in parallel
        try:
            await asyncio.gather(
                self._upload_bytes(
                    f"versions/{version_dir}/checkpoint.pb", checkpoint_data
                ),
                self._upload_json(
                    f"versions/{version_dir}/metadata.json",
                    {
                        "counter": version.counter,
                        "semantic_version": version.semantic_version,
                        "parent_hash": version.parent_hash,
                        "content_hash": version.content_hash,
                        "commit_timestamp": version.commit_timestamp,
                        "commit_message": version.commit_message,
                    },
                ),
                self._upload_bytes(
                    f"versions/{version_dir}/content_hash.txt",
                    content_hash.encode("utf-8"),
                ),
            )
        except Exception as e:
            # Rollback on upload failure
            await self._rollback_artifacts(version_dir)
            raise CommitError(f"Failed to upload artifacts: {e}")

        # Steps 5-7: Fetch chain.json + ETag, verify fast-forward, CAS write
        try:
            # Fetch current chain.json with ETag
            etag: Optional[str] = None
            try:
                response = await self._s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key="chain.json",
                )
                etag = response["ETag"].strip('"')  # Remove quotes from ETag
                chain_data_bytes = await response["Body"].read()
                chain_data = json.loads(chain_data_bytes.decode("utf-8"))

                # Verify fast-forward (parent hash matches current head)
                if (
                    current_head
                    and chain_data["content_hash"] != current_head.content_hash
                ):
                    # Someone committed between our get_head() and now
                    await self._rollback_artifacts(version_dir)
                    raise ConflictError(
                        f"Concurrent commit detected: expected head {current_head.content_hash[:8]}, "
                        f"got {chain_data['content_hash'][:8]}"
                    )
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "NoSuchKey":
                    # Genesis commit - no ETag required
                    etag = None
                else:
                    await self._rollback_artifacts(version_dir)
                    raise

            # CAS write: PUT chain.json with If-Match=<ETag>
            version_dict = {
                "counter": version.counter,
                "semantic_version": version.semantic_version,
                "parent_hash": version.parent_hash,
                "content_hash": version.content_hash,
                "commit_timestamp": version.commit_timestamp,
                "commit_message": version.commit_message,
            }

            put_kwargs: Dict[str, Any] = {
                "Bucket": self.bucket_name,
                "Key": "chain.json",
                "Body": json.dumps(version_dict, indent=2).encode("utf-8"),
                "ContentType": "application/json",
            }

            if etag is not None:
                # Non-genesis commit: use CAS
                put_kwargs["IfMatch"] = etag

            await self._s3_client.put_object(**put_kwargs)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            # Step 8: On conflict, rollback and raise
            if error_code in ("PreconditionFailed", "412"):
                await self._rollback_artifacts(version_dir)
                raise ConflictError("Concurrent commit detected during CAS write")

            # Other errors
            await self._rollback_artifacts(version_dir)
            raise CommitError(f"Failed to update chain.json: {e}")

        # Step 9: Append to audit log
        try:
            await self._append_audit_log(version)
        except Exception:
            # Don't fail commit if audit log fails (best-effort)
            pass

        # Step 10: Return ModelVersion
        return version

    @retry_on_throttle(max_retries=5)
    async def get_version(self, version_id: str) -> ModelVersion:
        """
        Retrieve a specific version by ID.

        Args:
            version_id: Version identifier (e.g., "v0000000000")

        Returns:
            ModelVersion

        Raises:
            VersionNotFoundError: If version doesn't exist
        """
        # List all version directories
        try:
            response = await self._s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="versions/",
                Delimiter="/",
            )

            if "CommonPrefixes" not in response:
                raise VersionNotFoundError(version_id)

            # Find matching version directory
            for prefix_obj in response["CommonPrefixes"]:
                prefix = prefix_obj[
                    "Prefix"
                ]  # e.g., "versions/v0000000000_1.0.0_abc123/"
                dir_name = prefix.rstrip("/").split("/")[-1]

                if dir_name.startswith(version_id):
                    # Load metadata
                    metadata_key = f"versions/{dir_name}/metadata.json"
                    data = await self._download_json(metadata_key)
                    return ModelVersion(**data)

            raise VersionNotFoundError(version_id)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise VersionNotFoundError(version_id)
            raise

    @retry_on_throttle(max_retries=5)
    async def load_checkpoint(self, version: ModelVersion) -> bytes:
        """
        Load checkpoint bytes for a version.

        Args:
            version: ModelVersion to load

        Returns:
            Checkpoint bytes

        Raises:
            VersionNotFoundError: If checkpoint doesn't exist
        """
        try:
            checkpoint_key = f"versions/{version.directory_name}/checkpoint.pb"
            return await self._download_bytes(checkpoint_key)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise VersionNotFoundError(version.version_id)
            raise


__all__ = ["AsyncBlockchainModelStore", "retry_on_throttle"]
