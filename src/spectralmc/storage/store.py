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
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import (
    Callable,
    Coroutine,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
)

import aioboto3  # Type stub: stubs/aioboto3/__init__.pyi
from botocore.config import Config
from botocore.exceptions import ClientError

from ..result import Failure, Result, Success
from .chain import ModelVersion, bump_semantic_version, create_genesis_version
from .errors import (
    AuditLogError,
    CommitError,
    ConflictError,
    HeadNotFoundError,
    VersionNotFoundError,
)
from .protocols import (
    S3ClientProtocol,
)
from .s3_errors import S3NetworkError, S3ObjectNotFound, S3OperationError, S3UnknownError
from .s3_operations import S3Operations


_logger = logging.getLogger(__name__)

# JSON type alias for nested dictionaries
JsonValue: TypeAlias = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]
JsonDict: TypeAlias = dict[str, JsonValue]


# Additional local Protocol for streaming body (not in shared protocols)
class _StreamingBodyProtocol(Protocol):
    """Protocol for S3 StreamingBody."""

    async def read(self) -> bytes: ...


# Type variables for retry decorator
P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class RetryScheduled:
    """Planned retry with bounded, explicit delay."""

    attempt: int
    delay_seconds: float
    scheduled_for: datetime


@dataclass(frozen=True)
class RetryExhausted(Exception):
    """Retry budget consumed (explicitly typed to avoid silent loops)."""

    attempts: int


@dataclass(frozen=True)
class RetryGiveUp:
    """Explicit stop signal when retries are not allowed for an error."""

    reason: str


RetryControl = RetryScheduled | RetryExhausted | RetryGiveUp


def _retry_schedule(max_retries: int, base_delay: float, max_delay: float) -> list[RetryScheduled]:
    """Deterministic retry plan to avoid unbounded sleeps."""
    now = datetime.now(UTC)
    schedule: list[RetryScheduled] = []
    cumulative_delay = 0.0
    for attempt in range(max_retries):
        delay_seconds = min(base_delay * (2**attempt), max_delay)
        cumulative_delay += delay_seconds
        schedule.append(
            RetryScheduled(
                attempt=attempt,
                delay_seconds=delay_seconds,
                scheduled_for=now + timedelta(seconds=cumulative_delay),
            )
        )
    return schedule


def _retry_decision(
    *,
    attempt: int,
    error_code: str,
    schedule: list[RetryScheduled],
    max_retries: int,
) -> RetryControl:
    """Map an error code and attempt to an explicit retry control signal."""
    if error_code in ("PreconditionFailed", "412"):
        return RetryGiveUp(reason="precondition_failed")

    if error_code in ("SlowDown", "RequestLimitExceeded", "ServiceUnavailable"):
        if attempt < max_retries:
            if attempt >= len(schedule):
                return RetryExhausted(attempts=attempt + 1)
            return schedule[attempt]
        return RetryExhausted(attempts=attempt + 1)

    return RetryGiveUp(reason=f"non_retryable:{error_code}")


async def _await_retry(schedule_entry: RetryScheduled) -> None:
    """Effectful wait for the next retry window (bounded by schedule)."""
    remaining = (schedule_entry.scheduled_for - datetime.now(UTC)).total_seconds()
    if remaining > 0:
        await asyncio.sleep(remaining)


def retry_on_throttle(
    max_retries: int = 5, base_delay: float = 0.1, max_delay: float = 5.0
) -> Callable[
    [Callable[P, Coroutine[object, object, R]]],
    Callable[P, Coroutine[object, object, R]],
]:
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

    def decorator(
        func: Callable[P, Coroutine[object, object, R]],
    ) -> Callable[P, Coroutine[object, object, R]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            schedule = _retry_schedule(max_retries, base_delay, max_delay)

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except ClientError as e:
                    error_code = e.response["Error"]["Code"]

                    match _retry_decision(
                        attempt=attempt,
                        error_code=error_code,
                        schedule=schedule,
                        max_retries=max_retries,
                    ):
                        case RetryGiveUp():
                            raise
                        case RetryExhausted():
                            # Exhausted retry budget: surface original error for callers/tests
                            raise e
                        case RetryScheduled() as retry_plan:
                            await _await_retry(retry_plan)
                            continue

            raise RuntimeError("Unexpected retry logic failure")

        # Manually preserve function metadata (avoiding @wraps to prevent Any)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        wrapper.__qualname__ = func.__qualname__
        wrapper.__annotations__ = func.__annotations__

        return wrapper

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
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
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
        self.aws_access_key_id = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
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
        # Typed as object to avoid cross-module Protocol incompatibility with aioboto3 stubs
        # Runtime behavior verified via hasattr checks; stored for lifecycle management
        self.session: object | None = None
        self._s3_client: S3ClientProtocol | None = None
        self._client_context: object | None = (
            None  # Context manager for client (object for aioboto3 compat)
        )
        # Functional S3 operations wrapper (initialized in __aenter__)
        self._s3_ops: S3Operations | None = None

    async def __aenter__(self) -> AsyncBlockchainModelStore:
        """Enter async context manager."""
        # Create session - aioboto3.Session returns a session object with .client() method
        session = aioboto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )
        self.session = session

        # Get client context manager from session
        # The session.client() method returns an async context manager
        client_context = session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            config=self.boto_config,
        )
        self._client_context = client_context

        # Enter the context manager to get the S3 client
        # Use await on the context manager's __aenter__ via the async with protocol
        self._s3_client = await client_context.__aenter__()

        # Initialize functional S3 operations wrapper
        self._s3_ops = S3Operations(self._s3_client)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit async context manager."""
        client_ctx = self._client_context
        if client_ctx is not None and hasattr(client_ctx, "__aexit__"):
            # Runtime check ensures client_ctx has __aexit__ (context manager protocol)
            aexit_method = client_ctx.__aexit__
            await aexit_method(exc_type, exc_val, exc_tb)
            self._s3_client = None
            self._client_context = None
        return None

    # -------------------------------------------------------------------------
    # Internal helper methods
    # -------------------------------------------------------------------------

    @retry_on_throttle(max_retries=5)
    async def _upload_bytes(self, key: str, data: bytes) -> None:
        """Upload bytes to S3 with retry logic."""
        if self._s3_client is None:
            raise RuntimeError("S3 client not initialized. Use 'async with' context manager.")
        await self._s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
        )

    @retry_on_throttle(max_retries=5)
    async def _upload_json(self, key: str, data: JsonDict) -> None:
        """Upload JSON to S3 with retry logic."""
        if self._s3_client is None:
            raise RuntimeError("S3 client not initialized. Use 'async with' context manager.")
        await self._s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(data, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    @retry_on_throttle(max_retries=5)
    async def _download_bytes(self, key: str) -> bytes:
        """Download bytes from S3 with retry logic."""
        if self._s3_client is None:
            raise RuntimeError("S3 client not initialized. Use 'async with' context manager.")
        response = await self._s3_client.get_object(
            Bucket=self.bucket_name,
            Key=key,
        )
        # Read the streaming body - extract Body field
        body = response["Body"]
        if not hasattr(body, "read"):
            raise TypeError(f"Expected streaming body with read() method, got {type(body)}")
        # Runtime check ensures body has read() method (StreamingBody has no stubs)
        data: bytes = await body.read()
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data)}")
        return data

    @retry_on_throttle(max_retries=5)
    async def _download_json(self, key: str) -> JsonDict:
        """Download JSON from S3 with retry logic."""
        data = await self._download_bytes(key)
        result = json.loads(data.decode("utf-8"))
        # Validate it's a dict at runtime
        if not isinstance(result, dict):
            raise TypeError(f"Expected dict, got {type(result)}")
        return result

    @retry_on_throttle(max_retries=5)
    async def _delete_object(self, key: str) -> None:
        """Delete object from S3 (used for rollback)."""
        if self._s3_client is None:
            raise RuntimeError("S3 client not initialized. Use 'async with' context manager.")
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
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        log_key = f"audit_log/{timestamp}_{version.version_id}.jsonl"

        await self._upload_bytes(log_key, log_line.encode("utf-8"))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def get_head(
        self,
    ) -> Result[ModelVersion, HeadNotFoundError | S3OperationError]:
        """
        Get current HEAD version with automatic retry on throttling.

        Retries up to 5 times with exponential backoff (0.1s, 0.2s, 0.4s, 0.8s, 1.6s)
        when encountering S3NetworkError (throttling/transient failures).

        Returns:
            Success(ModelVersion) if HEAD exists
            Failure(HeadNotFoundError) if no commits yet (empty chain)
            Failure(S3OperationError) for S3 operational failures:
                - S3BucketNotFound: Bucket doesn't exist
                - S3AccessDenied: Permission denied
                - S3NetworkError: Network/timeout error (after max retries)
                - S3UnknownError: Other S3 errors

        Example:
            ```python
            async with AsyncBlockchainModelStore("my-bucket") as store:
                result = await store.get_head()
                match result:
                    case Success(head):
                        print(f"Current HEAD: v{head.counter}")
                    case Failure(HeadNotFoundError()):
                        print("No commits yet")
                    case Failure(S3BucketNotFound(bucket=b)):
                        print(f"Bucket not found: {b}")
                    case Failure(error):
                        print(f"S3 error: {error}")
            ```
        """
        if self._s3_ops is None:
            # This shouldn't happen if using async context manager correctly
            return Failure(
                S3UnknownError(
                    error_code="NotInitialized",
                    message="S3Operations not initialized. Use 'async with' context manager.",
                )
            )

        # Retry logic for throttling errors
        max_retries = 5
        base_delay = 0.1
        max_delay = 5.0

        for attempt in range(max_retries + 1):
            # Use functional S3Operations to get chain.json
            result = await self._s3_ops.get_object(self.bucket_name, "chain.json")

            match result:
                case Failure(error):
                    # Check if it's S3ObjectNotFound (empty chain)
                    if isinstance(error, S3ObjectNotFound):
                        # Empty chain is valid state - return immediately
                        return Failure(HeadNotFoundError())

                    # Check if it's a network/throttling error that should be retried
                    if isinstance(error, S3NetworkError):
                        if attempt < max_retries:
                            # Retry with exponential backoff
                            delay = min(base_delay * (2**attempt), max_delay)
                            await asyncio.sleep(delay)
                            continue  # Retry

                        # Max retries exceeded - return the last network error
                        return Failure(error)

                    # All other S3 errors (BucketNotFound, AccessDenied, etc.) - fail immediately
                    return Failure(error)

                case Success(data):
                    # Success - break out of retry loop and continue with parsing
                    break

        # At this point, we must have Success(data) from the match statement
        # Otherwise we would have returned already within the retry loop
        match result:
            case Success(data):
                # Parse and validate JSON
                try:
                    parsed = json.loads(data.decode("utf-8"))
                    if not isinstance(parsed, dict):
                        return Failure(
                            S3UnknownError(
                                error_code="InvalidData",
                                message=f"Expected dict, got {type(parsed)}",
                            )
                        )

                    # Extract and validate fields
                    counter = parsed.get("counter")
                    semantic_version = parsed.get("semantic_version")
                    parent_hash = parsed.get("parent_hash")
                    content_hash = parsed.get("content_hash")
                    commit_timestamp = parsed.get("commit_timestamp")
                    commit_message = parsed.get("commit_message")

                    if not isinstance(counter, int):
                        return Failure(
                            S3UnknownError(
                                error_code="InvalidData",
                                message=f"counter must be int, got {type(counter)}",
                            )
                        )
                    if not isinstance(semantic_version, str):
                        return Failure(
                            S3UnknownError(
                                error_code="InvalidData",
                                message=f"semantic_version must be str, got {type(semantic_version)}",
                            )
                        )
                    if not isinstance(parent_hash, str):
                        return Failure(
                            S3UnknownError(
                                error_code="InvalidData",
                                message=f"parent_hash must be str, got {type(parent_hash)}",
                            )
                        )
                    if not isinstance(content_hash, str):
                        return Failure(
                            S3UnknownError(
                                error_code="InvalidData",
                                message=f"content_hash must be str, got {type(content_hash)}",
                            )
                        )
                    if not isinstance(commit_timestamp, str):
                        return Failure(
                            S3UnknownError(
                                error_code="InvalidData",
                                message=f"commit_timestamp must be str, got {type(commit_timestamp)}",
                            )
                        )
                    if not isinstance(commit_message, str):
                        return Failure(
                            S3UnknownError(
                                error_code="InvalidData",
                                message=f"commit_message must be str, got {type(commit_message)}",
                            )
                        )

                    version = ModelVersion(
                        counter=counter,
                        semantic_version=semantic_version,
                        parent_hash=parent_hash,
                        content_hash=content_hash,
                        commit_timestamp=commit_timestamp,
                        commit_message=commit_message,
                    )
                    return Success(version)

                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    return Failure(
                        S3UnknownError(
                            error_code="InvalidJSON",
                            message=f"Failed to parse chain.json: {e}",
                        )
                    )

            case Failure(error):
                # This should never happen - all Failure cases are handled in the retry loop
                # But we include it for exhaustiveness and type safety
                return Failure(error)

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
        head_result = await self.get_head()

        # Step 3: Build ModelVersion metadata
        current_head: ModelVersion | None = None
        match head_result:
            case Failure(_):
                # Genesis commit
                version = create_genesis_version(content_hash, message or "Genesis version")
            case Success(head):
                # Incremental commit
                current_head = head
                new_counter = current_head.counter + 1
                new_semver = bump_semantic_version(current_head.semantic_version, "patch")

                version = ModelVersion(
                    counter=new_counter,
                    semantic_version=new_semver,
                    parent_hash=current_head.content_hash,
                    content_hash=content_hash,
                    commit_timestamp=datetime.now(UTC).isoformat(),
                    commit_message=message,
                )

        version_dir = version.directory_name

        # Step 4: Upload artifacts in parallel
        upload_tasks = [
            asyncio.create_task(
                self._upload_bytes(f"versions/{version_dir}/checkpoint.pb", checkpoint_data)
            ),
            asyncio.create_task(
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
                )
            ),
            asyncio.create_task(
                self._upload_bytes(
                    f"versions/{version_dir}/content_hash.txt",
                    content_hash.encode("utf-8"),
                )
            ),
        ]

        upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        upload_errors = [res for res in upload_results if isinstance(res, BaseException)]
        if upload_errors:
            await self._rollback_artifacts(version_dir)
            root_error = upload_errors[0]
            if isinstance(root_error, (ClientError, OSError)):
                raise CommitError(f"Failed to upload artifacts: {root_error}") from root_error
            raise CommitError(f"Failed to upload artifacts: {root_error}")

        # Steps 5-7: Fetch chain.json + ETag, verify fast-forward, CAS write
        try:
            # Fetch current chain.json with ETag
            etag: str | None = None
            if self._s3_client is None:
                raise RuntimeError("S3 client not initialized. Use 'async with' context manager.")

            try:
                response = await self._s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key="chain.json",
                )
                # Extract ETag field
                etag_obj = response["ETag"]
                if not isinstance(etag_obj, str):
                    raise TypeError(f"ETag must be str, got {type(etag_obj)}")
                etag = etag_obj.strip('"')  # Remove quotes from ETag

                # Extract Body field and read
                body = response["Body"]
                if not hasattr(body, "read"):
                    raise TypeError(f"Expected streaming body with read() method, got {type(body)}")
                # Runtime check ensures body has read() method (StreamingBody has no stubs)
                chain_data_bytes: bytes = await body.read()
                if not isinstance(chain_data_bytes, bytes):
                    raise TypeError(f"Expected bytes, got {type(chain_data_bytes)}")

                chain_data = json.loads(chain_data_bytes.decode("utf-8"))

                # Verify fast-forward (parent hash matches current head)
                if (
                    current_head
                    and isinstance(chain_data, dict)
                    and chain_data.get("content_hash") != current_head.content_hash
                ):
                    # Someone committed between our get_head() and now
                    await self._rollback_artifacts(version_dir)
                    raise ConflictError(
                        f"Concurrent commit detected: expected head {current_head.content_hash[:8]}, "
                        f"got {str(chain_data.get('content_hash', ''))[:8]}"
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

            put_kwargs: dict[str, object] = {
                "Bucket": self.bucket_name,
                "Key": "chain.json",
                "Body": json.dumps(version_dict, indent=2).encode("utf-8"),
                "ContentType": "application/json",
            }

            if etag is not None:
                # Non-genesis commit: use CAS
                put_kwargs["IfMatch"] = etag

            if self._s3_client is None:
                raise RuntimeError("S3 client not initialized. Use 'async with' context manager.")
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
        except ClientError as e:
            _logger.warning(
                f"Audit log append failed (non-fatal) for version {version.version_id}: {e}"
            )
        except (AuditLogError, OSError) as e:
            _logger.error(
                f"Unexpected audit log error for version {version.version_id}: {e}",
                exc_info=True,
            )

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
        if self._s3_client is None:
            raise RuntimeError("S3 client not initialized. Use 'async with' context manager.")

        try:
            response = await self._s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="versions/",
                Delimiter="/",
            )

            # Runtime validation of response structure
            if not isinstance(response, dict):
                raise TypeError(f"Expected dict response, got {type(response)}")

            if "CommonPrefixes" not in response:
                raise VersionNotFoundError(version_id)

            common_prefixes = response["CommonPrefixes"]
            if not isinstance(common_prefixes, list):
                raise TypeError(f"Expected list for CommonPrefixes, got {type(common_prefixes)}")

            # Find matching version directory
            for prefix_obj in common_prefixes:
                if not isinstance(prefix_obj, dict):
                    continue
                prefix_val = prefix_obj.get("Prefix")
                if not isinstance(prefix_val, str):
                    continue
                # e.g., "versions/v0000000000_1.0.0_abc123/"
                dir_name = prefix_val.rstrip("/").split("/")[-1]

                if dir_name.startswith(version_id):
                    # Load metadata
                    metadata_key = f"versions/{dir_name}/metadata.json"
                    data = await self._download_json(metadata_key)

                    # Extract and validate fields
                    counter = data.get("counter")
                    semantic_version = data.get("semantic_version")
                    parent_hash = data.get("parent_hash")
                    content_hash = data.get("content_hash")
                    commit_timestamp = data.get("commit_timestamp")
                    commit_message = data.get("commit_message")

                    if not isinstance(counter, int):
                        raise TypeError(f"counter must be int, got {type(counter)}")
                    if not isinstance(semantic_version, str):
                        raise TypeError(
                            f"semantic_version must be str, got {type(semantic_version)}"
                        )
                    if not isinstance(parent_hash, str):
                        raise TypeError(f"parent_hash must be str, got {type(parent_hash)}")
                    if not isinstance(content_hash, str):
                        raise TypeError(f"content_hash must be str, got {type(content_hash)}")
                    if not isinstance(commit_timestamp, str):
                        raise TypeError(
                            f"commit_timestamp must be str, got {type(commit_timestamp)}"
                        )
                    if not isinstance(commit_message, str):
                        raise TypeError(f"commit_message must be str, got {type(commit_message)}")

                    return ModelVersion(
                        counter=counter,
                        semantic_version=semantic_version,
                        parent_hash=parent_hash,
                        content_hash=content_hash,
                        commit_timestamp=commit_timestamp,
                        commit_message=commit_message,
                    )

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
