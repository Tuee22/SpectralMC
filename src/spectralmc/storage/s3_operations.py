"""Pure functional S3 operations wrapper using Result types.

This module provides a functional interface to S3 operations that wraps all
boto3 ClientError exceptions into typed S3OperationError ADTs, enabling
exhaustive pattern matching and eliminating unhandled exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List

from botocore.exceptions import ClientError

from ..result import Failure, Result, Success
from .s3_errors import (
    S3AccessDenied,
    S3BucketNotFound,
    S3NetworkError,
    S3ObjectNotFound,
    S3OperationError,
    S3UnknownError,
)

if TYPE_CHECKING:
    # Protocol for S3 client - defined in store.py
    from typing import AsyncIterator, Protocol

    class _S3ResponseProtocol(Protocol):
        """Protocol for S3 get_object response."""

        def __getitem__(self, key: str) -> object: ...

    class _PaginatorProtocol(Protocol):
        """Protocol for S3 paginator."""

        def paginate(self, **kwargs: object) -> AsyncIterator[object]: ...

    class _S3ClientProtocol(Protocol):
        """Protocol for async S3 client."""

        async def put_object(self, **kwargs: object) -> object: ...
        async def get_object(self, **kwargs: object) -> _S3ResponseProtocol: ...
        async def delete_object(self, **kwargs: object) -> object: ...
        async def list_objects_v2(self, **kwargs: object) -> object: ...
        def get_paginator(self, operation: str) -> _PaginatorProtocol: ...

    S3Client = _S3ClientProtocol
else:
    from typing import Any as _ClientType
    S3Client = _ClientType


class S3Operations:
    """Pure functional interface for S3 operations.

    All methods return Result[T, S3OperationError] instead of raising exceptions,
    enabling type-safe error handling with exhaustive pattern matching.

    Example:
        ```python
        s3_ops = S3Operations(s3_client)
        result = await s3_ops.get_object("my-bucket", "my-key")

        match result:
            case Success(data):
                # Handle successful retrieval
                process(data)
            case Failure(S3BucketNotFound(bucket, msg)):
                # Handle missing bucket
                logger.error(f"Bucket {bucket} not found: {msg}")
            case Failure(S3ObjectNotFound(bucket, key, msg)):
                # Handle missing object
                logger.error(f"Object {key} not found in {bucket}")
            case Failure(error):
                # Handle other S3 errors
                logger.error(f"S3 error: {error}")
        ```
    """

    def __init__(self, s3_client: S3Client) -> None:
        """Initialize S3 operations wrapper.

        Args:
            s3_client: aioboto3 S3 client instance
        """
        self._client = s3_client

    async def get_object(
        self, bucket: str, key: str
    ) -> Result[bytes, S3OperationError]:
        """Get object from S3.

        Args:
            bucket: S3 bucket name
            key: Object key

        Returns:
            Success(bytes) if object retrieved successfully
            Failure(S3OperationError) for all error cases:
                - S3BucketNotFound: Bucket doesn't exist
                - S3ObjectNotFound: Object/key doesn't exist
                - S3AccessDenied: Permission denied
                - S3NetworkError: Network/timeout error
                - S3UnknownError: Other S3 errors
        """
        try:
            response = await self._client.get_object(Bucket=bucket, Key=key)
            # Read streaming body
            body = response["Body"]
            if not hasattr(body, "read"):
                # This shouldn't happen with real S3 client, but type-safe
                return Failure(
                    S3UnknownError(
                        error_code="InvalidResponse",
                        message=f"Expected streaming body with read() method, got {type(body)}",
                    )
                )
            data = await body.read()
            # Assert for type narrowing - boto3 always returns bytes
            assert isinstance(data, bytes), f"Expected bytes from S3, got {type(data)}"
            return Success(data)
        except ClientError as e:
            return Failure(self._classify_error(e, bucket, key, "GetObject"))

    async def put_object(
        self, bucket: str, key: str, body: bytes, **kwargs: object
    ) -> Result[None, S3OperationError]:
        """Put object to S3.

        Args:
            bucket: S3 bucket name
            key: Object key
            body: Object data as bytes
            **kwargs: Additional arguments to pass to put_object (e.g., Metadata, ContentType)

        Returns:
            Success(None) if object uploaded successfully
            Failure(S3OperationError) for all error cases
        """
        try:
            await self._client.put_object(Bucket=bucket, Key=key, Body=body, **kwargs)
            return Success(None)
        except ClientError as e:
            return Failure(self._classify_error(e, bucket, key, "PutObject"))

    async def delete_object(
        self, bucket: str, key: str
    ) -> Result[None, S3OperationError]:
        """Delete object from S3.

        Args:
            bucket: S3 bucket name
            key: Object key

        Returns:
            Success(None) if object deleted successfully (or didn't exist)
            Failure(S3OperationError) for all error cases
        """
        try:
            await self._client.delete_object(Bucket=bucket, Key=key)
            return Success(None)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            # NoSuchKey is success (idempotent delete)
            if error_code == "NoSuchKey":
                return Success(None)
            return Failure(self._classify_error(e, bucket, key, "DeleteObject"))

    async def list_objects(
        self, bucket: str, prefix: str = ""
    ) -> Result[List[str], S3OperationError]:
        """List object keys in S3 bucket with optional prefix.

        Args:
            bucket: S3 bucket name
            prefix: Optional key prefix to filter objects

        Returns:
            Success(List[str]) with list of object keys
            Failure(S3OperationError) for all error cases
        """
        try:
            keys: List[str] = []
            # Use pagination to handle large listings
            paginator = self._client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if isinstance(page, dict) and "Contents" in page:
                    contents = page["Contents"]
                    if isinstance(contents, list):
                        for obj in contents:
                            if isinstance(obj, dict) and "Key" in obj:
                                key = obj["Key"]
                                if isinstance(key, str):
                                    keys.append(key)
            return Success(keys)
        except ClientError as e:
            return Failure(self._classify_error(e, bucket, "", "ListObjectsV2"))

    def _classify_error(
        self, error: ClientError, bucket: str, key: str, operation: str
    ) -> S3OperationError:
        """Classify boto3 ClientError into specific S3OperationError ADT.

        Args:
            error: boto3 ClientError exception
            bucket: S3 bucket name
            key: Object key (may be empty for bucket-level operations)
            operation: S3 operation that failed (e.g., "GetObject", "PutObject")

        Returns:
            Specific S3OperationError variant based on error code
        """
        error_code = error.response["Error"]["Code"]
        message = error.response["Error"]["Message"]

        # Pattern match on error code to classify error
        match error_code:
            case "NoSuchBucket":
                return S3BucketNotFound(bucket_name=bucket, message=message)

            case "NoSuchKey":
                return S3ObjectNotFound(bucket_name=bucket, key=key, message=message)

            case "AccessDenied" | "Forbidden" | "InvalidAccessKeyId" | "SignatureDoesNotMatch":
                return S3AccessDenied(
                    bucket_name=bucket, operation=operation, message=message
                )

            case "RequestTimeout" | "ServiceUnavailable" | "SlowDown" | "InternalError":
                return S3NetworkError(message=message, retry_count=0)

            case _:
                # Catch-all for unknown error codes
                return S3UnknownError(error_code=error_code, message=message)
