"""S3 Error ADT - Algebraic Data Types for S3 operation failures.

This module defines frozen dataclasses representing all possible S3 errors,
enabling exhaustive pattern matching and type-safe error handling.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class S3BucketNotFound:
    """S3 bucket does not exist.

    Raised when attempting to access a bucket that doesn't exist.
    Corresponds to boto3 ClientError with code "NoSuchBucket".

    Attributes:
        bucket_name: Name of the bucket that was not found
        message: Error message from S3
    """

    bucket_name: str
    message: str


@dataclass(frozen=True)
class S3ObjectNotFound:
    """S3 object/key does not exist in bucket.

    Raised when attempting to access an object that doesn't exist in the bucket.
    Corresponds to boto3 ClientError with code "NoSuchKey".

    Attributes:
        bucket_name: Name of the bucket
        key: Object key that was not found
        message: Error message from S3
    """

    bucket_name: str
    key: str
    message: str


@dataclass(frozen=True)
class S3AccessDenied:
    """Permission denied for S3 operation.

    Raised when the credentials lack permission to perform the requested operation.
    Corresponds to boto3 ClientError with codes "AccessDenied" or "Forbidden".

    Attributes:
        bucket_name: Name of the bucket being accessed
        operation: S3 operation that was denied (e.g., "GetObject", "PutObject")
        message: Error message from S3
    """

    bucket_name: str
    operation: str
    message: str


@dataclass(frozen=True)
class S3NetworkError:
    """Network/connection error communicating with S3.

    Raised when S3 is unreachable due to network issues or service unavailability.
    Corresponds to boto3 ClientError with codes like "RequestTimeout", "ServiceUnavailable".

    Attributes:
        message: Error message from S3
        retry_count: Number of retries attempted before failure
    """

    message: str
    retry_count: int


@dataclass(frozen=True)
class S3UnknownError:
    """Unknown S3 error.

    Catch-all for S3 errors that don't match specific known patterns.
    Used to ensure exhaustive handling of all ClientError cases.

    Attributes:
        error_code: S3 error code from response
        message: Error message from S3
    """

    error_code: str
    message: str


# Union type for all S3 errors - enables exhaustive pattern matching
# Note: Named S3OperationError to avoid conflict with exception-based S3Error in errors.py
S3OperationError = (
    S3BucketNotFound
    | S3ObjectNotFound
    | S3AccessDenied
    | S3NetworkError
    | S3UnknownError
)
