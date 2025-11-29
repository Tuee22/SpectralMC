# src/spectralmc/storage/protocols.py
"""
Shared Protocol definitions for S3 operations.

This module centralizes all Protocol types used across the storage module
to avoid cross-module Protocol incompatibility issues with MyPy.

All modules that need S3 client Protocols should import from here:
- store.py
- s3_operations.py
- aioboto3 stub (stubs/aioboto3/__init__.pyi)
"""

from __future__ import annotations

from types import TracebackType
from typing import AsyncIterator, Protocol

from botocore.config import Config


# ---------------------------------------------------------------------------
# S3 Response Protocols
# ---------------------------------------------------------------------------


class StreamingBodyProtocol(Protocol):
    """Protocol for S3 StreamingBody."""

    async def read(self) -> bytes: ...


class S3ResponseProtocol(Protocol):
    """Protocol for S3 get_object response."""

    def __getitem__(self, key: str) -> object: ...


# ---------------------------------------------------------------------------
# S3 Paginator Protocols
# ---------------------------------------------------------------------------


class AsyncIteratorProtocol(Protocol):
    """Protocol for async iterator returned by paginate()."""

    def __aiter__(self) -> AsyncIteratorProtocol: ...
    async def __anext__(self) -> object: ...


class PaginatorProtocol(Protocol):
    """Protocol for S3 paginator returned by get_paginator()."""

    def paginate(self, **kwargs: object) -> AsyncIterator[object]: ...


# ---------------------------------------------------------------------------
# S3 Client Protocol
# ---------------------------------------------------------------------------


class S3ClientProtocol(Protocol):
    """
    Protocol for async S3 client.

    This is the shared Protocol that all modules use to type S3 clients.
    """

    async def put_object(self, **kwargs: object) -> object: ...
    async def get_object(self, **kwargs: object) -> S3ResponseProtocol: ...
    async def delete_object(self, **kwargs: object) -> object: ...
    async def list_objects_v2(self, **kwargs: object) -> object: ...
    async def head_object(self, **kwargs: object) -> object: ...
    async def create_bucket(self, **kwargs: object) -> object: ...
    async def delete_bucket(self, **kwargs: object) -> object: ...
    async def list_buckets(self, **kwargs: object) -> object: ...
    def get_paginator(self, operation_name: str) -> PaginatorProtocol: ...
    async def delete_objects(self, **kwargs: object) -> object: ...


# ---------------------------------------------------------------------------
# Async Context Manager Protocol
# ---------------------------------------------------------------------------


class AsyncContextManagerProtocol(Protocol):
    """Protocol for async context manager returned by session.client()."""

    async def __aenter__(self) -> S3ClientProtocol: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...


# ---------------------------------------------------------------------------
# Session Protocol
# ---------------------------------------------------------------------------


class SessionProtocol(Protocol):
    """Protocol for aioboto3.Session."""

    def client(
        self,
        service_name: str,
        endpoint_url: str | None = ...,
        config: Config | None = ...,
        **kwargs: object,
    ) -> AsyncContextManagerProtocol: ...


__all__ = [
    "StreamingBodyProtocol",
    "S3ResponseProtocol",
    "AsyncIteratorProtocol",
    "PaginatorProtocol",
    "S3ClientProtocol",
    "AsyncContextManagerProtocol",
    "SessionProtocol",
]
