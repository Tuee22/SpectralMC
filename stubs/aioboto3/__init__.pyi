"""
Minimal type stubs for aioboto3 S3 operations.

SpectralMC uses aioboto3 exclusively for S3 client operations in the blockchain
model storage system. This stub provides type coverage for only the S3 client
methods actually used by the codebase.

Style Guide Compliance:
- Zero tolerance: No `Any`, `cast()`, or `# type: ignore`
- Minimal surface: Only methods used by SpectralMC
- Type-pure: All signatures fully typed

Note: This stub is designed to be compatible with the Protocol types defined
in src/spectralmc/storage/store.py (_SessionProtocol, _S3ClientProtocol).
"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol

# Import Config from botocore for session configuration
from botocore.config import Config

# Async context manager protocol that matches store.py's expectations
class _AsyncContextManagerProtocol(Protocol):
    """Protocol for async context manager returned by session.client()."""

    async def __aenter__(self) -> object: ...  # Returns S3 client
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...

# Session protocol that matches store.py's _SessionProtocol
class _SessionProtocol(Protocol):
    """Session protocol - matches aioboto3.Session and store.py's _SessionProtocol."""

    def client(
        self,
        service_name: str,
        endpoint_url: str | None = None,
        config: Config | None = None,
        **kwargs: object,
    ) -> _AsyncContextManagerProtocol: ...

# Factory function
def Session(
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    region_name: str | None = None,
) -> _SessionProtocol:
    """Create a new aioboto3 session."""
    ...
