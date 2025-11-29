"""
Minimal type stubs for aioboto3 S3 operations.

SpectralMC uses aioboto3 exclusively for S3 client operations in the blockchain
model storage system. This stub imports Protocol types from the shared
spectralmc.storage.protocols module to ensure cross-module type compatibility.

Style Guide Compliance:
- Zero tolerance: No `Any`, `cast()`, or `# type: ignore`
- Minimal surface: Only methods used by SpectralMC
- Type-pure: All signatures fully typed
"""

from __future__ import annotations

# Import shared Protocol types from spectralmc.storage.protocols
# This ensures aioboto3 returns the same Protocol types that store.py expects
from spectralmc.storage.protocols import (
    AsyncContextManagerProtocol,
    SessionProtocol,
)

# Factory function returns the shared SessionProtocol
def Session(
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    region_name: str | None = None,
) -> SessionProtocol:
    """Create a new aioboto3 session."""
    ...
