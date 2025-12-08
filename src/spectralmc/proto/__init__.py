# src/spectralmc/proto/__init__.py
"""
Protobuf schema definitions for SpectralMC model versioning.

Generated Protocol Buffer modules are built into /opt/spectralmc_proto/
during Docker image build and exposed via .pth file mechanism.

This module provides backward-compatible imports by delegating to
/opt/spectralmc_proto/. Direct imports from spectralmc_proto are preferred.
"""

from __future__ import annotations

# Import from /opt/spectralmc_proto/ (available via spectralmc-proto.pth)
# These imports maintain backward compatibility for existing code
from spectralmc_proto import common_pb2
from spectralmc_proto import tensors_pb2
from spectralmc_proto import simulation_pb2
from spectralmc_proto import training_pb2
from spectralmc_proto import models_pb2

__all__ = [
    "common_pb2",
    "tensors_pb2",
    "simulation_pb2",
    "training_pb2",
    "models_pb2",
]
