# src/spectralmc/proto/__init__.py
"""
Protobuf schema definitions for SpectralMC model versioning.

Generated Protocol Buffer modules for blockchain model storage.
"""

from __future__ import annotations

# Re-export generated protobuf modules
from . import common_pb2
from . import tensors_pb2
from . import simulation_pb2
from . import training_pb2
from . import models_pb2

__all__ = [
    "common_pb2",
    "tensors_pb2",
    "simulation_pb2",
    "training_pb2",
    "models_pb2",
]
