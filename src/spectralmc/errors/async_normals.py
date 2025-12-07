"""Error ADTs for async_normals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class InvalidDType:
    """Unsupported CuPy dtype requested."""

    requested: str
    kind: Literal["InvalidDType"] = "InvalidDType"


@dataclass(frozen=True)
class InvalidShape:
    """Matrix shape is invalid (rows/cols <= 0)."""

    rows: int
    cols: int
    kind: Literal["InvalidShape"] = "InvalidShape"


@dataclass(frozen=True)
class SeedOutOfRange:
    """Seed must be positive and below SEED_LIMIT."""

    seed: int
    kind: Literal["SeedOutOfRange"] = "SeedOutOfRange"


@dataclass(frozen=True)
class QueueEmpty:
    """Attempted to read before enqueue."""

    kind: Literal["QueueEmpty"] = "QueueEmpty"


@dataclass(frozen=True)
class QueueBusy:
    """Attempted to enqueue before consuming prior matrix."""

    kind: Literal["QueueBusy"] = "QueueBusy"
