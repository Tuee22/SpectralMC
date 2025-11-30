"""
Metadata Effect ADTs for state management during effect execution.

This module defines frozen dataclasses representing metadata operations
for tracking training state like sobol_skip and global_step.

Type Safety:
    - All effect types are frozen dataclasses (immutable)
    - Literal discriminators enable exhaustive pattern matching
    - Union types define closed sets of effect variants

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ReadMetadata:
    """Request to read a value from the metadata registry.

    Attributes:
        kind: Discriminator for pattern matching. Always "ReadMetadata".
        key: Key to read from metadata registry.
        output_id: Identifier for storing the read value.
    """

    kind: Literal["ReadMetadata"] = "ReadMetadata"
    key: str = ""
    output_id: str = ""


@dataclass(frozen=True)
class UpdateMetadata:
    """Request to update a value in the metadata registry.

    Attributes:
        kind: Discriminator for pattern matching. Always "UpdateMetadata".
        key: Key to update in metadata registry.
        operation: Type of update operation.
        value: Value for the operation (interpretation depends on operation).

    Operations:
        - "set": Replace value with the provided value
        - "add": Add the provided value to current value (numeric)
        - "increment": Increment by 1 (ignores value parameter)
    """

    kind: Literal["UpdateMetadata"] = "UpdateMetadata"
    key: str = ""
    operation: Literal["set", "add", "increment"] = "set"
    value: int | float | str = 0


# Metadata Effect Union
MetadataEffect = ReadMetadata | UpdateMetadata
