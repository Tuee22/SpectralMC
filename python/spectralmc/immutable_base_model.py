"""
/spectralmc/python/spectralmc/immutable_base_model.py

Contains a Pydantic-based immutable model used by black_scholes.py.

No placeholders or ignoring imports. Must pass mypy --strict.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ImmutableBaseModel(BaseModel):
    """
    Pydantic BaseModel that is frozen (immutable).
    """

    model_config = ConfigDict(frozen=True)
