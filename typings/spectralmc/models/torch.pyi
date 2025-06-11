"""
Stub for ``spectralmc.models.torch`` exposing just ``AdamOptimizerState`` with
relaxed typing so mypy accepts the project’s usage.
"""

from typing import Any, Dict, Protocol

class _HasStateDict(Protocol):
    def state_dict(self) -> Dict[str, Any]: ...

class AdamOptimizerState:
    @classmethod
    def from_torch(cls, state: Dict[str, Any]) -> "AdamOptimizerState": ...
    def to_torch(self, *, device: Any) -> Dict[str, Any]: ...
