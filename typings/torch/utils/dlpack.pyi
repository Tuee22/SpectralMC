from __future__ import annotations
import torch
from typing import Protocol, runtime_checkable

@runtime_checkable
class _SupportsDLPack(Protocol):
    def __dlpack__(self, *, stream: int | None = ...) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

def from_dlpack(x: torch.Tensor | _SupportsDLPack, /) -> torch.Tensor: ...
