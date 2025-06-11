"""
Minimal stub for ``torch.cuda`` needed by *SpectralMC*.
"""

from __future__ import annotations
import torch
from typing import TypeAlias, Union
from .streams import Stream as Stream

Device: TypeAlias = "torch.device"

class Event: ...

def is_available() -> bool: ...
def empty_cache() -> None: ...

class _StreamContext:
    def __init__(self, stream: Stream): ...
    def __enter__(self) -> Stream: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool | None: ...

def stream(stream: Union[int, Device, Stream, None] = ...) -> _StreamContext: ...
