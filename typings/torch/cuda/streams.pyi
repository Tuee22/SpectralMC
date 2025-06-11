"""
Stub for ``torch.cuda.streams`` (subset).
"""

from __future__ import annotations
import torch
from types import TracebackType
from typing import Optional, TypeAlias, Union

Device: TypeAlias = "torch.device"
Event: TypeAlias = "torch.cuda.Event"

class Stream:
    def __init__(
        self,
        device: Optional[Union[int, Device]] = ...,
        *,
        priority: int = ...,
        capture: bool = ...,
    ) -> None: ...
    def __enter__(self) -> "Stream": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...
    def wait_event(self, event: Event) -> None: ...
    def wait_stream(self, stream: "Stream") -> None: ...
    def query(self) -> bool: ...
    def synchronize(self) -> None: ...
    def record_event(self, event: Optional[Event] = ...) -> Event: ...
    @property
    def priority(self) -> int: ...
    @property
    def device(self) -> Device: ...
    @property
    def pointer(self) -> int: ...
