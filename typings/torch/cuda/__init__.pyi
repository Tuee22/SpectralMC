from __future__ import annotations
from types import TracebackType

class Stream:
    def __init__(
        self,
        device: object | None = ...,  # accepts int, str, torch.device, etc.
        priority: int | None = ...,
    ) -> None: ...
    def synchronize(self) -> None: ...
    def __enter__(self) -> "Stream": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

def empty_cache() -> None: ...
