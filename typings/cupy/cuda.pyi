"""Tiny slice of cupy.cuda used by spectralmc.async_normals."""

from __future__ import annotations
from typing import Any

class Stream:
    def synchronize(self) -> None: ...
    def __enter__(self) -> "Stream": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None: ...

class Event:
    ptr: int
    def __init__(self, disable_timing: bool = ...) -> None: ...
    def record(self) -> None: ...

class runtime:
    @staticmethod
    def eventQuery(ptr: int) -> int: ...

class Device:
    def __init__(self, device_id: int | None = ...) -> None: ...
    def synchronize(self) -> None: ...
