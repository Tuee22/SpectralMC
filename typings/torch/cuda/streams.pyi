# torch/cuda/streams.pyi
"""
Typed stub for :pymod:`torch.cuda.streams`.
"""

from __future__ import annotations

from types import TracebackType
from typing import Optional, TypeAlias, Union

import torch
import torch.cuda as _cuda  # resolves Event from upstream runtime

Device: TypeAlias = torch.device
Event: TypeAlias = _cuda.Event

class Stream:
    """CUDA stream (independent GPU work queue)."""

    # ───────── Construction ─────────
    def __init__(
        self,
        device: Optional[Union[int, Device]] = ...,
        *,
        priority: int = ...,
        capture: bool = ...,
    ) -> None: ...

    # ─────── Context manager ───────
    def __enter__(self) -> "Stream": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...

    # ───── Synchronisation ──────
    def wait_event(self, event: Event) -> None: ...
    def wait_stream(self, stream: "Stream") -> None: ...
    def query(self) -> bool: ...
    def synchronize(self) -> None: ...

    # ─────── Recording ─────────
    def record_event(self, event: Optional[Event] = ...) -> Event: ...

    # ─────── Properties ────────
    @property
    def priority(self) -> int: ...
    @property
    def device(self) -> Device: ...
    @property
    def pointer(self) -> int: ...
