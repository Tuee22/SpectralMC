# torch/cuda/__init__.pyi
"""
Overlay stub for :pymod:`torch.cuda` adding the minimal API pieces that
PyTorch's own stubs omit.

Only ``is_available``, the context helper ``stream``, and the
``Stream``/``Event`` types are declared.
"""

from __future__ import annotations

from types import TracebackType
from typing import Optional, TypeAlias, Union

import torch
from torch.cuda.streams import Stream as Stream  # public re-export

Device: TypeAlias = torch.device

class Event:
    """CUDA event stub (timing and stream-sync primitive)."""

    def __init__(
        self,
        enable_timing: bool = ...,
        blocking: bool = ...,
        interprocess: bool = ...,
    ) -> None: ...

    # minimal public surface that your code (or Stream) touches
    def record(self, stream: Optional[Stream] = ...) -> None: ...
    def query(self) -> bool: ...
    def synchronize(self) -> None: ...

# ------------------------------------------------------------------ #
# helpers your project references                                    #
# ------------------------------------------------------------------ #
def is_available() -> bool: ...

"""Return ``True`` if CUDA is compiled in and a device is visible."""

class _StreamContext:
    """Context manager returned by :func:`stream`."""

    def __init__(self, stream: Stream): ...
    def __enter__(self) -> Stream: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...

def stream(
    stream: Union[int, Device, Stream, None] = ...,
) -> _StreamContext: ...

"""Return a context manager that sets *stream* as the default stream."""
