# typings/torch/utils/dlpack.pyi
"""
Stub for ``torch.utils.dlpack`` (“from_dlpack” helper only) without ``Any``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import torch

# A DLPack device is (device_type, device_id), e.g. (1, 0) for CUDA:0
DLPackDevice = tuple[int, int]

@runtime_checkable
class SupportsDLPack(Protocol):
    """Object that can hand its memory to another library via DLPack."""

    def __dlpack__(self, stream: int | None = None) -> object: ...
    def __dlpack_device__(self) -> DLPackDevice: ...

def from_dlpack(dlpack: SupportsDLPack) -> torch.Tensor: ...
