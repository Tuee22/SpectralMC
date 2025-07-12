"""
Strict stub for :pymod:`safetensors.torch` – only what SpectralMC needs.
"""

from __future__ import annotations
from typing import Mapping
import torch

def load(
    src: bytes | str, *, device: str | torch.device | None = ...
) -> dict[str, torch.Tensor]: ...
def save(tensors: Mapping[str, torch.Tensor]) -> bytes: ...
