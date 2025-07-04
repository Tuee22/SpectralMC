# typings/safetensors/torch.pyi
"""
Stub for :pymod:`safetensors.torch` – only ``save`` is required by
SpectralMC and it returns the raw SafeTensor bytes.
"""
from __future__ import annotations

import os
from io import BufferedIOBase
from typing import Mapping

import torch

_Path = str | bytes | os.PathLike[str]

def save(
    tensors: Mapping[str, torch.Tensor],
    filename: _Path | BufferedIOBase | None = ...,
) -> bytes: ...
