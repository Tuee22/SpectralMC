# typings/safetensors/__init__.pyi
"""
Typed stub for the *top‑level* :pymod:`safetensors` package
(only the bits SpectralMC touches).

No ``Any``, no ``cast``, no ``type: ignore`` – fully strict.
"""
from __future__ import annotations

import os
from io import BufferedIOBase
from types import TracebackType
from typing import ContextManager, Protocol, runtime_checkable

import torch

_Path = str | bytes | os.PathLike[str]

@runtime_checkable
class SafeTensorReader(Protocol):
    """Reader interface returned by :func:`safe_open().__enter__`."""

    def keys(self) -> list[str]: ...
    def get_tensor(self, name: str) -> torch.Tensor: ...

class _ReaderCM(ContextManager[SafeTensorReader], Protocol):
    def __enter__(self) -> SafeTensorReader: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...

def safe_open(
    file: _Path | bytes | BufferedIOBase,
    *,
    framework: str,
    device: str | torch.device | None = ...,
) -> _ReaderCM: ...
