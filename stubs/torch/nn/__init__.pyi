# typings/torch/nn/__init__.pyi
"""
Strict stub for ``torch.nn`` exposing only the symbols SpectralMC touches.

* Fully typed (no ``Any``).
* Compatible with ``mypy --strict``.
"""

from __future__ import annotations
from typing import Iterator, Sequence, Tuple, TypeVar, overload

import torch

TMod = TypeVar("TMod", bound="Module")

# ───────────────────────────────── Module ────────────────────────────────────
class Module:
    training: bool

    def __init__(self, *a: object, **kw: object) -> None: ...

    # ──────────── forward shortcuts (overloads kept minimal) ────────────
    @overload
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    # ─────────────────────── state helpers ─────────────────────────────
    def state_dict(
        self,
        destination: dict[str, torch.Tensor] | None = ...,
        prefix: str = ...,
        keep_vars: bool = ...,
    ) -> dict[str, torch.Tensor]: ...
    def load_state_dict(
        self: TMod,
        state_dict: dict[str, torch.Tensor],
        strict: bool = ...,
        *,
        assign: bool = ...,
    ) -> None: ...
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None: ...
    def parameters(self) -> Iterator["Parameter"]: ...

    # ──────────────────── mode & traversal helpers ─────────────────────
    def train(self, mode: bool = ...) -> None: ...
    def eval(self) -> None: ...
    def modules(self) -> Iterator["Module"]: ...

    # ───────────────────── device / dtype moves ────────────────────────
    def to(
        self: TMod,
        device: torch.device | str | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> TMod: ...
    def cuda(self: TMod, device: int | torch.device | str | None = ...) -> TMod: ...
    def named_parameters(
        self, prefix: str = ..., recurse: bool = ...
    ) -> Iterator[Tuple[str, "Parameter"]]: ...

# ─────────────────────────────── Parameter ──────────────────────────────────
class Parameter(torch.Tensor):
    data: torch.Tensor

    def __init__(
        self, data: torch.Tensor | None = ..., requires_grad: bool = ...
    ) -> None: ...

    # Must match Tensor.copy_ exactly except for the narrowed return type
    def copy_(
        self,
        other: torch.Tensor,
        *,
        non_blocking: bool | None = ...,
    ) -> "Parameter": ...
    def zero_(self) -> "Parameter": ...
    def fill_(self, value: int | float) -> "Parameter": ...
    def numel(self) -> int: ...

# ───────────────────── concrete layers used in the project ──────────────────
class Linear(Module):
    weight: Parameter
    bias: Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        device: torch.device | str | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> None: ...
    def cuda(self, device: int | torch.device | str | None = ...) -> "Linear": ...

class ReLU(Module):
    def __init__(self, inplace: bool = ...) -> None: ...

class Sequential(Module):
    def __init__(self, *args: Module) -> None: ...
    def cuda(self, device: int | torch.device | str | None = ...) -> "Sequential": ...

class BatchNorm1d(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = ...,
        momentum: float = ...,
        affine: bool = ...,
        track_running_stats: bool = ...,
    ) -> None: ...

class ModuleList(Module):
    def __init__(self, modules: Sequence[Module] | None = ...) -> None: ...
    def append(self, module: Module) -> None: ...
    def __iter__(self) -> Iterator[Module]: ...

# ─────────────────────────── init helpers (xavier etc.) ─────────────────────
class _InitModule(Module):
    def xavier_uniform_(self, tensor: torch.Tensor) -> torch.Tensor: ...
    def zeros_(self, tensor: torch.Tensor) -> torch.Tensor: ...

init: _InitModule

# ─────────────────────────── sub‑packages re‑exported ───────────────────────
from . import functional as functional  # noqa: F401
from . import utils as utils  # noqa: F401
