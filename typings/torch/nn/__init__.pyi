from __future__ import annotations

from typing import Iterator, Sequence, overload
import torch

# ─────────────────────────────────────────────────────
# Base nn.Module
# ─────────────────────────────────────────────────────
class Module:
    training: bool

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    # __call__ – two valid shapes used in cvnn.py
    @overload
    def __call__(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @overload
    def __call__(self, input_real: torch.Tensor) -> torch.Tensor: ...
    def state_dict(self) -> dict[str, torch.Tensor]: ...
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None: ...

# ─────────────────────────────────────────────────────
# Parameter
# ─────────────────────────────────────────────────────
class Parameter(torch.Tensor):
    def __init__(
        self,
        data: torch.Tensor | None = ...,
        requires_grad: bool = ...,
    ) -> None: ...

# ─────────────────────────────────────────────────────
# Layers referenced in cvnn.py
# ─────────────────────────────────────────────────────
class BatchNorm1d(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = ...,
        momentum: float = ...,
        affine: bool = ...,
        track_running_stats: bool = ...,
    ) -> None: ...

# ─────────────────────────────────────────────────────
# Container
# ─────────────────────────────────────────────────────
class ModuleList(Module):
    def __init__(self, modules: Sequence[Module] | None = ...) -> None: ...
    def append(self, module: Module) -> None: ...
    def __iter__(self) -> Iterator[Module]: ...

# ─────────────────────────────────────────────────────
# nn.init sub-module (only what cvnn.py touches)
# ─────────────────────────────────────────────────────
class _InitModule(Module):
    def xavier_uniform_(self, tensor: torch.Tensor) -> torch.Tensor: ...
    def zeros_(self, tensor: torch.Tensor) -> torch.Tensor: ...

init: _InitModule
