"""
Stub for ``torch.optim`` (only Optimizer & Adam) – deliberately *loose*
so project code can pass the raw dict that PyTorch returns.
"""

from __future__ import annotations
from typing import Iterable
import torch

class Optimizer:
    param_groups: list[dict[str, float]]

    def zero_grad(self, *, set_to_none: bool | None = ...) -> None: ...
    def step(self) -> None: ...
    def load_state_dict(self, state: dict[str, object]) -> None: ...
    def state_dict(self) -> dict[str, object]: ...

class Adam(Optimizer):
    def __init__(
        self, params: Iterable[torch.Tensor], lr: float = ..., **kw: object
    ) -> None: ...
