# typings/torch/optim/__init__.pyi
"""
Strict stub for :pymod:`torch.optim` that is *just* wide enough for
SpectralMC’s needs, yet fully typed (no ``Any``).

Key points
----------
* :pyclass:`_OptimizerStateDict` – exact shape returned by Adam(W).
* Adam, AdamW, SGD constructors keep their public signatures minimal.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Protocol, TypedDict

import torch

# -------------------------------------------------------------------- #
#  Optimiser‑internal typed‑dicts
# -------------------------------------------------------------------- #
class _AdamParamState(TypedDict, total=False):
    step: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    max_exp_avg_sq: torch.Tensor | None

class _OptimizerStateDict(TypedDict):
    state: dict[int, _AdamParamState]
    param_groups: list[dict[str, object]]

# -------------------------------------------------------------------- #
#  Base optimiser protocol
# -------------------------------------------------------------------- #
class _ParamLike(Protocol):
    grad: torch.Tensor | None

class Optimizer:
    """Skeleton of ``torch.optim.Optimizer``."""

    param_groups: list[dict[str, float]]

    def zero_grad(self, *, set_to_none: bool | None = ...) -> None: ...
    def step(self) -> None: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...
    def state_dict(self) -> Mapping[str, object]: ...

# -------------------------------------------------------------------- #
#  Concrete optimiser classes used in SpectralMC
# -------------------------------------------------------------------- #
class Adam(Optimizer):
    def __init__(
        self, params: Iterable[_ParamLike], lr: float = ..., **kw: object
    ) -> None: ...

class AdamW(Optimizer):
    def __init__(
        self, params: Iterable[_ParamLike], lr: float = ..., **kw: object
    ) -> None: ...

class SGD(Optimizer):
    def __init__(
        self, params: Iterable[_ParamLike], lr: float = ..., **kw: object
    ) -> None: ...
