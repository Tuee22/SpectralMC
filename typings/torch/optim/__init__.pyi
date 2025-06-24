"""
Stub for the :pymod:`torch.optim` package.

Only the small subset needed by SpectralMC’s code‑base is declared, but we keep
the public API surfaces **precise** to maintain type safety.
"""

from __future__ import annotations

import torch
from typing import Iterable, Protocol

# --------------------------------------------------------------------------- #
#  Core Optimizer base – very thin                                            #
# --------------------------------------------------------------------------- #
class _ParamLike(Protocol):
    grad: torch.Tensor | None

class Optimizer:
    """Skeleton of the real ``torch.optim.Optimizer``."""

    param_groups: list[dict[str, float]]

    def zero_grad(self, *, set_to_none: bool | None = ...) -> None: ...
    def step(self) -> None: ...
    def load_state_dict(self, state: dict[str, object]) -> None: ...
    def state_dict(self) -> dict[str, object]: ...

# --------------------------------------------------------------------------- #
#  Adam                                                                       #
# --------------------------------------------------------------------------- #
class Adam(Optimizer):
    """Type stub for :class:`torch.optim.Adam` (sufficient for tests)."""

    def __init__(
        self,
        params: Iterable[_ParamLike],
        lr: float = ...,
        **kw: object,
    ) -> None: ...

# --------------------------------------------------------------------------- #
#  SGD – required by the cvnn factory test‑suite                              #
# --------------------------------------------------------------------------- #
class SGD(Optimizer):
    """Stochastic Gradient Descent (no momentum parameters typed)."""

    def __init__(
        self,
        params: Iterable[_ParamLike],
        lr: float = ...,
        **kw: object,
    ) -> None: ...
