# typings/torch/random/__init__.pyi
"""
Strict stub for ``torch.random`` (only the public API that most projects use).
"""

from __future__ import annotations
import torch
from typing import TypeAlias, Sequence

Tensor: TypeAlias = torch.Tensor

# ─────────────────────────── RNG helper functions ───────────────────────────
def manual_seed(seed: int) -> Tensor: ...
def seed() -> int: ...
def initial_seed() -> int: ...
def get_rng_state() -> Tensor: ...
def set_rng_state(state: Tensor) -> None: ...

# ──────────────────────────── Generator object ──────────────────────────────
class Generator:
    def __init__(self, device: torch.device | str | None = ...) -> None: ...
    def manual_seed(self, seed: int) -> "Generator": ...
    def seed(self) -> int: ...
    def initial_seed(self) -> int: ...
    def get_state(self) -> Tensor: ...
    def set_state(self, state: Tensor) -> None: ...

# Global default RNG that torch exposes at run time
default_generator: Generator

# ─────────────────────── context-manager returned by fork_rng ──────────────
class _ForkRNG:
    """Context manager that snapshots RNG state and restores it on exit."""

    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool | None: ...
    # PyTorch returns False so exceptions propagate, but allowing None is fine.

# ─────────────────────────── public API we need ────────────────────────────
def fork_rng(
    *,
    devices: Sequence[int | torch.device] | None = ...,
    enabled: bool = ...,
    device_type: str | None = ...,
) -> _ForkRNG: ...
