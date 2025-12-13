from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Literal
from spectralmc.result import Result


@dataclass(frozen=True)
class TorchRuntime:
    kind: Literal["ready", "rejected"]
    cuda_version: str | None = None
    cudnn_version: int | None = None
    reason: str | None = None


def decide_torch_runtime() -> TorchRuntime: ...


def apply_torch_runtime(runtime: TorchRuntime) -> Result[ModuleType, TorchRuntime]: ...


def get_torch_handle() -> ModuleType: ...


__all__ = ["TorchRuntime", "apply_torch_runtime", "decide_torch_runtime", "get_torch_handle"]
