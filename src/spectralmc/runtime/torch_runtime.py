"""
Torch runtime decision + effect for deterministic execution.

This module models torch readiness as data and applies the deterministic
configuration effect. Use `get_torch_handle()` to cache a single configured torch
handle for downstream consumers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import torch
from spectralmc.result import Failure, Result, Success

_TORCH_HANDLE: ModuleType | None = None
_TORCH_RUNTIME: TorchRuntime | None = None


@dataclass(frozen=True)
class TorchRuntime:
    """Torch runtime readiness ADT.

    Attributes:
        kind: Discriminator indicating whether torch is ready.
        cuda_version: CUDA version string when ready.
        cudnn_version: cuDNN version when ready.
        reason: Failure reason when not ready.
    """

    kind: Literal["ready", "rejected"]
    cuda_version: str | None = None
    cudnn_version: int | None = None
    reason: str | None = None


def decide_torch_runtime() -> TorchRuntime:
    """Return deterministic torch runtime readiness.

    Returns:
        TorchRuntime: `ready` when CUDA and cuDNN are available, otherwise `rejected`.
    """
    cudnn_version = torch.backends.cudnn.version()
    match (torch.cuda.is_available(), cudnn_version):
        case (False, _):
            return TorchRuntime(kind="rejected", reason="cuda_unavailable")
        case (True, None):
            return TorchRuntime(kind="rejected", reason="cudnn_unavailable")
        case (True, ver):
            return TorchRuntime(
                kind="ready",
                cuda_version=torch.version.cuda,
                cudnn_version=ver,
            )
    return TorchRuntime(kind="rejected", reason="torch_runtime_unknown")


def apply_torch_runtime(runtime: TorchRuntime) -> Result[ModuleType, TorchRuntime]:
    """Apply deterministic torch configuration and return the torch handle result.

    Args:
        runtime: Torch runtime decision produced by `decide_torch_runtime`.

    Returns:
        Result[ModuleType, TorchRuntime]: Success with torch handle or Failure with runtime.
    """
    match runtime.kind:
        case "ready":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            return Success(torch)
        case _:
            return Failure(runtime)


def get_torch_handle() -> ModuleType:
    """Return a cached deterministically configured torch handle."""
    global _TORCH_HANDLE, _TORCH_RUNTIME
    match _TORCH_HANDLE:
        case None:
            runtime = decide_torch_runtime()
            _TORCH_RUNTIME = runtime
            match apply_torch_runtime(runtime):
                case Success(handle):
                    _TORCH_HANDLE = handle
                case Failure(reason):
                    message = reason.reason or reason.kind
                    raise AssertionError(f"invariant: torch runtime rejected ({message})")
        case _:
            _TORCH_HANDLE = _TORCH_HANDLE
    assert _TORCH_HANDLE is not None
    return _TORCH_HANDLE


__all__ = ["TorchRuntime", "apply_torch_runtime", "decide_torch_runtime", "get_torch_handle"]
