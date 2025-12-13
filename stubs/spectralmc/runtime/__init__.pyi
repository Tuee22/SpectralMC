"""Stub package for spectralmc.runtime."""

from spectralmc.runtime.torch_runtime import (
    TorchRuntime,
    apply_torch_runtime,
    decide_torch_runtime,
    get_torch_handle,
)

__all__ = ["TorchRuntime", "apply_torch_runtime", "decide_torch_runtime", "get_torch_handle"]
