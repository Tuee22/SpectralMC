# src/spectralmc/models/cpu_gpu_transfer.py
"""
Pure-functional helpers to

* move an arbitrarily-nested *TensorTree* CPU ↔ CUDA,
* detect the unique (Device, DType) of a tree, and
* derive that pair for model / optimiser ``state_dict`` objects.

No explicit ``if``/``for`` loops in user-level code; comprehensions,
pattern-matching, and expression-level guards do the work.

Fully ``mypy --strict`` clean.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch

# CRITICAL: Import facade BEFORE torch for deterministic algorithms
from spectralmc.errors.torch_facade import (
    CudaUnavailable,
    EmptyTensorTree,
    HeterogeneousTensorTree,
    NoOpTransfer,
    TorchFacadeError,
    TransferRejected,
    UnsupportedTorchDevice,
    UnsupportedTorchDType,
)
from spectralmc.models.torch import (
    _TORCH_DTYPE_TO_STR,
    AnyDType,
    Device,
    FullPrecisionDType,
    ReducedPrecisionDType,
)
from spectralmc.result import Failure, Result, Success


__all__ = [
    "TransferDestination",
    "TransferDecision",
    "plan_tensor_transfer",
    "Scalar",
    "TensorTree",
    "move_tensor_tree",
    "get_tree_device_dtype",
    "module_state_device_dtype",
    "optimizer_state_device_dtype",
]


# ────────────────────────────── enums ──────────────────────────────────────
class TransferDestination(Enum):
    """Transfer destination eliminating illegal (device, pin_memory) combinations.

    CPU: Transfer to CPU without pinned memory.
    CPU_PINNED: Transfer to CPU with pinned memory (faster GPU↔CPU transfers).
    CUDA: Transfer to CUDA device (pin_memory is meaningless here).
    """

    CPU = "cpu"
    CPU_PINNED = "cpu_pinned"
    CUDA = "cuda"

    def to_torch_device(self) -> torch.device:
        """Convert to torch.device."""
        match self:
            case TransferDestination.CPU | TransferDestination.CPU_PINNED:
                return Device.cpu.to_torch()
            case TransferDestination.CUDA:
                return Device.cuda.to_torch()

    def should_pin_memory(self) -> bool:
        """Return True if memory should be pinned (only for CPU_PINNED)."""
        return self == TransferDestination.CPU_PINNED


@dataclass(frozen=True)
class TransferDecision:
    """Pure transfer plan describing how to move a tensor."""

    kind: Literal["direct", "stage_and_copy", "reject"]
    dest: TransferDestination
    non_blocking: bool
    pin_output: bool
    reason: str | None = None


# ────────────────────────────── type aliases ────────────────────────────────
Scalar = int | float | bool | str | bytes | None
TensorTree = (
    torch.Tensor
    | Scalar
    | list["TensorTree"]
    | tuple["TensorTree", ...]
    | Mapping[Hashable, "TensorTree"]
)

# ──────────────────────────── global resources ──────────────────────────────
# NOTE: Conditional stream creation is intentional. torch.cuda.Stream() fails
# without CUDA. All CUDA operations fail-fast via the explicit check in
# move_tensor_tree(). This pattern is acceptable per CPU/GPU policy as
# infrastructure code that enables the TensorTree API.
_CUDA_STREAM: torch.cuda.Stream | None = (
    torch.cuda.Stream(Device.cuda.to_torch()) if torch.cuda.is_available() else None
)


# ───────────────────────── functional helpers ───────────────────────────────

def _plan_copy_tensor(
    src: torch.Tensor,
    *,
    dest: TransferDestination,
    allow_stage: bool,
) -> Result[TransferDecision, TorchFacadeError]:
    """Plan a transfer without performing any side effects."""
    target_dev = dest.to_torch_device()

    if target_dev.type == "cuda" and not torch.cuda.is_available():
        return Failure(CudaUnavailable())

    if src.device == target_dev:
        return Failure(NoOpTransfer(device=str(src.device)))

    match target_dev.type:
        case "cuda":
            is_pinned = src.is_pinned()
            if not is_pinned and not allow_stage:
                return Failure(TransferRejected(reason="staging_disabled"))
            return Success(
                TransferDecision(
                    kind="direct" if is_pinned else "stage_and_copy",
                    dest=dest,
                    non_blocking=False,
                    pin_output=False,
                    reason=None if is_pinned else "unpinned_host_to_cuda",
                )
            )
        case "cpu":
            pin_output = dest.should_pin_memory()
            non_blocking = pin_output and src.device.type == "cuda"
            return Success(
                TransferDecision(
                    kind="direct",
                    dest=dest,
                    non_blocking=non_blocking,
                    pin_output=pin_output,
                    reason=None,
                )
            )
        case _:
            return Failure(UnsupportedTorchDevice(device=str(target_dev)))




def plan_tensor_transfer(
    tensor: torch.Tensor,
    *,
    dest: TransferDestination,
    allow_stage: bool = True,
) -> Result[TransferDecision, TorchFacadeError]:
    """Public helper to plan a single-tensor transfer without executing it."""
    return _plan_copy_tensor(tensor, dest=dest, allow_stage=allow_stage)

def _execute_plan(
    src: torch.Tensor,
    *,
    plan: TransferDecision,
) -> Result[torch.Tensor, TorchFacadeError]:
    """Execute a planned transfer using the configured CUDA stream."""
    target_dev = plan.dest.to_torch_device()

    if plan.kind == "reject":
        return Failure(TransferRejected(reason=plan.reason or "transfer_rejected"))

    if plan.kind == "stage_and_copy":
        staging = torch.empty(
            *src.shape,
            dtype=src.dtype,
            device=Device.cpu.to_torch(),
            pin_memory=True,
        )
        staging.copy_(src.detach(), non_blocking=False)
        dst = torch.empty(*src.shape, dtype=src.dtype, device=target_dev)
        if plan.non_blocking and _CUDA_STREAM is not None:
            with torch.cuda.stream(_CUDA_STREAM):
                dst.copy_(staging, non_blocking=True)
        else:
            dst.copy_(staging, non_blocking=False)
        return Success(dst)

    if plan.kind == "direct":
        match target_dev.type:
            case "cuda":
                dst = torch.empty(*src.shape, dtype=src.dtype, device=target_dev)
                if plan.non_blocking and _CUDA_STREAM is not None:
                    with torch.cuda.stream(_CUDA_STREAM):
                        dst.copy_(src.detach(), non_blocking=True)
                else:
                    dst.copy_(src.detach(), non_blocking=False)
                return Success(dst)
            case "cpu":
                dst = torch.empty(
                    *src.shape,
                    dtype=src.dtype,
                    device=target_dev,
                    pin_memory=plan.pin_output,
                )
                if src.device.type == "cuda":
                    if plan.non_blocking:
                        assert _CUDA_STREAM
                        with torch.cuda.stream(_CUDA_STREAM):
                            dst.copy_(src.detach(), non_blocking=True)
                    else:
                        dst.copy_(src.detach(), non_blocking=False)
                else:
                    dst.copy_(src.detach(), non_blocking=False)
                return Success(dst)

    return Failure(TransferRejected(reason="unrecognized_transfer_plan"))


def _copy_tensor(
    src: torch.Tensor,
    *,
    dest: TransferDestination,
    allow_stage: bool,
) -> Result[torch.Tensor, TorchFacadeError]:
    """Clone *src* to *dest* using a pure plan + interpreter."""
    match _plan_copy_tensor(src, dest=dest, allow_stage=allow_stage):
        case Success(plan):
            return _execute_plan(src, plan=plan)
        case Failure(error):
            return Failure(error)


def _move(
    obj: TensorTree,
    *,
    dest: TransferDestination,
    allow_stage: bool,
) -> Result[TensorTree, TorchFacadeError]:
    """Pure-functional recursion via pattern matching."""
    match obj:
        case torch.Tensor():
            # torch.Tensor is a TensorTree - pattern match to widen type for mypy
            match _copy_tensor(obj, dest=dest, allow_stage=allow_stage):
                case Success(tensor):
                    return Success(tensor)  # Type widened: torch.Tensor -> TensorTree
                case Failure(error):
                    return Failure(error)
        case list() as seq:
            # Recursively move all elements, collecting Results
            moved_results = [_move(v, dest=dest, allow_stage=allow_stage) for v in seq]
            # Check for first failure
            first_failure = next((r for r in moved_results if isinstance(r, Failure)), None)
            if first_failure is not None:
                return first_failure
            # All succeeded - extract values
            return Success([r.value for r in moved_results if isinstance(r, Success)])
        case tuple() as seq:
            # Recursively move all elements, collecting Results
            moved_results = [_move(v, dest=dest, allow_stage=allow_stage) for v in seq]
            # Check for first failure
            first_failure = next((r for r in moved_results if isinstance(r, Failure)), None)
            if first_failure is not None:
                return first_failure
            # All succeeded - extract values as tuple
            return Success(tuple(r.value for r in moved_results if isinstance(r, Success)))
        case Mapping() as mp:
            # Recursively move all values, collecting Results
            moved_items: list[tuple[Hashable, Result[TensorTree, TorchFacadeError]]] = [
                (k, _move(v, dest=dest, allow_stage=allow_stage)) for k, v in mp.items()
            ]
            # Check for first failure
            first_failure = next((r for _, r in moved_items if isinstance(r, Failure)), None)
            if first_failure is not None:
                return first_failure
            # All succeeded - build dict
            return Success({k: r.value for k, r in moved_items if isinstance(r, Success)})
        case _:
            return Success(obj)  # scalar leaf


# ───────────────────────────── public API ───────────────────────────────────
def move_tensor_tree(
    tree: TensorTree,
    *,
    dest: TransferDestination,
    allow_stage: bool = True,
) -> Result[TensorTree, TorchFacadeError]:
    """Copy *tree* so every tensor lives on *dest* (CPU, CPU_PINNED, or CUDA)."""
    tgt = dest.to_torch_device()

    if tgt.type == "cuda" and not torch.cuda.is_available():
        return Failure(CudaUnavailable())

    result = _move(tree, dest=dest, allow_stage=allow_stage)
    (None if _CUDA_STREAM is None else _CUDA_STREAM.synchronize())
    return result


# ───────────────────────── tree inspection util ────────────────────────────
def get_tree_device_dtype(tree: TensorTree) -> Result[tuple[Device, AnyDType], TorchFacadeError]:
    """Return the unique ``(Device, AnyDType)`` shared by *all* tensors in *tree*."""

    def _pairs(node: TensorTree) -> set[tuple[torch.device, torch.dtype]]:
        match node:
            case torch.Tensor() as t:
                return {(t.device, t.dtype)}
            case list() | tuple() as seq:
                return {p for item in seq for p in _pairs(item)}
            case Mapping() as mp:
                return {p for v in mp.values() for p in _pairs(v)}
            case _:
                return set()

    pairs = _pairs(tree)
    if not pairs:
        return Failure(EmptyTensorTree())
    if len(pairs) != 1:
        return Failure(HeterogeneousTensorTree())

    dev, dt = next(iter(pairs))

    if dt not in _TORCH_DTYPE_TO_STR:
        return Failure(UnsupportedTorchDType(dtype=str(dt)))
    dtype_str = _TORCH_DTYPE_TO_STR[dt]

    dtype: AnyDType = (
        FullPrecisionDType(dtype_str)
        if dtype_str in ("float32", "float64", "complex64", "complex128")
        else ReducedPrecisionDType(dtype_str)
    )

    match Device.from_torch(dev):
        case Success(device):
            return Success((device, dtype))
        case Failure(error):
            return Failure(error)


# ───────────────────── state-dict helpers (strictly typed) ──────────────────
def module_state_device_dtype(
    state: Mapping[str, torch.Tensor],
) -> Result[tuple[Device, AnyDType], TorchFacadeError]:
    """Device/dtype for a *model* ``state_dict``."""
    return get_tree_device_dtype(tuple(state.values()))


def optimizer_state_device_dtype(
    state: Mapping[str, object],  # <-- keys are *str* now
) -> Result[tuple[Device, AnyDType], TorchFacadeError]:
    """Device/dtype for an *optimizer* ``state_dict`` (Adam, etc.)."""

    def _tensors(node: object) -> list[torch.Tensor]:
        match node:
            case torch.Tensor() as t:
                return [t]
            case list() | tuple() as seq:
                return [x for item in seq for x in _tensors(item)]
            case Mapping() as mp:
                return [x for v in mp.values() for x in _tensors(v)]
            case _:
                return []

    ts = _tensors(state)
    if not ts:
        return Failure(EmptyTensorTree())
    return get_tree_device_dtype(tuple(ts))
