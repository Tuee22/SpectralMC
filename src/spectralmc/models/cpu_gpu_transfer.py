# src/spectralmc/models/cpu_gpu_transfer.py
"""
Pure-functional helpers to

* move an arbitrarily-nested *TensorTree* CPU ↔ CUDA,
* detect the unique (Device, dtype) of a tree, and
* derive that pair for model / optimiser ``state_dict`` objects.

No explicit ``if``/``for`` loops in user-level code; comprehensions,
pattern-matching, and expression-level guards do the work.

Fully ``mypy --strict`` clean.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum

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
import torch
from spectralmc.runtime import get_torch_handle
from spectralmc.result import Failure, Result, Success


get_torch_handle()

__all__ = [
    "TransferDestination",
    "StagePolicy",
    "OutputPinning",
    "CopyMode",
    "TransferDecision",
    "plan_tensor_transfer",
    "Scalar",
    "TensorTree",
    "move_tensor_tree",
    "get_tree_device_dtype",
    "module_state_device_dtype",
    "optimizer_state_device_dtype",
]


# ────────────────────────────── enums / ADTs ───────────────────────────────
class TransferDestination(Enum):
    """Explicit target placement."""

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


class StagePolicy(str, Enum):
    """Stage-on-host policy expressed as an ADT (avoid boolean flags)."""

    ALLOW = "allow"
    FORBID = "forbid"

    def permits_staging(self) -> bool:
        return self is StagePolicy.ALLOW


class CopyMode(str, Enum):
    """Copy scheduling mode for the interpreter."""

    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"


class OutputPinning(str, Enum):
    """Output pinning intent for host targets."""

    PINNED = "pinned"
    UNPINNED = "unpinned"


@dataclass(frozen=True)
class HostPlacement:
    """Explicit host placement with pinning state."""

    pinned: bool

    def to_destination(self) -> TransferDestination:
        return TransferDestination.CPU_PINNED if self.pinned else TransferDestination.CPU


@dataclass(frozen=True)
class CudaPlacement:
    """Explicit CUDA placement with device identifier."""

    device: Device

    def to_destination(self) -> TransferDestination:
        return TransferDestination.CUDA


Placement = HostPlacement | CudaPlacement


@dataclass(frozen=True)
class StayOnPlacement:
    placement: Placement


@dataclass(frozen=True)
class DirectTransfer:
    destination: TransferDestination
    copy_mode: CopyMode
    output_pinning: OutputPinning


@dataclass(frozen=True)
class StageThenCopy:
    destination: TransferDestination
    copy_mode: CopyMode
    stage_reason: str


@dataclass(frozen=True)
class RejectTransfer:
    reason: str


TransferDecision = StayOnPlacement | DirectTransfer | StageThenCopy | RejectTransfer


@dataclass(frozen=True)
class CudaStreamPresent:
    stream: torch.cuda.Stream


@dataclass(frozen=True)
class CudaStreamAbsent:
    reason: str


CudaStreamState = CudaStreamPresent | CudaStreamAbsent


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
_MAX_CPU_BYTES = 64 * 1024 * 1024

# NOTE: Conditional stream creation is intentional. torch.cuda.Stream() fails
# without CUDA. All CUDA operations fail-fast via the explicit check in
# move_tensor_tree(). This pattern is acceptable per CPU/GPU policy as
# infrastructure code that enables the TensorTree API.
_CUDA_STREAM_STATE: CudaStreamState = (
    CudaStreamPresent(torch.cuda.Stream(Device.cuda.to_torch()))
    if torch.cuda.is_available()
    else CudaStreamAbsent(reason="cuda_unavailable")
)


# ───────────────────────── functional helpers ───────────────────────────────


def _placement_of(tensor: torch.Tensor) -> Result[Placement, TorchFacadeError]:
    """Infer explicit placement for a tensor without using truthy shortcuts."""
    device_result = Device.from_torch(tensor.device)
    match device_result:
        case Failure(error):
            return Failure(error)
        case Success(Device.cpu):
            return Success(HostPlacement(pinned=tensor.is_pinned()))
        case Success(Device.cuda):
            return Success(CudaPlacement(device=Device.cuda))
    return Failure(UnsupportedTorchDevice(device=str(tensor.device)))


def _target_placement(dest: TransferDestination) -> Placement:
    """Map destination intent to explicit placement."""
    match dest:
        case TransferDestination.CPU:
            return HostPlacement(pinned=False)
        case TransferDestination.CPU_PINNED:
            return HostPlacement(pinned=True)
        case TransferDestination.CUDA:
            return CudaPlacement(device=Device.cuda)


def _placements_match(source: Placement, target: Placement) -> bool:
    """Return True when no move is required (device + pinning already aligned)."""
    match (source, target):
        case (HostPlacement(pinned=sp), HostPlacement(pinned=tp)):
            return sp == tp
        case (CudaPlacement(device=sd), CudaPlacement(device=td)):
            return sd == td
        case _:
            return False


def _plan_copy_tensor(
    src: torch.Tensor,
    *,
    dest: TransferDestination,
    stage_policy: StagePolicy,
) -> Result[TransferDecision, TorchFacadeError]:
    """Plan a transfer without performing any side effects."""
    target_dev = dest.to_torch_device()

    if dest is TransferDestination.CUDA and not torch.cuda.is_available():
        return Failure(CudaUnavailable())

    source_placement_result = _placement_of(src)
    if isinstance(source_placement_result, Failure):
        return source_placement_result
    source_placement = source_placement_result.value
    target_placement = _target_placement(dest)

    if _placements_match(source_placement, target_placement):
        return Failure(NoOpTransfer(device=str(src.device)))

    match target_placement:
        case CudaPlacement():
            match source_placement:
                case HostPlacement(pinned=False):
                    if not stage_policy.permits_staging():
                        return Failure(TransferRejected(reason="staging_disabled"))
                    return Success(
                        StageThenCopy(
                            destination=dest,
                            copy_mode=CopyMode.BLOCKING,
                            stage_reason="unpinned_host_to_cuda",
                        )
                    )
                case HostPlacement(pinned=True):
                    return Success(
                        DirectTransfer(
                            destination=dest,
                            copy_mode=CopyMode.BLOCKING,
                            output_pinning=OutputPinning.UNPINNED,
                        )
                    )
                case CudaPlacement():
                    # Same CUDA device is handled by _placements_match earlier
                    return Success(
                        DirectTransfer(
                            destination=dest,
                            copy_mode=CopyMode.BLOCKING,
                            output_pinning=OutputPinning.UNPINNED,
                        )
                    )
        case HostPlacement(pinned=target_pinned):
            transfer_bytes = src.element_size() * src.numel()
            if transfer_bytes > _MAX_CPU_BYTES:
                return Failure(TransferRejected(reason="oversized_host_transfer"))

            copy_mode = (
                CopyMode.NON_BLOCKING
                if isinstance(source_placement, CudaPlacement) and target_pinned
                else CopyMode.BLOCKING
            )
            output_pinning = OutputPinning.PINNED if target_pinned else OutputPinning.UNPINNED
            return Success(
                DirectTransfer(
                    destination=dest,
                    copy_mode=copy_mode,
                    output_pinning=output_pinning,
                )
            )

    return Failure(UnsupportedTorchDevice(device=str(target_dev)))


def plan_tensor_transfer(
    tensor: torch.Tensor,
    *,
    dest: TransferDestination,
    stage_policy: StagePolicy = StagePolicy.ALLOW,
) -> Result[TransferDecision, TorchFacadeError]:
    """Public helper to plan a single-tensor transfer without executing it."""
    return _plan_copy_tensor(tensor, dest=dest, stage_policy=stage_policy)


def _execute_plan(
    src: torch.Tensor,
    *,
    plan: TransferDecision,
) -> Result[torch.Tensor, TorchFacadeError]:
    """Execute a planned transfer using the configured CUDA stream."""
    stream_state = _CUDA_STREAM_STATE

    match plan:
        case RejectTransfer(reason=reason):
            return Failure(TransferRejected(reason=reason))
        case StayOnPlacement(placement=p):
            return Failure(
                TransferRejected(reason=f"noop_transfer_for_{p.__class__.__name__.lower()}")
            )
        case StageThenCopy(destination=dest, copy_mode=copy_mode, stage_reason=_):
            target_dev = dest.to_torch_device()
            staging = torch.empty(
                *src.shape,
                dtype=src.dtype,
                device=Device.cpu.to_torch(),
                pin_memory=True,
            )
            staging.copy_(src.detach(), non_blocking=False)
            dst = torch.empty(*src.shape, dtype=src.dtype, device=target_dev)
            if copy_mode is CopyMode.NON_BLOCKING and isinstance(stream_state, CudaStreamPresent):
                with torch.cuda.stream(stream_state.stream):
                    dst.copy_(staging, non_blocking=True)
            else:
                dst.copy_(staging, non_blocking=False)
            return Success(dst)
        case DirectTransfer(
            destination=dest,
            copy_mode=copy_mode,
            output_pinning=output_pinning,
        ):
            target_dev = dest.to_torch_device()
            match dest:
                case TransferDestination.CUDA:
                    dst = torch.empty(*src.shape, dtype=src.dtype, device=target_dev)
                    if copy_mode is CopyMode.NON_BLOCKING and isinstance(
                        stream_state, CudaStreamPresent
                    ):
                        with torch.cuda.stream(stream_state.stream):
                            dst.copy_(src.detach(), non_blocking=True)
                    else:
                        dst.copy_(src.detach(), non_blocking=False)
                    return Success(dst)
                case TransferDestination.CPU | TransferDestination.CPU_PINNED:
                    dst = torch.empty(
                        *src.shape,
                        dtype=src.dtype,
                        device=target_dev,
                        pin_memory=output_pinning is OutputPinning.PINNED,
                    )
                    if copy_mode is CopyMode.NON_BLOCKING:
                        assert isinstance(stream_state, CudaStreamPresent)
                        with torch.cuda.stream(stream_state.stream):
                            dst.copy_(src.detach(), non_blocking=True)
                    else:
                        dst.copy_(src.detach(), non_blocking=False)
                    return Success(dst)
    raise AssertionError(f"Unhandled transfer plan: {plan!r}")


def _copy_tensor(
    src: torch.Tensor,
    *,
    dest: TransferDestination,
    stage_policy: StagePolicy,
) -> Result[torch.Tensor, TorchFacadeError]:
    """Clone *src* to *dest* using a pure plan + interpreter."""
    match _plan_copy_tensor(src, dest=dest, stage_policy=stage_policy):
        case Success(plan):
            return _execute_plan(src, plan=plan)
        case Failure(error):
            return Failure(error)


def _move(
    obj: TensorTree,
    *,
    dest: TransferDestination,
    stage_policy: StagePolicy,
) -> Result[TensorTree, TorchFacadeError]:
    """Pure-functional recursion via pattern matching."""
    match obj:
        case torch.Tensor():
            # torch.Tensor is a TensorTree - pattern match to widen type for mypy
            match _copy_tensor(obj, dest=dest, stage_policy=stage_policy):
                case Success(tensor):
                    return Success(tensor)  # Type widened: torch.Tensor -> TensorTree
                case Failure(error):
                    return Failure(error)
        case list() as seq:
            # Recursively move all elements, collecting Results
            moved_results = [_move(v, dest=dest, stage_policy=stage_policy) for v in seq]
            # Check for first failure
            first_failure = next((r for r in moved_results if isinstance(r, Failure)), None)
            if first_failure is not None:
                return first_failure
            # All succeeded - extract values
            return Success([r.value for r in moved_results if isinstance(r, Success)])
        case tuple() as seq:
            # Recursively move all elements, collecting Results
            moved_results = [_move(v, dest=dest, stage_policy=stage_policy) for v in seq]
            # Check for first failure
            first_failure = next((r for r in moved_results if isinstance(r, Failure)), None)
            if first_failure is not None:
                return first_failure
            # All succeeded - extract values as tuple
            return Success(tuple(r.value for r in moved_results if isinstance(r, Success)))
        case Mapping() as mp:
            # Recursively move all values, collecting Results
            moved_items: list[tuple[Hashable, Result[TensorTree, TorchFacadeError]]] = [
                (k, _move(v, dest=dest, stage_policy=stage_policy)) for k, v in mp.items()
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
    stage_policy: StagePolicy = StagePolicy.ALLOW,
) -> Result[TensorTree, TorchFacadeError]:
    """Copy *tree* so every tensor lives on *dest* (CPU, CPU_PINNED, or CUDA)."""
    tgt = dest.to_torch_device()

    if tgt.type == "cuda" and not torch.cuda.is_available():
        return Failure(CudaUnavailable())

    result = _move(tree, dest=dest, stage_policy=stage_policy)
    match _CUDA_STREAM_STATE:
        case CudaStreamPresent(stream):
            stream.synchronize()
        case CudaStreamAbsent():
            pass
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
