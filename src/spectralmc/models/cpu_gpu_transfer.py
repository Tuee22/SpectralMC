# src/spectralmc/models/cpu_gpu_transfer.py
from __future__ import annotations

"""
Pure‑functional helpers to

* move an arbitrarily‑nested *TensorTree* CPU ↔ CUDA,
* detect the unique (Device, DType) of a tree, and
* derive that pair for model / optimiser ``state_dict`` objects.

No explicit ``if``/``for`` loops in user‑level code; comprehensions,
pattern‑matching, and expression‑level guards do the work.

Fully ``mypy --strict`` clean.
"""

from collections.abc import Hashable, Mapping
from typing import List, Optional, Tuple, Union, NoReturn

import torch
from spectralmc.models.torch import Device, DType

__all__ = [
    "Scalar",
    "TensorTree",
    "move_tensor_tree",
    "get_tree_device_dtype",
    "module_state_device_dtype",
    "optimizer_state_device_dtype",
]

# ────────────────────────────── type aliases ────────────────────────────────
Scalar = Union[int, float, bool, str, bytes, None]
TensorTree = Union[
    torch.Tensor,
    Scalar,
    List["TensorTree"],
    Tuple["TensorTree", ...],
    Mapping[Hashable, "TensorTree"],
]

# ──────────────────────────── global resources ──────────────────────────────
_CUDA_STREAM: Optional[torch.cuda.Stream] = (
    torch.cuda.Stream(Device.cuda.to_torch()) if torch.cuda.is_available() else None
)


# ───────────────────────── functional helpers ───────────────────────────────
def _raise(exc: Exception) -> NoReturn:  # expression‑level raise
    raise exc


def _copy_tensor(
    src: torch.Tensor,
    *,
    target_dev: torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    """Clone *src* onto *target_dev* (non‑blocking, global stream)."""
    src.device == target_dev and _raise(
        ValueError("Attempted to move a tensor to its current device.")
    )

    dims = tuple(src.shape)

    match target_dev.type:
        case "cpu":
            assert _CUDA_STREAM
            dst = torch.empty(
                *dims,
                dtype=src.dtype,
                device=Device.cpu.to_torch(),
                pin_memory=pin_memory,
            )
            with torch.cuda.stream(_CUDA_STREAM):
                dst.copy_(src.detach(), non_blocking=True)
            return dst

        case "cuda":
            assert _CUDA_STREAM
            dst = torch.empty(*dims, dtype=src.dtype, device=target_dev)
            with torch.cuda.stream(_CUDA_STREAM):
                dst.copy_(src.detach(), non_blocking=True)
            return dst

        case _:
            _raise(RuntimeError(f"Unsupported destination device: {target_dev!r}"))


def _move(
    obj: TensorTree,
    *,
    target_dev: torch.device,
    pin_memory: bool,
) -> TensorTree:
    """Pure‑functional recursion via pattern matching."""
    match obj:
        case torch.Tensor():
            return _copy_tensor(obj, target_dev=target_dev, pin_memory=pin_memory)
        case list() as seq:
            return [_move(v, target_dev=target_dev, pin_memory=pin_memory) for v in seq]
        case tuple() as seq:
            return tuple(
                _move(v, target_dev=target_dev, pin_memory=pin_memory) for v in seq
            )
        case Mapping() as mp:
            return {
                k: _move(v, target_dev=target_dev, pin_memory=pin_memory)
                for k, v in mp.items()
            }
        case _:
            return obj  # scalar leaf


# ───────────────────────────── public API ───────────────────────────────────
def move_tensor_tree(
    tree: TensorTree,
    *,
    dest: Device,
    pin_memory: bool = True,
) -> TensorTree:
    """Copy *tree* so every tensor lives on *dest* (CPU or cuda:0)."""
    tgt = dest.to_torch()

    (tgt.type == "cuda" and not torch.cuda.is_available()) and _raise(
        RuntimeError("CUDA destination requested but CUDA is unavailable.")
    )

    result = _move(tree, target_dev=tgt, pin_memory=pin_memory)
    (None if _CUDA_STREAM is None else _CUDA_STREAM.synchronize())
    return result


# ───────────────────────── tree inspection util ────────────────────────────
def get_tree_device_dtype(tree: TensorTree) -> Tuple[Device, DType]:
    """Return the unique ``(Device, DType)`` shared by *all* tensors in *tree*."""

    def _pairs(node: TensorTree) -> set[Tuple[torch.device, torch.dtype]]:
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
    not pairs and _raise(RuntimeError("TensorTree contains no tensors."))
    len(pairs) != 1 and _raise(
        RuntimeError("TensorTree contains heterogeneous tensors.")
    )

    dev, dt = next(iter(pairs))
    return Device.from_torch(dev), DType.from_torch(dt)


# ───────────────────── state‑dict helpers (strictly typed) ──────────────────
def module_state_device_dtype(
    state: Mapping[str, torch.Tensor],
) -> Tuple[Device, DType]:
    """Device/dtype for a *model* ``state_dict``."""
    return get_tree_device_dtype(tuple(state.values()))


def optimizer_state_device_dtype(
    state: Mapping[str, object],  # <-- keys are *str* now
) -> Tuple[Device, DType]:
    """Device/dtype for an *optimizer* ``state_dict`` (Adam, etc.)."""

    def _tensors(node: object) -> List[torch.Tensor]:
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
    not ts and _raise(RuntimeError("state_dict() contains no tensors."))
    return get_tree_device_dtype(tuple(ts))
