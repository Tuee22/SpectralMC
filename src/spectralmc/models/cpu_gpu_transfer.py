# src/spectralmc/models/cpu_gpu_transfer.py
from __future__ import annotations

"""
cpu_transfer
============

Move an arbitrarily‑nested *TensorTree* to :data:`device.cpu`
or :data:`device.cuda` without allocating on the source GPU.

* One global CUDA stream (`cuda:0`) reused for every copy.
* Raises :class:`ValueError` as soon as a tensor is already on *dest*.
* Passes ``mypy --strict``.
"""

from spectralmc.models.torch import Device, DType
from typing import Dict, List, Optional, Tuple, Union
import torch

__all__ = ["Scalar", "TensorTree", "move_tensor_tree"]

# ────────────────────────────── type aliases ────────────────────────────────
Scalar = Union[int, float, bool, str, bytes, None]
TensorTree = Union[
    torch.Tensor,
    Scalar,
    List["TensorTree"],
    Tuple["TensorTree", ...],
    Dict[str, "TensorTree"],
]

# ──────────────────────────── global resources ──────────────────────────────
_CUDA_STREAM: Optional[torch.cuda.Stream] = (
    torch.cuda.Stream(device=0) if torch.cuda.is_available() else None
)


# ───────────────────────── internal helpers ─────────────────────────────────
def _copy_tensor(
    src: torch.Tensor,
    *,
    target_dev: torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    """
    Copy *src* onto *target_dev*.

    Raises
    ------
    ValueError
        If *src* already lives on *target_dev*.
    """
    if src.device == target_dev:
        raise ValueError("Attempted to move a tensor to its current device.")

    dims = tuple(src.shape)  # explicit; avoids .size() missing in stubs

    match target_dev.type:
        case "cpu":
            # GPU → CPU
            assert _CUDA_STREAM is not None
            dst = torch.empty(
                *dims,
                dtype=src.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            with torch.cuda.stream(_CUDA_STREAM):
                dst.copy_(src.detach(), non_blocking=True)
            return dst

        case "cuda":
            # CPU → GPU  (cuda:0 only in this codebase)
            assert _CUDA_STREAM is not None
            dst = torch.empty(
                *dims,
                dtype=src.dtype,
                device=target_dev,
            )
            with torch.cuda.stream(_CUDA_STREAM):
                dst.copy_(src.detach(), non_blocking=True)
            return dst

        case _:
            raise RuntimeError(f"Unsupported destination device: {target_dev!r}")


def _move(
    obj: TensorTree,
    *,
    target_dev: torch.device,
    pin_memory: bool,
) -> TensorTree:
    """Recursively copy every tensor in *obj* to *target_dev*."""
    if isinstance(obj, torch.Tensor):
        return _copy_tensor(obj, target_dev=target_dev, pin_memory=pin_memory)

    if isinstance(obj, list):
        return [_move(v, target_dev=target_dev, pin_memory=pin_memory) for v in obj]

    if isinstance(obj, tuple):
        return tuple(
            _move(v, target_dev=target_dev, pin_memory=pin_memory) for v in obj
        )

    if isinstance(obj, dict):
        return {
            k: _move(v, target_dev=target_dev, pin_memory=pin_memory)
            for k, v in obj.items()
        }

    # Scalar leaf
    return obj


# ───────────────────────────── public API ───────────────────────────────────
def move_tensor_tree(
    tree: TensorTree,
    *,
    dest: Device,
    pin_memory: bool = True,
) -> TensorTree:
    """
    Return a deep‑copy of *tree* where **all tensors now live on *dest***.

    A :class:`ValueError` is raised immediately if a tensor encountered
    is already on *dest*.
    """
    target_dev: torch.device = dest.to_torch()

    if target_dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA destination requested but CUDA is unavailable.")

    result: TensorTree = _move(tree, target_dev=target_dev, pin_memory=pin_memory)

    if _CUDA_STREAM is not None:
        _CUDA_STREAM.synchronize()

    return result


# ───────────────────────── tree inspection util ────────────────────────────
def tree_device_dtype(tree: TensorTree) -> Tuple[Device, DType]:
    """
    Return the unique ``(Device, DType)`` shared by **all** tensors in *tree*.

    Raises
    ------
    RuntimeError
        If *tree* contains **no tensors**.
    ValueError
        If tensors differ in *device* **or** *dtype*.

    Notes
    -----
    * Scalar leaves (ints, floats, ``None`` …) are ignored.
    * Pure‑functional traversal: recursion, set comprehensions, pattern matching.
    * No ``Any``, ``cast``, or ``type: ignore`` needed.
    """

    # one‑pass recursion emitting {(device, dtype)} pairs
    def _pairs(node: TensorTree) -> set[Tuple[torch.device, torch.dtype]]:
        match node:
            case torch.Tensor() as t:
                return {(t.device, t.dtype)}
            case list() | tuple() as seq:
                return {p for item in seq for p in _pairs(item)}
            case dict() as mapping:
                return {p for item in mapping.values() for p in _pairs(item)}
            case _:
                return set()

    pairs = _pairs(tree)

    if not pairs:
        raise RuntimeError("TensorTree contains no tensors.")
    if len(pairs) != 1:
        raise ValueError("TensorTree contains tensors on different devices or dtypes.")

    dev, dt = next(iter(pairs))
    return Device.from_torch(dev), DType.from_torch(dt)
