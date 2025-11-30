"""
GPU Effect ADTs for tensor transfers and stream operations.

This module defines frozen dataclasses representing all GPU-related side effects,
enabling exhaustive pattern matching and type-safe effect composition.

Type Safety:
    - All effect types are frozen dataclasses (immutable)
    - Literal discriminators enable exhaustive pattern matching
    - Union types define closed sets of effect variants
    - __post_init__ validation prevents illegal state construction

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from spectralmc.models.torch import Device


@dataclass(frozen=True)
class TensorTransfer:
    """Request to transfer tensor between devices.

    Attributes:
        kind: Discriminator for pattern matching. Always "TensorTransfer".
        source_device: Device to transfer from.
        target_device: Device to transfer to.
        tensor_id: Opaque identifier for the tensor or TensorTree.

    Raises:
        ValueError: If source_device equals target_device (illegal state).

    Example:
        >>> effect = TensorTransfer(
        ...     source_device=Device.cuda,
        ...     target_device=Device.cpu,
        ...     tensor_id="model_weights",
        ... )
        >>> match effect:
        ...     case TensorTransfer(source_device=src, target_device=dst):
        ...         print(f"Transfer {src} -> {dst}")
    """

    kind: Literal["TensorTransfer"] = "TensorTransfer"
    source_device: Device = Device.cuda
    target_device: Device = Device.cpu
    tensor_id: str = ""

    def __post_init__(self) -> None:
        if self.source_device == self.target_device:
            raise ValueError(f"Invalid transfer: source and target are both {self.source_device}")


@dataclass(frozen=True)
class StreamSync:
    """Request to synchronize a CUDA stream.

    Attributes:
        kind: Discriminator for pattern matching. Always "StreamSync".
        stream_type: Which stream type to synchronize.
    """

    kind: Literal["StreamSync"] = "StreamSync"
    stream_type: Literal["torch", "cupy", "numba"] = "torch"


@dataclass(frozen=True)
class KernelLaunch:
    """Request to launch a CUDA kernel.

    Attributes:
        kind: Discriminator for pattern matching. Always "KernelLaunch".
        kernel_name: Name of the kernel function.
        grid_config: Grid dimensions for kernel launch.
        block_config: Block dimensions for kernel launch.
    """

    kind: Literal["KernelLaunch"] = "KernelLaunch"
    kernel_name: str = ""
    grid_config: tuple[int, ...] = ()
    block_config: tuple[int, ...] = ()


@dataclass(frozen=True)
class DLPackTransfer:
    """Request to transfer tensor between frameworks via DLPack protocol.

    Enables zero-copy tensor sharing between CuPy and PyTorch on GPU.

    Attributes:
        kind: Discriminator for pattern matching. Always "DLPackTransfer".
        source_tensor_id: Identifier for the source tensor.
        source_framework: Framework of the source tensor.
        target_framework: Framework to convert to.
        output_tensor_id: Identifier for storing the converted tensor.

    Raises:
        ValueError: If source and target frameworks are the same.

    Example:
        >>> effect = DLPackTransfer(
        ...     source_tensor_id="cupy_fft_result",
        ...     source_framework="cupy",
        ...     target_framework="torch",
        ...     output_tensor_id="torch_fft_tensor",
        ... )
    """

    kind: Literal["DLPackTransfer"] = "DLPackTransfer"
    source_tensor_id: str = ""
    source_framework: Literal["cupy", "torch"] = "cupy"
    target_framework: Literal["cupy", "torch"] = "torch"
    output_tensor_id: str = ""

    def __post_init__(self) -> None:
        if self.source_framework == self.target_framework:
            raise ValueError(
                f"Invalid DLPack transfer: source and target are both {self.source_framework}"
            )


# GPU Effect Union - enables exhaustive pattern matching
GPUEffect = TensorTransfer | StreamSync | KernelLaunch | DLPackTransfer
