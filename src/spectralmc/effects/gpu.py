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
from spectralmc.result import Result, Success, Failure


@dataclass(frozen=True)
class InvalidTransferError:
    """Error when source and target device are identical."""

    device: Device
    kind: Literal["InvalidTransferError"] = "InvalidTransferError"


@dataclass(frozen=True)
class InvalidDLPackTransferError:
    """Error when source and target framework are identical."""

    framework: Literal["cupy", "torch"]
    kind: Literal["InvalidDLPackTransferError"] = "InvalidDLPackTransferError"


@dataclass(frozen=True)
class TensorTransfer:
    """Request to transfer tensor between devices.

    NOTE: Only constructible via tensor_transfer() factory function.
    Direct construction is not validated.

    Attributes:
        kind: Discriminator for pattern matching. Always "TensorTransfer".
        source_device: Device to transfer from.
        target_device: Device to transfer to.
        tensor_id: Opaque identifier for the tensor or TensorTree.

    Example:
        >>> match tensor_transfer(Device.cuda, Device.cpu, "weights"):
        ...     case Success(effect):
        ...         print(f"Transfer {effect.source_device} -> {effect.target_device}")
        ...     case Failure(error):
        ...         print(f"Cannot transfer to same device: {error.device}")
    """

    kind: Literal["TensorTransfer"] = "TensorTransfer"
    source_device: Device = Device.cuda
    target_device: Device = Device.cpu
    tensor_id: str = ""


def tensor_transfer(
    source_device: Device,
    target_device: Device,
    tensor_id: str = "",
) -> Result[TensorTransfer, InvalidTransferError]:
    """Create TensorTransfer, returning Failure if devices are identical.

    This is the preferred way to construct TensorTransfer with validation.

    Args:
        source_device: Device to transfer from
        target_device: Device to transfer to
        tensor_id: Opaque identifier for the tensor

    Returns:
        Success(TensorTransfer) if valid transfer, else Failure(InvalidTransferError)

    Example:
        >>> match tensor_transfer(Device.cuda, Device.cpu, "weights"):
        ...     case Success(effect):
        ...         # Valid transfer - use it
        ...         pass
        ...     case Failure(error):
        ...         # Invalid: same device
        ...         print(f"Cannot transfer to same device: {error.device}")
    """
    return (
        Failure(InvalidTransferError(device=source_device))
        if source_device == target_device
        else Success(
            TensorTransfer(
                source_device=source_device,
                target_device=target_device,
                tensor_id=tensor_id,
            )
        )
    )


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

    NOTE: Only constructible via dlpack_transfer() factory function.
    Direct construction is not validated.

    Enables zero-copy tensor sharing between CuPy and PyTorch on GPU.

    Attributes:
        kind: Discriminator for pattern matching. Always "DLPackTransfer".
        source_tensor_id: Identifier for the source tensor.
        source_framework: Framework of the source tensor.
        target_framework: Framework to convert to.
        output_tensor_id: Identifier for storing the converted tensor.

    Example:
        >>> match dlpack_transfer("cupy_fft", "cupy", "torch", "torch_fft"):
        ...     case Success(effect):
        ...         # Valid transfer
        ...         pass
        ...     case Failure(error):
        ...         # Invalid: same framework
        ...         print(f"Cannot transfer to same framework: {error.framework}")
    """

    kind: Literal["DLPackTransfer"] = "DLPackTransfer"
    source_tensor_id: str = ""
    source_framework: Literal["cupy", "torch"] = "cupy"
    target_framework: Literal["cupy", "torch"] = "torch"
    output_tensor_id: str = ""


def dlpack_transfer(
    source_tensor_id: str,
    source_framework: Literal["cupy", "torch"],
    target_framework: Literal["cupy", "torch"],
    output_tensor_id: str,
) -> Result[DLPackTransfer, InvalidDLPackTransferError]:
    """Create DLPackTransfer, returning Failure if frameworks are identical.

    This is the preferred way to construct DLPackTransfer with validation.

    Args:
        source_tensor_id: Identifier for the source tensor
        source_framework: Framework of the source tensor
        target_framework: Framework to convert to
        output_tensor_id: Identifier for storing the converted tensor

    Returns:
        Success(DLPackTransfer) if valid transfer, else Failure(error)

    Example:
        >>> match dlpack_transfer("fft_result", "cupy", "torch", "torch_tensor"):
        ...     case Success(effect):
        ...         # Valid transfer - frameworks are different
        ...         pass
        ...     case Failure(error):
        ...         # Invalid: same framework
        ...         print(f"Cannot transfer to same framework: {error.framework}")
    """
    return (
        Failure(InvalidDLPackTransferError(framework=source_framework))
        if source_framework == target_framework
        else Success(
            DLPackTransfer(
                source_tensor_id=source_tensor_id,
                source_framework=source_framework,
                target_framework=target_framework,
                output_tensor_id=output_tensor_id,
            )
        )
    )


# GPU Effect Union - enables exhaustive pattern matching
GPUEffect = TensorTransfer | StreamSync | KernelLaunch | DLPackTransfer
