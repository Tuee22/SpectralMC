"""
Shared registry for effect interpreter data flow.

This module provides a centralized registry for tensor, bytes, metadata, and other
data storage that is shared across all sub-interpreters, enabling data flow between
effects in a sequence.

Type Safety:
    - Result types for all retrieval operations
    - Type-specific retrieval methods with runtime validation
    - Frozen error ADTs for pattern matching

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cupy as cp
import torch

from spectralmc.result import Failure, Result, Success


@dataclass(frozen=True)
class RegistryKeyNotFound:
    """Error when a registry key is not found.

    Attributes:
        kind: Discriminator for pattern matching. Always "RegistryKeyNotFound".
        key: The key that was not found.
        expected_type: The type of value that was expected.
    """

    kind: Literal["RegistryKeyNotFound"] = "RegistryKeyNotFound"
    key: str = ""
    expected_type: str = ""


@dataclass(frozen=True)
class RegistryTypeMismatch:
    """Error when a registry value has an unexpected type.

    Attributes:
        kind: Discriminator for pattern matching. Always "RegistryTypeMismatch".
        key: The key that was accessed.
        expected_type: The type that was expected.
        actual_type: The actual type of the value.
    """

    kind: Literal["RegistryTypeMismatch"] = "RegistryTypeMismatch"
    key: str = ""
    expected_type: str = ""
    actual_type: str = ""


RegistryError = RegistryKeyNotFound | RegistryTypeMismatch


class SharedRegistry:
    """Centralized registry for all effect interpreter data.

    Provides type-safe storage and retrieval of:
    - Tensors (PyTorch tensors, CuPy arrays)
    - Bytes (for storage content, RNG state)
    - Metadata values (int, float, str)
    - Models (PyTorch modules)
    - Optimizers (PyTorch optimizers)
    - Kernels (Numba CUDA kernels)

    All retrieval methods return Result types for explicit error handling.

    Example:
        >>> registry = SharedRegistry()
        >>> registry.register_tensor("normals", cupy_array)
        >>> result = registry.get_cupy_array("normals")
        >>> match result:
        ...     case Success(arr):
        ...         use(arr)
        ...     case Failure(err):
        ...         handle(err)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tensors: dict[str, object] = {}
        self._bytes: dict[str, bytes] = {}
        self._metadata: dict[str, int | float | str] = {}
        self._models: dict[str, object] = {}
        self._optimizers: dict[str, object] = {}
        self._kernels: dict[str, object] = {}

    # ========== Tensor Operations ==========

    def register_tensor(self, tensor_id: str, tensor: object) -> None:
        """Register a tensor (PyTorch or CuPy) in the registry.

        Args:
            tensor_id: Unique identifier for the tensor.
            tensor: The tensor to store (PyTorch Tensor or CuPy ndarray).
        """
        self._tensors[tensor_id] = tensor

    def get_tensor(self, tensor_id: str) -> Result[object, RegistryError]:
        """Get a tensor by ID (untyped).

        Args:
            tensor_id: Identifier of the tensor to retrieve.

        Returns:
            Success with tensor, or Failure with RegistryKeyNotFound.
        """
        match self._tensors.get(tensor_id):
            case None:
                return Failure(RegistryKeyNotFound(key=tensor_id, expected_type="tensor"))
            case value:
                return Success(value)

    def get_torch_tensor(self, tensor_id: str) -> Result[torch.Tensor, RegistryError]:
        """Get a PyTorch tensor by ID with type validation.

        Args:
            tensor_id: Identifier of the tensor to retrieve.

        Returns:
            Success with torch.Tensor, or Failure with error.
        """
        match self._tensors.get(tensor_id):
            case None:
                return Failure(RegistryKeyNotFound(key=tensor_id, expected_type="torch.Tensor"))
            case torch.Tensor() as value:
                return Success(value)
            case value:
                return Failure(
                    RegistryTypeMismatch(
                        key=tensor_id,
                        expected_type="torch.Tensor",
                        actual_type=type(value).__name__,
                    )
                )

    def get_cupy_array(self, tensor_id: str) -> Result[cp.ndarray, RegistryError]:
        """Get a CuPy array by ID with type validation.

        Args:
            tensor_id: Identifier of the array to retrieve.

        Returns:
            Success with cp.ndarray, or Failure with error.
        """
        match self._tensors.get(tensor_id):
            case None:
                return Failure(RegistryKeyNotFound(key=tensor_id, expected_type="cp.ndarray"))
            case cp.ndarray() as value:
                return Success(value)
            case value:
                return Failure(
                    RegistryTypeMismatch(
                        key=tensor_id,
                        expected_type="cp.ndarray",
                        actual_type=type(value).__name__,
                    )
                )

    def has_tensor(self, tensor_id: str) -> bool:
        """Check if a tensor ID exists in the registry.

        Args:
            tensor_id: Identifier to check.

        Returns:
            True if tensor exists, False otherwise.
        """
        return tensor_id in self._tensors

    # ========== Bytes Operations ==========

    def register_bytes(self, key: str, data: bytes) -> None:
        """Register bytes content by key.

        Args:
            key: Unique identifier for the bytes.
            data: The bytes to store.
        """
        self._bytes[key] = data

    def get_bytes(self, key: str) -> Result[bytes, RegistryError]:
        """Get bytes content by key.

        Args:
            key: Identifier of the bytes to retrieve.

        Returns:
            Success with bytes, or Failure with RegistryKeyNotFound.
        """
        match self._bytes.get(key):
            case None:
                return Failure(RegistryKeyNotFound(key=key, expected_type="bytes"))
            case value:
                return Success(value)

    def has_bytes(self, key: str) -> bool:
        """Check if a bytes key exists in the registry.

        Args:
            key: Identifier to check.

        Returns:
            True if bytes exists, False otherwise.
        """
        return key in self._bytes

    # ========== Metadata Operations ==========

    def register_metadata(self, key: str, value: int | float | str) -> None:
        """Register a metadata value.

        Args:
            key: Unique identifier for the metadata.
            value: The value to store.
        """
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Result[int | float | str, RegistryError]:
        """Get a metadata value by key.

        Args:
            key: Identifier of the metadata to retrieve.

        Returns:
            Success with value, or Failure with RegistryKeyNotFound.
        """
        match self._metadata.get(key):
            case None:
                return Failure(RegistryKeyNotFound(key=key, expected_type="metadata"))
            case value:
                return Success(value)

    def update_metadata(
        self,
        key: str,
        operation: Literal["set", "add", "increment"],
        value: int | float | str,
    ) -> Result[int | float | str, RegistryError]:
        """Update metadata with an operation.

        Args:
            key: Identifier of the metadata to update.
            operation: The operation to perform ("set", "add", "increment").
            value: The value to use in the operation.

        Returns:
            Success with new value, or Failure with error.
        """
        match operation:
            case "set":
                self._metadata[key] = value
                return Success(value)
            case "add":
                current = self._metadata.get(key, 0)
                match (current, value):
                    case (str(), _) | (_, str()):
                        return Failure(
                            RegistryTypeMismatch(
                                key=key, expected_type="numeric", actual_type="str"
                            )
                        )
                    case _:
                        new_value: int | float = current + value
                        self._metadata[key] = new_value
                        return Success(new_value)
            case "increment":
                current = self._metadata.get(key, 0)
                match current:
                    case str():
                        return Failure(
                            RegistryTypeMismatch(
                                key=key, expected_type="numeric", actual_type="str"
                            )
                        )
                    case _:
                        new_value_inc: int | float = current + 1
                        self._metadata[key] = new_value_inc
                        return Success(new_value_inc)

    def has_metadata(self, key: str) -> bool:
        """Check if a metadata key exists in the registry.

        Args:
            key: Identifier to check.

        Returns:
            True if metadata exists, False otherwise.
        """
        return key in self._metadata

    # ========== Model Operations ==========

    def register_model(self, model_id: str, model: object) -> None:
        """Register a model (PyTorch module).

        Args:
            model_id: Unique identifier for the model.
            model: The model to store.
        """
        self._models[model_id] = model

    def get_model(self, model_id: str) -> Result[object, RegistryError]:
        """Get a model by ID.

        Args:
            model_id: Identifier of the model to retrieve.

        Returns:
            Success with model, or Failure with RegistryKeyNotFound.
        """
        match self._models.get(model_id):
            case None:
                return Failure(RegistryKeyNotFound(key=model_id, expected_type="model"))
            case value:
                return Success(value)

    def has_model(self, model_id: str) -> bool:
        """Check if a model ID exists in the registry.

        Args:
            model_id: Identifier to check.

        Returns:
            True if model exists, False otherwise.
        """
        return model_id in self._models

    # ========== Optimizer Operations ==========

    def register_optimizer(self, optimizer_id: str, optimizer: object) -> None:
        """Register an optimizer.

        Args:
            optimizer_id: Unique identifier for the optimizer.
            optimizer: The optimizer to store.
        """
        self._optimizers[optimizer_id] = optimizer

    def get_optimizer(self, optimizer_id: str) -> Result[object, RegistryError]:
        """Get an optimizer by ID.

        Args:
            optimizer_id: Identifier of the optimizer to retrieve.

        Returns:
            Success with optimizer, or Failure with RegistryKeyNotFound.
        """
        match self._optimizers.get(optimizer_id):
            case None:
                return Failure(RegistryKeyNotFound(key=optimizer_id, expected_type="optimizer"))
            case value:
                return Success(value)

    def has_optimizer(self, optimizer_id: str) -> bool:
        """Check if an optimizer ID exists in the registry.

        Args:
            optimizer_id: Identifier to check.

        Returns:
            True if optimizer exists, False otherwise.
        """
        return optimizer_id in self._optimizers

    # ========== Kernel Operations ==========

    def register_kernel(self, kernel_name: str, kernel_fn: object) -> None:
        """Register a kernel function.

        Args:
            kernel_name: Unique identifier for the kernel.
            kernel_fn: The kernel function to store.
        """
        self._kernels[kernel_name] = kernel_fn

    def get_kernel(self, kernel_name: str) -> Result[object, RegistryError]:
        """Get a kernel by name.

        Args:
            kernel_name: Name of the kernel to retrieve.

        Returns:
            Success with kernel, or Failure with RegistryKeyNotFound.
        """
        match self._kernels.get(kernel_name):
            case None:
                return Failure(RegistryKeyNotFound(key=kernel_name, expected_type="kernel"))
            case value:
                return Success(value)

    def has_kernel(self, kernel_name: str) -> bool:
        """Check if a kernel name exists in the registry.

        Args:
            kernel_name: Name to check.

        Returns:
            True if kernel exists, False otherwise.
        """
        return kernel_name in self._kernels

    # ========== Bulk Operations ==========

    def clear(self) -> None:
        """Clear all registry contents."""
        self._tensors.clear()
        self._bytes.clear()
        self._metadata.clear()
        self._models.clear()
        self._optimizers.clear()
        self._kernels.clear()

    def clear_tensors(self) -> None:
        """Clear only tensor registry (preserves models, content, etc.)."""
        self._tensors.clear()

    def clear_bytes(self) -> None:
        """Clear only bytes registry."""
        self._bytes.clear()

    def clear_metadata(self) -> None:
        """Clear only metadata registry."""
        self._metadata.clear()
