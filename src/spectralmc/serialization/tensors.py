# src/spectralmc/serialization/tensors.py
"""Converters for tensor state and optimizer state."""

from __future__ import annotations

from typing import Dict, List

# CRITICAL: Import facade BEFORE torch for deterministic algorithms
import spectralmc.models.torch as sm_torch  # noqa: E402
import torch  # noqa: E402

from spectralmc.models.torch import (  # noqa: E402
    AdamOptimizerState,
    AdamParamState,
    AdamParamGroup,
    Device,
    AnyDType,
    FullPrecisionDType,
    ReducedPrecisionDType,
)
from spectralmc.proto import tensors_pb2
from spectralmc.serialization.common import DTypeConverter, DeviceConverter


class TensorStateConverter:
    """Convert torch.Tensor to/from TensorStateProto."""

    @staticmethod
    def to_proto(tensor: torch.Tensor) -> tensors_pb2.TensorStateProto:
        """
        Serialize a torch.Tensor to protobuf.

        Args:
            tensor: PyTorch tensor to serialize

        Returns:
            TensorStateProto message
        """
        proto = tensors_pb2.TensorStateProto()

        # Shape
        proto.shape.extend(tensor.shape)

        # DType - convert to DType enum then to proto
        torch_dtype = tensor.dtype
        dtype_enum: AnyDType

        # Map torch.dtype to our DType enum
        if torch_dtype == torch.float32:
            dtype_enum = FullPrecisionDType.float32
        elif torch_dtype == torch.float64:
            dtype_enum = FullPrecisionDType.float64
        elif torch_dtype == torch.complex64:
            dtype_enum = FullPrecisionDType.complex64
        elif torch_dtype == torch.complex128:
            dtype_enum = FullPrecisionDType.complex128
        elif torch_dtype == torch.float16:
            dtype_enum = ReducedPrecisionDType.float16
        elif torch_dtype == torch.bfloat16:
            dtype_enum = ReducedPrecisionDType.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {torch_dtype}")

        proto.dtype = DTypeConverter.to_proto(dtype_enum)

        # Device
        device_enum = Device.from_torch(tensor.device)
        proto.device = DeviceConverter.to_proto(device_enum)

        # Data - serialize via numpy
        # Move to CPU first to ensure consistent serialization
        cpu_tensor = tensor.cpu().detach()
        proto.data = cpu_tensor.numpy().tobytes()

        # Requires grad
        proto.requires_grad = tensor.requires_grad

        return proto

    @staticmethod
    def from_proto(proto: tensors_pb2.TensorStateProto) -> torch.Tensor:
        """
        Deserialize a TensorStateProto to torch.Tensor.

        Args:
            proto: TensorStateProto message

        Returns:
            PyTorch tensor
        """
        # Dtype
        dtype_enum = DTypeConverter.from_proto(proto.dtype)

        # Map DType enum to torch.dtype
        if isinstance(dtype_enum, FullPrecisionDType):
            if dtype_enum == FullPrecisionDType.float32:
                torch_dtype = torch.float32
            elif dtype_enum == FullPrecisionDType.float64:
                torch_dtype = torch.float64
            elif dtype_enum == FullPrecisionDType.complex64:
                torch_dtype = torch.complex64
            elif dtype_enum == FullPrecisionDType.complex128:
                torch_dtype = torch.complex128
            else:
                raise ValueError(f"Unknown FullPrecisionDType: {dtype_enum}")
        elif isinstance(dtype_enum, ReducedPrecisionDType):
            if dtype_enum == ReducedPrecisionDType.float16:
                torch_dtype = torch.float16
            elif dtype_enum == ReducedPrecisionDType.bfloat16:
                torch_dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown ReducedPrecisionDType: {dtype_enum}")
        else:
            raise TypeError(f"Unknown dtype type: {type(dtype_enum)}")

        # Shape
        shape = tuple(proto.shape)

        # Deserialize data from bytes via numpy
        import numpy as np

        # Map torch dtype to numpy dtype
        # Use explicit union type to avoid Any
        np_dtype: (
            type[np.float32]
            | type[np.float64]
            | type[np.float16]
            | type[np.complex64]
            | type[np.complex128]
            | type[np.uint16]
        )
        if torch_dtype == torch.float32:
            np_dtype = np.float32
        elif torch_dtype == torch.float16:
            np_dtype = np.float16
        elif torch_dtype == torch.float64:
            np_dtype = np.float64
        elif torch_dtype == torch.complex64:
            np_dtype = np.complex64
        elif torch_dtype == torch.complex128:
            np_dtype = np.complex128
        elif torch_dtype == torch.bfloat16:
            # NumPy doesn't have bfloat16, use uint16 and reinterpret
            np_dtype = np.uint16
        else:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

        # Reconstruct array from bytes
        flat_array = np.frombuffer(proto.data, dtype=np_dtype)
        reshaped = flat_array.reshape(shape)

        # Convert to torch tensor
        if torch_dtype == torch.bfloat16:
            # Special handling for bfloat16
            tensor = torch.from_numpy(reshaped).view(torch.bfloat16)
        else:
            tensor = torch.from_numpy(reshaped.copy())

        # Device
        device_enum = DeviceConverter.from_proto(proto.device)
        device = device_enum.to_torch()
        tensor = tensor.to(device)

        # Requires grad
        if proto.requires_grad:
            tensor.requires_grad_(True)

        return tensor


class AdamOptimizerStateConverter:
    """Convert AdamOptimizerState to/from AdamOptimizerStateProto."""

    @staticmethod
    def to_proto(
        optimizer_state: AdamOptimizerState,
    ) -> tensors_pb2.AdamOptimizerStateProto:
        """
        Serialize AdamOptimizerState to protobuf.

        Args:
            optimizer_state: AdamOptimizerState instance

        Returns:
            AdamOptimizerStateProto message
        """
        proto = tensors_pb2.AdamOptimizerStateProto()

        # Serialize param states (map of int → AdamParamState)
        for param_id, param_state in optimizer_state.param_states.items():
            param_proto = tensors_pb2.AdamParamStateProto()
            param_proto.step = param_state.step
            param_proto.exp_avg.CopyFrom(
                TensorStateConverter.to_proto(param_state.exp_avg.to_torch())
            )
            param_proto.exp_avg_sq.CopyFrom(
                TensorStateConverter.to_proto(param_state.exp_avg_sq.to_torch())
            )
            proto.state[param_id].CopyFrom(param_proto)

        # Serialize param groups
        for group in optimizer_state.param_groups:
            group_proto = tensors_pb2.AdamParamGroupProto()
            group_proto.lr = group.lr
            group_proto.beta1 = group.betas[0]
            group_proto.beta2 = group.betas[1]
            group_proto.eps = group.eps
            group_proto.weight_decay = group.weight_decay
            group_proto.amsgrad = group.amsgrad
            proto.param_groups.append(group_proto)

        return proto

    @staticmethod
    def from_proto(
        proto: tensors_pb2.AdamOptimizerStateProto,
    ) -> AdamOptimizerState:
        """
        Deserialize AdamOptimizerStateProto to AdamOptimizerState.

        Args:
            proto: AdamOptimizerStateProto message

        Returns:
            AdamOptimizerState instance
        """
        # Deserialize param states
        param_states: Dict[int, AdamParamState] = {}
        for param_id, param_proto in proto.state.items():
            exp_avg_tensor = TensorStateConverter.from_proto(param_proto.exp_avg)
            exp_avg_sq_tensor = TensorStateConverter.from_proto(param_proto.exp_avg_sq)

            param_state = AdamParamState.from_torch(
                {
                    "step": param_proto.step,
                    "exp_avg": exp_avg_tensor,
                    "exp_avg_sq": exp_avg_sq_tensor,
                }
            )
            param_states[param_id] = param_state

        # Deserialize param groups
        param_groups: List[AdamParamGroup] = []
        for group_proto in proto.param_groups:
            group = AdamParamGroup(
                params=[],  # Not serialized in proto
                lr=group_proto.lr,
                betas=(group_proto.beta1, group_proto.beta2),
                eps=group_proto.eps,
                weight_decay=group_proto.weight_decay,
                amsgrad=group_proto.amsgrad,
            )
            param_groups.append(group)

        return AdamOptimizerState(param_states=param_states, param_groups=param_groups)


class RNGStateConverter:
    """Convert RNG state to/from RNGStateProto."""

    @staticmethod
    def to_proto(
        torch_cpu_rng_state: bytes, torch_cuda_rng_states: List[bytes]
    ) -> tensors_pb2.RNGStateProto:
        """
        Serialize RNG state to protobuf.

        Args:
            torch_cpu_rng_state: PyTorch CPU RNG state
            torch_cuda_rng_states: List of CUDA RNG states (one per device)

        Returns:
            RNGStateProto message
        """
        proto = tensors_pb2.RNGStateProto()
        proto.torch_cpu_rng_state = torch_cpu_rng_state
        proto.torch_cuda_rng_states.extend(torch_cuda_rng_states)
        return proto

    @staticmethod
    def from_proto(proto: tensors_pb2.RNGStateProto) -> tuple[bytes, List[bytes]]:
        """
        Deserialize RNGStateProto to RNG state.

        Args:
            proto: RNGStateProto message

        Returns:
            Tuple of (cpu_rng_state, cuda_rng_states)
        """
        return proto.torch_cpu_rng_state, list(proto.torch_cuda_rng_states)


__all__ = [
    "ModelCheckpointConverter",
    "TensorStateConverter",
    "AdamOptimizerStateConverter",
    "RNGStateConverter",
]


class ModelCheckpointConverter:
    """Convert complete model checkpoint to/from ModelCheckpointProto."""

    @staticmethod
    def to_proto(
        model_state_dict: Dict[str, torch.Tensor],
        optimizer_state: AdamOptimizerState,
        torch_cpu_rng_state: bytes,
        torch_cuda_rng_states: List[bytes],
        global_step: int,
    ) -> tensors_pb2.ModelCheckpointProto:
        """
        Serialize complete model checkpoint to protobuf.

        Args:
            model_state_dict: Model parameters (name → tensor)
            optimizer_state: AdamOptimizerState instance
            torch_cpu_rng_state: PyTorch CPU RNG state
            torch_cuda_rng_states: List of CUDA RNG states
            global_step: Training step counter

        Returns:
            ModelCheckpointProto message
        """
        proto = tensors_pb2.ModelCheckpointProto()

        # Serialize model state dict
        for name, tensor in model_state_dict.items():
            proto.model_state_dict[name].CopyFrom(TensorStateConverter.to_proto(tensor))

        # Serialize optimizer state
        proto.optimizer_state.CopyFrom(
            AdamOptimizerStateConverter.to_proto(optimizer_state)
        )

        # Serialize RNG state
        proto.rng_state.CopyFrom(
            RNGStateConverter.to_proto(torch_cpu_rng_state, torch_cuda_rng_states)
        )

        # Global step
        proto.global_step = global_step

        return proto

    @staticmethod
    def from_proto(
        proto: tensors_pb2.ModelCheckpointProto,
    ) -> tuple[Dict[str, torch.Tensor], AdamOptimizerState, bytes, List[bytes], int]:
        """
        Deserialize ModelCheckpointProto to checkpoint components.

        Args:
            proto: ModelCheckpointProto message

        Returns:
            Tuple of (model_state_dict, optimizer_state, cpu_rng, cuda_rngs, global_step)
        """
        # Deserialize model state dict
        model_state_dict: Dict[str, torch.Tensor] = {}
        for name, tensor_proto in proto.model_state_dict.items():
            model_state_dict[name] = TensorStateConverter.from_proto(tensor_proto)

        # Deserialize optimizer state
        optimizer_state = AdamOptimizerStateConverter.from_proto(proto.optimizer_state)

        # Deserialize RNG state
        cpu_rng, cuda_rngs = RNGStateConverter.from_proto(proto.rng_state)

        # Global step
        global_step = int(proto.global_step)

        return model_state_dict, optimizer_state, cpu_rng, cuda_rngs, global_step
