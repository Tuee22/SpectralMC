# src/spectralmc/serialization/tensors.py
"""Converters for tensor state and optimizer state."""

from __future__ import annotations


import numpy as np
import torch

from spectralmc.errors.serialization import InvalidTensorState, SerializationError
from spectralmc.errors.torch_facade import TorchFacadeError
from spectralmc.models.torch import (
    AdamOptimizerState,
    AdamParamGroup,
    AdamParamState,
    AnyDType,
    Device,
    FullPrecisionDType,
    ReducedPrecisionDType,
)
from spectralmc.proto import tensors_pb2
from spectralmc.serialization.common import DeviceConverter, DTypeConverter
from spectralmc.result import Failure, Result, Success


class TensorStateConverter:
    """Convert torch.Tensor to/from TensorStateProto."""

    @staticmethod
    def to_proto(tensor: torch.Tensor) -> Result[tensors_pb2.TensorStateProto, InvalidTensorState]:
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
        # Map torch.dtype to our DType enum via match/case
        match torch_dtype:
            case torch.float32:
                dtype_enum: AnyDType = FullPrecisionDType.float32
            case torch.float64:
                dtype_enum = FullPrecisionDType.float64
            case torch.complex64:
                dtype_enum = FullPrecisionDType.complex64
            case torch.complex128:
                dtype_enum = FullPrecisionDType.complex128
            case torch.float16:
                dtype_enum = ReducedPrecisionDType.float16
            case torch.bfloat16:
                dtype_enum = ReducedPrecisionDType.bfloat16
            case _:
                return Failure(InvalidTensorState(message=f"Unsupported dtype: {torch_dtype}"))

        match DTypeConverter.to_proto(dtype_enum):
            case Failure(error):
                return Failure(InvalidTensorState(message=str(error)))
            case Success(dtype_proto):
                proto.dtype = dtype_proto

        # Device
        match Device.from_torch(tensor.device):
            case Failure(device_error):
                return Failure(InvalidTensorState(message=str(device_error)))
            case Success(device_enum):
                proto.device = DeviceConverter.to_proto(device_enum)

        # Data - serialize via numpy
        # NOTE: .cpu() is acceptable here per CPU/GPU policy - this is serialization
        # boundary I/O, not compute. The TensorTree API cannot be used because
        # it raises on no-op moves (tensors may already be on CPU).
        cpu_tensor = tensor.cpu().detach()
        proto.data = cpu_tensor.numpy().tobytes()

        # Requires grad
        proto.requires_grad = tensor.requires_grad

        return Success(proto)

    @staticmethod
    def from_proto(proto: tensors_pb2.TensorStateProto) -> Result[torch.Tensor, InvalidTensorState]:
        """
        Deserialize a TensorStateProto to torch.Tensor.

        Args:
            proto: TensorStateProto message

        Returns:
            Result containing PyTorch tensor or deserialization error
        """
        # Dtype
        match DTypeConverter.from_proto(proto.dtype):
            case Failure(error):
                return Failure(InvalidTensorState(message=str(error)))
            case Success(dtype_enum):
                pass

        # Map DType enum to torch.dtype via match/case
        match dtype_enum:
            case FullPrecisionDType.float32:
                torch_dtype = torch.float32
            case FullPrecisionDType.float64:
                torch_dtype = torch.float64
            case FullPrecisionDType.complex64:
                torch_dtype = torch.complex64
            case FullPrecisionDType.complex128:
                torch_dtype = torch.complex128
            case ReducedPrecisionDType.float16:
                torch_dtype = torch.float16
            case ReducedPrecisionDType.bfloat16:
                torch_dtype = torch.bfloat16

        # Shape
        shape = tuple(proto.shape)

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
        match torch_dtype:
            case torch.float32:
                np_dtype = np.float32
            case torch.float16:
                np_dtype = np.float16
            case torch.float64:
                np_dtype = np.float64
            case torch.complex64:
                np_dtype = np.complex64
            case torch.complex128:
                np_dtype = np.complex128
            case torch.bfloat16:
                # NumPy doesn't have bfloat16, use uint16 and reinterpret
                np_dtype = np.uint16
            case _:
                return Failure(
                    InvalidTensorState(message=f"Unsupported torch dtype: {torch_dtype}")
                )

        # Reconstruct array from bytes
        flat_array = np.frombuffer(proto.data, dtype=np_dtype)
        reshaped = flat_array.reshape(shape)

        # Convert to torch tensor
        tensor = (
            torch.from_numpy(reshaped).view(torch.bfloat16)
            if torch_dtype == torch.bfloat16
            else torch.from_numpy(reshaped.copy())
        )

        # Device
        # NOTE: .to(device) is acceptable here per CPU/GPU policy - this is
        # deserialization boundary I/O, not compute. The TensorTree API cannot
        # be used because it raises on no-op moves (target may be CPU).
        device_enum = DeviceConverter.from_proto(proto.device)
        device = device_enum.to_torch()
        tensor = tensor.to(device)

        # Requires grad
        if proto.requires_grad:
            tensor.requires_grad_(True)

        return Success(tensor)


def _convert_param_state_to_proto(
    param_id: int, param_state: AdamParamState
) -> Result[tuple[int, tensors_pb2.AdamParamStateProto], SerializationError | TorchFacadeError]:
    """Convert single param state to proto, returning Result.

    Pure helper function for AdamOptimizerStateConverter.to_proto().
    """
    entry = tensors_pb2.AdamParamStateProto(step=param_state.step)

    # Convert exp_avg
    match param_state.exp_avg.to_torch():
        case Failure() as f:
            return f
        case Success(exp_avg_tensor):
            match TensorStateConverter.to_proto(exp_avg_tensor):
                case Failure() as f:
                    return f
                case Success(exp_avg_proto):
                    entry.exp_avg.CopyFrom(exp_avg_proto)

    # Convert exp_avg_sq
    match param_state.exp_avg_sq.to_torch():
        case Failure() as f:
            return f
        case Success(exp_avg_sq_tensor):
            match TensorStateConverter.to_proto(exp_avg_sq_tensor):
                case Failure() as f:
                    return f
                case Success(exp_avg_sq_proto):
                    entry.exp_avg_sq.CopyFrom(exp_avg_sq_proto)

    return Success((param_id, entry))


def _convert_param_state_from_proto(
    param_id: int, param_proto: tensors_pb2.AdamParamStateProto
) -> Result[tuple[int, AdamParamState], SerializationError | TorchFacadeError]:
    """Convert single param proto to AdamParamState, returning Result.

    Pure helper function for AdamOptimizerStateConverter.from_proto().
    """
    # Convert exp_avg
    match TensorStateConverter.from_proto(param_proto.exp_avg):
        case Failure() as f:
            return f
        case Success(exp_avg_tensor):
            pass

    # Convert exp_avg_sq
    match TensorStateConverter.from_proto(param_proto.exp_avg_sq):
        case Failure() as f:
            return f
        case Success(exp_avg_sq_tensor):
            pass

    # Build param state
    match AdamParamState.from_torch(
        {
            "step": param_proto.step,
            "exp_avg": exp_avg_tensor,
            "exp_avg_sq": exp_avg_sq_tensor,
        }
    ):
        case Success(param_state):
            return Success((param_id, param_state))
        case Failure() as f:
            return f


class AdamOptimizerStateConverter:
    """Convert AdamOptimizerState to/from AdamOptimizerStateProto."""

    @staticmethod
    def to_proto(
        optimizer_state: AdamOptimizerState,
    ) -> Result[tensors_pb2.AdamOptimizerStateProto, SerializationError | TorchFacadeError]:
        """
        Serialize AdamOptimizerState to protobuf.

        Args:
            optimizer_state: AdamOptimizerState instance

        Returns:
            AdamOptimizerStateProto message
        """
        proto = tensors_pb2.AdamOptimizerStateProto()

        # Pure: convert all param states to proto entries via comprehension
        param_entries_results: list[
            Result[
                tuple[int, tensors_pb2.AdamParamStateProto], SerializationError | TorchFacadeError
            ]
        ] = [
            _convert_param_state_to_proto(param_id, param_state)
            for param_id, param_state in optimizer_state.param_states.items()
        ]

        # Check for first failure and return it early
        first_failure = next(
            (result for result in param_entries_results if isinstance(result, Failure)), None
        )
        match first_failure:
            case None:
                pass  # No failures, continue
            case failure:
                return failure

        # All conversions succeeded - populate proto (mutation required by protobuf API)
        # Use comprehension for iteration (CopyFrom mutation unavoidable with protobuf)
        _ = [
            proto.state[param_id].CopyFrom(entry_proto)
            for result in param_entries_results
            if isinstance(result, Success)
            for param_id, entry_proto in [result.value]  # Unpack tuple from Success
        ]

        proto.param_groups.extend(
            [
                tensors_pb2.AdamParamGroupProto(
                    lr=group.lr,
                    beta1=group.betas[0],
                    beta2=group.betas[1],
                    eps=group.eps,
                    weight_decay=group.weight_decay,
                    amsgrad=group.amsgrad,
                )
                for group in optimizer_state.param_groups
            ]
        )

        return Success(proto)

    @staticmethod
    def from_proto(
        proto: tensors_pb2.AdamOptimizerStateProto,
    ) -> Result[AdamOptimizerState, SerializationError | TorchFacadeError]:
        """
        Deserialize AdamOptimizerStateProto to AdamOptimizerState.

        Args:
            proto: AdamOptimizerStateProto message

        Returns:
            Result containing AdamOptimizerState instance or serialization error
        """
        # Pure: convert all param states from proto via comprehension
        param_state_results: list[
            Result[tuple[int, AdamParamState], SerializationError | TorchFacadeError]
        ] = [
            _convert_param_state_from_proto(param_id, param_proto)
            for param_id, param_proto in proto.state.items()
        ]

        # Check for first failure
        first_failure = next(
            (result for result in param_state_results if isinstance(result, Failure)), None
        )
        match first_failure:
            case None:
                pass  # No failures, continue
            case failure:
                return failure

        # All conversions succeeded - build dict
        param_states = {
            param_id: param_state
            for result in param_state_results
            if isinstance(result, Success)
            for param_id, param_state in [result.value]  # Unpack tuple from Success
        }

        param_groups: list[AdamParamGroup] = [
            AdamParamGroup(
                params=[],  # Not serialized in proto
                lr=group_proto.lr,
                betas=(group_proto.beta1, group_proto.beta2),
                eps=group_proto.eps,
                weight_decay=group_proto.weight_decay,
                amsgrad=group_proto.amsgrad,
            )
            for group_proto in proto.param_groups
        ]

        return Success(AdamOptimizerState(param_states=param_states, param_groups=param_groups))


class RNGStateConverter:
    """Convert RNG state to/from RNGStateProto."""

    @staticmethod
    def to_proto(
        torch_cpu_rng_state: bytes, torch_cuda_rng_states: list[bytes]
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
    def from_proto(proto: tensors_pb2.RNGStateProto) -> tuple[bytes, list[bytes]]:
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
        model_state_dict: dict[str, torch.Tensor],
        optimizer_state: AdamOptimizerState,
        torch_cpu_rng_state: bytes,
        torch_cuda_rng_states: list[bytes],
        global_step: int,
    ) -> Result[tensors_pb2.ModelCheckpointProto, SerializationError | TorchFacadeError]:
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

        # Convert model state dict with error propagation via comprehension
        # Build list of (name, Result[proto, Error]) tuples
        tensor_conversion_results = [
            (name, TensorStateConverter.to_proto(tensor))
            for name, tensor in model_state_dict.items()
        ]

        # Check for first failure using next() with generator
        first_failure_tuple = next(
            (
                (name, result)
                for name, result in tensor_conversion_results
                if isinstance(result, Failure)
            ),
            None,
        )
        match first_failure_tuple:
            case None:
                pass  # No failures, continue
            case (_, failure_result):
                return Failure(failure_result.error)

        # Populate proto (mutation required by protobuf API)
        _ = [
            proto.model_state_dict[name].CopyFrom(result.value)
            for name, result in tensor_conversion_results
            if isinstance(result, Success)
        ]

        opt_result = AdamOptimizerStateConverter.to_proto(optimizer_state)
        match opt_result:
            case Failure():
                return opt_result
            case Success(optimizer_proto):
                proto.optimizer_state.CopyFrom(optimizer_proto)

        # Serialize RNG state
        proto.rng_state.CopyFrom(
            RNGStateConverter.to_proto(torch_cpu_rng_state, torch_cuda_rng_states)
        )

        # Global step
        proto.global_step = global_step

        return Success(proto)

    @staticmethod
    def from_proto(
        proto: tensors_pb2.ModelCheckpointProto,
    ) -> Result[
        tuple[dict[str, torch.Tensor], AdamOptimizerState, bytes, list[bytes], int],
        SerializationError | TorchFacadeError,
    ]:
        """
        Deserialize ModelCheckpointProto to checkpoint components.

        Args:
            proto: ModelCheckpointProto message

        Returns:
            Result containing tuple of (model_state_dict, optimizer_state, cpu_rng, cuda_rngs, global_step)
        """
        # Convert tensors from proto via comprehension
        # Build list of (name, Result[tensor, Error]) tuples
        tensor_conversion_results = [
            (name, TensorStateConverter.from_proto(tensor_proto))
            for name, tensor_proto in proto.model_state_dict.items()
        ]

        # Check for first failure using next() with generator
        first_failure_tuple = next(
            (
                (name, result)
                for name, result in tensor_conversion_results
                if isinstance(result, Failure)
            ),
            None,
        )
        match first_failure_tuple:
            case None:
                pass  # No failures, continue
            case (_, failure_result):
                return Failure(failure_result.error)

        # Build dict from successful results
        model_state_dict: dict[str, torch.Tensor] = {
            name: result.value
            for name, result in tensor_conversion_results
            if isinstance(result, Success)
        }

        match AdamOptimizerStateConverter.from_proto(proto.optimizer_state):
            case Failure(error):
                return Failure(error)
            case Success(optimizer_state):
                pass

        # Deserialize RNG state
        cpu_rng, cuda_rngs = RNGStateConverter.from_proto(proto.rng_state)

        # Global step
        global_step = int(proto.global_step)

        return Success((model_state_dict, optimizer_state, cpu_rng, cuda_rngs, global_step))
