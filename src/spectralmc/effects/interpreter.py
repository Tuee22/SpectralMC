"""
Effect Interpreter Protocol and implementations.

The interpreter is the ONLY place where side effects are executed.
All other code produces pure effect descriptions.

Type Safety:
    - Protocol defines the interpreter interface
    - assert_never ensures exhaustive pattern matching
    - Result types make error handling explicit

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

import asyncio
import hashlib
import pickle
from dataclasses import dataclass
from typing import Literal, Never, Protocol, Sequence, TypeVar

import cupy as cp
from cupy import stack as cupy_stack
import numpy as np
from numba import cuda
from numba.cuda import synchronize as numba_sync

import torch
from torch.utils.tensorboard import SummaryWriter
from spectralmc.runtime import get_torch_handle

from spectralmc.effects.composition import EffectParallel, EffectSequence
from spectralmc.effects.errors import (
    EffectError,
    GPUError,
    LoggingError,
    MetadataError,
    MonteCarloError,
    RNGError,
    StorageError,
    TrainingError,
)
from spectralmc.effects.gpu import (
    DLPackTransfer,
    GPUEffect,
    KernelLaunch,
    SplitInputs,
    StreamSync,
    TensorTransfer,
)
from spectralmc.effects.logging import LogMessage, LoggingInterpreter
from spectralmc.effects.metadata import MetadataEffect, ReadMetadata, UpdateMetadata
from spectralmc.effects.montecarlo import (
    ComputeFFT,
    ComputeMeanFFT,
    ForwardNormalization,
    GenerateNormals,
    MonteCarloEffect,
    PathScheme,
    ProcessBatch,
    SampleContracts,
    SimulatePaths,
    StackTensors,
)
from spectralmc.effects.registry import SharedRegistry, SupportsContracts
from spectralmc.effects.rng import CaptureRNGState, RestoreRNGState, RNGEffect
from spectralmc.effects.storage import (
    CommitCheckpoint,
    CommitVersion,
    ReadObject,
    StorageEffect,
    WriteObject,
)
from spectralmc.effects.training import (
    BackwardPass,
    ComputeComplexLoss,
    ComputeGradNorm,
    ComputeLoss,
    ForwardPass,
    ForwardPassComplex,
    LogMetrics,
    OptimizerStep,
    TrainingEffect,
    UpdateLearningRate,
    ZeroGrad,
)
from spectralmc.effects.types import Effect
from spectralmc.gbm import BlackScholes, SimulateBlackScholes
from spectralmc.models.cpu_gpu_transfer import (
    OutputPinning,
    TransferDestination,
    move_tensor_tree,
    plan_tensor_transfer,
)
from spectralmc.models.torch import Device
from spectralmc.result import Failure, Result, Success
from spectralmc.storage.s3_operations import S3Operations
from spectralmc.storage.store import AsyncBlockchainModelStore

get_torch_handle()

T = TypeVar("T")


@dataclass(frozen=True)
class CudaRngEnvAvailable:
    device_count: int


@dataclass(frozen=True)
class CudaRngEnvMissing:
    reason: str


CudaRngEnv = CudaRngEnvAvailable | CudaRngEnvMissing


def assert_never(value: Never) -> Never:
    """Type-safe exhaustiveness check for pattern matching.

    Use this in the default case of match statements to ensure
    all variants are handled. If a new variant is added but not
    handled, mypy will report an error.

    Example:
        >>> match effect:
        ...     case TensorTransfer(): ...
        ...     case StreamSync(): ...
        ...     case _:
        ...         assert_never(effect)  # mypy error if variants missing
    """
    raise AssertionError(f"Unhandled case: {value!r}")


def _cuda_rng_env() -> CudaRngEnv:
    """Pure CUDA RNG environment probe."""
    match torch.cuda.is_available():
        case False:
            return CudaRngEnvMissing(reason="cuda_unavailable")
        case True:
            count = torch.cuda.device_count()
            return (
                CudaRngEnvAvailable(device_count=count)
                if count > 0
                else CudaRngEnvMissing(reason="no_cuda_devices")
            )


class EffectInterpreter(Protocol):
    """Protocol for effect interpretation.

    Each interpreter handles a specific category of effects and
    returns a Result type for explicit error handling.
    """

    async def interpret(self, effect: Effect) -> Result[object, EffectError]:
        """Execute a single effect and return the result."""
        ...


class GPUInterpreter:
    """Interpreter for GPU effects.

    Handles tensor transfers, stream synchronization, and kernel launches.
    Coordinates between PyTorch, CuPy, and Numba CUDA streams.
    Uses SharedRegistry for tensor and kernel storage.
    """

    def __init__(
        self,
        torch_stream: torch.cuda.Stream,
        cupy_stream: cp.cuda.Stream,
        registry: SharedRegistry,
    ) -> None:
        """Initialize GPU interpreter with CUDA streams.

        Args:
            torch_stream: PyTorch CUDA stream for PyTorch operations.
            cupy_stream: CuPy CUDA stream for CuPy operations.
            registry: Shared registry for tensor and kernel storage.
        """
        self._torch_stream = torch_stream
        self._cupy_stream = cupy_stream
        self._registry = registry

    async def interpret(self, effect: GPUEffect) -> Result[object, GPUError]:
        """Execute GPU effect with stream coordination."""
        match effect:
            case TensorTransfer() as transfer:
                return await self._transfer_tensor(transfer)
            case StreamSync(stream_type=st):
                return await self._sync_stream(st)
            case KernelLaunch(kernel_name=name, grid_config=grid, block_config=block):
                return await self._launch_kernel(name, grid, block)
            case DLPackTransfer():
                return await self._dlpack_transfer(effect)
            case SplitInputs():
                return await self._split_inputs(effect)
            case _:
                assert_never(effect)

    async def _transfer_tensor(self, transfer: TensorTransfer) -> Result[object, GPUError]:
        """Transfer tensor between devices.

        Uses spectralmc.models.cpu_gpu_transfer for actual transfer.
        """

        tensor_result = self._registry.get_torch_tensor(transfer.tensor_id)
        match tensor_result:
            case Failure(_):
                return Failure(GPUError(message=f"Tensor not found: {transfer.tensor_id}"))
            case Success(tensor):
                pass

        # Determine transfer destination
        if transfer.target_device == Device.cpu:
            dest = (
                TransferDestination.CPU_PINNED
                if transfer.output_pinning is OutputPinning.PINNED
                else TransferDestination.CPU
            )
        else:
            dest = TransferDestination.CUDA

        if isinstance(tensor, torch.Tensor):
            plan_result = plan_tensor_transfer(
                tensor,
                dest=dest,
                stage_policy=transfer.stage_policy,
            )
            if isinstance(plan_result, Success):
                plan = plan_result.value
                match self._registry.register_metadata(
                    f"transfer_plan:{transfer.tensor_id}", repr(plan)
                ):
                    case Failure(err):
                        return Failure(GPUError(message=f"Registry metadata error: {err}"))
                    case Success(_):
                        pass
            else:
                match self._registry.register_metadata(
                    f"transfer_plan:{transfer.tensor_id}", f"error:{plan_result.error}"
                ):
                    case Failure(err):
                        return Failure(GPUError(message=f"Registry metadata error: {err}"))
                    case Success(_):
                        pass

        match move_tensor_tree(tensor, dest=dest, stage_policy=transfer.stage_policy):
            case Failure(error):
                return Failure(GPUError(message=str(error)))
            case Success(moved_tensor):
                match self._registry.register_tensor(transfer.tensor_id, moved_tensor):
                    case Failure(err):
                        return Failure(GPUError(message=f"Registry tensor error: {err}"))
                    case Success(_):
                        return Success(moved_tensor)

    async def _sync_stream(
        self, stream_type: Literal["torch", "cupy", "numba"]
    ) -> Result[object, GPUError]:
        """Synchronize the specified CUDA stream."""
        try:
            match stream_type:
                case "torch":
                    self._torch_stream.synchronize()
                case "cupy":
                    self._cupy_stream.synchronize()
                case "numba":
                    # Numba synchronize via cuda module
                    numba_sync()
                case _:
                    assert_never(stream_type)
            return Success(None)
        except RuntimeError as e:
            return Failure(GPUError(message=str(e)))

    async def _launch_kernel(
        self,
        name: str,
        grid: tuple[int, ...],
        block: tuple[int, ...],
    ) -> Result[object, GPUError]:
        """Launch a registered CUDA kernel.

        Supports Numba CUDA kernels registered via register_kernel().
        The kernel must be pre-configured with its argument tensors.

        Args:
            name: Registered kernel name.
            grid: CUDA grid configuration (blocks).
            block: CUDA block configuration (threads per block).

        Returns:
            Result containing None on success or GPUError on failure.
        """
        kernel_result = self._registry.get_kernel(name)
        match kernel_result:
            case Failure(_):
                return Failure(GPUError(message=f"Kernel not found: {name}"))
            case Success(kernel_fn):
                pass

        try:
            # For Numba CUDA kernels, use [] indexing syntax
            # The registered kernel should be a configured callable
            if callable(kernel_fn):
                # Launch with grid/block configuration
                # For Numba, this requires kernel[grid, block](*args) syntax
                # Since args are bound at registration, we launch with empty args
                # This is a simplified implementation - real usage would need
                # a more sophisticated argument passing mechanism
                kernel_fn()
            return Success(None)
        except RuntimeError as e:
            return Failure(GPUError(message=str(e)))

    async def _dlpack_transfer(
        self,
        effect: DLPackTransfer,
    ) -> Result[object, GPUError]:
        """Transfer tensor between frameworks via DLPack protocol.

        Enables zero-copy tensor sharing between CuPy and PyTorch on GPU.

        Args:
            effect: DLPackTransfer effect with source/target framework info.

        Returns:
            Result containing transferred tensor or GPUError.
        """
        tensor_result = self._registry.get_tensor(effect.source_tensor_id)
        match tensor_result:
            case Failure(_):
                return Failure(GPUError(message=f"Tensor not found: {effect.source_tensor_id}"))
            case Success(tensor):
                pass

        try:
            transferred: object
            if effect.source_framework == "cupy" and effect.target_framework == "torch":
                if not isinstance(tensor, cp.ndarray):
                    return Failure(GPUError(message="Source tensor is not a CuPy array"))
                # CuPy to PyTorch via DLPack
                transferred = torch.from_dlpack(tensor)
            elif effect.source_framework == "torch" and effect.target_framework == "cupy":
                if not isinstance(tensor, torch.Tensor):
                    return Failure(GPUError(message="Source tensor is not a PyTorch tensor"))
                # PyTorch to CuPy via DLPack - use asarray which supports __dlpack__
                transferred = cp.asarray(tensor)
            else:
                return Failure(
                    GPUError(
                        message=f"Unsupported transfer: {effect.source_framework} -> {effect.target_framework}"
                    )
                )

            # Store result in shared registry
            match self._registry.register_tensor(effect.output_tensor_id, transferred):
                case Failure(err):
                    return Failure(GPUError(message=f"Registry tensor error: {err}"))
                case Success(_):
                    return Success(transferred)
        except (RuntimeError, TypeError) as e:
            return Failure(GPUError(message=str(e)))

    async def _split_inputs(self, effect: SplitInputs) -> Result[object, GPUError]:
        """Split stored contracts into real/imaginary tensors."""
        contracts_result = self._registry.get_contracts(effect.contracts_tensor_id)
        match contracts_result:
            case Failure(err):
                return Failure(GPUError(message=f"Contracts not found: {err}"))
            case Success(contracts):
                pass

        try:
            rows = [
                [
                    float(contract.X0),
                    float(contract.K),
                    float(contract.T),
                    float(contract.r),
                    float(contract.d),
                    float(contract.v),
                ]
                for contract in contracts
            ]
            real = torch.tensor(rows, device="cuda", dtype=torch.float32)
            imag = torch.zeros_like(real)

            match self._registry.register_tensor(effect.real_output_tensor_id, real):
                case Failure(reg_err):
                    return Failure(GPUError(message=f"Registry tensor error: {reg_err}"))
                case Success(_):
                    pass

            match self._registry.register_tensor(effect.imag_output_tensor_id, imag):
                case Failure(reg_err):
                    return Failure(GPUError(message=f"Registry tensor error: {reg_err}"))
                case Success(_):
                    return Success((real, imag))
        except Exception as exc:  # noqa: BLE001
            return Failure(GPUError(message=str(exc)))


class TrainingInterpreter:
    """Interpreter for training effects.

    Handles forward/backward passes and optimizer steps.
    Uses SharedRegistry for tensor, model, and optimizer storage.
    """

    def __init__(self, registry: SharedRegistry) -> None:
        """Initialize training interpreter.

        Args:
            registry: Shared registry for tensor, model, and optimizer storage.
        """
        self._registry = registry

    async def interpret(self, effect: TrainingEffect) -> Result[object, TrainingError]:
        """Execute training effect."""
        match effect:
            case ForwardPass():
                return await self._forward_pass(effect)
            case ForwardPassComplex():
                return await self._forward_pass_complex(effect)
            case BackwardPass(loss_tensor_id=tid):
                return await self._backward_pass(tid)
            case OptimizerStep(optimizer_id=oid):
                return await self._optimizer_step(oid)
            case ComputeLoss():
                return await self._compute_loss(effect)
            case ComputeComplexLoss():
                return await self._compute_complex_loss(effect)
            case ZeroGrad():
                return await self._zero_grad(effect)
            case ComputeGradNorm():
                return await self._compute_grad_norm(effect)
            case UpdateLearningRate():
                return await self._update_learning_rate(effect)
            case LogMetrics():
                return await self._log_metrics(effect)
            case _:
                assert_never(effect)

    async def _forward_pass(
        self,
        effect: ForwardPass,
    ) -> Result[object, TrainingError]:
        """Execute forward pass through model.

        Args:
            effect: ForwardPass effect with model_id, input_tensor_id, output_tensor_id.

        Returns:
            Result containing output tensor or TrainingError.
        """
        model_result = self._registry.get_model(effect.model_id)
        match model_result:
            case Failure(_):
                return Failure(TrainingError(message=f"Model not found: {effect.model_id}"))
            case Success(model):
                pass

        tensor_result = self._registry.get_tensor(effect.input_tensor_id)
        match tensor_result:
            case Failure(_):
                return Failure(TrainingError(message=f"Tensor not found: {effect.input_tensor_id}"))
            case Success(tensor):
                pass

        try:
            # Type-safe model call requires proper protocol
            # Use callable() for reliable type checking
            if callable(model):
                output = model(tensor)
                # Store output in shared registry with specified ID
                match self._registry.register_tensor(effect.output_tensor_id, output):
                    case Failure(err):
                        return Failure(TrainingError(message=f"Registry tensor error: {err}"))
                    case Success(_):
                        return Success(output)
            return Failure(TrainingError(message=f"Model {effect.model_id} is not callable"))
        except RuntimeError as e:
            return Failure(TrainingError(message=str(e)))

    async def _forward_pass_complex(
        self,
        effect: ForwardPassComplex,
    ) -> Result[object, TrainingError]:
        """Execute complex-valued forward pass."""
        model_result = self._registry.get_model(effect.model_id)
        match model_result:
            case Failure(_):
                return Failure(TrainingError(message=f"Model not found: {effect.model_id}"))
            case Success(model):
                pass

        real_result = self._registry.get_torch_tensor(effect.real_input_tensor_id)
        match real_result:
            case Failure(_):
                return Failure(
                    TrainingError(message=f"Tensor not found: {effect.real_input_tensor_id}")
                )
            case Success(real_input):
                pass

        imag_result = self._registry.get_torch_tensor(effect.imag_input_tensor_id)
        match imag_result:
            case Failure(_):
                return Failure(
                    TrainingError(message=f"Tensor not found: {effect.imag_input_tensor_id}")
                )
            case Success(imag_input):
                pass

        try:
            if callable(model):
                outputs = model(real_input, imag_input)
            else:
                return Failure(TrainingError(message=f"Model {effect.model_id} is not callable"))

            match outputs:
                case (pred_real, pred_imag):
                    pass
                case _:
                    return Failure(
                        TrainingError(
                            message=f"Model {effect.model_id} did not return complex outputs"
                        )
                    )

            for tensor_id, tensor in (
                (effect.real_output_tensor_id, pred_real),
                (effect.imag_output_tensor_id, pred_imag),
            ):
                match self._registry.register_tensor(tensor_id, tensor):
                    case Failure(err):
                        return Failure(
                            TrainingError(message=f"Registry tensor error: {err}", step=None)
                        )
                    case Success(_):
                        pass

            return Success((pred_real, pred_imag))
        except RuntimeError as exc:
            return Failure(TrainingError(message=str(exc)))

    async def _backward_pass(self, loss_tensor_id: str) -> Result[object, TrainingError]:
        """Compute gradients via backpropagation."""
        tensor_result = self._registry.get_tensor(loss_tensor_id)
        match tensor_result:
            case Failure(_):
                return Failure(TrainingError(message=f"Loss tensor not found: {loss_tensor_id}"))
            case Success(tensor):
                pass

        try:
            # PyTorch backward requires tensor with grad
            if hasattr(tensor, "backward"):
                backward_fn = tensor.backward
                backward_fn()
            return Success(None)
        except RuntimeError as e:
            return Failure(TrainingError(message=str(e)))

    async def _optimizer_step(self, optimizer_id: str) -> Result[object, TrainingError]:
        """Update model parameters using optimizer (zero_grad handled separately)."""
        optimizer_result = self._registry.get_optimizer(optimizer_id)
        match optimizer_result:
            case Failure(_):
                return Failure(TrainingError(message=f"Optimizer not found: {optimizer_id}"))
            case Success(optimizer):
                pass

        try:
            if hasattr(optimizer, "step"):
                step_fn = optimizer.step
                step_fn()
            return Success(None)
        except RuntimeError as e:
            return Failure(TrainingError(message=str(e)))

    async def _compute_loss(self, effect: ComputeLoss) -> Result[object, TrainingError]:
        """Compute loss between predictions and targets.

        Args:
            effect: ComputeLoss effect with tensor IDs and loss type.

        Returns:
            Result containing loss tensor or TrainingError.
        """
        pred_result = self._registry.get_torch_tensor(effect.pred_tensor_id)
        match pred_result:
            case Failure(_):
                return Failure(
                    TrainingError(message=f"Prediction tensor not found: {effect.pred_tensor_id}")
                )
            case Success(pred):
                pass

        target_result = self._registry.get_torch_tensor(effect.target_tensor_id)
        match target_result:
            case Failure(_):
                return Failure(
                    TrainingError(message=f"Target tensor not found: {effect.target_tensor_id}")
                )
            case Success(target):
                pass

        try:
            loss: torch.Tensor
            loss_kind = str(effect.loss_type)
            match loss_kind:
                case "mse":
                    loss = torch.nn.functional.mse_loss(pred, target)
                case "mae":
                    loss = torch.nn.functional.l1_loss(pred, target)
                case "huber":
                    loss = torch.nn.functional.smooth_l1_loss(pred, target)
                case other:
                    return Failure(TrainingError(message=f"unsupported_loss_type:{other}"))

            # Store loss in shared registry for BackwardPass
            match self._registry.register_tensor(effect.output_tensor_id, loss):
                case Failure(err):
                    return Failure(TrainingError(message=f"Registry tensor error: {err}"))
                case Success(_):
                    return Success(loss)
        except RuntimeError as e:
            return Failure(TrainingError(message=str(e)))

    async def _compute_complex_loss(
        self, effect: ComputeComplexLoss
    ) -> Result[object, TrainingError]:
        """Compute loss for complex-valued predictions."""
        pred_real_result = self._registry.get_torch_tensor(effect.pred_real_tensor_id)
        match pred_real_result:
            case Failure(_):
                return Failure(
                    TrainingError(
                        message=f"Prediction tensor not found: {effect.pred_real_tensor_id}"
                    )
                )
            case Success(pred_real):
                pass

        pred_imag_result = self._registry.get_torch_tensor(effect.pred_imag_tensor_id)
        match pred_imag_result:
            case Failure(_):
                return Failure(
                    TrainingError(
                        message=f"Prediction tensor not found: {effect.pred_imag_tensor_id}"
                    )
                )
            case Success(pred_imag):
                pass

        target_result = self._registry.get_torch_tensor(effect.target_tensor_id)
        match target_result:
            case Failure(_):
                return Failure(
                    TrainingError(message=f"Target tensor not found: {effect.target_tensor_id}")
                )
            case Success(target):
                target_real = torch.real(target)

        try:
            zero_like_imag = torch.zeros_like(pred_imag)
            loss_kind = str(effect.loss_type)
            match loss_kind:
                case "mse":
                    loss_real = torch.nn.functional.mse_loss(pred_real, target_real)
                    loss_imag = torch.nn.functional.mse_loss(pred_imag, zero_like_imag)
                case "mae":
                    loss_real = torch.nn.functional.l1_loss(pred_real, target_real)
                    loss_imag = torch.nn.functional.l1_loss(pred_imag, zero_like_imag)
                case "huber":
                    loss_real = torch.nn.functional.smooth_l1_loss(pred_real, target_real)
                    loss_imag = torch.nn.functional.smooth_l1_loss(pred_imag, zero_like_imag)
                case other:
                    return Failure(TrainingError(message=f"unsupported_loss_type:{other}"))

            loss = loss_real + loss_imag
            match self._registry.register_tensor(effect.output_tensor_id, loss):
                case Failure(err):
                    return Failure(TrainingError(message=f"Registry tensor error: {err}"))
                case Success(_):
                    return Success(loss)
        except RuntimeError as exc:
            return Failure(TrainingError(message=str(exc)))

    async def _zero_grad(self, effect: ZeroGrad) -> Result[object, TrainingError]:
        """Explicit optimizer.zero_grad()."""
        optimizer_result = self._registry.get_optimizer(effect.optimizer_id)
        match optimizer_result:
            case Failure(_):
                return Failure(TrainingError(message=f"Optimizer not found: {effect.optimizer_id}"))
            case Success(optimizer):
                pass

        try:
            if hasattr(optimizer, "zero_grad"):
                optimizer.zero_grad(set_to_none=effect.set_to_none)
            return Success(None)
        except RuntimeError as exc:
            return Failure(TrainingError(message=str(exc)))

    async def _compute_grad_norm(self, effect: ComputeGradNorm) -> Result[object, TrainingError]:
        """Compute gradient norm for monitoring."""
        model_result = self._registry.get_model(effect.model_id)
        match model_result:
            case Failure(_):
                return Failure(TrainingError(message=f"Model not found: {effect.model_id}"))
            case Success(model):
                pass

        try:
            parameters = list(model.parameters()) if hasattr(model, "parameters") else []
            if not parameters:
                return Failure(TrainingError(message="Model has no parameters"))

            grad_norm_result: torch.Tensor | float = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=effect.max_norm
            )
            param_device = parameters[0].device
            param_dtype = parameters[0].dtype
            if isinstance(grad_norm_result, torch.Tensor):
                grad_tensor = grad_norm_result.to(device=param_device, dtype=param_dtype).detach()
            else:
                grad_tensor = torch.tensor(grad_norm_result, device=param_device, dtype=param_dtype)
            match self._registry.register_tensor(effect.output_tensor_id, grad_tensor):
                case Failure(err):
                    return Failure(TrainingError(message=f"Registry tensor error: {err}"))
                case Success(_):
                    return Success(grad_tensor)
        except RuntimeError as exc:
            return Failure(TrainingError(message=str(exc)))

    async def _update_learning_rate(
        self, effect: UpdateLearningRate
    ) -> Result[object, TrainingError]:
        """Update optimizer learning rate."""
        optimizer_result = self._registry.get_optimizer(effect.optimizer_id)
        match optimizer_result:
            case Failure(_):
                return Failure(TrainingError(message=f"Optimizer not found: {effect.optimizer_id}"))
            case Success(optimizer):
                pass

        try:
            groups = getattr(optimizer, "param_groups", None)
            if groups is None:
                return Failure(TrainingError(message="optimizer_missing_param_groups"))

            for group in groups:
                group["lr"] = effect.lr
            return Success(effect.lr)
        except Exception as exc:  # noqa: BLE001
            return Failure(TrainingError(message=str(exc)))

    async def _log_metrics(self, effect: LogMetrics) -> Result[object, TrainingError]:
        """Log metrics to TensorBoard.

        Args:
            effect: LogMetrics effect with metrics and step.

        Returns:
            Result containing None or TrainingError.
        """
        try:
            # Check if a writer is registered in shared registry
            writer_result = self._registry.get_tensor("_tensorboard_writer")
            match writer_result:
                case Success(writer) if isinstance(writer, SummaryWriter):
                    for metric_name, metric_value in effect.metrics:
                        resolved_value: float
                        match metric_value:
                            case str() as placeholder if placeholder.startswith(
                                "{"
                            ) and placeholder.endswith("}"):
                                tensor_id = placeholder.strip("{}")
                                metric_tensor_result = self._registry.get_tensor(tensor_id)
                                match metric_tensor_result:
                                    case Failure(err):
                                        return Failure(
                                            TrainingError(message=f"metric_tensor_missing:{err}")
                                        )
                                    case Success(metric_tensor):
                                        if hasattr(metric_tensor, "item"):
                                            resolved_value = float(metric_tensor.item())
                                        else:
                                            return Failure(
                                                TrainingError(
                                                    message=f"metric_tensor_invalid:{tensor_id}"
                                                )
                                            )
                            case _:
                                resolved_value = float(metric_value)

                        writer.add_scalar(metric_name, resolved_value, effect.step)
                case Success(_):
                    return Failure(TrainingError(message="tensorboard_writer_invalid"))
                case Failure(_):
                    return Success(None)

            return Success(None)
        except (ImportError, RuntimeError) as exc:
            return Failure(TrainingError(message=f"tensorboard_logging_failed:{exc}"))


class MonteCarloInterpreter:
    """Interpreter for Monte Carlo effects.

    Handles random number generation, path simulation, and FFT computation.
    Uses SharedRegistry for tensor storage to enable data flow between effects.
    """

    def __init__(self, registry: SharedRegistry, mc_engine: BlackScholes | None = None) -> None:
        """Initialize Monte Carlo interpreter.

        Args:
            registry: Shared registry for tensor storage across interpreters.
            mc_engine: Optional BlackScholes engine for contract pricing.
        """
        self._registry = registry
        self._mc_engine = mc_engine

    async def interpret(self, effect: MonteCarloEffect) -> Result[object, MonteCarloError]:
        """Execute Monte Carlo effect."""
        match effect:
            case GenerateNormals():
                return await self._generate_normals(effect)
            case SimulatePaths():
                return await self._simulate_paths(effect)
            case ComputeFFT():
                return await self._compute_fft(effect)
            case SampleContracts():
                return await self._sample_contracts(effect)
            case ProcessBatch():
                return await self._process_batch(effect)
            case StackTensors():
                return await self._stack_tensors(
                    effect.input_tensor_ids, effect.dim, effect.output_tensor_id
                )
            case ComputeMeanFFT():
                return await self._compute_mean_fft(effect)
            case _:
                assert_never(effect)

    async def _generate_normals(
        self,
        effect: GenerateNormals,
    ) -> Result[object, MonteCarloError]:
        """Generate standard normal random matrix on GPU.

        Args:
            effect: GenerateNormals effect with dimensions, seed, and output_tensor_id.

        Returns:
            Result containing the generated matrix or MonteCarloError.
        """
        try:
            # Use CuPy's random generator with seed
            rng = cp.random.default_rng(effect.seed)
            # Skip values if resuming (use tuple shape for skip)
            if effect.skip > 0:
                _ = rng.standard_normal((effect.skip,))
            # Generate the matrix
            matrix = rng.standard_normal((effect.rows, effect.cols), dtype=cp.float32)

            # Store in shared registry with specified output ID
            match self._registry.register_tensor(effect.output_tensor_id, matrix):
                case Failure(err):
                    return Failure(MonteCarloError(message=f"Registry tensor error: {err}"))
                case Success(_):
                    return Success(matrix)
        except RuntimeError as e:
            return Failure(MonteCarloError(message=str(e)))

    async def _simulate_paths(
        self,
        effect: SimulatePaths,
    ) -> Result[object, MonteCarloError]:
        """Simulate GBM price paths using Numba CUDA kernel.

        Executes the SimulateBlackScholes kernel to generate price paths
        from pre-generated normal random numbers.

        Args:
            effect: SimulatePaths effect with all simulation parameters.

        Returns:
            Result containing simulated paths tensor or MonteCarloError.
        """
        try:
            # Get input normals from shared registry
            normals_result = self._registry.get_cupy_array(effect.input_normals_id)
            match normals_result:
                case Failure(_):
                    return Failure(
                        MonteCarloError(
                            message=f"Normals tensor not found: {effect.input_normals_id}"
                        )
                    )
                case Success(normals):
                    pass

            # Make a copy since the kernel operates in-place
            sims: cp.ndarray = cp.array(normals)

            # Compute kernel launch parameters
            threads_per_block = 256
            total_paths: int = int(sims.shape[1])
            blocks = (total_paths + threads_per_block - 1) // threads_per_block

            # Compute timestep size
            dt = effect.expiry / effect.timesteps

            # Get the default CUDA stream for kernel launch
            stream = cuda.default_stream()

            match effect.path_scheme:
                case PathScheme.LOG_EULER:
                    simulate_log_return = True
                case PathScheme.SIMPLE_EULER:
                    simulate_log_return = False
                case _:
                    assert_never(effect.path_scheme)

            # Launch the SimulateBlackScholes kernel (blocks, threads, stream)
            SimulateBlackScholes[blocks, threads_per_block, stream](
                cuda.as_cuda_array(sims),
                effect.timesteps,
                dt,
                effect.spot,
                effect.rate,
                effect.dividend,
                effect.vol,
                simulate_log_return,
            )

            # Synchronize to ensure kernel completes
            cuda.synchronize()

            # Optional forward normalization
            match effect.normalization:
                case ForwardNormalization.NORMALIZE:
                    times = cp.linspace(dt, effect.expiry, effect.timesteps, dtype=sims.dtype)
                    forwards = effect.spot * cp.exp((effect.rate - effect.dividend) * times)
                    row_means = cp.mean(sims, axis=1, keepdims=True).squeeze()
                    sims *= cp.expand_dims(forwards / row_means, 1)
                case ForwardNormalization.RAW:
                    pass
                case _:
                    assert_never(effect.normalization)

            # Store result in shared registry with specified output ID
            match self._registry.register_tensor(effect.output_tensor_id, sims):
                case Failure(err):
                    return Failure(MonteCarloError(message=f"Registry tensor error: {err}"))
                case Success(_):
                    return Success(sims)
        except RuntimeError as e:
            return Failure(MonteCarloError(message=str(e)))

    async def _compute_fft(
        self,
        effect: ComputeFFT,
    ) -> Result[object, MonteCarloError]:
        """Compute FFT on tensor.

        Args:
            effect: ComputeFFT effect with input_tensor_id, axis, and output_tensor_id.

        Returns:
            Result containing the FFT result or MonteCarloError.
        """
        tensor_result = self._registry.get_cupy_array(effect.input_tensor_id)
        match tensor_result:
            case Failure(_):
                return Failure(
                    MonteCarloError(message=f"Tensor not found: {effect.input_tensor_id}")
                )
            case Success(tensor):
                pass

        try:
            # Compute FFT using CuPy
            result = cp.fft.fft(tensor, axis=effect.axis)

            # Store in shared registry with specified output ID
            match self._registry.register_tensor(effect.output_tensor_id, result):
                case Failure(err):
                    return Failure(MonteCarloError(message=f"Registry tensor error: {err}"))
                case Success(_):
                    return Success(result)
        except RuntimeError as e:
            return Failure(MonteCarloError(message=str(e)))

    async def _sample_contracts(self, effect: SampleContracts) -> Result[object, MonteCarloError]:
        """Sample contracts using a registered sampler."""
        sampler_result = self._registry.get_sampler(effect.sampler_id)
        match sampler_result:
            case Failure(reg_err):
                return Failure(MonteCarloError(message=f"Sampler not found: {reg_err}"))
            case Success(sampler):
                pass

        def _normalize_contracts(
            raw_contracts: Sequence[SupportsContracts] | Sequence[BlackScholes.Inputs],
        ) -> Result[tuple[BlackScholes.Inputs, ...], MonteCarloError]:
            normalized: list[BlackScholes.Inputs] = []
            for contract in raw_contracts:
                if not isinstance(contract, BlackScholes.Inputs):
                    return Failure(MonteCarloError(message="Invalid contract type in sampler"))
                normalized.append(contract)
            return Success(tuple(normalized))

        try:
            sampled = sampler.sample(effect.num_samples)
            match sampled:
                case Failure(err):
                    return Failure(MonteCarloError(message=str(err)))
                case Success(contracts):
                    norm_result = _normalize_contracts(contracts)
                case list_contracts if isinstance(list_contracts, list):
                    norm_result = _normalize_contracts(list_contracts)

            match norm_result:
                case Failure(norm_err):
                    return Failure(norm_err)
                case Success(normalized_contracts):
                    match self._registry.register_contracts(
                        effect.output_tensor_id, normalized_contracts
                    ):
                        case Failure(reg_err):
                            return Failure(
                                MonteCarloError(message=f"Registry contracts error: {reg_err}")
                            )
                        case Success(_):
                            return Success(normalized_contracts)
        except Exception as exc:  # noqa: BLE001
            return Failure(MonteCarloError(message=str(exc)))

    async def _process_batch(self, effect: ProcessBatch) -> Result[object, MonteCarloError]:
        """Generate FFT targets for a batch of contracts."""
        contracts_result = self._registry.get_contracts(effect.contracts_tensor_id)
        match contracts_result:
            case Failure(reg_err):
                return Failure(MonteCarloError(message=f"Contracts not found: {reg_err}"))
            case Success(contracts):
                pass

        if self._mc_engine is None:
            return Failure(MonteCarloError(message="mc_engine_missing"))

        fft_results: list[cp.ndarray] = []

        batches_per_run = int(effect.config.get("batches_per_mc_run", 1))
        network_size = int(effect.config.get("network_size", 1))

        try:
            for contract in contracts:
                if not isinstance(contract, BlackScholes.Inputs):
                    return Failure(
                        MonteCarloError(
                            message=f"Unsupported contract type: {type(contract).__name__}"
                        )
                    )
                price_result = self._mc_engine.price(inputs=contract)
                match price_result:
                    case Failure(mc_err):
                        return Failure(MonteCarloError(message=str(mc_err)))
                    case Success(pricing):
                        pass

                mat = pricing.put_price.reshape(batches_per_run, network_size)
                fft_val = cp.mean(cp.fft.fft(mat, axis=1), axis=0)
                fft_results.append(fft_val)

            stacked = cupy_stack(fft_results, axis=0)
            match self._registry.register_tensor(effect.output_tensor_id, stacked):
                case Failure(reg_err):
                    return Failure(MonteCarloError(message=f"Registry tensor error: {reg_err}"))
                case Success(_):
                    return Success(stacked)
        except Exception as exc:  # noqa: BLE001
            return Failure(MonteCarloError(message=str(exc)))

    async def _stack_tensors(
        self, input_tensor_ids: tuple[str, ...], dim: int, output_tensor_id: str
    ) -> Result[object, MonteCarloError]:
        """Stack multiple tensors or arrays."""
        try:
            tensors: list[object] = []
            for tensor_id in input_tensor_ids:
                tensor_result = self._registry.get_tensor(tensor_id)
                match tensor_result:
                    case Failure(err):
                        return Failure(MonteCarloError(message=f"Tensor not found: {err}"))
                    case Success(tensor):
                        tensors.append(tensor)

            if not tensors:
                empty = cp.array([])
                match self._registry.register_tensor(output_tensor_id, empty):
                    case Failure(err):
                        return Failure(MonteCarloError(message=f"Registry tensor error: {err}"))
                    case Success(_):
                        return Success(empty)

            first = tensors[0]
            if isinstance(first, cp.ndarray):
                cupy_tensors: list[cp.ndarray] = []
                for t in tensors:
                    if not isinstance(t, cp.ndarray):
                        return Failure(
                            MonteCarloError(message=f"Unsupported tensor type: {type(t).__name__}")
                        )
                    cupy_tensors.append(t)
                stacked_cu = cupy_stack(cupy_tensors, axis=dim)
                match self._registry.register_tensor(output_tensor_id, stacked_cu):
                    case Failure(err):
                        return Failure(MonteCarloError(message=f"Registry tensor error: {err}"))
                    case Success(_):
                        return Success(stacked_cu)
            elif isinstance(first, torch.Tensor):
                torch_tensors: list[torch.Tensor] = []
                for t in tensors:
                    if not isinstance(t, torch.Tensor):
                        return Failure(
                            MonteCarloError(message=f"Unsupported tensor type: {type(t).__name__}")
                        )
                    torch_tensors.append(t)
                stacked_torch = torch.stack(torch_tensors, dim=dim)
                match self._registry.register_tensor(output_tensor_id, stacked_torch):
                    case Failure(err):
                        return Failure(MonteCarloError(message=f"Registry tensor error: {err}"))
                    case Success(_):
                        return Success(stacked_torch)
            else:
                return Failure(
                    MonteCarloError(message=f"Unsupported tensor type: {type(first).__name__}")
                )
        except Exception as exc:  # noqa: BLE001
            return Failure(MonteCarloError(message=str(exc)))

    async def _compute_mean_fft(self, effect: ComputeMeanFFT) -> Result[object, MonteCarloError]:
        """Compute FFT with mean reduction."""
        tensor_result = self._registry.get_tensor(effect.input_tensor_id)
        match tensor_result:
            case Failure(_):
                return Failure(
                    MonteCarloError(message=f"Tensor not found: {effect.input_tensor_id}")
                )
            case Success(tensor):
                pass

        try:
            if isinstance(tensor, cp.ndarray):
                fft_val = cp.mean(cp.fft.fft(tensor, axis=1), axis=0)
            elif isinstance(tensor, torch.Tensor):
                fft_val = torch.fft.fft(tensor, dim=1).mean(dim=0)
            else:
                return Failure(
                    MonteCarloError(message=f"Unsupported tensor type: {type(tensor).__name__}")
                )

            match self._registry.register_tensor(effect.output_tensor_id, fft_val):
                case Failure(err):
                    return Failure(MonteCarloError(message=f"Registry tensor error: {err}"))
                case Success(_):
                    return Success(fft_val)
        except Exception as exc:  # noqa: BLE001
            return Failure(MonteCarloError(message=str(exc)))


class StorageInterpreter:
    """Interpreter for storage effects.

    Handles S3 reads, writes, and blockchain version commits.
    Uses SharedRegistry for bytes content storage.
    """

    def __init__(self, bucket: str, registry: SharedRegistry) -> None:
        """Initialize storage interpreter.

        Args:
            bucket: Default S3 bucket for operations.
            registry: Shared registry for bytes content storage.
        """
        self._bucket = bucket
        self._registry = registry

    async def interpret(self, effect: StorageEffect) -> Result[object, StorageError]:
        """Execute storage effect."""
        match effect:
            case ReadObject():
                return await self._read_object(effect)
            case WriteObject(bucket=b, key=k, content_hash=h):
                return await self._write_object(b or self._bucket, k, h)
            case CommitVersion(parent_counter=p, checkpoint_hash=h, message=m):
                return await self._commit_version(p, h, m)
            case CommitCheckpoint():
                return await self._commit_checkpoint(effect)
            case _:
                assert_never(effect)

    async def _read_object(self, effect: ReadObject) -> Result[object, StorageError]:
        """Read object from S3.

        Args:
            effect: ReadObject effect with bucket, key, and output_id.

        Returns:
            Result containing the read data or StorageError.
        """
        bucket = effect.bucket or self._bucket
        key = effect.key
        try:
            async with AsyncBlockchainModelStore(bucket) as store:
                # Use public API via S3Operations
                if store._s3_client is None:
                    return Failure(
                        StorageError(message="S3 client not initialized", bucket=bucket, key=key)
                    )
                ops = S3Operations(store._s3_client)
                result = await ops.get_object(bucket, key)
                match result:
                    case Success(data):
                        # Store in shared registry with specified output ID
                        if isinstance(data, bytes):
                            match self._registry.register_bytes(effect.output_id, data):
                                case Failure(err):
                                    return Failure(
                                        StorageError(
                                            message=f"Registry bytes error: {err}",
                                            bucket=bucket,
                                            key=key,
                                        )
                                    )
                                case Success(_):
                                    pass
                        return Success(data)
                    case Failure(err):
                        return Failure(StorageError(message=str(err), bucket=bucket, key=key))
        except Exception as e:
            return Failure(StorageError(message=str(e), bucket=bucket, key=key))

    async def _write_object(
        self,
        bucket: str,
        key: str,
        content_hash: str,
    ) -> Result[object, StorageError]:
        """Write object to S3."""
        # Get content from shared registry using the content hash as key
        content_result = self._registry.get_bytes(content_hash)
        match content_result:
            case Failure(_):
                return Failure(
                    StorageError(
                        message=f"Content not found for hash: {content_hash}",
                        bucket=bucket,
                        key=key,
                    )
                )
            case Success(content):
                pass

        try:
            async with AsyncBlockchainModelStore(bucket) as store:
                if store._s3_client is None:
                    return Failure(
                        StorageError(message="S3 client not initialized", bucket=bucket, key=key)
                    )
                ops = S3Operations(store._s3_client)
                result = await ops.put_object(bucket, key, content)
                match result:
                    case Success(_):
                        return Success(None)
                    case Failure(s3_err):
                        return Failure(StorageError(message=str(s3_err), bucket=bucket, key=key))
        except Exception as e:
            return Failure(StorageError(message=str(e), bucket=bucket, key=key))

    async def _commit_version(
        self,
        parent_counter: int | None,
        checkpoint_hash: str,
        message: str,
    ) -> Result[object, StorageError]:
        """Commit new model version."""
        # Get checkpoint data from shared registry
        checkpoint_result = self._registry.get_bytes(checkpoint_hash)
        match checkpoint_result:
            case Failure(_):
                return Failure(StorageError(message=f"Checkpoint not found: {checkpoint_hash}"))
            case Success(checkpoint_data):
                pass

        try:
            async with AsyncBlockchainModelStore(self._bucket) as store:
                # store.commit returns ModelVersion directly, not Result
                version = await store.commit(checkpoint_data, checkpoint_hash, message)
                return Success(version)
        except Exception as e:
            return Failure(StorageError(message=str(e)))

    async def _commit_checkpoint(self, effect: CommitCheckpoint) -> Result[object, StorageError]:
        """Conditionally commit checkpoint bytes to blockchain storage."""
        plan = effect.commit_plan

        def _parse_interval(plan_str: str) -> int | None:
            try:
                return int(plan_str.split(":", maxsplit=1)[1])
            except (IndexError, ValueError):
                return None

        should_commit = False
        if plan == "NoCommit":
            should_commit = False
        elif plan == "FinalCommit":
            should_commit = effect.current_step >= effect.total_steps - 1
        elif plan.startswith("IntervalCommit"):
            interval = _parse_interval(plan)
            should_commit = interval is not None and effect.current_step % interval == 0
        elif plan.startswith("FinalAndIntervalCommit"):
            interval = _parse_interval(plan)
            should_commit = effect.current_step % (interval or 1) == 0 or (
                effect.current_step >= effect.total_steps - 1
            )
        else:
            should_commit = False

        if not should_commit:
            return Success(None)

        checkpoint_result = self._registry.get_bytes(effect.checkpoint_id)
        match checkpoint_result:
            case Failure(err):
                return Failure(StorageError(message=f"Checkpoint not found: {err}"))
            case Success(checkpoint_bytes):
                pass

        store_result = self._registry.get_blockchain_store(effect.blockchain_store_id)
        match store_result:
            case Failure(err):
                return Failure(StorageError(message=f"Blockchain store not found: {err}"))
            case Success(store):
                pass

        try:
            content_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
            version = await store.commit(
                checkpoint_data=checkpoint_bytes,
                content_hash=content_hash,
                message=f"step:{effect.current_step}",
            )
            return Success(version)
        except Exception as exc:  # noqa: BLE001
            return Failure(StorageError(message=str(exc)))


class RNGInterpreter:
    """Interpreter for RNG effects.

    Captures and restores RNG states for reproducibility.
    Uses SharedRegistry for state bytes storage.
    See reproducibility_proofs.md for determinism guarantees.
    """

    def __init__(self, registry: SharedRegistry) -> None:
        """Initialize RNG interpreter.

        Args:
            registry: Shared registry for state bytes storage.
        """
        self._registry = registry

    async def interpret(self, effect: RNGEffect) -> Result[object, RNGError]:
        """Execute RNG effect."""
        match effect:
            case CaptureRNGState():
                return await self._capture_state(effect)
            case RestoreRNGState(rng_type=rt, state_bytes=sb):
                return await self._restore_state(rt, sb)
            case _:
                assert_never(effect)

    async def _capture_state(self, effect: CaptureRNGState) -> Result[object, RNGError]:
        """Capture RNG state as bytes for serialization.

        Args:
            effect: CaptureRNGState effect with rng_type and output_id.

        Returns:
            Result containing captured state bytes or RNGError.
        """
        rng_type = effect.rng_type
        try:
            state: bytes
            match rng_type:
                case "torch_cpu":
                    state = torch.get_rng_state().cpu().numpy().tobytes()
                case "torch_cuda":
                    match _cuda_rng_env():
                        case CudaRngEnvMissing(reason):
                            return Failure(RNGError(message=reason, rng_type=rng_type))
                        case CudaRngEnvAvailable(device_count=count):
                            states = torch.cuda.get_rng_state_all()
                            assert (
                                len(states) == count
                            ), f"CUDA RNG states ({len(states)}) differ from device count ({count})"
                            state = b"".join(s.cpu().numpy().tobytes() for s in states)
                case "numpy":
                    # Get numpy random state as bytes via pickle
                    state_dict = np.random.get_state(legacy=False)
                    state = pickle.dumps(state_dict)
                case "cupy":
                    # CuPy doesn't have direct state export
                    return Failure(
                        RNGError(message="CuPy state capture not implemented", rng_type=rng_type)
                    )

            # Store in shared registry with specified output ID
            match self._registry.register_bytes(effect.output_id, state):
                case Failure(err):
                    return Failure(
                        RNGError(message=f"Registry bytes error: {err}", rng_type=rng_type)
                    )
                case Success(_):
                    return Success(state)
        except RuntimeError as e:
            return Failure(RNGError(message=str(e), rng_type=rng_type))

    async def _restore_state(self, rng_type: str, state_bytes: bytes) -> Result[object, RNGError]:
        """Restore previously captured RNG state."""
        try:
            match rng_type:
                case "torch_cpu":
                    state_array = np.frombuffer(state_bytes, dtype=np.uint8)
                    state_tensor = torch.from_numpy(state_array.copy())
                    torch.set_rng_state(state_tensor)
                    return Success(None)
                case "torch_cuda":
                    match _cuda_rng_env():
                        case CudaRngEnvMissing(reason):
                            return Failure(RNGError(message=reason, rng_type=rng_type))
                        case CudaRngEnvAvailable(device_count=device_count):
                            state_size = len(state_bytes) // device_count
                            states = []
                            for i in range(device_count):
                                chunk = state_bytes[i * state_size : (i + 1) * state_size]
                                state_array = np.frombuffer(chunk, dtype=np.uint8)
                                states.append(torch.from_numpy(state_array.copy()))
                            torch.cuda.set_rng_state_all(states)
                            return Success(None)
                case "numpy":
                    state = pickle.loads(state_bytes)  # noqa: S301
                    np.random.set_state(state)
                    return Success(None)
                case "cupy":
                    return Failure(
                        RNGError(message="CuPy state restore not implemented", rng_type=rng_type)
                    )
                case _:
                    return Failure(RNGError(message=f"Unknown RNG type: {rng_type}", rng_type=""))
        except RuntimeError as e:
            return Failure(RNGError(message=str(e), rng_type=rng_type))


class MetadataInterpreter:
    """Interpreter for metadata effects.

    Handles reading and updating metadata state during effect execution.
    Uses SharedRegistry for metadata storage.
    This enables tracking of training state like sobol_skip and global_step.
    """

    def __init__(self, registry: SharedRegistry) -> None:
        """Initialize metadata interpreter.

        Args:
            registry: Shared registry for metadata storage.
        """
        self._registry = registry

    async def interpret(self, effect: MetadataEffect) -> Result[object, MetadataError]:
        """Execute metadata effect."""
        match effect:
            case ReadMetadata(key=k, output_id=oid):
                return await self._read_metadata(k, oid)
            case UpdateMetadata(key=k, operation=op, value=v):
                return await self._update_metadata(k, op, v)
            case _:
                assert_never(effect)

    async def _read_metadata(
        self,
        key: str,
        output_id: str,
    ) -> Result[object, MetadataError]:
        """Read a value from the metadata registry.

        Args:
            key: Key to read.
            output_id: Identifier for storing the read value (unused, value is returned).

        Returns:
            Result containing the value or MetadataError if key not found.
        """
        result = self._registry.get_metadata(key)
        match result:
            case Failure(_):
                return Failure(MetadataError(message=f"Metadata key not found: {key}", key=key))
            case Success(value):
                return Success(value)

    async def _update_metadata(
        self,
        key: str,
        operation: Literal["set", "add", "increment"],
        value: int | float | str,
    ) -> Result[object, MetadataError]:
        """Update a value in the metadata registry.

        Args:
            key: Key to update.
            operation: Type of operation (set, add, increment).
            value: Value for the operation.

        Returns:
            Result containing the new value or MetadataError.
        """
        # Use SharedRegistry's update_metadata method which handles all operations
        if operation not in ("set", "add", "increment"):
            return Failure(MetadataError(message=f"Unknown operation: {operation}", key=key))

        result = self._registry.update_metadata(key, operation, value)
        match result:
            case Failure(err):
                return Failure(MetadataError(message=str(err), key=key))
            case Success(new_value):
                return Success(new_value)


class SpectralMCInterpreter:
    """Master interpreter composing all effect interpreters.

    Routes effects to the appropriate sub-interpreter based on type.
    This is the ONLY entry point for effect execution in SpectralMC.

    All sub-interpreters share a single SharedRegistry for data flow.

    Example:
        >>> registry = SharedRegistry()
        >>> gpu = GPUInterpreter(torch_stream, cupy_stream, registry)
        >>> training = TrainingInterpreter(registry)
        >>> mc = MonteCarloInterpreter(registry)
        >>> storage = StorageInterpreter(bucket, registry)
        >>> rng = RNGInterpreter(registry)
        >>> metadata = MetadataInterpreter(registry)
        >>> logging_interpreter = LoggingInterpreter()
        >>> interpreter = SpectralMCInterpreter(
        ...     gpu,
        ...     training,
        ...     mc,
        ...     storage,
        ...     rng,
        ...     metadata,
        ...     logging_interpreter,
        ...     registry,
        ... )
        >>> result = await interpreter.interpret(effect)
    """

    def __init__(
        self,
        gpu: GPUInterpreter,
        training: TrainingInterpreter,
        montecarlo: MonteCarloInterpreter,
        storage: StorageInterpreter,
        rng: RNGInterpreter,
        metadata: MetadataInterpreter,
        logging_interpreter: LoggingInterpreter,
        registry: SharedRegistry,
    ) -> None:
        """Initialize master interpreter with all sub-interpreters.

        Args:
            gpu: Interpreter for GPU effects.
            training: Interpreter for training effects.
            montecarlo: Interpreter for Monte Carlo effects.
            storage: Interpreter for storage effects.
            rng: Interpreter for RNG effects.
            metadata: Interpreter for metadata effects.
            logging_interpreter: Interpreter for logging effects.
            registry: Shared registry used by all sub-interpreters.
        """
        self._gpu = gpu
        self._training = training
        self._montecarlo = montecarlo
        self._storage = storage
        self._rng = rng
        self._metadata = metadata
        self._logging = logging_interpreter
        self._registry = registry

    async def interpret(self, effect: Effect) -> Result[object, EffectError]:
        """Route effect to appropriate sub-interpreter."""
        match effect:
            case (
                TensorTransfer() | StreamSync() | KernelLaunch() | DLPackTransfer() | SplitInputs()
            ):
                gpu_result = await self._gpu.interpret(effect)
                return _widen_gpu_error(gpu_result)
            case (
                ForwardPass()
                | ForwardPassComplex()
                | BackwardPass()
                | OptimizerStep()
                | ComputeLoss()
                | ComputeComplexLoss()
                | ZeroGrad()
                | ComputeGradNorm()
                | UpdateLearningRate()
                | LogMetrics()
            ):
                training_result = await self._training.interpret(effect)
                return _widen_training_error(training_result)
            case (
                GenerateNormals()
                | SimulatePaths()
                | ComputeFFT()
                | SampleContracts()
                | ProcessBatch()
                | StackTensors()
                | ComputeMeanFFT()
            ):
                mc_result = await self._montecarlo.interpret(effect)
                return _widen_montecarlo_error(mc_result)
            case ReadObject() | WriteObject() | CommitVersion() | CommitCheckpoint():
                storage_result = await self._storage.interpret(effect)
                return _widen_storage_error(storage_result)
            case CaptureRNGState() | RestoreRNGState():
                rng_result = await self._rng.interpret(effect)
                return _widen_rng_error(rng_result)
            case ReadMetadata() | UpdateMetadata():
                metadata_result = await self._metadata.interpret(effect)
                return _widen_metadata_error(metadata_result)
            case LogMessage():
                logging_result = await self._logging.interpret(effect)
                return _widen_logging_error(logging_result)
            case _:
                assert_never(effect)

    async def interpret_sequence(
        self,
        sequence: EffectSequence[T],
    ) -> Result[T, EffectError]:
        """Execute a sequence of effects, threading registry state.

        Executes each effect in order. If any effect fails, returns that
        failure immediately. On success, applies the continuation function
        to all results.

        Args:
            sequence: EffectSequence containing effects and continuation.

        Returns:
            Result containing the continuation result or the first error.

        Example:
            >>> seq = sequence_effects(
            ...     GenerateNormals(rows=1024, cols=252, seed=42),
            ...     SimulatePaths(spot=100.0, vol=0.2),
            ...     StreamSync(stream_type="cupy"),
            ... )
            >>> result = await interpreter.interpret_sequence(seq)
        """
        results: list[object] = []

        for effect in sequence.effects:
            result = await self.interpret(effect)
            match result:
                case Failure(error):
                    return Failure(error)
                case Success(value):
                    results.append(value)

        # Apply continuation to collected results
        final_value = sequence.continuation(results)
        return Success(final_value)

    async def interpret_parallel(
        self,
        parallel: EffectParallel[T],
    ) -> Result[T, EffectError]:
        """Execute effects in parallel (concurrently).

        All effects are started concurrently using asyncio.gather.
        If any effect fails, the entire parallel block fails.

        Args:
            parallel: EffectParallel containing effects and combiner.

        Returns:
            Result containing the combined result or the first error.

        Example:
            >>> par = parallel_effects(
            ...     WriteObject(bucket="models", key="v1/a.pb"),
            ...     WriteObject(bucket="models", key="v1/b.pb"),
            ... )
            >>> result = await interpreter.interpret_parallel(par)
        """
        # Start all effects concurrently
        tasks = [self.interpret(effect) for effect in parallel.effects]
        results = await asyncio.gather(*tasks)

        # Collect successful values or return first failure
        values: list[object] = []
        for result in results:
            match result:
                case Failure(error):
                    return Failure(error)
                case Success(value):
                    values.append(value)

        # Apply combiner to collected results
        final_value = parallel.combiner(values)
        return Success(final_value)

    @property
    def registry(self) -> SharedRegistry:
        """Get the shared registry for direct access."""
        return self._registry

    @property
    def gpu_interpreter(self) -> GPUInterpreter:
        """Access GPU interpreter."""
        return self._gpu

    @property
    def training_interpreter(self) -> TrainingInterpreter:
        """Access training interpreter."""
        return self._training

    @property
    def montecarlo_interpreter(self) -> MonteCarloInterpreter:
        """Access Monte Carlo interpreter."""
        return self._montecarlo

    @property
    def storage_interpreter(self) -> StorageInterpreter:
        """Access storage interpreter."""
        return self._storage

    @property
    def rng_interpreter(self) -> RNGInterpreter:
        """Access RNG interpreter."""
        return self._rng

    @property
    def metadata_interpreter(self) -> MetadataInterpreter:
        """Access metadata interpreter."""
        return self._metadata

    @property
    def logging_interpreter(self) -> LoggingInterpreter:
        """Access logging interpreter."""
        return self._logging

    @classmethod
    def create(
        cls,
        torch_stream: torch.cuda.Stream,
        cupy_stream: cp.cuda.Stream,
        storage_bucket: str,
        mc_engine: BlackScholes | None = None,
        logging_interpreter: LoggingInterpreter | None = None,
    ) -> SpectralMCInterpreter:
        """Factory method to create a SpectralMCInterpreter with a shared registry.

        Creates all sub-interpreters with a single shared registry for data flow.

        Args:
            torch_stream: PyTorch CUDA stream.
            cupy_stream: CuPy CUDA stream.
            storage_bucket: Default S3 bucket for storage operations.
            mc_engine: Optional BlackScholes engine for Monte Carlo processing.
            logging_interpreter: Optional logging interpreter instance.

        Returns:
            Configured SpectralMCInterpreter with shared registry.

        Example:
            >>> import cupy as cp
            >>> import torch
            >>> torch_stream = torch.cuda.Stream()
            >>> cupy_stream = cp.cuda.Stream()
            >>> interpreter = SpectralMCInterpreter.create(
            ...     torch_stream, cupy_stream, "my-bucket"
            ... )
        """
        registry = SharedRegistry()
        logging_impl = logging_interpreter or LoggingInterpreter()
        return cls(
            gpu=GPUInterpreter(torch_stream, cupy_stream, registry),
            training=TrainingInterpreter(registry),
            montecarlo=MonteCarloInterpreter(registry, mc_engine=mc_engine),
            storage=StorageInterpreter(storage_bucket, registry),
            rng=RNGInterpreter(registry),
            metadata=MetadataInterpreter(registry),
            logging_interpreter=logging_impl,
            registry=registry,
        )


def _widen_gpu_error(result: Result[object, GPUError]) -> Result[object, EffectError]:
    """Widen GPU error to EffectError union."""
    match result:
        case Success(value):
            return Success(value)
        case Failure(error):
            widened: EffectError = error
            return Failure(widened)


def _widen_training_error(result: Result[object, TrainingError]) -> Result[object, EffectError]:
    """Widen Training error to EffectError union."""
    match result:
        case Success(value):
            return Success(value)
        case Failure(error):
            widened: EffectError = error
            return Failure(widened)


def _widen_montecarlo_error(result: Result[object, MonteCarloError]) -> Result[object, EffectError]:
    """Widen MonteCarlo error to EffectError union."""
    match result:
        case Success(value):
            return Success(value)
        case Failure(error):
            widened: EffectError = error
            return Failure(widened)


def _widen_storage_error(result: Result[object, StorageError]) -> Result[object, EffectError]:
    """Widen Storage error to EffectError union."""
    match result:
        case Success(value):
            return Success(value)
        case Failure(error):
            widened: EffectError = error
            return Failure(widened)


def _widen_rng_error(result: Result[object, RNGError]) -> Result[object, EffectError]:
    """Widen RNG error to EffectError union."""
    match result:
        case Success(value):
            return Success(value)
        case Failure(error):
            widened: EffectError = error
            return Failure(widened)


def _widen_metadata_error(result: Result[object, MetadataError]) -> Result[object, EffectError]:
    """Widen Metadata error to EffectError union."""
    match result:
        case Success(value):
            return Success(value)
        case Failure(error):
            widened: EffectError = error
            return Failure(widened)


def _widen_logging_error(result: Result[object, LoggingError]) -> Result[object, EffectError]:
    """Widen Logging error to EffectError union."""
    match result:
        case Success(value):
            return Success(value)
        case Failure(error):
            widened: EffectError = error
            return Failure(widened)
