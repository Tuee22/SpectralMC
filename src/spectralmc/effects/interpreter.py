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

from typing import TYPE_CHECKING, Never, Protocol, TypeVar


T = TypeVar("T")

from spectralmc.effects.composition import EffectParallel, EffectSequence
from spectralmc.effects.errors import (
    EffectError,
    GPUError,
    MetadataError,
    MonteCarloError,
    RNGError,
    StorageError,
    TrainingError,
)
from spectralmc.effects.registry import SharedRegistry
from spectralmc.effects.gpu import (
    DLPackTransfer,
    GPUEffect,
    KernelLaunch,
    StreamSync,
    TensorTransfer,
)
from spectralmc.effects.metadata import MetadataEffect, ReadMetadata, UpdateMetadata
from spectralmc.effects.montecarlo import (
    ComputeFFT,
    GenerateNormals,
    MonteCarloEffect,
    SimulatePaths,
)
from spectralmc.effects.rng import CaptureRNGState, RestoreRNGState, RNGEffect
from spectralmc.effects.storage import CommitVersion, ReadObject, StorageEffect, WriteObject
from spectralmc.effects.training import (
    BackwardPass,
    ComputeLoss,
    ForwardPass,
    LogMetrics,
    OptimizerStep,
    TrainingEffect,
)
from spectralmc.effects.types import Effect
from spectralmc.result import Failure, Result, Success


if TYPE_CHECKING:
    import cupy as cp
    import torch


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
    """

    def __init__(
        self,
        torch_stream: torch.cuda.Stream,
        cupy_stream: cp.cuda.Stream,
    ) -> None:
        """Initialize GPU interpreter with CUDA streams.

        Args:
            torch_stream: PyTorch CUDA stream for PyTorch operations.
            cupy_stream: CuPy CUDA stream for CuPy operations.
        """
        self._torch_stream = torch_stream
        self._cupy_stream = cupy_stream
        # Registry for tensor storage (tensor_id -> tensor)
        self._tensor_registry: dict[str, object] = {}
        # Registry for kernel functions (kernel_name -> callable)
        self._kernel_registry: dict[str, object] = {}

    async def interpret(self, effect: GPUEffect) -> Result[object, GPUError]:
        """Execute GPU effect with stream coordination."""
        match effect:
            case TensorTransfer(source_device=src, target_device=dst, tensor_id=tid):
                return await self._transfer_tensor(src, dst, tid)
            case StreamSync(stream_type=st):
                return await self._sync_stream(st)
            case KernelLaunch(kernel_name=name, grid_config=grid, block_config=block):
                return await self._launch_kernel(name, grid, block)
            case DLPackTransfer():
                return await self._dlpack_transfer(effect)
            case _:
                assert_never(effect)

    async def _transfer_tensor(
        self,
        src: object,
        dst: object,
        tensor_id: str,
    ) -> Result[object, GPUError]:
        """Transfer tensor between devices.

        Uses spectralmc.models.cpu_gpu_transfer for actual transfer.
        """
        import torch

        from spectralmc.models.cpu_gpu_transfer import TransferDestination, move_tensor_tree
        from spectralmc.models.torch import Device

        try:
            tensor = self._tensor_registry.get(tensor_id)
            if tensor is None:
                return Failure(GPUError(message=f"Tensor not found: {tensor_id}"))

            # Determine transfer destination
            if dst == Device.cpu:
                dest = TransferDestination.CPU
            else:
                dest = TransferDestination.CUDA

            # Validate tensor is a torch.Tensor (base TensorTree type)
            if not isinstance(tensor, torch.Tensor):
                return Failure(
                    GPUError(message=f"Registry item {tensor_id} is not a Tensor: {type(tensor)}")
                )

            result = move_tensor_tree(tensor, dest=dest)
            self._tensor_registry[tensor_id] = result
            return Success(result)
        except (RuntimeError, ValueError) as e:
            return Failure(GPUError(message=str(e)))

    async def _sync_stream(self, stream_type: str) -> Result[object, GPUError]:
        """Synchronize the specified CUDA stream."""
        try:
            match stream_type:
                case "torch":
                    self._torch_stream.synchronize()
                case "cupy":
                    self._cupy_stream.synchronize()
                case "numba":
                    # Numba synchronize via cuda module
                    from numba.cuda import synchronize as numba_sync

                    numba_sync()
                case _:
                    return Failure(GPUError(message=f"Unknown stream type: {stream_type}"))
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
        kernel_fn = self._kernel_registry.get(name)
        if kernel_fn is None:
            return Failure(GPUError(message=f"Kernel not found: {name}"))

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
        tensor = self._tensor_registry.get(effect.source_tensor_id)
        if tensor is None:
            return Failure(GPUError(message=f"Tensor not found: {effect.source_tensor_id}"))

        try:
            import cupy as cp
            import torch

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
                # Type stubs incomplete for torch->cupy DLPack transfer
                transferred = cp.asarray(tensor)  # type: ignore[arg-type]
            else:
                return Failure(
                    GPUError(
                        message=f"Unsupported transfer: {effect.source_framework} -> {effect.target_framework}"
                    )
                )

            # Store result in registry
            self._tensor_registry[effect.output_tensor_id] = transferred
            return Success(transferred)
        except (RuntimeError, TypeError) as e:
            return Failure(GPUError(message=str(e)))

    def register_tensor(self, tensor_id: str, tensor: object) -> None:
        """Register a tensor in the interpreter's registry."""
        self._tensor_registry[tensor_id] = tensor

    def register_kernel(self, kernel_name: str, kernel_fn: object) -> None:
        """Register a kernel function in the interpreter's registry."""
        self._kernel_registry[kernel_name] = kernel_fn


class TrainingInterpreter:
    """Interpreter for training effects.

    Handles forward/backward passes and optimizer steps.
    """

    def __init__(self) -> None:
        """Initialize training interpreter."""
        # Registry for models (model_id -> model)
        self._model_registry: dict[str, object] = {}
        # Registry for optimizers (optimizer_id -> optimizer)
        self._optimizer_registry: dict[str, object] = {}
        # Registry for tensors (tensor_id -> tensor)
        self._tensor_registry: dict[str, object] = {}

    async def interpret(self, effect: TrainingEffect) -> Result[object, TrainingError]:
        """Execute training effect."""
        match effect:
            case ForwardPass():
                return await self._forward_pass(effect)
            case BackwardPass(loss_tensor_id=tid):
                return await self._backward_pass(tid)
            case OptimizerStep(optimizer_id=oid):
                return await self._optimizer_step(oid)
            case ComputeLoss():
                return await self._compute_loss(effect)
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
        model = self._model_registry.get(effect.model_id)
        if model is None:
            return Failure(TrainingError(message=f"Model not found: {effect.model_id}"))

        tensor = self._tensor_registry.get(effect.input_tensor_id)
        if tensor is None:
            return Failure(TrainingError(message=f"Tensor not found: {effect.input_tensor_id}"))

        try:
            # Type-safe model call requires proper protocol
            # Use callable() for reliable type checking
            if callable(model):
                output = model(tensor)
                # Store output in registry with specified ID
                self._tensor_registry[effect.output_tensor_id] = output
                return Success(output)
            return Failure(TrainingError(message=f"Model {effect.model_id} is not callable"))
        except RuntimeError as e:
            return Failure(TrainingError(message=str(e)))

    async def _backward_pass(self, loss_tensor_id: str) -> Result[object, TrainingError]:
        """Compute gradients via backpropagation."""
        tensor = self._tensor_registry.get(loss_tensor_id)
        if tensor is None:
            return Failure(TrainingError(message=f"Loss tensor not found: {loss_tensor_id}"))

        try:
            # PyTorch backward requires tensor with grad
            if hasattr(tensor, "backward"):
                backward_fn = tensor.backward
                backward_fn()
            return Success(None)
        except RuntimeError as e:
            return Failure(TrainingError(message=str(e)))

    async def _optimizer_step(self, optimizer_id: str) -> Result[object, TrainingError]:
        """Update model parameters using optimizer."""
        optimizer = self._optimizer_registry.get(optimizer_id)
        if optimizer is None:
            return Failure(TrainingError(message=f"Optimizer not found: {optimizer_id}"))

        try:
            if hasattr(optimizer, "step"):
                step_fn = optimizer.step
                step_fn()
            if hasattr(optimizer, "zero_grad"):
                zero_grad_fn = optimizer.zero_grad
                zero_grad_fn()
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
        pred = self._tensor_registry.get(effect.pred_tensor_id)
        if pred is None:
            return Failure(
                TrainingError(message=f"Prediction tensor not found: {effect.pred_tensor_id}")
            )

        target = self._tensor_registry.get(effect.target_tensor_id)
        if target is None:
            return Failure(
                TrainingError(message=f"Target tensor not found: {effect.target_tensor_id}")
            )

        try:
            import torch

            if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
                return Failure(TrainingError(message="Tensors must be torch.Tensor"))

            loss: torch.Tensor
            if effect.loss_type == "mse":
                loss = torch.nn.functional.mse_loss(pred, target)
            elif effect.loss_type == "mae":
                # Type stubs incomplete for torch.mean
                loss = torch.mean(torch.abs(pred - target))  # type: ignore[attr-defined]
            else:
                # effect.loss_type == "huber"
                # Type stubs incomplete for smooth_l1_loss
                loss = torch.nn.functional.smooth_l1_loss(pred, target)  # type: ignore[attr-defined]

            # Store loss in registry for BackwardPass
            self._tensor_registry[effect.output_tensor_id] = loss
            return Success(loss)
        except RuntimeError as e:
            return Failure(TrainingError(message=str(e)))

    async def _log_metrics(self, effect: LogMetrics) -> Result[object, TrainingError]:
        """Log metrics to TensorBoard.

        Args:
            effect: LogMetrics effect with metrics and step.

        Returns:
            Result containing None or TrainingError.
        """
        try:
            # Import TensorBoard writer if available
            # This is a simplified implementation - real usage would use
            # a pre-registered SummaryWriter
            from torch.utils.tensorboard import SummaryWriter

            # Check if a writer is registered
            writer = self._tensor_registry.get("_tensorboard_writer")
            if writer is not None and isinstance(writer, SummaryWriter):
                for metric_name, metric_value in effect.metrics:
                    writer.add_scalar(metric_name, metric_value, effect.step)

            return Success(None)
        except (ImportError, RuntimeError):
            # Log metrics is optional - don't fail if TensorBoard unavailable
            return Success(None)

    def register_model(self, model_id: str, model: object) -> None:
        """Register a model in the interpreter's registry."""
        self._model_registry[model_id] = model

    def register_optimizer(self, optimizer_id: str, optimizer: object) -> None:
        """Register an optimizer in the interpreter's registry."""
        self._optimizer_registry[optimizer_id] = optimizer

    def register_tensor(self, tensor_id: str, tensor: object) -> None:
        """Register a tensor in the interpreter's registry."""
        self._tensor_registry[tensor_id] = tensor


class MonteCarloInterpreter:
    """Interpreter for Monte Carlo effects.

    Handles random number generation, path simulation, and FFT computation.
    """

    def __init__(self) -> None:
        """Initialize Monte Carlo interpreter."""
        # Registry for tensors
        self._tensor_registry: dict[str, object] = {}

    async def interpret(self, effect: MonteCarloEffect) -> Result[object, MonteCarloError]:
        """Execute Monte Carlo effect."""
        match effect:
            case GenerateNormals(rows=r, cols=c, seed=s, skip=sk):
                return await self._generate_normals(r, c, s, sk)
            case SimulatePaths():
                return await self._simulate_paths(effect)
            case ComputeFFT(input_tensor_id=tid, axis=ax):
                return await self._compute_fft(tid, ax)
            case _:
                assert_never(effect)

    async def _generate_normals(
        self,
        rows: int,
        cols: int,
        seed: int,
        skip: int,
    ) -> Result[object, MonteCarloError]:
        """Generate standard normal random matrix on GPU."""
        try:
            import cupy as cp

            # Use CuPy's random generator with seed
            rng = cp.random.default_rng(seed)
            # Skip values if resuming (use tuple shape for skip)
            if skip > 0:
                _ = rng.standard_normal((skip,))
            # Generate the matrix
            matrix = rng.standard_normal((rows, cols), dtype=cp.float32)
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
            import cupy as cp
            from numba import cuda

            from spectralmc.gbm import SimulateBlackScholes

            # Get input normals from registry
            normals = self._tensor_registry.get(effect.input_normals_id)
            if normals is None:
                return Failure(
                    MonteCarloError(message=f"Normals tensor not found: {effect.input_normals_id}")
                )

            if not isinstance(normals, cp.ndarray):
                return Failure(MonteCarloError(message="Input normals must be a CuPy array"))

            # Make a copy since the kernel operates in-place
            # Type stubs incomplete for cp.array
            sims: cp.ndarray = cp.array(normals)  # type: ignore[attr-defined]

            # Compute kernel launch parameters
            threads_per_block = 256
            total_paths: int = int(sims.shape[1])
            blocks = (total_paths + threads_per_block - 1) // threads_per_block

            # Compute timestep size
            dt = effect.expiry / effect.timesteps

            # Get the default CUDA stream for kernel launch
            # Type stubs incomplete for cuda.default_stream
            stream = cuda.default_stream()  # type: ignore[attr-defined]

            # Launch the SimulateBlackScholes kernel (blocks, threads, stream)
            SimulateBlackScholes[blocks, threads_per_block, stream](
                cuda.as_cuda_array(sims),
                effect.timesteps,
                dt,
                effect.spot,
                effect.rate,
                effect.dividend,
                effect.vol,
                effect.simulate_log_return,
            )

            # Synchronize to ensure kernel completes
            cuda.synchronize()

            # Optional forward normalization
            if effect.normalize_forwards:
                times = cp.linspace(dt, effect.expiry, effect.timesteps, dtype=sims.dtype)
                forwards = effect.spot * cp.exp((effect.rate - effect.dividend) * times)
                row_means = cp.mean(sims, axis=1, keepdims=True).squeeze()
                sims *= cp.expand_dims(forwards / row_means, 1)

            # Store result in registry for downstream effects
            output_id = f"paths_{id(effect)}"
            self._tensor_registry[output_id] = sims

            return Success(sims)
        except RuntimeError as e:
            return Failure(MonteCarloError(message=str(e)))

    async def _compute_fft(
        self,
        input_tensor_id: str,
        axis: int,
    ) -> Result[object, MonteCarloError]:
        """Compute FFT on tensor."""
        tensor = self._tensor_registry.get(input_tensor_id)
        if tensor is None:
            return Failure(MonteCarloError(message=f"Tensor not found: {input_tensor_id}"))

        try:
            import cupy as cp

            # Compute FFT using CuPy - tensor must be a cupy array
            if isinstance(tensor, cp.ndarray):
                result = cp.fft.fft(tensor, axis=axis)
                return Success(result)
            return Failure(MonteCarloError(message="Tensor is not a CuPy array"))
        except RuntimeError as e:
            return Failure(MonteCarloError(message=str(e)))

    def register_tensor(self, tensor_id: str, tensor: object) -> None:
        """Register a tensor in the interpreter's registry."""
        self._tensor_registry[tensor_id] = tensor


class StorageInterpreter:
    """Interpreter for storage effects.

    Handles S3 reads, writes, and blockchain version commits.
    """

    def __init__(self, bucket: str) -> None:
        """Initialize storage interpreter.

        Args:
            bucket: Default S3 bucket for operations.
        """
        self._bucket = bucket
        # Content registry for write operations (hash -> bytes)
        self._content_registry: dict[str, bytes] = {}

    async def interpret(self, effect: StorageEffect) -> Result[object, StorageError]:
        """Execute storage effect."""
        match effect:
            case ReadObject(bucket=b, key=k):
                return await self._read_object(b or self._bucket, k)
            case WriteObject(bucket=b, key=k, content_hash=h):
                return await self._write_object(b or self._bucket, k, h)
            case CommitVersion(parent_counter=p, checkpoint_hash=h, message=m):
                return await self._commit_version(p, h, m)
            case _:
                assert_never(effect)

    async def _read_object(self, bucket: str, key: str) -> Result[object, StorageError]:
        """Read object from S3."""
        try:
            # Import AsyncBlockchainModelStore for S3 operations
            from spectralmc.storage.s3_operations import S3Operations
            from spectralmc.storage.store import AsyncBlockchainModelStore

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
        content = self._content_registry.get(content_hash)
        if content is None:
            return Failure(
                StorageError(
                    message=f"Content not found for hash: {content_hash}",
                    bucket=bucket,
                    key=key,
                )
            )

        try:
            from spectralmc.storage.s3_operations import S3Operations
            from spectralmc.storage.store import AsyncBlockchainModelStore

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
                    case Failure(err):
                        return Failure(StorageError(message=str(err), bucket=bucket, key=key))
        except Exception as e:
            return Failure(StorageError(message=str(e), bucket=bucket, key=key))

    async def _commit_version(
        self,
        parent_counter: int | None,
        checkpoint_hash: str,
        message: str,
    ) -> Result[object, StorageError]:
        """Commit new model version."""
        checkpoint_data = self._content_registry.get(checkpoint_hash)
        if checkpoint_data is None:
            return Failure(StorageError(message=f"Checkpoint not found: {checkpoint_hash}"))

        try:
            from spectralmc.storage.store import AsyncBlockchainModelStore

            async with AsyncBlockchainModelStore(self._bucket) as store:
                # store.commit returns ModelVersion directly, not Result
                version = await store.commit(checkpoint_data, checkpoint_hash, message)
                return Success(version)
        except Exception as e:
            return Failure(StorageError(message=str(e)))

    def register_content(self, content_hash: str, content: bytes) -> None:
        """Register content in the interpreter's registry."""
        self._content_registry[content_hash] = content


class RNGInterpreter:
    """Interpreter for RNG effects.

    Captures and restores RNG states for reproducibility.
    See reproducibility_proofs.md for determinism guarantees.
    """

    async def interpret(self, effect: RNGEffect) -> Result[object, RNGError]:
        """Execute RNG effect."""
        match effect:
            case CaptureRNGState(rng_type=rt):
                return await self._capture_state(rt)
            case RestoreRNGState(rng_type=rt, state_bytes=sb):
                return await self._restore_state(rt, sb)
            case _:
                assert_never(effect)

    async def _capture_state(self, rng_type: str) -> Result[object, RNGError]:
        """Capture RNG state as bytes for serialization."""
        try:
            match rng_type:
                case "torch_cpu":
                    import torch

                    state = torch.get_rng_state().cpu().numpy().tobytes()
                    return Success(state)
                case "torch_cuda":
                    import torch

                    if not torch.cuda.is_available():
                        return Failure(RNGError(message="CUDA not available", rng_type=rng_type))
                    states = torch.cuda.get_rng_state_all()
                    # Serialize all CUDA device states
                    combined = b"".join(s.cpu().numpy().tobytes() for s in states)
                    return Success(combined)
                case "numpy":
                    import pickle

                    import numpy as np

                    # Get numpy random state as bytes via pickle
                    state_dict = np.random.get_state(legacy=False)
                    state_bytes = pickle.dumps(state_dict)
                    return Success(state_bytes)
                case "cupy":
                    # CuPy doesn't have direct state export
                    return Failure(
                        RNGError(message="CuPy state capture not implemented", rng_type=rng_type)
                    )
                case _:
                    return Failure(RNGError(message=f"Unknown RNG type: {rng_type}", rng_type=""))
        except RuntimeError as e:
            return Failure(RNGError(message=str(e), rng_type=rng_type))

    async def _restore_state(self, rng_type: str, state_bytes: bytes) -> Result[object, RNGError]:
        """Restore previously captured RNG state."""
        try:
            match rng_type:
                case "torch_cpu":
                    import numpy as np
                    import torch

                    state_array = np.frombuffer(state_bytes, dtype=np.uint8)
                    state_tensor = torch.from_numpy(state_array.copy())
                    torch.set_rng_state(state_tensor)
                    return Success(None)
                case "torch_cuda":
                    import numpy as np
                    import torch

                    if not torch.cuda.is_available():
                        return Failure(RNGError(message="CUDA not available", rng_type=rng_type))
                    # Restore requires knowing device count
                    device_count = torch.cuda.device_count()
                    state_size = len(state_bytes) // device_count
                    states = []
                    for i in range(device_count):
                        chunk = state_bytes[i * state_size : (i + 1) * state_size]
                        state_array = np.frombuffer(chunk, dtype=np.uint8)
                        states.append(torch.from_numpy(state_array.copy()))
                    torch.cuda.set_rng_state_all(states)
                    return Success(None)
                case "numpy":
                    import pickle

                    import numpy as np

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
    This enables tracking of training state like sobol_skip and global_step.
    """

    def __init__(self) -> None:
        """Initialize metadata interpreter with empty registry."""
        self._registry: dict[str, int | float | str] = {}

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
        value = self._registry.get(key)
        if value is None:
            return Failure(MetadataError(message=f"Metadata key not found: {key}", key=key))
        return Success(value)

    async def _update_metadata(
        self,
        key: str,
        operation: str,
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
        try:
            match operation:
                case "set":
                    self._registry[key] = value
                    return Success(value)
                case "add":
                    current = self._registry.get(key, 0)
                    if isinstance(current, str) or isinstance(value, str):
                        return Failure(MetadataError(message="Cannot add to string value", key=key))
                    new_value = current + value
                    self._registry[key] = new_value
                    return Success(new_value)
                case "increment":
                    current = self._registry.get(key, 0)
                    if isinstance(current, str):
                        return Failure(
                            MetadataError(message="Cannot increment string value", key=key)
                        )
                    new_value = current + 1
                    self._registry[key] = new_value
                    return Success(new_value)
                case _:
                    return Failure(
                        MetadataError(message=f"Unknown operation: {operation}", key=key)
                    )
        except (TypeError, ValueError) as e:
            return Failure(MetadataError(message=str(e), key=key))

    def register(self, key: str, value: int | float | str) -> None:
        """Pre-register a metadata value."""
        self._registry[key] = value


class SpectralMCInterpreter:
    """Master interpreter composing all effect interpreters.

    Routes effects to the appropriate sub-interpreter based on type.
    This is the ONLY entry point for effect execution in SpectralMC.

    Example:
        >>> interpreter = SpectralMCInterpreter(gpu, training, mc, storage, rng)
        >>> result = await interpreter.interpret(effect)
        >>> match result:
        ...     case Success(value): ...
        ...     case Failure(error): ...
    """

    def __init__(
        self,
        gpu: GPUInterpreter,
        training: TrainingInterpreter,
        montecarlo: MonteCarloInterpreter,
        storage: StorageInterpreter,
        rng: RNGInterpreter,
        metadata: MetadataInterpreter | None = None,
    ) -> None:
        """Initialize master interpreter with all sub-interpreters.

        Args:
            gpu: Interpreter for GPU effects.
            training: Interpreter for training effects.
            montecarlo: Interpreter for Monte Carlo effects.
            storage: Interpreter for storage effects.
            rng: Interpreter for RNG effects.
            metadata: Interpreter for metadata effects (optional, created if None).
        """
        self._gpu = gpu
        self._training = training
        self._montecarlo = montecarlo
        self._storage = storage
        self._rng = rng
        self._metadata = metadata or MetadataInterpreter()

    async def interpret(self, effect: Effect) -> Result[object, EffectError]:
        """Route effect to appropriate sub-interpreter."""
        match effect:
            case TensorTransfer() | StreamSync() | KernelLaunch() | DLPackTransfer():
                gpu_result = await self._gpu.interpret(effect)
                return _widen_gpu_error(gpu_result)
            case ForwardPass() | BackwardPass() | OptimizerStep() | ComputeLoss() | LogMetrics():
                training_result = await self._training.interpret(effect)
                return _widen_training_error(training_result)
            case GenerateNormals() | SimulatePaths() | ComputeFFT():
                mc_result = await self._montecarlo.interpret(effect)
                return _widen_montecarlo_error(mc_result)
            case ReadObject() | WriteObject() | CommitVersion():
                storage_result = await self._storage.interpret(effect)
                return _widen_storage_error(storage_result)
            case CaptureRNGState() | RestoreRNGState():
                rng_result = await self._rng.interpret(effect)
                return _widen_rng_error(rng_result)
            case ReadMetadata() | UpdateMetadata():
                metadata_result = await self._metadata.interpret(effect)
                return _widen_metadata_error(metadata_result)
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
        import asyncio

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
