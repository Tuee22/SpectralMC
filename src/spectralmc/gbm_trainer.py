# src/spectralmc/gbm_trainer.py
"""
CUDA-accelerated trainer for learning discounted Black-Scholes pay-off
spectra with a complex-valued neural network (CVNN).

Overview
--------
The :class:`GbmCVNNPricer` orchestrates four tightly-coupled components:

1. **Quasi-random contract generator** - `SobolSampler` draws input
   contracts inside user-supplied bounds.
2. **Monte-Carlo engine** - `spectralmc.gbm.BlackScholes` prices a batch
   of contracts entirely on **CUDA**, then applies an FFT along the time
   axis.
3. **Complex-valued neural network** - any user model that satisfies the
   :class:`ComplexValuedModel` protocol.  The trainer infers its
   execution *device* and *real dtype* directly from the network's first
   parameter and asserts that **all** parameters conform.
4. **Optimiser & logging** - an *Adam* optimiser operating in
   half-open-loop: model parameters live on the active CUDA device while
   the optimiser *state* is moved to CPU **before** every snapshot to
   satisfy :pyclass:`spectralmc.models.torch.AdamOptimizerState`.

Determinism
-----------
*   Global PRNG seeding is delegated to callers.
*   Model and optimiser states are serialised via pure-CPU
    :pyclass:`~spectralmc.models.torch.TensorState` objects; this avoids
    device-specific artefacts in unit tests.

Implementation notes
--------------------
* ``_move_optimizer_state`` is a tiny helper that migrates *all* tensors
  in an `torch.optim.Optimizer.state` dict between devices.
* The trainer *always* stores the optimiser snapshot on **CPU**.  During
  a restart we immediately migrate the state back to the training
  device.
* Strict typing is enforced with **mypy --strict**; no `Any`, `cast` or
  `type: ignore` are required.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import warnings
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Protocol,
    Sequence,
    runtime_checkable,
)


if TYPE_CHECKING:
    from spectralmc.storage import AsyncBlockchainModelStore

from spectralmc.storage.errors import (
    CommitError,
    ConflictError,
    NotFastForwardError,
    StorageError,
)


_logger = logging.getLogger(__name__)

import cupy as cp
import numpy as np
import torch
from cupy.cuda import Stream as CuPyStream
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import (
    nn,
    optim,
)
from torch.utils.tensorboard import SummaryWriter

# CRITICAL: Import facade BEFORE torch for deterministic algorithms
from spectralmc.effects import (
    BackwardPass,
    CaptureRNGState,
    CommitVersion,
    ComputeFFT,
    ComputeLoss,
    DLPackTransfer,
    EffectSequence,
    ForwardPass,
    GenerateNormals,
    LogMetrics,
    OptimizerStep,
    ReadMetadata,
    SimulatePaths,
    StreamSync,
    UpdateMetadata,
    WriteObject,
    sequence_effects,
)
from spectralmc.gbm import BlackScholes, BlackScholesConfig, SimulationParams
from spectralmc.models.cpu_gpu_transfer import module_state_device_dtype
from spectralmc.models.numerical import Precision
from spectralmc.models.torch import (
    AdamOptimizerState,
    AnyDType,
    Device,
    DType,
    FullPrecisionDType,
)
from spectralmc.sobol_sampler import BoundSpec, SobolConfig, SobolSampler


__all__: tuple[str, ...] = (
    "GbmCVNNPricerConfig",
    "StepMetrics",
    "TrainingResult",
    "CudaExecutionContext",
    "TensorBoardLogger",
    "GbmCVNNPricer",
    "TrainingConfig",
)

# =============================================================================
# Protocols & helpers
# =============================================================================


@runtime_checkable
class ComplexValuedModel(Protocol):
    """Callable complex-valued network ``(real, imag) -> (real, imag)``."""

    def __call__(
        self, __real: torch.Tensor, __imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    # minimal subset of the nn.Module API
    def parameters(self) -> Iterable[nn.Parameter]: ...
    def named_parameters(self) -> Iterable[tuple[str, nn.Parameter]]: ...
    def state_dict(
        self,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]: ...
    def to(self, device: torch.device, dtype: torch.dtype) -> nn.Module: ...
    def train(self, mode: bool = True) -> None: ...
    def eval(self) -> None: ...


# =============================================================================
# Immutable dataclasses
# =============================================================================

StepLogger = Callable[["StepMetrics"], None]
"""User-supplied callback executed once after every optimiser step."""


class TrainingConfig(BaseModel):
    """Validated training hyperparameters."""

    num_batches: PositiveInt
    batch_size: PositiveInt
    learning_rate: float = Field(gt=0.0, lt=1.0)

    model_config = ConfigDict(frozen=True, extra="forbid")


class GbmCVNNPricerConfig(BaseModel):
    """Frozen snapshot used for (de-)serialising a trainer."""

    cfg: BlackScholesConfig
    domain_bounds: dict[str, BoundSpec]
    cvnn: ComplexValuedModel
    optimizer_state: AdamOptimizerState | None = None
    global_step: int = 0
    sobol_skip: int = 0
    torch_cpu_rng_state: bytes | None = None
    torch_cuda_rng_states: list[bytes] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")


class StepMetrics(BaseModel):
    """Scalar diagnostics emitted at the end of every optimiser step."""

    step: int
    batch_time: float
    loss: float
    grad_norm: float
    lr: float
    optimizer: optim.Adam
    model: ComplexValuedModel

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")


class TrainingResult(BaseModel):
    """Immutable outcome of a training run.

    Note: blockchain_version field intentionally omitted as _commit_to_blockchain
    does not return the committed version. Users can query the store separately
    if they need the version information after training.
    """

    updated_config: GbmCVNNPricerConfig
    final_loss: float
    total_batches: int
    final_grad_norm: float

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")


# =============================================================================
#  Execution Context ADT
# =============================================================================


@dataclass(frozen=True)
class CudaExecutionContext:
    """CUDA execution context with streams and Monte Carlo engine.

    Encapsulates all GPU-specific resources required for training and inference.
    The GbmCVNNPricer is CUDA-only for training (enforced by train() method).
    """

    torch_stream: torch.cuda.Stream
    cupy_stream: CuPyStream
    mc_engine: BlackScholes
    device: Device

    def run_cvnn_inference(
        self, cvnn: ComplexValuedModel, real_in: torch.Tensor, imag_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute CVNN inference on GPU with stream synchronization."""
        with torch.cuda.stream(self.torch_stream):
            pred_r, pred_i = cvnn(real_in, imag_in)
        self.torch_stream.synchronize()
        return pred_r, pred_i

    def simulate_fft(
        self,
        contract: BlackScholes.Inputs,
        sim_params: SimulationParams,
        cupy_cdtype: cp.dtype,
    ) -> cp.ndarray:
        """Run Monte Carlo simulation and return batch-mean FFT."""
        with self.cupy_stream:
            prices = self.mc_engine.price(inputs=contract).put_price
            mat = prices.reshape(sim_params.batches_per_mc_run, sim_params.network_size)
            result = cp.mean(cp.fft.fft(mat, axis=1), axis=0)
        self.cupy_stream.synchronize()
        return result


# =============================================================================
#  TensorBoard logger
# =============================================================================


class TensorBoardLogger:
    """Light-weight TensorBoard logger for :class:`StepMetrics`."""

    def __init__(
        self,
        *,
        logdir: str = ".logs/gbm_trainer",
        hist_every: int = 10,
        flush_every: int = 100,
    ) -> None:
        self._writer = SummaryWriter(log_dir=logdir)
        self._hist_every = max(hist_every, 1)
        self._flush_every = max(flush_every, 1)

    # ------------------------------------------------------------------ #
    # Callable interface                                                 #
    # ------------------------------------------------------------------ #

    def __call__(self, metrics: StepMetrics) -> None:
        """Write one optimisation step to disk."""
        w = self._writer
        step = metrics.step

        w.add_scalar("Loss/train", metrics.loss, step)
        w.add_scalar("LR", metrics.lr, step)
        w.add_scalar("GradNorm", metrics.grad_norm, step)
        w.add_scalar("BatchTime", metrics.batch_time, step)

        if step % self._hist_every == 0:
            for name, param in metrics.model.named_parameters():
                w.add_histogram(name, param, step)
                if param.grad is not None:
                    w.add_histogram(f"{name}.grad", param.grad, step)

        if step % self._flush_every == 0:
            w.flush()

    def close(self) -> None:
        """Close the underlying :class:`~torch.utils.tensorboard.SummaryWriter`."""
        self._writer.close()


# =============================================================================
#  Trainer
# =============================================================================


class GbmCVNNPricer:
    """Coordinate MC pricing, FFT and CVNN optimisation."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: GbmCVNNPricerConfig) -> None:
        """Instantiate a trainer from its immutable configuration."""
        # User payload -------------------------------------------------- #
        self._sim_params: SimulationParams = cfg.cfg.sim_params
        self._cvnn: ComplexValuedModel = cfg.cvnn
        self._domain_bounds: dict[str, BoundSpec] = cfg.domain_bounds
        self._optimizer_state: AdamOptimizerState | None = cfg.optimizer_state
        self._global_step: int = cfg.global_step
        self._sobol_skip: int = cfg.sobol_skip

        self._device: Device
        dtype_any: AnyDType
        self._device, dtype_any = module_state_device_dtype(self._cvnn.state_dict())

        # GBM simulation requires full precision (not float16/bfloat16)
        assert isinstance(dtype_any, FullPrecisionDType), (
            f"CVNN must use full precision dtype (float32/64, complex64/128), " f"got {dtype_any}"
        )
        self._dtype: FullPrecisionDType = dtype_any

        assert (
            self._dtype.to_precision() == cfg.cfg.sim_params.dtype
        ), f"Error: gbm sim dtype {cfg.cfg.sim_params.dtype} does not match cvnn dtype {self._dtype}"

        # Complex dtypes & streams -------------------------------------- #
        self._complex_dtype: Precision = self._dtype.to_precision().to_complex()
        self._torch_cdtype: torch.dtype = DType.from_precision(self._complex_dtype).to_torch()
        self._cupy_cdtype: cp.dtype = self._complex_dtype.to_cupy()

        # Execution context (CUDA-only) --------------------------------- #
        if self._device != Device.cuda:
            raise RuntimeError(
                "GbmCVNNPricer requires CUDA. "
                f"Model is on device={self._device}, but CUDA is required for training."
            )

        self._context: CudaExecutionContext = CudaExecutionContext(
            torch_stream=torch.cuda.Stream(device=self._device.to_torch()),
            cupy_stream=CuPyStream(),
            mc_engine=BlackScholes(cfg.cfg),
            device=self._device,
        )
        self._sampler: SobolSampler[BlackScholes.Inputs] = SobolSampler(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self._domain_bounds,
            config=SobolConfig(
                seed=self._sim_params.mc_seed,
                skip=self._sobol_skip,
            ),
        )

        # Restore RNG states for deterministic reproducibility ----------- #
        if cfg.torch_cpu_rng_state is not None:
            torch.set_rng_state(
                torch.from_numpy(np.frombuffer(cfg.torch_cpu_rng_state, dtype=np.uint8).copy())
            )
        if cfg.torch_cuda_rng_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [
                    torch.from_numpy(np.frombuffer(state_bytes, dtype=np.uint8).copy())
                    for state_bytes in cfg.torch_cuda_rng_states
                ]
            )

    # ------------------------------------------------------------------ #
    # Checkpointing                                                      #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> GbmCVNNPricerConfig:
        """
        Return a fully-deterministic snapshot.

        Note
        ----
        The optimiser state is **moved to CPU** so that the resulting
        :class:`AdamOptimizerState` contains only CPU tensors - a hard
        requirement for the strict serialisation helper.
        """
        # No need to check CUDA - _context is always CudaExecutionContext
        # (enforced by __init__ which raises RuntimeError if not on CUDA)

        # Capture RNG states for reproducibility
        # NOTE: .cpu() calls are acceptable here per CPU/GPU policy - these are
        # serialization boundary conversions for checkpoint I/O, not compute
        # operations. The TensorTree API cannot be used because it raises on
        # no-op moves (CPU RNG state is already on CPU).
        torch_cpu_rng = torch.get_rng_state().cpu().numpy().tobytes()
        torch_cuda_rng: list[bytes] | None = (
            [state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else None
        )

        return GbmCVNNPricerConfig(
            cfg=self._context.mc_engine.snapshot(),
            domain_bounds=self._domain_bounds,
            cvnn=self._cvnn,
            optimizer_state=self._optimizer_state,
            global_step=self._global_step,
            sobol_skip=self._sobol_skip,
            torch_cpu_rng_state=torch_cpu_rng,
            torch_cuda_rng_states=torch_cuda_rng,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _simulate_fft(self, contract: BlackScholes.Inputs) -> cp.ndarray:
        """Simulate one contract and return the batch-mean FFT."""
        # No need to check - _context.mc_engine is always initialized
        prices = self._context.mc_engine.price(inputs=contract).put_price
        mat = prices.reshape(self._sim_params.batches_per_mc_run, self._sim_params.network_size)
        return cp.mean(cp.fft.fft(mat, axis=1), axis=0)

    def _torch_step(
        self,
        real_in: torch.Tensor,
        imag_in: torch.Tensor,
        targets: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> tuple[torch.Tensor, float]:
        """One forward/backward/optimiser step; returns ``(loss, grad_norm)``."""
        pred_r, pred_i = self._cvnn(real_in, imag_in)
        loss = nn.functional.mse_loss(pred_r, torch.real(targets)) + nn.functional.mse_loss(
            pred_i, torch.imag(targets)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self._cvnn.parameters(), float("inf")))
        return loss, grad_norm

    # ------------------------------------------------------------------ #
    # Effect builders                                                    #
    # ------------------------------------------------------------------ #

    def build_training_step_effects(
        self, batch_idx: int, config: TrainingConfig
    ) -> EffectSequence[list[object]]:
        """Build pure effect sequence describing a single training step.

        This method produces an immutable effect description that can be:
        - Inspected and tested without GPU hardware
        - Serialized for reproducibility tracking
        - Composed with other effects in larger workflows

        The actual execution happens when the interpreter processes these effects.

        Registry ID naming convention:
            - "normals_{batch_idx}" - Generated normal random matrix
            - "paths_{batch_idx}" - Simulated price paths
            - "fft_{batch_idx}" - FFT of price paths
            - "targets_{batch_idx}" - DLPack-transferred targets tensor
            - "pred_{batch_idx}" - Model predictions
            - "loss_{batch_idx}" - Computed loss tensor

        Args:
            batch_idx: Index of this batch within the training run.
            config: Training hyperparameters.

        Returns:
            EffectSequence describing: data generation → forward → backward → optimizer → sync.

        Example:
            >>> effects = trainer.build_training_step_effects(batch_idx=0, config=config)
            >>> # Pure description - no side effects yet
            >>> result = await interpreter.interpret_sequence(effects)
        """
        mc_snapshot = self._context.mc_engine.snapshot()
        ngen_snapshot = self._context.mc_engine._ngen.snapshot()

        # Compute current sobol skip for this batch
        current_skip = ngen_snapshot.skips + (batch_idx * config.batch_size)

        # Note: Market parameters (spot, rate, dividend, vol, expiry) come from
        # Sobol-sampled contracts at runtime. The effect description uses placeholder
        # values that will be overridden by the interpreter when processing actual
        # contracts from the sampler.
        return sequence_effects(
            # Phase 1: Monte Carlo data generation
            GenerateNormals(
                rows=mc_snapshot.sim_params.timesteps,
                cols=mc_snapshot.sim_params.total_paths(),
                seed=ngen_snapshot.seed,
                skip=current_skip,
                output_tensor_id=f"normals_{batch_idx}",
            ),
            # Phase 2: Simulate GBM paths using Numba kernel
            # Market params are placeholders; real values come from Sobol sampler
            SimulatePaths(
                spot=100.0,
                rate=0.05,
                dividend=0.0,
                vol=0.2,
                expiry=1.0,
                timesteps=mc_snapshot.sim_params.timesteps,
                batches=mc_snapshot.sim_params.total_paths(),
                simulate_log_return=mc_snapshot.simulate_log_return,
                normalize_forwards=mc_snapshot.normalize_forwards,
                input_normals_id=f"normals_{batch_idx}",
                output_tensor_id=f"paths_{batch_idx}",
            ),
            StreamSync(stream_type="cupy"),
            # Phase 3: FFT of price paths
            ComputeFFT(
                input_tensor_id=f"paths_{batch_idx}",
                axis=1,
                output_tensor_id=f"fft_{batch_idx}",
            ),
            # Phase 4: Transfer FFT result from CuPy to PyTorch via DLPack
            DLPackTransfer(
                source_tensor_id=f"fft_{batch_idx}",
                source_framework="cupy",
                target_framework="torch",
                output_tensor_id=f"targets_{batch_idx}",
            ),
            StreamSync(stream_type="torch"),
            # Phase 5: Training step (forward/backward/optimizer)
            ForwardPass(
                model_id="cvnn",
                input_tensor_id=f"batch_{batch_idx}",
                output_tensor_id=f"pred_{batch_idx}",
            ),
            ComputeLoss(
                pred_tensor_id=f"pred_{batch_idx}",
                target_tensor_id=f"targets_{batch_idx}",
                loss_type="mse",
                output_tensor_id=f"loss_{batch_idx}",
            ),
            BackwardPass(loss_tensor_id=f"loss_{batch_idx}"),
            OptimizerStep(optimizer_id="adam"),
            StreamSync(stream_type="torch"),
            # Phase 6: Update metadata for tracking
            UpdateMetadata(key="global_step", operation="increment"),
            UpdateMetadata(key="sobol_skip", operation="add", value=config.batch_size),
            # Phase 7: Log metrics (optional, uses registered TensorBoard writer)
            LogMetrics(
                metrics=(),  # Populated by interpreter from loss value
                step=batch_idx,
            ),
            # Phase 8: RNG state capture for reproducibility
            CaptureRNGState(rng_type="torch_cuda", output_id=f"rng_state_{batch_idx}"),
        )

    def build_epoch_effects(self, config: TrainingConfig) -> EffectSequence[list[object]]:
        """Build pure effect sequence describing a complete training epoch.

        An epoch consists of config.num_batches training steps executed sequentially.

        Args:
            config: Training hyperparameters.

        Returns:
            EffectSequence containing all training steps for one epoch.
        """
        step_effects = [
            effect
            for i in range(config.num_batches)
            for effect in self.build_training_step_effects(i, config).effects
        ]
        return sequence_effects(*step_effects)

    def build_training_effects(
        self,
        config: TrainingConfig,
        *,
        num_epochs: int = 1,
        checkpoint_bucket: str | None = None,
        checkpoint_interval: int | None = None,
    ) -> EffectSequence[list[object]]:
        """Build pure effect sequence describing a complete training run.

        This method produces an immutable effect description representing
        the entire training process as a data structure. The training run
        includes optional periodic checkpointing.

        Args:
            config: Training hyperparameters.
            num_epochs: Number of training epochs to run.
            checkpoint_bucket: S3 bucket for checkpoints (enables checkpointing if set).
            checkpoint_interval: Checkpoint every N epochs (requires checkpoint_bucket).

        Returns:
            EffectSequence describing the complete training run.

        Example:
            >>> effects = trainer.build_training_effects(
            ...     config,
            ...     num_epochs=10,
            ...     checkpoint_bucket="my-models",
            ...     checkpoint_interval=5,
            ... )
            >>> # Pure description of entire training - no side effects yet
            >>> result = await interpreter.interpret_sequence(effects)
        """
        all_effects: list[object] = []

        for epoch in range(num_epochs):
            # Add epoch training effects
            epoch_effects = self.build_epoch_effects(config)
            all_effects.extend(epoch_effects.effects)

            # Add checkpoint effects if configured
            if (
                checkpoint_bucket is not None
                and checkpoint_interval is not None
                and (epoch + 1) % checkpoint_interval == 0
            ):
                checkpoint_effects = [
                    CaptureRNGState(rng_type="torch_cuda"),
                    WriteObject(
                        bucket=checkpoint_bucket,
                        key=f"epoch_{epoch + 1}/checkpoint.pb",
                    ),
                    CommitVersion(
                        parent_counter=epoch,
                        message=f"Epoch {epoch + 1}",
                    ),
                ]
                all_effects.extend(checkpoint_effects)

        # Type safety: sequence_effects expects Effect, but we've collected objects
        # Cast back to the proper tuple for the EffectSequence
        from spectralmc.effects.types import Effect

        typed_effects: list[Effect] = []
        for e in all_effects:
            if isinstance(
                e,
                (
                    GenerateNormals,
                    SimulatePaths,
                    ComputeFFT,
                    DLPackTransfer,
                    StreamSync,
                    ForwardPass,
                    BackwardPass,
                    OptimizerStep,
                    ComputeLoss,
                    LogMetrics,
                    UpdateMetadata,
                    ReadMetadata,
                    CaptureRNGState,
                    WriteObject,
                    CommitVersion,
                ),
            ):
                typed_effects.append(e)

        return sequence_effects(*typed_effects)

    # ------------------------------------------------------------------ #
    # Blockchain storage integration                                     #
    # ------------------------------------------------------------------ #

    def _commit_to_blockchain(
        self,
        blockchain_store: AsyncBlockchainModelStore,
        adam: optim.Optimizer,
        commit_message_template: str,
        loss: float,
        batch: int,
    ) -> None:
        """
        Commit current trainer snapshot to blockchain storage.

        Executes async commit synchronously using asyncio.run().
        Handles errors gracefully to avoid interrupting training.

        Args:
            blockchain_store: AsyncBlockchainModelStore instance
            adam: Current Adam optimizer (for snapshotting state)
            commit_message_template: Template string for commit message
            loss: Current training loss
            batch: Current batch/step number
        """
        try:
            # Snapshot optimizer state before committing
            # NOTE: Explicit .cpu() calls acceptable per CPU/GPU policy for checkpoint I/O.
            # The TensorTree API cannot be used here due to type narrowing constraints
            # with optimizer state_dict's complex nested type structure.
            state_dict = adam.state_dict()
            for param_id in state_dict["state"]:
                for key in state_dict["state"][param_id]:
                    val = state_dict["state"][param_id][key]
                    if isinstance(val, torch.Tensor):
                        state_dict["state"][param_id][key] = val.cpu()
            self._optimizer_state = AdamOptimizerState.from_torch(state_dict)

            # Create snapshot
            snapshot = self.snapshot()

            # Format commit message
            commit_message = commit_message_template.format(
                step=self._global_step,
                loss=loss,
                batch=batch,
            )

            # Execute async commit synchronously
            async def _do_commit() -> None:
                # NOTE: Function-level import required to break circular dependency:
                # storage.__init__ -> inference.py -> gbm_trainer.py -> storage
                # This is an acceptable exception per coding_standards.md.
                from spectralmc.storage import commit_snapshot

                version = await commit_snapshot(
                    blockchain_store,
                    snapshot,
                    commit_message,
                )
                _logger.info(
                    f"Committed version {version.counter}: {version.content_hash[:8]}... "
                    f"(step={self._global_step}, loss={loss:.6f})"
                )

            # Handle both sync and async contexts
            try:
                # Check if we're already in an event loop
                asyncio.get_running_loop()
                # We're in an event loop - cannot use asyncio.run(), skip commit
                _logger.warning(
                    f"Skipping blockchain commit at step {self._global_step}: "
                    "Cannot commit from async context (event loop already running). "
                    "Call commit_snapshot() manually after training completes."
                )
            except RuntimeError:
                # No event loop running - safe to use asyncio.run()
                asyncio.run(_do_commit())

        except (CommitError, NotFastForwardError, ConflictError, StorageError) as e:
            _logger.error(
                f"Failed to commit to blockchain at step {self._global_step}: {e}",
                exc_info=True,
            )
            # Don't raise - training should continue even if commit fails

    async def _async_commit_to_blockchain(
        self,
        blockchain_store: AsyncBlockchainModelStore,
        adam: optim.Optimizer,
        commit_message_template: str,
        loss: float,
        batch: int,
    ) -> None:
        """
        Async commit current trainer snapshot to blockchain storage.

        Native async version of _commit_to_blockchain for use in train_via_effects().
        Handles errors gracefully to avoid interrupting training.

        Args:
            blockchain_store: AsyncBlockchainModelStore instance
            adam: Current Adam optimizer (for snapshotting state)
            commit_message_template: Template string for commit message
            loss: Current training loss
            batch: Current batch/step number
        """
        try:
            # Snapshot optimizer state before committing
            # NOTE: Explicit .cpu() calls acceptable per CPU/GPU policy for checkpoint I/O.
            state_dict = adam.state_dict()
            for param_id in state_dict["state"]:
                for key in state_dict["state"][param_id]:
                    val = state_dict["state"][param_id][key]
                    if isinstance(val, torch.Tensor):
                        state_dict["state"][param_id][key] = val.cpu()
            self._optimizer_state = AdamOptimizerState.from_torch(state_dict)

            # Create snapshot
            snapshot = self.snapshot()

            # Format commit message
            commit_message = commit_message_template.format(
                step=self._global_step,
                loss=loss,
                batch=batch,
            )

            # Execute async commit
            # NOTE: Function-level import required to break circular dependency:
            # storage.__init__ -> inference.py -> gbm_trainer.py -> storage
            from spectralmc.storage import commit_snapshot

            version = await commit_snapshot(
                blockchain_store,
                snapshot,
                commit_message,
            )
            _logger.info(
                f"Committed version {version.counter}: {version.content_hash[:8]}... "
                f"(step={self._global_step}, loss={loss:.6f})"
            )

        except (CommitError, NotFastForwardError, ConflictError, StorageError) as e:
            _logger.error(
                f"Failed to commit to blockchain at step {self._global_step}: {e}",
                exc_info=True,
            )
            # Don't raise - training should continue even if commit fails

    # ------------------------------------------------------------------ #
    # Public API: training                                               #
    # ------------------------------------------------------------------ #

    def train(
        self,
        config: TrainingConfig,
        *,
        logger: StepLogger | None = None,
        blockchain_store: AsyncBlockchainModelStore | None = None,
        auto_commit: bool = False,
        commit_interval: int | None = None,
        commit_message_template: str = "Training checkpoint at step {step}",
    ) -> TrainingResult:
        """
        Run **CUDA-only** optimisation for *config.num_batches* steps.

        Args:
            config: Training hyperparameters (num_batches, batch_size, learning_rate)
            logger: Optional callback executed after each step
            blockchain_store: Optional AsyncBlockchainModelStore for automatic commits
            auto_commit: If True, commit snapshot after training completes (requires blockchain_store)
            commit_interval: If set, commit every N batches during training (requires blockchain_store)
            commit_message_template: Template for commit messages (can use {step}, {loss}, {batch})

        Returns:
            TrainingResult with updated config and training metrics

        Note:
            Blockchain commits are executed synchronously within the training loop using asyncio.run().
            This may add latency; for production, consider committing in a separate process/thread.
        """
        # No need to check CUDA - _context guarantees we're on CUDA
        # (enforced by __init__ which raises RuntimeError if not on CUDA)

        if (auto_commit or commit_interval is not None) and blockchain_store is None:
            raise ValueError(
                "auto_commit or commit_interval requires blockchain_store to be provided"
            )

        # Track state in local variables (functional approach)
        current_sobol_skip = self._sobol_skip
        current_global_step = self._global_step
        final_loss = 0.0
        final_grad_norm = 0.0

        adam = optim.Adam(self._cvnn.parameters(), lr=config.learning_rate)

        # (re-)attach previous optimiser state -------------------------- #
        if self._optimizer_state is not None:
            adam.load_state_dict(self._optimizer_state.to_torch())

        self._cvnn.train()

        # main loop ------------------------------------------------------ #
        for _ in range(config.num_batches):
            tic = time.perf_counter()

            # ── Monte-Carlo + FFT (CuPy) ─────────────────────────────── #
            sobol_inputs = self._sampler.sample(config.batch_size)
            current_sobol_skip += config.batch_size
            with self._context.cupy_stream:
                fft_buf = cp.asarray(
                    [self._simulate_fft(c) for c in sobol_inputs],
                    dtype=self._cupy_cdtype,
                )
            self._context.cupy_stream.synchronize()

            # ── CVNN step (Torch) ────────────────────────────────────── #
            with torch.cuda.stream(self._context.torch_stream):
                # torch.from_dlpack() is the modern, non-deprecated API (replaces torch.utils.dlpack.from_dlpack)
                # Type stub added in stubs/torch/__init__.pyi
                targets = torch.from_dlpack(fft_buf).to(self._torch_cdtype).detach()
                real_in, imag_in = _split_inputs(
                    sobol_inputs,
                    dtype=self._dtype.to_torch(),
                    device=self._device.to_torch(),
                )
                loss, grad_norm = self._torch_step(real_in, imag_in, targets, adam)
            self._context.torch_stream.synchronize()

            # ── Logging ──────────────────────────────────────────────── #
            final_loss = loss.item()
            final_grad_norm = grad_norm
            if logger is not None:
                logger(
                    StepMetrics(
                        step=current_global_step,
                        batch_time=time.perf_counter() - tic,
                        loss=final_loss,
                        grad_norm=final_grad_norm,
                        lr=float(adam.param_groups[0]["lr"]),
                        optimizer=adam,
                        model=self._cvnn,
                    )
                )
            current_global_step += 1

            # ── Periodic blockchain commit ───────────────────────────── #
            if (
                blockchain_store is not None
                and commit_interval is not None
                and current_global_step % commit_interval == 0
            ):
                _logger.info(f"Periodic commit at step {current_global_step}")
                # Temporarily sync self._* for snapshot
                self._global_step = current_global_step
                self._sobol_skip = current_sobol_skip
                self._commit_to_blockchain(
                    blockchain_store,
                    adam,
                    commit_message_template,
                    final_loss,
                    batch=current_global_step,
                )

        # ── Snapshot optimiser      ─────────────────────────────────── #
        # NOTE: Explicit .cpu() calls acceptable per CPU/GPU policy for checkpoint I/O.
        # The TensorTree API cannot be used here due to type narrowing constraints
        # with optimizer state_dict's complex nested type structure.
        state_dict = adam.state_dict()
        for param_id in state_dict["state"]:
            for key in state_dict["state"][param_id]:
                val = state_dict["state"][param_id][key]
                if isinstance(val, torch.Tensor):
                    state_dict["state"][param_id][key] = val.cpu()
        final_optimizer_state = AdamOptimizerState.from_torch(state_dict)

        # Update self._* with final values for snapshot
        self._optimizer_state = final_optimizer_state
        self._global_step = current_global_step
        self._sobol_skip = current_sobol_skip

        # ── Final blockchain commit ─────────────────────────────────── #
        if blockchain_store is not None and auto_commit:
            _logger.info(f"Final commit after training at step {current_global_step}")
            self._commit_to_blockchain(
                blockchain_store,
                adam,
                commit_message_template,
                final_loss,
                batch=config.num_batches,
            )

        # ── Return immutable training result ────────────────────────── #
        updated_config = self.snapshot()
        return TrainingResult(
            updated_config=updated_config,
            final_loss=final_loss,
            total_batches=config.num_batches,
            final_grad_norm=final_grad_norm,
        )

    async def train_via_effects(
        self,
        config: TrainingConfig,
        *,
        logger: StepLogger | None = None,
        blockchain_store: AsyncBlockchainModelStore | None = None,
        auto_commit: bool = False,
        commit_interval: int | None = None,
        commit_message_template: str = "Training checkpoint at step {step}",
    ) -> TrainingResult:
        """
        Run effect-driven training for *config.num_batches* steps.

        This method executes training through the Effect Interpreter pattern,
        where all side effects are described as pure effect ADTs and then
        interpreted by the SpectralMCInterpreter.

        Benefits:
        - Reproducibility: Complete execution trace as immutable data
        - Testability: Effect sequences can be inspected/tested without GPU
        - Composability: Effects compose without coupling to execution details

        Args:
            config: Training hyperparameters (num_batches, batch_size, learning_rate)
            logger: Optional callback executed after each step
            blockchain_store: Optional AsyncBlockchainModelStore for automatic commits
            auto_commit: If True, commit snapshot after training completes (requires blockchain_store)
            commit_interval: If set, commit every N batches during training (requires blockchain_store)
            commit_message_template: Template for commit messages (can use {step}, {loss}, {batch})

        Returns:
            TrainingResult with updated config and training metrics

        Example:
            >>> result = await trainer.train_via_effects(config)
            >>> print(f"Final loss: {result.final_loss:.6f}")
        """
        from spectralmc.effects import (
            SpectralMCInterpreter,
        )
        from spectralmc.result import Failure, Success

        # Validate blockchain parameters
        if (auto_commit or commit_interval is not None) and blockchain_store is None:
            raise ValueError(
                "auto_commit or commit_interval requires blockchain_store to be provided"
            )

        # Create interpreter with shared registry using factory method
        interpreter = SpectralMCInterpreter.create(
            torch_stream=self._context.torch_stream,
            cupy_stream=self._context.cupy_stream,
            storage_bucket="",  # Not used for training
        )

        # Get shared registry for pre-registration
        registry = interpreter.registry

        # Register model and optimizer
        adam = optim.Adam(self._cvnn.parameters(), lr=config.learning_rate)
        if self._optimizer_state is not None:
            adam.load_state_dict(self._optimizer_state.to_torch())

        registry.register_model("cvnn", self._cvnn)
        registry.register_optimizer("adam", adam)

        # Initialize metadata
        registry.register_metadata("global_step", self._global_step)
        registry.register_metadata("sobol_skip", self._sobol_skip)

        self._cvnn.train()

        # Track final metrics
        final_loss = 0.0
        final_grad_norm = 0.0

        # Execute training steps via effects
        for batch_idx in range(config.num_batches):
            tic = time.perf_counter()

            # Build effect sequence for this step
            effects = self.build_training_step_effects(batch_idx, config)

            # Execute via interpreter
            result = await interpreter.interpret_sequence(effects)

            match result:
                case Failure(error):
                    raise RuntimeError(f"Effect execution failed: {error}")
                case Success(_):
                    pass

            # Get loss from shared registry for logging
            loss_result = registry.get_tensor(f"loss_{batch_idx}")
            match loss_result:
                case Success(loss_tensor) if hasattr(loss_tensor, "item"):
                    final_loss = float(loss_tensor.item())
                case _:
                    pass  # No loss tensor found

            # Compute grad norm
            final_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self._cvnn.parameters(), float("inf"))
            )

            # Logging
            current_global_step = self._global_step + batch_idx + 1
            if logger is not None:
                logger(
                    StepMetrics(
                        step=self._global_step + batch_idx,
                        batch_time=time.perf_counter() - tic,
                        loss=final_loss,
                        grad_norm=final_grad_norm,
                        lr=float(adam.param_groups[0]["lr"]),
                        optimizer=adam,
                        model=self._cvnn,
                    )
                )

            # Periodic blockchain commit
            if (
                blockchain_store is not None
                and commit_interval is not None
                and current_global_step % commit_interval == 0
            ):
                _logger.info(f"Periodic commit at step {current_global_step}")
                await self._async_commit_to_blockchain(
                    blockchain_store,
                    adam,
                    commit_message_template,
                    final_loss,
                    batch=current_global_step,
                )

        # Update internal state from shared registry metadata
        global_step_result = registry.get_metadata("global_step")
        sobol_skip_result = registry.get_metadata("sobol_skip")

        match global_step_result:
            case Success(val) if isinstance(val, int):
                self._global_step = val
            case _:
                pass

        match sobol_skip_result:
            case Success(val) if isinstance(val, int):
                self._sobol_skip = val
            case _:
                pass

        # Snapshot optimizer state
        # NOTE: Explicit .cpu() calls acceptable per CPU/GPU policy for checkpoint I/O.
        state_dict = adam.state_dict()
        for param_id in state_dict["state"]:
            for key in state_dict["state"][param_id]:
                state_val = state_dict["state"][param_id][key]
                if isinstance(state_val, torch.Tensor):
                    state_dict["state"][param_id][key] = state_val.cpu()
        self._optimizer_state = AdamOptimizerState.from_torch(state_dict)

        # Final blockchain commit
        if blockchain_store is not None and auto_commit:
            final_global_step = self._global_step + config.num_batches
            _logger.info(f"Final commit after training at step {final_global_step}")
            await self._async_commit_to_blockchain(
                blockchain_store,
                adam,
                commit_message_template,
                final_loss,
                batch=config.num_batches,
            )

        # Return immutable training result
        updated_config = self.snapshot()
        return TrainingResult(
            updated_config=updated_config,
            final_loss=final_loss,
            total_batches=config.num_batches,
            final_grad_norm=final_grad_norm,
        )

    # ------------------------------------------------------------------ #
    # Public API: inference                                              #
    # ------------------------------------------------------------------ #

    def predict_price(
        self, inputs: Sequence[BlackScholes.Inputs]
    ) -> list[BlackScholes.HostPricingResults]:
        """Vectorised valuation of plain-vanilla European options."""
        if not inputs:
            return []

        self._cvnn.eval()
        real_in, imag_in = _split_inputs(
            inputs, dtype=self._dtype.to_torch(), device=self._device.to_torch()
        )

        # Always use CUDA stream - _context guarantees it exists
        with torch.cuda.stream(self._context.torch_stream):
            pred_r, pred_i = self._cvnn(real_in, imag_in)
        self._context.torch_stream.synchronize()

        spectrum = torch.view_as_complex(torch.stack((pred_r, pred_i), dim=-1))
        avg_ifft = torch.fft.ifft(spectrum, dim=1).mean(dim=1)

        results: list[BlackScholes.HostPricingResults] = []
        for coeff, contract in zip(avg_ifft, inputs, strict=True):
            real_val = float(torch.real(coeff).item())
            imag_val = float(torch.imag(coeff).item())

            # Option prices should be real-valued. Large imaginary components
            # indicate poor CVNN training or numerical instability.
            # Filtered in tests (pyproject.toml) but active in production.
            if abs(imag_val) > 1.0e-6:
                warnings.warn(
                    f"IFFT imaginary component {imag_val:.3e} exceeds tolerance.",
                    RuntimeWarning,
                )

            discount = math.exp(-contract.r * contract.T)
            forward = contract.X0 * math.exp((contract.r - contract.d) * contract.T)
            put_price = real_val
            call_price = put_price + forward - contract.K * discount

            results.append(
                BlackScholes.HostPricingResults(
                    underlying=forward,
                    put_price=put_price,
                    call_price=call_price,
                    put_price_intrinsic=discount * max(contract.K - forward, 0.0),
                    call_price_intrinsic=discount * max(forward - contract.K, 0.0),
                    put_convexity=put_price - discount * max(contract.K - forward, 0.0),
                    call_convexity=call_price - discount * max(forward - contract.K, 0.0),
                )
            )
        return results


# =============================================================================
#  Pure helpers
# =============================================================================


def _split_inputs(
    inputs: Sequence[BlackScholes.Inputs], *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Pydantic input contracts into ``(real, imag)`` tensors."""
    fields = list(BlackScholes.Inputs.model_fields.keys())
    rows = [[float(getattr(inp, f)) for f in fields] for inp in inputs]
    real = torch.tensor(rows, dtype=dtype, device=device)
    imag = torch.zeros_like(real)
    return real, imag
