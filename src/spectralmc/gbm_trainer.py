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
* Strict typing is enforced with **mypy --strict**; no `Any`, `cast`, or
  `type: ignore` are used.
"""

from __future__ import annotations

import asyncio
import math
import time
import warnings
from collections import deque
from dataclasses import dataclass
from itertools import chain, starmap
from typing import (
    Callable,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

import cupy as cp
import numpy as np
from cupy.cuda import Stream as CuPyStream
from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError

import torch
from torch.utils.tensorboard import SummaryWriter
from spectralmc.runtime import get_torch_handle
from spectralmc.models.torch import (
    AdamOptimizerState,
    AnyDType,
    Device,
    FullPrecisionDType,
    build_adam_optimizer_state,
)

from spectralmc.effects import (
    BackwardPass,
    CaptureRNGState,
    CommitVersion,
    ComputeFFT,
    ComputeLoss,
    DLPackTransfer,
    dlpack_transfer,
    EffectSequence,
    ForwardPass,
    GenerateNormals,
    LogMessage,
    LogMetrics,
    LoggingInterpreter,
    OptimizerStep,
    ReadMetadata,
    SimulatePaths,
    SpectralMCInterpreter,
    StreamSync,
    UpdateMetadata,
    WriteObject,
    sequence_effects,
)
from spectralmc.effects.types import Effect
from spectralmc.errors.gbm import CudaRNGUnavailable, NormalsGenerationFailed, NormalsUnavailable
from spectralmc.errors.serialization import SerializationError
from spectralmc.errors.torch_facade import TorchFacadeError
from spectralmc.errors.trainer import (
    InvalidTrainerConfig,
    InvalidTrainingConfig,
    OptimizerStateSerializationFailed,
    PredictionFailed,
    SamplerInitFailed,
    TrainerError,
)
from spectralmc.gbm import BlackScholes, BlackScholesConfig, SimulationParams
from spectralmc.models.cpu_gpu_transfer import module_state_device_dtype
from spectralmc.models.numerical import Precision
from spectralmc.result import Failure, Result, Success, collect_results, fold_results
from spectralmc.serialization import compute_sha256
from spectralmc.serialization.tensors import ModelCheckpointConverter
from spectralmc.sobol_sampler import DomainBounds, SobolSampler, build_sobol_config
from spectralmc.storage.errors import (
    CommitError,
    ConflictError,
    NotFastForwardError,
    StorageError,
)
from spectralmc.storage.store import AsyncBlockchainModelStore
from spectralmc.validation import validate_model

get_torch_handle()
nn = torch.nn
optim = torch.optim

LOGGER_NAME = __name__
LogLevel = Literal["debug", "info", "warning", "error", "critical"]

S = TypeVar("S")


def _consume_log_task_exception(task: asyncio.Task[object]) -> None:
    """Consume logging task exceptions with specific handling.

    Handles expected exceptions from optional TensorBoard logging.
    Logging failures are intentionally non-fatal and should not crash training.
    """
    try:
        task.exception()
    except asyncio.CancelledError:
        # Expected during shutdown
        return
    except (ImportError, RuntimeError, IOError, OSError):
        # Expected exceptions from TensorBoard operations
        # Silently consume - logging is optional and should not crash training
        return


__all__: tuple[str, ...] = (
    "GbmCVNNPricerConfig",
    "StepMetrics",
    "TrainingResult",
    "CudaExecutionContext",
    "TensorBoardLogger",
    "GbmCVNNPricer",
    "TrainingConfig",
    "build_training_config",
    "NoCommit",
    "FinalCommit",
    "IntervalCommit",
    "FinalAndIntervalCommit",
    "CommitPlan",
)

# =============================================================================
# Constants
# =============================================================================


# Commit plan ADT to avoid boolean toggles for blockchain effects
@dataclass(frozen=True)
class NoCommit:
    kind: Literal["NoCommit"] = "NoCommit"


@dataclass(frozen=True)
class FinalCommit:
    kind: Literal["FinalCommit"] = "FinalCommit"
    commit_message_template: str = "Training checkpoint at step {step}"


@dataclass(frozen=True)
class IntervalCommit:
    interval: PositiveInt
    kind: Literal["IntervalCommit"] = "IntervalCommit"
    commit_message_template: str = "Training checkpoint at step {step}"


@dataclass(frozen=True)
class FinalAndIntervalCommit:
    interval: PositiveInt
    kind: Literal["FinalAndIntervalCommit"] = "FinalAndIntervalCommit"
    commit_message_template: str = "Training checkpoint at step {step}"


CommitPlan = NoCommit | FinalCommit | IntervalCommit | FinalAndIntervalCommit

# Allowed Effect types for filtering in effect sequence building
_ALLOWED_EFFECT_TYPES: tuple[type[Effect], ...] = (
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
)


def _is_allowed_effect(e: object) -> TypeGuard[Effect]:
    """Type guard for allowed Effect types.

    Enables mypy type narrowing in list comprehensions where isinstance()
    alone doesn't narrow the type correctly.
    """
    return isinstance(e, _ALLOWED_EFFECT_TYPES)


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
    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None: ...
    def to(self, device: torch.device, dtype: torch.dtype) -> nn.Module: ...
    def train(self, mode: bool = True) -> None: ...
    def eval(self) -> None: ...


# =============================================================================
# Immutable dataclasses
# =============================================================================

StepLogger = Callable[["StepMetrics"], None]
"""User-supplied callback executed once after every optimiser step."""


@dataclass(frozen=True)
class TrainingConfig:
    """Validated training hyperparameters."""

    num_batches: int
    batch_size: int
    learning_rate: float


def build_training_config(
    *, num_batches: int, batch_size: int, learning_rate: float
) -> Result[TrainingConfig, InvalidTrainingConfig]:
    """Validate and construct TrainingConfig."""
    if num_batches <= 0:
        return Failure(
            InvalidTrainingConfig(
                num_batches=num_batches,
                batch_size=batch_size,
                learning_rate=learning_rate,
                message="num_batches must be > 0",
            )
        )
    if batch_size <= 0:
        return Failure(
            InvalidTrainingConfig(
                num_batches=num_batches,
                batch_size=batch_size,
                learning_rate=learning_rate,
                message="batch_size must be > 0",
            )
        )
    if not (0.0 < learning_rate < 1.0):
        return Failure(
            InvalidTrainingConfig(
                num_batches=num_batches,
                batch_size=batch_size,
                learning_rate=learning_rate,
                message="learning_rate must be in (0, 1)",
            )
        )
    return Success(
        TrainingConfig(
            num_batches=num_batches,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    )


class GbmCVNNPricerConfig(BaseModel):
    """Frozen snapshot used for (de-)serialising a trainer."""

    cfg: BlackScholesConfig
    domain_bounds: DomainBounds[BlackScholes.Inputs]
    cvnn: ComplexValuedModel
    optimizer_state: AdamOptimizerState | None = None
    global_step: int = 0
    sobol_skip: int = 0
    torch_cpu_rng_state: bytes | None = None
    torch_cuda_rng_states: list[bytes] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")


@dataclass(frozen=True)
class _BatchState:
    """Immutable state for training batch processing."""

    sobol_skip: int
    global_step: int
    loss: float
    grad_norm: float

    @staticmethod
    def initial(sobol_skip: int, global_step: int) -> _BatchState:
        """Create initial batch state."""
        return _BatchState(
            sobol_skip=sobol_skip,
            global_step=global_step,
            loss=0.0,
            grad_norm=0.0,
        )


@dataclass(frozen=True)
class StepMetrics:
    """Scalar diagnostics emitted at the end of every optimiser step."""

    step: int
    batch_time: float
    loss: float
    grad_norm: float
    lr: float
    optimizer: optim.Optimizer
    model: ComplexValuedModel


@dataclass(frozen=True)
class TrainingResult:
    """Immutable outcome of a training run.

    Note: blockchain_version field intentionally omitted as _commit_to_blockchain
    does not return the committed version. Users can query the store separately
    if they need the version information after training.
    """

    updated_config: GbmCVNNPricerConfig
    final_loss: float
    total_batches: int
    final_grad_norm: float


def build_gbm_cvnn_pricer_config(
    **kwargs: object,
) -> Result[GbmCVNNPricerConfig, ValidationError]:
    return validate_model(GbmCVNNPricerConfig, **kwargs)


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
    ) -> Result[cp.ndarray, NormalsUnavailable | NormalsGenerationFailed]:
        """Run Monte Carlo simulation and return batch-mean FFT."""
        with self.cupy_stream:
            match self.mc_engine.price(inputs=contract):
                case Failure(error):
                    return Failure(error)
                case Success(pricing):
                    mat = pricing.put_price.reshape(
                        sim_params.batches_per_mc_run, sim_params.network_size
                    )
                    result = cp.mean(cp.fft.fft(mat, axis=1), axis=0)
        self.cupy_stream.synchronize()
        return Success(result)


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
        """Write one optimisation step to disk.

        Pure: Uses conditional expressions for policy checks.
        """
        w = self._writer
        step = metrics.step

        # Always log scalar metrics
        w.add_scalar("Loss/train", metrics.loss, step)
        w.add_scalar("LR", metrics.lr, step)
        w.add_scalar("GradNorm", metrics.grad_norm, step)
        w.add_scalar("BatchTime", metrics.batch_time, step)

        # Conditional histogram logging (pure: match on boolean predicate)
        match step % self._hist_every == 0:
            case True:
                self._log_histograms(w, metrics.model, step)
            case False:
                pass

        # Conditional flush (pure: match on boolean predicate)
        match step % self._flush_every == 0:
            case True:
                w.flush()
            case False:
                pass

    def _log_histograms(
        self,
        writer: SummaryWriter,
        model: ComplexValuedModel,
        step: int,
    ) -> None:
        """Write parameter and gradient histograms to TensorBoard.

        Pure: Extracted helper to enable conditional expression in __call__().
        """
        param_pairs = list(model.named_parameters())
        self._write_parameter_histograms(writer, param_pairs, step)
        self._write_gradient_histograms(writer, param_pairs, step)

    @staticmethod
    def _write_parameter_histograms(
        writer: SummaryWriter,
        param_pairs: list[tuple[str, nn.Parameter]],
        step: int,
    ) -> None:
        """Write parameter histograms to TensorBoard.

        Pure: Uses deque to consume generator for side effects without building list.
        """
        deque(
            starmap(lambda name, param: writer.add_histogram(name, param, step), param_pairs),
            maxlen=0,
        )

    @staticmethod
    def _write_gradient_histograms(
        writer: SummaryWriter,
        param_pairs: list[tuple[str, nn.Parameter]],
        step: int,
    ) -> None:
        """Write gradient histograms to TensorBoard.

        Pure: Uses deque to consume filtered generator for side effects.
        """
        deque(
            starmap(
                lambda name, grad: writer.add_histogram(f"{name}.grad", grad, step),
                ((name, param.grad) for name, param in param_pairs if param.grad is not None),
            ),
            maxlen=0,
        )

    def close(self) -> None:
        """Close the underlying :class:`~torch.utils.tensorboard.SummaryWriter`."""
        self._writer.close()


# =============================================================================
#  Trainer Error Types
# =============================================================================


@dataclass(frozen=True)
class DeviceDTypeError:
    """Failed to determine device/dtype from model state."""

    kind: Literal["DeviceDTypeError"] = "DeviceDTypeError"
    message: str = ""
    underlying_error: object | None = None


@dataclass(frozen=True)
class DeviceNotCUDA:
    """Model is not on CUDA device but CUDA is required."""

    kind: Literal["DeviceNotCUDA"] = "DeviceNotCUDA"
    device: Device = Device.cpu
    message: str = "GbmCVNNPricer requires CUDA device"


@dataclass(frozen=True)
class CudaUnavailableForRNGRestore:
    """Checkpoint contains CUDA RNG state but CUDA is not available."""

    kind: Literal["CudaUnavailableForRNGRestore"] = "CudaUnavailableForRNGRestore"
    message: str = (
        "Cannot restore CUDA RNG state: CUDA not available but checkpoint contains CUDA RNG state"
    )


# Union type for pricer creation errors
GbmPricerError = DeviceDTypeError | DeviceNotCUDA | CudaUnavailableForRNGRestore


@dataclass(frozen=True)
class CudaEnvAvailable:
    """CUDA environment is usable with at least one device."""

    device_count: int


@dataclass(frozen=True)
class CudaEnvMissing:
    """CUDA environment unavailable or unusable."""

    reason: str


CudaEnv = CudaEnvAvailable | CudaEnvMissing


def _cuda_env() -> CudaEnv:
    """Total CUDA environment probe that forbids silent CPU fallbacks."""
    match torch.cuda.is_available():
        case False:
            return CudaEnvMissing(reason="cuda_unavailable")
        case True:
            count = torch.cuda.device_count()
            return (
                CudaEnvAvailable(device_count=count)
                if count > 0
                else CudaEnvMissing(reason="no_cuda_devices")
            )


# =============================================================================
#  Trainer
# =============================================================================


class GbmCVNNPricer:
    """Coordinate MC pricing, FFT and CVNN optimisation."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def create(cfg: GbmCVNNPricerConfig) -> Result[GbmCVNNPricer, GbmPricerError]:
        """Create GbmCVNNPricer with validation.

        This is the ONLY way to create a GbmCVNNPricer. The constructor is private.

        Args:
            cfg: Configuration for the pricer

        Returns:
            Success[GbmCVNNPricer] if validation passes
            Failure[GbmPricerError] if device validation fails

        Example:
            >>> match GbmCVNNPricer.create(config):
            ...     case Success(pricer):
            ...         # Use pricer
            ...     case Failure(error):
            ...         # Handle error
        """
        # Step 1: Get device/dtype from model
        device_dtype_result = module_state_device_dtype(cfg.cvnn.state_dict())
        match device_dtype_result:
            case Failure(dtype_err):
                return Failure(
                    DeviceDTypeError(
                        message=f"Failed to get device/dtype from CVNN: {dtype_err}",
                        underlying_error=dtype_err,
                    )
                )
            case Success((device, dtype)):
                pass  # Continue validation

        # Step 2: Validate CUDA requirement
        match device:
            case Device.cuda:
                pass  # Valid device
            case other_device:
                return Failure(
                    DeviceNotCUDA(
                        device=other_device,
                        message=f"Model on {other_device}, but CUDA required for training",
                    )
                )

        # Step 3: Validate CUDA availability for RNG restoration
        match cfg.torch_cuda_rng_states:
            case None:
                pass  # No CUDA RNG state to restore, validation passes
            case _:
                match _cuda_env():
                    case CudaEnvMissing():
                        return Failure(
                            CudaUnavailableForRNGRestore(
                                message="Checkpoint contains CUDA RNG state but CUDA is not available"
                            )
                        )
                    case CudaEnvAvailable():
                        pass  # CUDA available, validation passes

        # Step 4: Create instance (all validation passed) - call private constructor
        pricer = GbmCVNNPricer.__new__(GbmCVNNPricer)
        pricer._initialize(cfg, device, dtype)
        return Success(pricer)

    def _initialize(self, cfg: GbmCVNNPricerConfig, device: Device, dtype: AnyDType) -> None:
        """Private initialization after validation. Called by create() factory."""
        # User payload -------------------------------------------------- #
        self._sim_params: SimulationParams = cfg.cfg.sim_params
        self._cvnn: ComplexValuedModel = cfg.cvnn
        self._domain_bounds: DomainBounds[BlackScholes.Inputs] = cfg.domain_bounds
        self._optimizer_state: AdamOptimizerState | None = cfg.optimizer_state
        self._global_step: int = cfg.global_step
        self._sobol_skip: int = cfg.sobol_skip

        # Device and dtype (validated by factory, passed as parameters)
        self._device: Device = device

        # GBM simulation requires full precision (not float16/bfloat16)
        assert isinstance(dtype, FullPrecisionDType), (
            f"CVNN must use full precision dtype (float32/64, complex64/128), " f"got {dtype}"
        )
        self._dtype: FullPrecisionDType = dtype

        assert (
            self._dtype.to_precision() == cfg.cfg.sim_params.dtype
        ), f"Error: gbm sim dtype {cfg.cfg.sim_params.dtype} does not match cvnn dtype {self._dtype}"

        # Complex dtypes & streams -------------------------------------- #
        self._complex_dtype: Precision = self._dtype.to_precision().to_complex().unwrap()
        self._torch_cdtype: torch.dtype = FullPrecisionDType.from_precision(
            self._complex_dtype
        ).to_torch()
        self._cupy_cdtype: cp.dtype = self._complex_dtype.to_cupy()

        # Execution context (CUDA-only, validated by factory) ----------- #

        self._context: CudaExecutionContext = CudaExecutionContext(
            torch_stream=torch.cuda.Stream(device=self._device.to_torch()),
            cupy_stream=CuPyStream(),
            mc_engine=BlackScholes(cfg.cfg),
            device=self._device,
        )
        sobol_cfg = build_sobol_config(
            seed=self._sim_params.mc_seed, skip=self._sobol_skip
        ).unwrap()
        sampler_result = SobolSampler.create(
            pydantic_class=BlackScholes.Inputs,
            dimensions=self._domain_bounds,
            config=sobol_cfg,
        )
        self._sampler_result = sampler_result

        # Restore RNG states for deterministic reproducibility ----------- #
        match cfg.torch_cpu_rng_state:
            case None:
                pass  # No CPU RNG state to restore
            case cpu_state:
                torch.set_rng_state(
                    torch.from_numpy(np.frombuffer(cpu_state, dtype=np.uint8).copy())
                )

        match cfg.torch_cuda_rng_states:
            case None:
                pass  # No CUDA RNG state to restore
            case cuda_states:
                # Factory validation guarantees CUDA is available if we reach here
                env = _cuda_env()
                assert isinstance(
                    env, CudaEnvAvailable
                ), f"CUDA RNG restore invariant violated: {_cuda_env()!r}"
                count = env.device_count
                assert count == len(cuda_states), (
                    f"CUDA RNG state count ({len(cuda_states)}) does not match "
                    f"device count ({count})"
                )
                torch.cuda.set_rng_state_all(
                    list(
                        map(
                            lambda state_bytes: torch.from_numpy(
                                np.frombuffer(state_bytes, dtype=np.uint8).copy()
                            ),
                            cuda_states,
                        )
                    )
                )

        # Dedicated logging interpreter (side effects executed outside business logic)
        self._logging_interpreter: LoggingInterpreter = LoggingInterpreter(
            default_logger_name=LOGGER_NAME
        )

    # ------------------------------------------------------------------ #
    # Checkpointing                                                      #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Result[GbmCVNNPricerConfig, NormalsUnavailable]:
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
        match _cuda_env():
            case CudaEnvAvailable():
                torch_cuda_rng: list[bytes] = [
                    state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()
                ]
            case CudaEnvMissing(reason):
                return Failure(NormalsUnavailable(error=CudaRNGUnavailable(reason=reason)))

        match self._context.mc_engine.snapshot():
            case Failure(error):
                return Failure(error)
            case Success(cfg):
                match build_gbm_cvnn_pricer_config(
                    cfg=cfg,
                    domain_bounds=self._domain_bounds,
                    cvnn=self._cvnn,
                    optimizer_state=self._optimizer_state,
                    global_step=self._global_step,
                    sobol_skip=self._sobol_skip,
                    torch_cpu_rng_state=torch_cpu_rng,
                    torch_cuda_rng_states=torch_cuda_rng,
                ):
                    case Success(snapshot_cfg):
                        return Success(snapshot_cfg)
                    case Failure(err):
                        raise AssertionError(f"GbmCVNNPricerConfig validation failed: {err}")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _simulate_fft(
        self, contract: BlackScholes.Inputs
    ) -> Result[cp.ndarray, NormalsUnavailable | NormalsGenerationFailed]:
        """Simulate one contract and return the batch-mean FFT."""
        match self._context.mc_engine.price(inputs=contract):
            case Failure(error):
                return Failure(error)
            case Success(pricing):
                mat = pricing.put_price.reshape(
                    self._sim_params.batches_per_mc_run, self._sim_params.network_size
                )
                return Success(cp.mean(cp.fft.fft(mat, axis=1), axis=0))

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

    def _log_effect(
        self,
        message: str,
        *,
        level: LogLevel = "info",
        exc_info: bool = False,
    ) -> LogMessage:
        """Build a LogMessage effect with consistent logger name."""
        return LogMessage(
            level=level,
            message=message,
            logger_name=LOGGER_NAME,
            exc_info=exc_info,
        )

    def _log_sync_message(
        self,
        message: str,
        *,
        level: LogLevel = "info",
        exc_info: bool = False,
    ) -> None:
        """Execute a logging effect in synchronous contexts."""
        effect = self._log_effect(message, level=level, exc_info=exc_info)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._logging_interpreter.interpret(effect))
            return

        task = loop.create_task(self._logging_interpreter.interpret(effect))
        task.add_done_callback(_consume_log_task_exception)

    async def _log_async_internal(
        self,
        message: str,
        *,
        level: LogLevel = "info",
        exc_info: bool = False,
    ) -> None:
        """Execute a logging effect using the internal interpreter."""
        effect = self._log_effect(message, level=level, exc_info=exc_info)
        log_result = await self._logging_interpreter.interpret(effect)
        match log_result:
            case Failure(_):
                return
            case Success(_):
                return

    async def _log_async_with_interpreter(
        self,
        interpreter: SpectralMCInterpreter,
        message: str,
        *,
        level: LogLevel = "info",
        exc_info: bool = False,
    ) -> None:
        """Execute a logging effect via the master interpreter."""
        effect = self._log_effect(message, level=level, exc_info=exc_info)
        log_result = await interpreter.interpret(effect)
        match log_result:
            case Failure(_):
                return
            case Success(_):
                return

    # ------------------------------------------------------------------ #
    # Effect builders                                                    #
    # ------------------------------------------------------------------ #

    def build_training_step_effects(
        self, batch_idx: int, config: TrainingConfig
    ) -> Result[EffectSequence[list[object]], NormalsUnavailable]:
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
        match self._context.mc_engine.snapshot():
            case Failure(error):
                return Failure(error)
            case Success(mc_snapshot):
                pass
        match self._context.mc_engine._ngen_result:
            case Failure(ngen_err):
                return Failure(NormalsUnavailable(error=ngen_err))
            case Success(ngen):
                ngen_snapshot = ngen.snapshot()

        # Compute current sobol skip for this batch
        current_skip = ngen_snapshot.skips + (batch_idx * config.batch_size)

        # Note: Market parameters (spot, rate, dividend, vol, expiry) come from
        # Sobol-sampled contracts at runtime. The effect description uses placeholder
        # values that will be overridden by the interpreter when processing actual
        # contracts from the sampler.
        return Success(
            sequence_effects(
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
                    path_scheme=mc_snapshot.path_scheme,
                    normalization=mc_snapshot.normalization,
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
                # Note: Frameworks are always different ("cupy" -> "torch"), so .unwrap() is safe
                dlpack_transfer(
                    source_tensor_id=f"fft_{batch_idx}",
                    source_framework="cupy",
                    target_framework="torch",
                    output_tensor_id=f"targets_{batch_idx}",
                ).unwrap(),
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
        )

    def build_epoch_effects(
        self, config: TrainingConfig
    ) -> Result[EffectSequence[list[object]], NormalsUnavailable]:
        """Build pure effect sequence describing a complete training epoch.

        An epoch consists of config.num_batches training steps executed sequentially.

        Args:
            config: Training hyperparameters.

        Returns:
            EffectSequence containing all training steps for one epoch.
        """
        steps = [self.build_training_step_effects(i, config) for i in range(config.num_batches)]
        match collect_results(steps):
            case Failure(error):
                return Failure(error)
            case Success(step_sequences):
                step_effects = list(chain.from_iterable(seq.effects for seq in step_sequences))
                return Success(sequence_effects(*step_effects))

    def build_training_effects(
        self,
        config: TrainingConfig,
        *,
        num_epochs: int = 1,
        checkpoint_bucket: str | None = None,
        checkpoint_interval: int | None = None,
    ) -> Result[EffectSequence[list[object]], NormalsUnavailable]:
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
        checkpoint_builder: Callable[[int], list[Effect]] = (
            (lambda epoch: [])
            if checkpoint_bucket is None or checkpoint_interval is None
            else lambda epoch: (
                [
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
                if (epoch + 1) % checkpoint_interval == 0
                else []
            )
        )

        # Pure: build list of Results via comprehension instead of imperative loop
        # Avoid lambda to work around mypy generic type inference limitation
        epoch_effects_results: list[Result[list[object], NormalsUnavailable]] = [
            self._transform_epoch_effects(
                self.build_epoch_effects(config), checkpoint_builder(epoch)
            )
            for epoch in range(num_epochs)
        ]

        match collect_results(epoch_effects_results):
            case Failure(error):
                return Failure(error)
            case Success(epoch_effects_lists):
                all_effects: list[object] = list(chain.from_iterable(epoch_effects_lists))

        # Pure: filter effects using comprehension instead of imperative loop
        # TypeGuard enables mypy type narrowing in list comprehension
        typed_effects: list[Effect] = [e for e in all_effects if _is_allowed_effect(e)]

        return Success(sequence_effects(*typed_effects))

    @staticmethod
    def _transform_epoch_effects(
        epoch_result: Result[EffectSequence[list[object]], NormalsUnavailable],
        checkpoint_effects: list[Effect],
    ) -> Result[list[object], NormalsUnavailable]:
        """Transform epoch effects Result by combining with checkpoint effects.

        Extracted to helper function to avoid mypy lambda type inference limitation.

        Args:
            epoch_result: Result containing epoch effects sequence
            checkpoint_effects: Checkpoint effects to append

        Returns:
            Result containing combined list of effects (epoch + checkpoint)
        """
        match epoch_result:
            case Success(effects_seq):
                return Success([*effects_seq.effects, *checkpoint_effects])
            case Failure(error):
                return Failure(error)

    # ------------------------------------------------------------------ #
    # Blockchain storage integration                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialize_snapshot(
        snapshot: GbmCVNNPricerConfig,
    ) -> Result[tuple[bytes, str], SerializationError | TorchFacadeError]:
        """Serialize a snapshot to checkpoint bytes and content hash.

        Returns:
            Result containing (checkpoint_bytes, content_hash) or SerializationError/TorchFacadeError
        """
        # Use Result-wrapped factory for fallback case
        if snapshot.optimizer_state is not None:
            optimizer_state = snapshot.optimizer_state
        else:
            # Create empty optimizer state with Result pattern
            optimizer_state_result = build_adam_optimizer_state(param_states={}, param_groups=[])
            match optimizer_state_result:
                case Success(state):
                    optimizer_state = state
                case Failure(err):
                    # Empty state should never fail validation
                    raise RuntimeError(f"Failed to create empty optimizer state: {err}")
        checkpoint_result = ModelCheckpointConverter.to_proto(
            model_state_dict=snapshot.cvnn.state_dict(),
            optimizer_state=optimizer_state,
            torch_cpu_rng_state=snapshot.torch_cpu_rng_state or b"",
            torch_cuda_rng_states=snapshot.torch_cuda_rng_states or [],
            global_step=snapshot.global_step,
        )
        match checkpoint_result:
            case Failure(error):
                return Failure(error)
            case Success(checkpoint_proto):
                checkpoint_bytes = checkpoint_proto.SerializeToString()
                return Success((checkpoint_bytes, compute_sha256(checkpoint_bytes)))

    def _commit_to_blockchain(
        self,
        blockchain_store: AsyncBlockchainModelStore,
        adam: optim.Optimizer,
        commit_message_template: str,
        loss: float,
        batch: int,
        interpreter: SpectralMCInterpreter | None = None,
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
            state_dict["state"] = dict(
                map(
                    lambda item: (
                        item[0],
                        dict(
                            map(
                                lambda kv: (
                                    kv[0],
                                    kv[1].cpu() if isinstance(kv[1], torch.Tensor) else kv[1],
                                ),
                                item[1].items(),
                            )
                        ),
                    ),
                    state_dict["state"].items(),
                )
            )
            match AdamOptimizerState.from_torch(state_dict):
                case Failure(opt_err):
                    self._log_sync_message(
                        f"Skipping blockchain commit at step {self._global_step}: optimizer state serialization failed ({opt_err})",
                        level="error",
                        exc_info=True,
                    )
                    return
                case Success(state):
                    self._optimizer_state = state

            # Create snapshot
            snapshot_result = self.snapshot()
            match snapshot_result:
                case Failure(snap_err):
                    self._log_sync_message(
                        f"Skipping blockchain commit at step {self._global_step}: snapshot unavailable ({snap_err})",
                        level="error",
                    )
                    return
                case Success(snapshot):
                    pass

            # Format commit message
            commit_message = commit_message_template.format(
                step=self._global_step,
                loss=loss,
                batch=batch,
            )

            # Execute async commit synchronously
            async def _do_commit() -> None:
                match self._serialize_snapshot(snapshot):
                    case Failure(error):
                        await self._log_async_internal(
                            f"Serialization failed, skipping commit: {error}",
                            level="error",
                        )
                        return
                    case Success((checkpoint_bytes, content_hash)):
                        pass

                version = await blockchain_store.commit(
                    checkpoint_data=checkpoint_bytes,
                    content_hash=content_hash,
                    message=commit_message,
                )
                await self._log_async_internal(
                    f"Committed version {version.counter}: {version.content_hash[:8]}... "
                    f"(step={self._global_step}, loss={loss:.6f})",
                    level="info",
                )

            # Handle both sync and async contexts
            try:
                # Check if we're already in an event loop
                asyncio.get_running_loop()
                # We're in an event loop - cannot use asyncio.run(), skip commit
                self._log_sync_message(
                    (
                        f"Skipping blockchain commit at step {self._global_step}: "
                        "Cannot commit from async context (event loop already running). "
                        "Call AsyncBlockchainModelStore.commit() manually after training completes."
                    ),
                    level="warning",
                )
            except RuntimeError:
                # No event loop running - safe to use asyncio.run()
                asyncio.run(_do_commit())

        except (CommitError, NotFastForwardError, ConflictError, StorageError) as e:
            self._log_sync_message(
                f"Failed to commit to blockchain at step {self._global_step}: {e}",
                level="error",
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
            state_dict["state"] = dict(
                map(
                    lambda item: (
                        item[0],
                        dict(
                            map(
                                lambda kv: (
                                    kv[0],
                                    kv[1].cpu() if isinstance(kv[1], torch.Tensor) else kv[1],
                                ),
                                item[1].items(),
                            )
                        ),
                    ),
                    state_dict["state"].items(),
                )
            )
            match AdamOptimizerState.from_torch(state_dict):
                case Failure(opt_err):
                    await self._log_async_internal(
                        f"Skipping blockchain commit at step {self._global_step}: optimizer state serialization failed ({opt_err})",
                        level="error",
                        exc_info=True,
                    )
                    return
                case Success(state):
                    self._optimizer_state = state

            # Create snapshot
            snapshot_result = self.snapshot()
            match snapshot_result:
                case Failure(snap_err):
                    await self._log_async_internal(
                        f"Skipping blockchain commit at step {self._global_step}: snapshot unavailable ({snap_err})",
                        level="error",
                    )
                    return
                case Success(snapshot):
                    pass

            # Format commit message
            commit_message = commit_message_template.format(
                step=self._global_step,
                loss=loss,
                batch=batch,
            )

            # Serialize snapshot
            match self._serialize_snapshot(snapshot):
                case Failure(error):
                    await self._log_async_internal(
                        f"Serialization failed, skipping commit: {error}",
                        level="error",
                    )
                    return
                case Success((checkpoint_bytes, content_hash)):
                    pass

            version = await blockchain_store.commit(
                checkpoint_data=checkpoint_bytes,
                content_hash=content_hash,
                message=commit_message,
            )
            await self._log_async_internal(
                f"Committed version {version.counter}: {version.content_hash[:8]}... "
                f"(step={self._global_step}, loss={loss:.6f})",
                level="info",
            )

        except (CommitError, NotFastForwardError, ConflictError, StorageError) as e:
            await self._log_async_internal(
                f"Failed to commit to blockchain at step {self._global_step}: {e}",
                level="error",
                exc_info=True,
            )
            # Don't raise - training should continue even if commit fails

    # ------------------------------------------------------------------ #
    # Public API: training                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _maybe_log_metrics(
        logger: StepLogger | None,
        step: int,
        batch_time: float,
        loss: float,
        grad_norm: float,
        lr: float,
        optimizer: optim.Optimizer,
        model: ComplexValuedModel,
    ) -> None:
        """Log metrics if logger is provided. No-op if logger is None."""
        match logger:
            case None:
                pass
            case log_fn:
                log_fn(
                    StepMetrics(
                        step=step,
                        batch_time=batch_time,
                        loss=loss,
                        grad_norm=grad_norm,
                        lr=lr,
                        optimizer=optimizer,
                        model=model,
                    )
                )

    @staticmethod
    def _should_commit_now(
        blockchain_store: AsyncBlockchainModelStore | None,
        commit_plan: CommitPlan,
        global_step: int,
    ) -> tuple[bool, str]:
        """Return (should_commit, template) for periodic commits."""
        match commit_plan:
            case IntervalCommit(interval=interval, commit_message_template=template):
                return (blockchain_store is not None and global_step % interval == 0, template)
            case FinalAndIntervalCommit(interval=interval, commit_message_template=template):
                return (blockchain_store is not None and global_step % interval == 0, template)
            case NoCommit() | FinalCommit():
                return (False, "")

    @staticmethod
    def _should_commit_final(
        blockchain_store: AsyncBlockchainModelStore | None,
        commit_plan: CommitPlan,
    ) -> tuple[bool, str]:
        """Return (should_commit, template) for final commit."""
        match commit_plan:
            case FinalCommit(commit_message_template=template):
                return (blockchain_store is not None, template)
            case FinalAndIntervalCommit(commit_message_template=template):
                return (blockchain_store is not None, template)
            case NoCommit() | IntervalCommit():
                return (False, "")

    @staticmethod
    def _validate_commit_plan(plan: CommitPlan) -> Result[CommitPlan, InvalidTrainerConfig]:
        """Validate commit plan variants."""
        match plan:
            case NoCommit() | FinalCommit():
                return Success(plan)
            case IntervalCommit(interval=interval) | FinalAndIntervalCommit(interval=interval) if (
                interval <= 0
            ):
                return Failure(
                    InvalidTrainerConfig(
                        message="commit interval must be positive for blockchain commit plan"
                    )
                )
            case IntervalCommit() | FinalAndIntervalCommit():
                return Success(plan)

    def train(
        self,
        config: TrainingConfig,
        *,
        logger: StepLogger | None = None,
        blockchain_store: AsyncBlockchainModelStore | None = None,
        commit_plan: CommitPlan = NoCommit(),
    ) -> Result[TrainingResult, TrainerError]:
        """
        Run **CUDA-only** optimisation for *config.num_batches* steps.

        Args:
            config: Training hyperparameters (num_batches, batch_size, learning_rate)
            logger: Optional callback executed after each step
            blockchain_store: Optional AsyncBlockchainModelStore for automatic commits
            commit_plan: Explicit blockchain commit plan (FinalCommit, IntervalCommit, etc.)

        Returns:
            TrainingResult with updated config and training metrics

        Note:
            Blockchain commits are executed synchronously within the training loop using asyncio.run().
            This may add latency; for production, consider committing in a separate process/thread.
        """
        # No need to check CUDA - _context guarantees we're on CUDA
        # (enforced by __init__ which raises RuntimeError if not on CUDA)

        match self._validate_commit_plan(commit_plan):
            case Failure(plan_error):
                return Failure(plan_error)
            case Success(_):
                pass

        match commit_plan:
            case NoCommit():
                pass
            case _ if blockchain_store is None:
                return Failure(
                    InvalidTrainerConfig(
                        message="commit_plan requires blockchain_store to be provided"
                    )
                )
            case _:
                pass

        match self._sampler_result:
            case Failure(sampler_error):
                return Failure(SamplerInitFailed(error=sampler_error))
            case Success(_):
                pass

        # Track state in local variables (functional approach)
        current_sobol_skip = self._sobol_skip
        current_global_step = self._global_step
        final_loss = 0.0
        final_grad_norm = 0.0

        adam = optim.Adam(self._cvnn.parameters(), lr=config.learning_rate)

        # (re-)attach previous optimiser state -------------------------- #
        match self._optimizer_state:
            case None:
                pass  # No previous optimizer state to restore
            case optimizer_state:
                match optimizer_state.to_torch():
                    case Failure(opt_err):
                        return Failure(
                            OptimizerStateSerializationFailed(
                                message=f"Failed to deserialize optimizer state: {opt_err}"
                            )
                        )
                    case Success(state_dict):
                        adam.load_state_dict(state_dict)

        self._cvnn.train()

        def _run_batch(state: _BatchState, batch_idx: int) -> Result[_BatchState, TrainerError]:
            sobol_skip = state.sobol_skip
            global_step = state.global_step
            tic = time.perf_counter()

            match self._sampler_result:
                case Failure(sampler_err):
                    return Failure(SamplerInitFailed(error=sampler_err))
                case Success(sampler):
                    sample_result = sampler.sample(config.batch_size)
                    match sample_result:
                        case Success(sobol_inputs):
                            sobol_skip += config.batch_size
                        case Failure(sample_err):
                            return Failure(SamplerInitFailed(error=sample_err))

            fft_result = collect_results([self._simulate_fft(c) for c in sobol_inputs])
            match fft_result:
                case Failure(error):
                    return Failure(error)
                case Success(fft_values):
                    with self._context.cupy_stream:
                        fft_buf = cp.asarray(fft_values, dtype=self._cupy_cdtype)
                    self._context.cupy_stream.synchronize()

            with torch.cuda.stream(self._context.torch_stream):
                targets = torch.from_dlpack(fft_buf).to(self._torch_cdtype).detach()
                real_in, imag_in = _split_inputs(
                    sobol_inputs,
                    dtype=self._dtype.to_torch(),
                    device=self._device.to_torch(),
                )
                loss, grad_norm = self._torch_step(real_in, imag_in, targets, adam)
            self._context.torch_stream.synchronize()

            updated_loss = loss.item()
            updated_grad_norm = grad_norm
            self._maybe_log_metrics(
                logger=logger,
                step=global_step,
                batch_time=time.perf_counter() - tic,
                loss=updated_loss,
                grad_norm=updated_grad_norm,
                lr=float(adam.param_groups[0]["lr"]),
                optimizer=adam,
                model=self._cvnn,
            )
            global_step += 1

            # Periodic blockchain commits
            match self._should_commit_now(blockchain_store, commit_plan, global_step):
                case True, template:
                    self._log_sync_message(
                        f"Periodic commit at step {global_step}",
                        level="info",
                    )
                    self._global_step = global_step
                    self._sobol_skip = sobol_skip
                    # Type narrowing: blockchain_store guaranteed non-None by _should_commit_now
                    assert blockchain_store is not None
                    self._commit_to_blockchain(
                        blockchain_store,
                        adam,
                        template,
                        updated_loss,
                        batch=global_step,
                    )
                case False, _:
                    pass

            return Success(
                _BatchState(
                    sobol_skip=sobol_skip,
                    global_step=global_step,
                    loss=updated_loss,
                    grad_norm=updated_grad_norm,
                )
            )

        initial_state = _BatchState.initial(current_sobol_skip, current_global_step)
        final_state_result = fold_results(
            list(range(config.num_batches)),
            _run_batch,
            initial_state,
        )

        match final_state_result:
            case Failure(batch_err):
                return Failure(batch_err)
            case Success(final_state):
                current_sobol_skip = final_state.sobol_skip
                current_global_step = final_state.global_step
                final_loss = final_state.loss
                final_grad_norm = final_state.grad_norm

        # ── Snapshot optimiser      ─────────────────────────────────── #
        # NOTE: Explicit .cpu() calls acceptable per CPU/GPU policy for checkpoint I/O.
        # The TensorTree API cannot be used here due to type narrowing constraints
        # with optimizer state_dict's complex nested type structure.
        state_dict = adam.state_dict()
        state_dict["state"] = dict(
            map(
                lambda item: (
                    item[0],
                    dict(
                        map(
                            lambda kv: (
                                kv[0],
                                kv[1].cpu() if isinstance(kv[1], torch.Tensor) else kv[1],
                            ),
                            item[1].items(),
                        )
                    ),
                ),
                state_dict["state"].items(),
            )
        )
        match AdamOptimizerState.from_torch(state_dict):
            case Failure(final_opt_err):
                return Failure(
                    OptimizerStateSerializationFailed(
                        message=f"Failed to capture optimizer snapshot: {final_opt_err}"
                    )
                )
            case Success(state):
                final_optimizer_state = state

        # Update self._* with final values for snapshot
        self._optimizer_state = final_optimizer_state
        self._global_step = current_global_step
        self._sobol_skip = current_sobol_skip

        # ── Final blockchain commit ─────────────────────────────────── #
        match self._should_commit_final(blockchain_store, commit_plan):
            case True, template:
                self._log_sync_message(
                    f"Final commit after training at step {current_global_step}",
                    level="info",
                )
                # Type narrowing: blockchain_store guaranteed non-None by _should_commit_final
                assert blockchain_store is not None
                self._commit_to_blockchain(
                    blockchain_store,
                    adam,
                    template,
                    final_loss,
                    batch=config.num_batches,
                )
            case False, _:
                pass

        # ── Return immutable training result ────────────────────────── #
        updated_config = self.snapshot()
        match updated_config:
            case Failure(snapshot_err):
                return Failure(snapshot_err)
            case Success(cfg):
                return Success(
                    TrainingResult(
                        updated_config=cfg,
                        final_loss=final_loss,
                        total_batches=config.num_batches,
                        final_grad_norm=final_grad_norm,
                    )
                )

    async def train_via_effects(
        self,
        config: TrainingConfig,
        *,
        logger: StepLogger | None = None,
        blockchain_store: AsyncBlockchainModelStore | None = None,
        commit_plan: CommitPlan = NoCommit(),
    ) -> Result[TrainingResult, TrainerError]:
        """
        Effect-based training currently delegates to the synchronous implementation
        while the Result-based refactor is in progress.
        """
        return self.train(
            config,
            logger=logger,
            blockchain_store=blockchain_store,
            commit_plan=commit_plan,
        )

    # ------------------------------------------------------------------ #
    # Public API: inference                                              #
    # ------------------------------------------------------------------ #

    def predict_price(
        self, inputs: Sequence[BlackScholes.Inputs]
    ) -> Result[list[BlackScholes.HostPricingResults], TrainerError]:
        """Vectorised valuation of plain-vanilla European options (returns `Success`/`Failure`)."""
        match inputs:
            case [] | ():
                return Success([])
            case _:
                pass

        self._cvnn.eval()
        real_in, imag_in = _split_inputs(
            inputs, dtype=self._dtype.to_torch(), device=self._device.to_torch()
        )

        try:
            with torch.cuda.stream(self._context.torch_stream):
                pred_r, pred_i = self._cvnn(real_in, imag_in)
            self._context.torch_stream.synchronize()

            spectrum = torch.view_as_complex(torch.stack((pred_r, pred_i), dim=-1))
            avg_ifft = torch.fft.ifft(spectrum, dim=1).mean(dim=1)

            def _price_contract(
                entry: tuple[torch.Tensor, BlackScholes.Inputs],
            ) -> BlackScholes.HostPricingResults:
                coeff, contract = entry
                real_val = float(torch.real(coeff).item())
                imag_val = float(torch.imag(coeff).item())
                _ = (
                    warnings.warn(
                        f"IFFT imaginary component {imag_val:.3e} exceeds tolerance.",
                        RuntimeWarning,
                    )
                    if abs(imag_val) > 1.0e-6
                    else None
                )
                discount = math.exp(-contract.r * contract.T)
                forward = contract.X0 * math.exp((contract.r - contract.d) * contract.T)
                put_price = real_val
                call_price = put_price + forward - contract.K * discount
                match validate_model(
                    BlackScholes.HostPricingResults,
                    underlying=forward,
                    put_price=put_price,
                    call_price=call_price,
                    put_price_intrinsic=discount * max(contract.K - forward, 0.0),
                    call_price_intrinsic=discount * max(forward - contract.K, 0.0),
                    put_convexity=put_price - discount * max(contract.K - forward, 0.0),
                    call_convexity=call_price - discount * max(forward - contract.K, 0.0),
                ):
                    case Success(host):
                        return host
                    case Failure(err):
                        raise AssertionError(f"HostPricingResults validation failed: {err}")

            return Success(list(map(_price_contract, zip(avg_ifft, inputs, strict=True))))
        except Exception as exc:
            return Failure(PredictionFailed(message=str(exc)))


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
