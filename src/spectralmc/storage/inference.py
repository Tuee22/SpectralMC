# src/spectralmc/storage/inference.py
"""Inference client with version pinning and automatic tracking modes."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import torch
from spectralmc.runtime import get_torch_handle
from ..errors.storage import (
    StartError,
    ValidationError as StorageValidationError,
    VersionNotFoundError as StorageVersionNotFoundError,
)
from ..gbm_trainer import GbmCVNNPricerConfig
from ..result import Failure, Result, Success
from .chain import ModelVersion
from .checkpoint import load_snapshot_from_checkpoint
from .errors import StorageError, VersionNotFoundError
from .store import AsyncBlockchainModelStore


get_torch_handle()

logger = logging.getLogger(__name__)


# ============================================================================
# InferenceMode ADT
# ============================================================================


def pinned_mode(counter: int) -> Result[PinnedMode, StorageValidationError]:
    """Create PinnedMode with validation.

    Args:
        counter: Version counter (must be >= 0)

    Returns:
        Success with PinnedMode if valid, Failure with ValidationError if invalid
    """
    return (
        Failure(
            StorageValidationError(
                field="counter",
                value=counter,
                message=f"Version counter must be >= 0, got {counter}",
            )
        )
        if counter < 0
        else Success(PinnedMode(counter=counter))
    )


@dataclass(frozen=True)
class PinnedMode:
    """Pin to a specific version counter.

    Attributes:
        counter: Version counter to pin to (must be >= 0)
    """

    counter: int


@dataclass(frozen=True)
class TrackingMode:
    """Track latest version automatically with hot-swapping."""

    pass


# Union type for inference modes
InferenceMode = PinnedMode | TrackingMode


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for inference client.

    Attributes:
        mode: Inference mode (PinnedMode or TrackingMode)
        poll_interval: Seconds between polls in tracking mode
        store: Blockchain model store
    """

    mode: InferenceMode
    poll_interval: float
    store: AsyncBlockchainModelStore


class InferenceClient:
    """
    Inference client with pinned or tracking mode.

    **Pinned Mode** (PinnedMode):
    - Loads specific version on start
    - Never updates automatically
    - Use for production stability, A/B testing, reproducibility

    **Tracking Mode** (TrackingMode):
    - Loads latest version on start
    - Polls for updates every poll_interval seconds
    - Hot-swaps model atomically when new version available
    - Use for development, continuous learning systems

    Thread Safety:
    - Model reference swap is atomic (Python GIL)
    - Prediction operations are thread-safe
    - Multiple predict() calls can run concurrently

    Usage:
        ```python
        from spectralmc.storage import InferenceClient, pinned_mode, TrackingMode
        from spectralmc.result import Success, Failure

        # Pinned mode (production) - use factory function
        match pinned_mode(42):
            case Success(mode):
                client = InferenceClient(
                    mode=mode,
                    poll_interval=60.0,
                    store=store,
                    model_template=model,
                    config_template=config
                )
            case Failure(error):
                raise RuntimeError(f"Invalid version counter: {error.message}")

        async with client:
            model = client.get_model()
            # Run inference...

        # Tracking mode (development)
        client = InferenceClient(
            mode=TrackingMode(),
            poll_interval=30.0,
            store=store,
            model_template=model,
            config_template=config
        )

        async with client:
            # Model auto-updates every 30 seconds
            model = client.get_model()
            # Run inference...
        ```
    """

    def __init__(
        self,
        mode: InferenceMode,
        poll_interval: float,
        store: AsyncBlockchainModelStore,
        model_template: torch.nn.Module,
        config_template: GbmCVNNPricerConfig,
        max_consecutive_failures: int = 5,
    ) -> None:
        """
        Initialize inference client.

        Args:
            mode: InferenceMode (PinnedMode or TrackingMode)
            poll_interval: Seconds between version polls (tracking mode only)
            store: AsyncBlockchainModelStore instance
            model_template: Empty model for loading weights into
            config_template: Config template for snapshot loading
            max_consecutive_failures: Stop polling after this many failures (default: 5)
        """
        self.mode = mode
        self.poll_interval = poll_interval
        self.store = store
        self.model_template = model_template
        self.config_template = config_template
        self.max_consecutive_failures = max_consecutive_failures

        # Runtime state
        self._current_version: ModelVersion | None = None
        self._current_snapshot: GbmCVNNPricerConfig | None = None
        self._polling_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._consecutive_failures: int = 0

        match mode:
            case PinnedMode(counter):
                logger.info(
                    f"InferenceClient initialized: mode=pinned, "
                    f"version={counter}, poll_interval={poll_interval}s"
                )
            case TrackingMode():
                logger.info(
                    f"InferenceClient initialized: mode=tracking, "
                    f"poll_interval={poll_interval}s"
                )

    async def __aenter__(self) -> InferenceClient:
        """Enter async context manager, start client."""
        match await self.start():
            case Success(_):
                return self
            case Failure(error):
                raise RuntimeError(f"Failed to start InferenceClient: {error.message}")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Exit async context manager, stop client."""
        await self.stop()

    async def start(self) -> Result[None, StartError]:
        """
        Start inference client.

        - Loads initial model (specific version or latest)
        - Spawns background polling task if tracking mode

        Returns:
            Success(None) if started successfully
            Failure(StartError) if cannot start (e.g., empty store in tracking mode)
        """
        logger.info("Starting InferenceClient...")

        # Load initial model based on mode
        match self.mode:
            case PinnedMode(counter):
                # Pinned mode: load specific version
                match await self._fetch_version_by_counter(counter):
                    case Success(version):
                        logger.info(f"Pinned mode: loading version {counter}")
                    case Failure(error):
                        return Failure(
                            StartError(
                                message=f"Cannot start pinned mode: version {counter} not found",
                                underlying_error=error if isinstance(error, Exception) else None,
                            )
                        )

            case TrackingMode():
                # Tracking mode: load latest
                head_result = await self.store.get_head()
                match head_result:
                    case Success(version):
                        logger.info(f"Tracking mode: loading latest version {version.counter}")
                    case Failure():
                        return Failure(
                            StartError(
                                message="Cannot start tracking mode: no versions in store",
                                underlying_error=None,
                            )
                        )

        await self._load_version(version)

        # Start polling task in tracking mode
        match self.mode:
            case TrackingMode():
                self._shutdown_event.clear()
                self._polling_task = asyncio.create_task(self._poll_loop())
                logger.info(f"Started polling task (interval={self.poll_interval}s)")
            case PinnedMode(_):
                pass  # No polling in pinned mode

        logger.info(f"InferenceClient started with version {version.counter}")
        return Success(None)

    async def stop(self) -> None:
        """
        Stop inference client.

        - Cancels background polling task
        - Cleans up resources
        """
        logger.info("Stopping InferenceClient...")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel polling task
        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
            logger.info("Polling task stopped")

        logger.info("InferenceClient stopped")

    def get_model(self) -> GbmCVNNPricerConfig:
        """
        Get currently loaded model for inference.

        Thread-safe: Model reference is atomically swapped during updates.
        Use this to run predictions directly on the model.

        Returns:
            Current model (in eval mode)

        Raises:
            RuntimeError: If client not started

        Example:
            ```python
            model = client.get_model()
            model.eval()
            with torch.no_grad():
                # For complex-valued models:
                real_out, imag_out = model(real_in, imag_in)
            ```
        """
        if self._current_snapshot is None:
            raise RuntimeError("InferenceClient not started - call start() first")

        # Return snapshot (atomic reference)
        return self._current_snapshot

    def get_current_version(self) -> ModelVersion | None:
        """Get currently loaded version metadata."""
        return self._current_version

    async def _poll_loop(self) -> None:
        """Background task that polls for new versions (tracking mode only)."""
        logger.info("Polling loop started")

        while not self._shutdown_event.is_set():
            try:
                # Wait for poll interval or shutdown
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=self.poll_interval)
                    # Shutdown signaled
                    break
                except TimeoutError:
                    # Poll interval elapsed
                    pass

                # Fetch latest version
                head_result = await self.store.get_head()

                match head_result:
                    case Success(head):
                        # Reset failure counter on success
                        self._consecutive_failures = 0

                        # Check if newer version available
                        if self._current_version is None:
                            continue

                        if head.counter > self._current_version.counter:
                            logger.info(
                                f"New version detected: {head.counter} "
                                f"(current: {self._current_version.counter})"
                            )
                            await self._load_version(head)
                            logger.info(f"Hot-swapped to version {head.counter}")

                    case Failure(error):
                        self._consecutive_failures += 1
                        logger.warning(
                            f"Failed to fetch HEAD during poll (attempt {self._consecutive_failures}/{self.max_consecutive_failures}): {error}"
                        )

                        if self._consecutive_failures >= self.max_consecutive_failures:
                            logger.error(
                                f"Circuit breaker triggered: {self._consecutive_failures} consecutive failures. "
                                "Stopping polling loop."
                            )
                            break

            except (VersionNotFoundError, StorageError, OSError) as e:
                self._consecutive_failures += 1
                logger.error(
                    f"Unexpected error in polling loop (attempt {self._consecutive_failures}/{self.max_consecutive_failures}): {e}",
                    exc_info=True,
                )

                if self._consecutive_failures >= self.max_consecutive_failures:
                    logger.error(
                        f"Circuit breaker triggered: {self._consecutive_failures} consecutive failures. "
                        "Stopping polling loop."
                    )
                    break

        logger.info("Polling loop stopped")

    async def _load_version(self, version: ModelVersion) -> None:
        """
        Load version into current model.

        Atomic swap of model reference.

        Args:
            version: Version to load
        """
        logger.info(f"Loading version {version.counter}...")

        # Load snapshot from checkpoint
        snapshot_result = await load_snapshot_from_checkpoint(
            self.store, version, self.model_template, self.config_template
        )
        match snapshot_result:
            case Failure(load_err):
                raise RuntimeError(
                    f"Failed to load snapshot for version {version.counter}: {load_err}"
                )
            case Success(snapshot):
                pass

        # Atomic swap (Python GIL makes this thread-safe)
        self._current_version = version
        self._current_snapshot = snapshot

        logger.info(
            f"Loaded version {version.counter}: "
            f"global_step={snapshot.global_step}, "
            f"params={sum(p.numel() for p in snapshot.cvnn.parameters())}"
        )

    async def _fetch_version_by_counter(
        self, counter: int
    ) -> Result[ModelVersion, StorageVersionNotFoundError]:
        """
        Fetch specific version by counter.

        Args:
            counter: Version counter

        Returns:
            Success(ModelVersion) if found
            Failure(VersionNotFoundError) if not found
        """
        # Get HEAD to find version ID
        head_result = await self.store.get_head()

        match head_result:
            case Success(head):
                # If requesting HEAD, return it
                if counter == head.counter:
                    return Success(head)
            case Failure():
                return Failure(StorageVersionNotFoundError(counter=counter, available_versions=[]))

        # Otherwise, fetch by version ID
        # Version ID format: v{counter:010d}
        version_id = f"v{counter:010d}"
        version = await self.store.get_version(version_id)

        return Success(version)
