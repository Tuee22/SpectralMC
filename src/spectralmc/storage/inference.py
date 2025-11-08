# src/spectralmc/storage/inference.py
"""Inference client with version pinning and automatic tracking modes."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass

import torch

from .store import AsyncBlockchainModelStore
from .chain import ModelVersion
from .checkpoint import load_snapshot_from_checkpoint
from ..gbm_trainer import GbmCVNNPricerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for inference client.

    Attributes:
        version_counter: Specific version to pin to (None = tracking mode)
        poll_interval: Seconds between polls in tracking mode
        store: Blockchain model store
    """

    version_counter: Optional[int]
    poll_interval: float
    store: AsyncBlockchainModelStore


class InferenceClient:
    """
    Inference client with pinned or tracking mode.

    **Pinned Mode** (version_counter is set):
    - Loads specific version on start
    - Never updates automatically
    - Use for production stability, A/B testing, reproducibility

    **Tracking Mode** (version_counter is None):
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
        # Pinned mode (production)
        client = InferenceClient(
            version_counter=42,
            poll_interval=60.0,
            store=store,
            model_template=model,
            config_template=config
        )

        async with client:
            predictions = await client.predict(inputs)

        # Tracking mode (development)
        client = InferenceClient(
            version_counter=None,  # Auto-track latest
            poll_interval=30.0,
            store=store,
            model_template=model,
            config_template=config
        )

        async with client:
            # Model auto-updates every 30 seconds
            predictions = await client.predict(inputs)
        ```
    """

    def __init__(
        self,
        version_counter: Optional[int],
        poll_interval: float,
        store: AsyncBlockchainModelStore,
        model_template: torch.nn.Module,
        config_template: GbmCVNNPricerConfig,
    ) -> None:
        """
        Initialize inference client.

        Args:
            version_counter: Specific version to load (None = tracking mode)
            poll_interval: Seconds between version polls (tracking mode only)
            store: AsyncBlockchainModelStore instance
            model_template: Empty model for loading weights into
            config_template: Config template for snapshot loading
        """
        self.version_counter = version_counter
        self.poll_interval = poll_interval
        self.store = store
        self.model_template = model_template
        self.config_template = config_template

        # Runtime state
        self._current_version: Optional[ModelVersion] = None
        self._current_snapshot: Optional[GbmCVNNPricerConfig] = None
        self._polling_task: Optional[asyncio.Task[None]] = None
        self._shutdown_event = asyncio.Event()

        logger.info(
            f"InferenceClient initialized: mode={'pinned' if version_counter is not None else 'tracking'}, "
            f"version={version_counter}, poll_interval={poll_interval}s"
        )

    async def __aenter__(self) -> InferenceClient:
        """Enter async context manager, start client."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Exit async context manager, stop client."""
        await self.stop()

    async def start(self) -> None:
        """
        Start inference client.

        - Loads initial model (specific version or latest)
        - Spawns background polling task if tracking mode
        """
        logger.info("Starting InferenceClient...")

        # Load initial model
        if self.version_counter is not None:
            # Pinned mode: load specific version
            version = await self._fetch_version_by_counter(self.version_counter)
            logger.info(f"Pinned mode: loading version {self.version_counter}")
        else:
            # Tracking mode: load latest
            head = await self.store.get_head()
            if head is None:
                raise ValueError("Cannot start tracking mode: no versions in store")
            version = head
            logger.info(f"Tracking mode: loading latest version {version.counter}")

        await self._load_version(version)

        # Start polling task in tracking mode
        if self.version_counter is None:
            self._shutdown_event.clear()
            self._polling_task = asyncio.create_task(self._poll_loop())
            logger.info(f"Started polling task (interval={self.poll_interval}s)")

        logger.info(f"InferenceClient started with version {version.counter}")

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

    def get_current_version(self) -> Optional[ModelVersion]:
        """Get currently loaded version metadata."""
        return self._current_version

    async def _poll_loop(self) -> None:
        """Background task that polls for new versions (tracking mode only)."""
        logger.info("Polling loop started")

        while not self._shutdown_event.is_set():
            try:
                # Wait for poll interval or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self.poll_interval
                    )
                    # Shutdown signaled
                    break
                except asyncio.TimeoutError:
                    # Poll interval elapsed
                    pass

                # Fetch latest version
                head = await self.store.get_head()
                if head is None:
                    logger.warning("No HEAD version found during poll")
                    continue

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

            except Exception as e:
                logger.error(f"Error in polling loop: {e}", exc_info=True)
                # Continue polling despite errors

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
        snapshot = await load_snapshot_from_checkpoint(
            self.store, version, self.model_template, self.config_template
        )

        # Atomic swap (Python GIL makes this thread-safe)
        self._current_version = version
        self._current_snapshot = snapshot

        logger.info(
            f"Loaded version {version.counter}: "
            f"global_step={snapshot.global_step}, "
            f"params={sum(p.numel() for p in snapshot.cvnn.parameters())}"  # type: ignore[attr-defined]
        )

    async def _fetch_version_by_counter(self, counter: int) -> ModelVersion:
        """
        Fetch specific version by counter.

        Args:
            counter: Version counter

        Returns:
            ModelVersion

        Raises:
            ValueError: If version not found
        """
        # Get HEAD to find version ID
        head = await self.store.get_head()
        if head is None:
            raise ValueError(f"Version {counter} not found: store is empty")

        # If requesting HEAD, return it
        if counter == head.counter:
            return head

        # Otherwise, fetch by version ID
        # Version ID format: v{counter:010d}
        version_id = f"v{counter:010d}"
        version = await self.store.get_version(version_id)

        return version
