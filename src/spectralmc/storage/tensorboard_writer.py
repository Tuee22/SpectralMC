# src/spectralmc/storage/tensorboard_writer.py
"""TensorBoard logging for blockchain model versions.

Logs version metadata, training metrics, and model statistics to TensorBoard.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from spectralmc.runtime import get_torch_handle
from ..gbm_trainer import GbmCVNNPricerConfig
from ..result import Failure, Success
from .chain import ModelVersion
from .checkpoint import load_snapshot_from_checkpoint
from .errors import StorageError, VersionNotFoundError
from .store import AsyncBlockchainModelStore


get_torch_handle()

logger = logging.getLogger(__name__)


class TensorBoardWriter:
    """
    TensorBoard logger for blockchain model storage.

    Logs:
    - Version metadata (counter, semver, content_hash)
    - Training metrics (global_step, loss) from checkpoints
    - Model statistics (parameter count, checkpoint size)
    - Commit timeline

    Usage:
        ```python
        writer = TensorBoardWriter(
            store=store,
            log_dir="runs/blockchain_models"
        )

        # Log all existing versions
        await writer.log_all_versions(
            model_template=model,
            config_template=config
        )

        # Or log incrementally as new versions are committed
        await writer.log_version(version)

        writer.close()
        ```
    """

    def __init__(
        self, store: AsyncBlockchainModelStore, log_dir: str = "runs/blockchain_models"
    ) -> None:
        """
        Initialize TensorBoard writer.

        Args:
            store: AsyncBlockchainModelStore instance
            log_dir: TensorBoard log directory
        """
        self.store = store
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        logger.info(f"TensorBoardWriter initialized: log_dir={self.log_dir}")

    async def log_version(
        self,
        version: ModelVersion,
        model_template: torch.nn.Module | None = None,
        config_template: GbmCVNNPricerConfig | None = None,
    ) -> None:
        """
        Log a single version to TensorBoard.

        Args:
            version: ModelVersion to log
            model_template: Model template for loading checkpoint (optional)
            config_template: Config template for loading checkpoint (optional)

        Note:
            If model_template and config_template are provided, will also log
            training metrics from checkpoint. Otherwise, only logs version metadata.
        """
        counter = version.counter

        # Log version metadata
        self.writer.add_text(
            "version/semantic_version", version.semantic_version, global_step=counter
        )

        self.writer.add_text("version/content_hash", version.content_hash, global_step=counter)

        self.writer.add_text("version/commit_message", version.commit_message, global_step=counter)

        # Log timestamp as scalar (seconds since epoch)
        commit_time = datetime.fromisoformat(version.commit_timestamp)
        self.writer.add_scalar(
            "version/commit_timestamp", commit_time.timestamp(), global_step=counter
        )

        # If templates provided, load checkpoint and log metrics
        if model_template is not None and config_template is not None:
            try:
                snapshot_result = await load_snapshot_from_checkpoint(
                    self.store, version, model_template, config_template
                )
                match snapshot_result:
                    case Failure(load_err):
                        logger.warning(
                            f"Failed to load checkpoint for version {counter}: {load_err}"
                        )
                        return
                    case Success(snapshot):
                        pass

                # Log global_step from training
                self.writer.add_scalar(
                    "training/global_step", snapshot.global_step, global_step=counter
                )

                # Log model statistics
                param_count = sum(p.numel() for p in snapshot.cvnn.parameters())
                self.writer.add_scalar("model/param_count", param_count, global_step=counter)

                # Log Sobol skip
                self.writer.add_scalar(
                    "training/sobol_skip", snapshot.sobol_skip, global_step=counter
                )

                logger.info(f"Logged version {counter} with checkpoint metrics")

            except (VersionNotFoundError, StorageError, RuntimeError, OSError) as e:
                logger.warning(f"Failed to load checkpoint for version {counter}: {e}")
        else:
            logger.info(f"Logged version {counter} metadata only")

    async def log_all_versions(
        self,
        model_template: torch.nn.Module | None = None,
        config_template: GbmCVNNPricerConfig | None = None,
    ) -> None:
        """
        Log all versions in the blockchain to TensorBoard.

        Args:
            model_template: Model template for loading checkpoints (optional)
            config_template: Config template for loading checkpoints (optional)

        Note:
            This is useful for initializing TensorBoard with historical data.
            For large chains, this may take a while.
        """
        head_result = await self.store.get_head()

        match head_result:
            case Success(head):
                logger.info(f"Logging {head.counter + 1} versions to TensorBoard...")
            case Failure(_):
                logger.info("No versions to log (empty chain)")
                return

        for counter in range(head.counter + 1):
            version_id = f"v{counter:010d}"
            version = await self.store.get_version(version_id)

            await self.log_version(version, model_template, config_template)

        self.writer.flush()
        logger.info(f"Logged {head.counter + 1} versions to {self.log_dir}")

    async def log_summary_statistics(self) -> None:
        """
        Log high-level summary statistics about the blockchain.

        Includes:
        - Total version count
        - Total storage size estimate
        - Average versions per day
        """
        head_result = await self.store.get_head()

        match head_result:
            case Success(head):
                total_versions = head.counter + 1
            case Failure(_):
                return

        # Log total versions
        self.writer.add_scalar("summary/total_versions", total_versions, global_step=0)

        # Calculate time span
        if total_versions > 1:
            v0 = await self.store.get_version("v0000000000")
            v_head = head

            t0 = datetime.fromisoformat(v0.commit_timestamp)
            t_head = datetime.fromisoformat(v_head.commit_timestamp)

            days_elapsed = (t_head - t0).total_seconds() / 86400
            if days_elapsed > 0:
                versions_per_day = total_versions / days_elapsed
                self.writer.add_scalar("summary/versions_per_day", versions_per_day, global_step=0)

        self.writer.flush()
        logger.info("Logged summary statistics")

    def close(self) -> None:
        """Close the TensorBoard writer and flush remaining data."""
        self.writer.close()
        logger.info("TensorBoardWriter closed")

    def __enter__(self) -> TensorBoardWriter:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Context manager exit."""
        self.close()


async def log_blockchain_to_tensorboard(
    store: AsyncBlockchainModelStore,
    log_dir: str = "runs/blockchain_models",
    model_template: torch.nn.Module | None = None,
    config_template: GbmCVNNPricerConfig | None = None,
) -> None:
    """
    Convenience function to log entire blockchain to TensorBoard.

    Args:
        store: AsyncBlockchainModelStore instance
        log_dir: TensorBoard log directory
        model_template: Model template for loading checkpoints (optional)
        config_template: Config template for loading checkpoints (optional)

    Example:
        ```python
        async with AsyncBlockchainModelStore("bucket") as store:
            await log_blockchain_to_tensorboard(
                store,
                log_dir="runs/my_experiment",
                model_template=torch.nn.Linear(5, 5),
                config_template=config
            )
        ```

        Then view with:
        ```bash
        tensorboard --logdir=runs/
        ```
    """
    with TensorBoardWriter(store, log_dir) as writer:
        await writer.log_all_versions(model_template, config_template)
        await writer.log_summary_statistics()
