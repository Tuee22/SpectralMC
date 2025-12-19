# src/spectralmc/storage/gc.py
"""Garbage collection for blockchain model storage.

Removes old model versions while preserving chain integrity and recent checkpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..errors.storage import GCError
from ..result import Failure, Result, Success
from .chain import ModelVersion
from .store import AsyncBlockchainModelStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionPolicy:
    """Garbage collection retention policy.

    Attributes:
        keep_versions: Number of most recent versions to keep (None = keep all)
        keep_min_versions: Minimum versions to always keep (protects against accidents)
        protect_tags: Tuple of version counters to always protect (e.g., production releases)
    """

    keep_versions: int | None = None  # None = keep all
    keep_min_versions: int = 3  # Always keep at least this many
    protect_tags: tuple[int, ...] = field(
        default_factory=tuple
    )  # Always protect these version counters


@dataclass(frozen=True)
class GCReport:
    """Garbage collection execution report.

    Attributes:
        deleted_versions: Tuple of deleted version counters
        protected_versions: Tuple of protected version counters
        bytes_freed: Approximate bytes freed (sum of checkpoint sizes)
        dry_run: Whether this was a dry run
    """

    deleted_versions: tuple[int, ...]
    protected_versions: tuple[int, ...]
    bytes_freed: int
    dry_run: bool


@dataclass(frozen=True)
class PreviewGC:
    """Preview mode for garbage collection (no deletions)."""

    kind: str = "PreviewGC"


@dataclass(frozen=True)
class ExecuteGC:
    """Execution mode for garbage collection (perform deletions)."""

    kind: str = "ExecuteGC"


GCMode = PreviewGC | ExecuteGC


class GarbageCollector:
    """
    Garbage collector for blockchain model storage.

    Removes old versions according to retention policy while preserving:
    - Genesis version (v0) - always protected
    - Recent versions (configurable via policy)
    - Tagged/protected versions (e.g., production releases)
    - Chain integrity (never orphans versions)

    Supports explicit preview/execute modes instead of boolean toggles.

    Usage:
        ```python
        # Keep last 10 versions
        policy = RetentionPolicy(keep_versions=10)
        gc = GarbageCollector(store, policy)

        # Preview first
        report = await gc.collect(mode=PreviewGC())
        print(f"Would delete: {report.deleted_versions}")
        print(f"Would free: {report.bytes_freed} bytes")

        # Actually delete
        report = await gc.collect(mode=ExecuteGC())
        ```
    """

    def __init__(self, store: AsyncBlockchainModelStore, policy: RetentionPolicy) -> None:
        """
        Initialize garbage collector.

        Args:
            store: AsyncBlockchainModelStore instance
            policy: Retention policy
        """
        self.store = store
        self.policy = policy

    async def collect(self, mode: GCMode = PreviewGC()) -> Result[GCReport, GCError]:
        """
        Run garbage collection.

        Args:
            mode: Explicit GC mode (PreviewGC or ExecuteGC)

        Returns:
            Success(GCReport) with deletion summary
            Failure(GCError) if policy would violate minimum version requirement
        """
        dry_run = isinstance(mode, PreviewGC)

        # Fetch all versions
        head_result = await self.store.get_head()

        match head_result:
            case Failure(_):
                # Empty chain, nothing to collect
                return Success(
                    GCReport(
                        deleted_versions=(),
                        protected_versions=(),
                        bytes_freed=0,
                        dry_run=dry_run,
                    )
                )
            case Success(head):
                pass  # Continue with GC logic

        # Collect all version metadata
        versions: list[ModelVersion] = []
        for counter in range(head.counter + 1):
            version_id = f"v{counter:010d}"
            version = await self.store.get_version(version_id)
            versions.append(version)

        # Determine what to delete
        protected, to_delete = self._plan_deletion(versions)

        # Safety check: ensure minimum versions retained
        if len(protected) < self.policy.keep_min_versions:
            return Failure(
                GCError(
                    message="Retention policy would violate minimum version requirement",
                    protected_count=len(protected),
                    minimum_required=self.policy.keep_min_versions,
                )
            )

        # Calculate bytes to free
        dry_run = isinstance(mode, PreviewGC)
        bytes_freed = 0
        if not dry_run:
            # Actually delete versions
            for version in to_delete:
                freed = await self._delete_version(version)
                bytes_freed += freed
                logger.info(f"Deleted version {version.counter} (freed {freed} bytes)")
        else:
            # Dry run: estimate bytes
            for version in to_delete:
                freed = await self._estimate_version_size(version)
                bytes_freed += freed

        return Success(
            GCReport(
                deleted_versions=tuple(v.counter for v in to_delete),
                protected_versions=tuple(v.counter for v in protected),
                bytes_freed=bytes_freed,
                dry_run=dry_run,
            )
        )

    def _plan_deletion(
        self, versions: list[ModelVersion]
    ) -> tuple[list[ModelVersion], list[ModelVersion]]:
        """
        Plan which versions to delete and which to protect.

        Args:
            versions: All versions in chain (sorted by counter)

        Returns:
            Tuple of (protected_versions, to_delete_versions)
        """
        # Genesis (v0) is always protected
        protected = {0}

        # Protect tagged versions
        for tag in self.policy.protect_tags:
            protected.add(tag)

        # Protect recent versions
        if self.policy.keep_versions is not None:
            # Keep N most recent
            total_versions = len(versions)
            keep_from = max(0, total_versions - self.policy.keep_versions)
            for i in range(keep_from, total_versions):
                protected.add(versions[i].counter)
        else:
            # keep_versions=None means keep all
            protected.update(v.counter for v in versions)

        # Split into protected and to_delete
        protected_list = [v for v in versions if v.counter in protected]
        to_delete = [v for v in versions if v.counter not in protected]

        return protected_list, to_delete

    async def _delete_version(self, version: ModelVersion) -> int:
        """
        Delete a version's artifacts from S3.

        Args:
            version: Version to delete

        Returns:
            Approximate bytes freed
        """
        if self.store._s3_client is None:
            raise RuntimeError("S3 client not initialized. Use 'async with' context manager.")

        version_dir = version.directory_name
        prefix = f"versions/{version_dir}/"

        # List all objects in version directory
        total_size = 0
        objects_to_delete = []

        paginator = self.store._s3_client.get_paginator("list_objects_v2")
        # paginator.paginate returns an async iterator
        async for page in paginator.paginate(Bucket=self.store.bucket_name, Prefix=prefix):
            # Runtime validation of page structure
            if not isinstance(page, dict):
                continue
            if "Contents" not in page:
                continue

            contents = page["Contents"]
            if not isinstance(contents, list):
                continue

            for obj in contents:
                if not isinstance(obj, dict):
                    continue
                key = obj.get("Key")
                if isinstance(key, str):
                    objects_to_delete.append({"Key": key})
                size = obj.get("Size", 0)
                if isinstance(size, int):
                    total_size += size

        # Delete objects
        if objects_to_delete:
            await self.store._s3_client.delete_objects(
                Bucket=self.store.bucket_name, Delete={"Objects": objects_to_delete}
            )

        return total_size

    async def _estimate_version_size(self, version: ModelVersion) -> int:
        """
        Estimate version size without deleting.

        Args:
            version: Version to estimate

        Returns:
            Approximate size in bytes
        """
        if self.store._s3_client is None:
            raise RuntimeError("S3 client not initialized. Use 'async with' context manager.")

        version_dir = version.directory_name
        prefix = f"versions/{version_dir}/"

        total_size = 0
        paginator = self.store._s3_client.get_paginator("list_objects_v2")
        # paginator.paginate returns an async iterator
        async for page in paginator.paginate(Bucket=self.store.bucket_name, Prefix=prefix):
            # Runtime validation of page structure
            if not isinstance(page, dict):
                continue
            if "Contents" not in page:
                continue

            contents = page["Contents"]
            if not isinstance(contents, list):
                continue

            for obj in contents:
                if not isinstance(obj, dict):
                    continue
                size = obj.get("Size", 0)
                if isinstance(size, int):
                    total_size += size

        return total_size


async def run_gc(
    store: AsyncBlockchainModelStore,
    keep_versions: int | None = None,
    keep_min_versions: int = 3,
    protect_tags: list[int] | None = None,
    mode: GCMode = PreviewGC(),
) -> Result[GCReport, GCError]:
    """
    Convenience function to run garbage collection.

    Args:
        store: AsyncBlockchainModelStore instance
        keep_versions: Number of recent versions to keep (None = keep all)
        keep_min_versions: Minimum versions to always keep
        protect_tags: Version counters to always protect
        mode: Explicit GC mode (PreviewGC or ExecuteGC)

    Returns:
        Success(GCReport) with deletion summary
        Failure(GCError) if policy would violate minimum version requirement

    Example:
        ```python
        async with AsyncBlockchainModelStore("bucket") as store:
            # Dry run: preview deletions
            match await run_gc(store, keep_versions=10, mode=PreviewGC()):
                case Success(report):
                    print(f"Would delete: {len(report.deleted_versions)} versions")
                case Failure(error):
                    print(f"GC error: {error.message}")

            # Actually delete
            match await run_gc(store, keep_versions=10, mode=ExecuteGC()):
                case Success(report):
                    print(f"Deleted {len(report.deleted_versions)} versions")
                case Failure(error):
                    print(f"GC failed: {error.message}")
        ```
    """
    policy = RetentionPolicy(
        keep_versions=keep_versions,
        keep_min_versions=keep_min_versions,
        protect_tags=tuple(protect_tags) if protect_tags else (),
    )

    gc = GarbageCollector(store, policy)
    return await gc.collect(mode=mode)
