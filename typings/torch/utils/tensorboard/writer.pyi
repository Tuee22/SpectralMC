# torch/utils/tensorboard/writer.pyi
"""
Typed stub for :pymod:`torch.utils.tensorboard.writer`.

Only the public API needed by most projects is covered.  Everything is
fully typed (no ``Any``), so mypy will fail loudly if upstream signatures
change.
"""

from __future__ import annotations

from typing import (
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

Tensor: TypeAlias = torch.Tensor
"""Alias for a PyTorch tensor with unconstrained dtype and shape."""

NDArrayFloat: TypeAlias = NDArray[np.floating]
"""A NumPy array of floating-point values (dtype ``np.float_``)."""

Scalar = Union[int, float]
"""A real scalar supported by TensorBoard and PyTorch."""

class SummaryWriter:
    """Thin wrapper around TensorBoard’s event writer.

    The writer flushes events and summaries to *log_dir* so that
    TensorBoard can pick them up.

    Notes
    -----
    Only a subset of TensorBoard features is declared here—enough for
    scalar curves, images, histograms, graphs, embeddings, and
    hyper-parameters.  Extend as needed.
    """

    log_dir: str
    """Directory where event files are written."""

    # ──────────────────────────── Construction ────────────────────────────
    def __init__(
        self,
        log_dir: Optional[str] = ...,
        *,
        comment: str = ...,
        purge_step: Optional[int] = ...,
        max_queue: int = ...,
        flush_secs: int = ...,
        filename_suffix: str = ...,
        enable_flush_thread: bool = ...,
        write_to_disk: bool = ...,
        log_dir_suffix: str = ...,
    ) -> None:
        """Create a writer.

        Args:
            log_dir: Root directory for all event files.  If *None*,
                defaults to a timestamped sub-directory of
                ``runs/``.
            comment: Tag that is appended to the directory name.
            purge_step: First global step to *keep* – steps before this are
                discarded from the log on construction.
            max_queue: How many summary items to accumulate before writing
                to disk.
            flush_secs: How often, in seconds, to flush the queue.
            filename_suffix: Extra string appended to event-file names.
            enable_flush_thread: Whether to flush periodically in a
                background thread.
            write_to_disk: Disable to keep everything in memory (useful
                for tests).
            log_dir_suffix: Extra string appended *after* :pyattr:`comment`.
        """
    # ─────────────────────────────── Scalars ──────────────────────────────
    def add_scalar(
        self,
        tag: str,
        scalar_value: Scalar,
        global_step: Optional[int] = ...,
        *,
        walltime: Optional[float] = ...,
        new_style: bool = ...,
        double_precision: int = ...,
    ) -> None:
        """Log a single scalar value."""

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Mapping[str, Scalar],
        global_step: Optional[int] = ...,
        *,
        walltime: Optional[float] = ...,
    ) -> None:
        """Log several related scalar values in one call."""
    # ─────────────────────────────── Images ───────────────────────────────
    def add_image(
        self,
        tag: str,
        img_tensor: Union[Tensor, NDArrayFloat],
        global_step: Optional[int] = ...,
        *,
        walltime: Optional[float] = ...,
        dataformats: Literal["CHW", "HWC", "HW"] = ...,
    ) -> None:
        """Write a single image."""

    def add_images(
        self,
        tag: str,
        img_tensor: Union[Tensor, NDArrayFloat],
        global_step: Optional[int] = ...,
        *,
        walltime: Optional[float] = ...,
        dataformats: Literal["NCHW", "NHWC", "NHW"] = ...,
    ) -> None:
        """Write a batch of images."""
    # ─────────────────────────────── Text ────────────────────────────────
    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: Optional[int] = ...,
        *,
        walltime: Optional[float] = ...,
    ) -> None:
        """Add arbitrary markdown text."""
    # ──────────────────────────── Histograms ──────────────────────────────
    def add_histogram(
        self,
        tag: str,
        values: Union[Tensor, NDArrayFloat],
        global_step: Optional[int] = ...,
        *,
        bins: Union[
            int,
            Sequence[float],
            NDArrayFloat,
            Literal[
                "auto",
                "fd",
                "doane",
                "scott",
                "stone",
                "rice",
                "sturges",
                "sqrt",
            ],
        ] = ...,
        walltime: Optional[float] = ...,
    ) -> None:
        """Add a histogram of *values*."""
    # ─────────────────────────────── Graph ────────────────────────────────
    def add_graph(
        self,
        model: nn.Module,
        input_to_model: Union[
            Tensor,
            Tuple[Tensor, ...],
            Dict[str, Tensor],
            None,
        ] = ...,
        *,
        verbose: bool = ...,
    ) -> None:
        """Trace *model* and write its computational graph."""
    # ─────────────────────────── Hyper-parameters ─────────────────────────
    def add_hparams(
        self,
        hparam_dict: Mapping[str, Scalar],
        metric_dict: Mapping[str, Scalar],
        *,
        run_name: Optional[str] = ...,
    ) -> None:
        """Log a single hyper-parameter run."""
    # ───────────────────────────── Embedding ──────────────────────────────
    def add_embedding(
        self,
        mat: Union[Tensor, NDArrayFloat],
        *,
        metadata: Optional[Sequence[str]] = ...,
        label_img: Optional[Tensor] = ...,
        global_step: Optional[int] = ...,
        tag: str = ...,
        metadata_header: Optional[Sequence[str]] = ...,
    ) -> None:
        """Visualize high-dimensional data in TensorBoard’s projector."""
    # ────────────────────────────── Misc. ─────────────────────────────────
    def flush(self) -> None:
        """Force all pending events to be written to disk."""

    def close(self) -> None:
        """Flush and close the underlying file writers."""

    def get_logdir(self) -> str:
        """Return the directory this writer logs to."""
