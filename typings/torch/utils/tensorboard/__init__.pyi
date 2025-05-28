from __future__ import annotations
import torch
from typing import Sequence, Optional, Union

class SummaryWriter:
    def __init__(
        self,
        log_dir: str | None = ...,
        comment: str | None = ...,
        purge_step: int | None = ...,
        max_queue: int = ...,
        flush_secs: int = ...,
        filename_suffix: str = ...,
        write_to_disk: bool = ...,
    ) -> None: ...
    # Scalars
    def add_scalar(
        self,
        tag: str,
        scalar_value: float | int,
        global_step: int | None = ...,
        walltime: float | None = ...,
        *,
        new_style: bool = ...,
        double_precision: int = ...,
    ) -> None: ...
    # Histograms
    def add_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, Sequence[int], Sequence[float]],
        global_step: int | None = ...,
        bins: int | str = ...,
        walltime: float | None = ...,
    ) -> None: ...
    # House-keeping
    def flush(self) -> None: ...
    def close(self) -> None: ...
