from __future__ import annotations
import torch
from typing import Optional

def mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    size_average: Optional[bool] = ...,
    reduce: Optional[bool] = ...,
    reduction: str = ...,
) -> torch.Tensor: ...
