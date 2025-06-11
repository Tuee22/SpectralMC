"""
Stub for ``torch.nn.utils`` exposing only clip_grad_norm_.
"""

from __future__ import annotations
from typing import Iterable
import torch

def clip_grad_norm_(
    parameters: Iterable[torch.Tensor], max_norm: float, norm_type: float | int = ...
) -> float: ...
