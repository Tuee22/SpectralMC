# torch/nn/functional.py
"""
Tiny stub for ``torch.nn.functional`` (only mse_loss).
"""

from __future__ import annotations
import torch

def mse_loss(
    input: torch.Tensor, target: torch.Tensor, *, reduction: str = ...
) -> torch.Tensor: ...
