"""
Stub for ``torch.utils.dlpack`` (“from_dlpack” helper only).
"""

from __future__ import annotations
from typing import Any
import torch

def from_dlpack(dlpack: Any) -> torch.Tensor: ...
