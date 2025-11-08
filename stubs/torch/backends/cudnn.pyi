# typings/torch/backends/cudnn.pyi
"""
Strict stub for :pymod:`torch.backends.cudnn` (SpectralMC subset).
"""

from __future__ import annotations

def version() -> int | None: ...

deterministic: bool
benchmark: bool
allow_tf32: bool  # ← added: matches real PyTorch API
