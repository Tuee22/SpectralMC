"""
Strict stub for :pymod:`torch.backends.cuda` (subset needed by SpectralMC).
"""

from __future__ import annotations

class _MatmulConfig:
    allow_tf32: bool  # public attribute toggled by user‑code

matmul: _MatmulConfig  # e.g. ``torch.backends.cuda.matmul.allow_tf32``
