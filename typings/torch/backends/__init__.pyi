# typings/torch/backends/__init__.pyi
"""
Stub for :pymod:`torch.backends` that re‑exports strict sub‑modules needed by
SpectralMC so mypy recognises their attributes.
"""

from __future__ import annotations

from . import cudnn as cudnn  # exact‑typed re‑export
from . import cuda as cuda  # new: exposes ``torch.backends.cuda``
