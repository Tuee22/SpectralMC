# typings/torch/backends/__init__.pyi
"""
Stub for :pymod:`torch.backends` that re‑exports the fully‑typed
:cudnn: submodule so mypy recognises attributes such as
``deterministic`` and ``benchmark``.
"""
from __future__ import annotations

from . import cudnn as cudnn  # exact‑typed re‑export
