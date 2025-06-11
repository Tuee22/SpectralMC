"""
Stub package for ``torch.utils`` exposing the ``dlpack`` sub-module.
"""

from __future__ import annotations
from types import ModuleType

# Re-export sub-modules that the repo imports
import importlib as _il

dlpack: ModuleType = _il.import_module("torch.utils.dlpack")  # noqa: E702
