# File: tools/purity/__init__.py
"""Purity checking tools for SpectralMC business logic.

This package provides AST-based static analysis to enforce zero-tolerance
purity requirements in Tier 2 business logic files.

See documents/engineering/purity_enforcement.md for complete documentation.
"""

from __future__ import annotations

__all__ = [
    "PurityChecker",
    "FileClassifier",
    "PurityViolation",
    "load_purity_config",
    "format_violation",
]
