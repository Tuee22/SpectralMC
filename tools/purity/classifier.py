# File: tools/purity/classifier.py
"""File tier classification for purity enforcement.

Classifies Python files into:
- Tier 1: Infrastructure (exempt from purity)
- Tier 2: Business Logic (zero-tolerance purity enforcement)
- Tier 3: Effect Interpreter (exempt from purity)
- None: Not in src/spectralmc/ (exempt)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path


class FileTier(Enum):
    """File tier classification for purity enforcement."""

    TIER1_INFRASTRUCTURE = 1  # Exempt: PyTorch facade, GPU transfer, etc.
    TIER2_BUSINESS_LOGIC = 2  # Zero tolerance: Pure functional code
    TIER3_EFFECTS = 3  # Exempt: Effect interpreter, storage layer


@dataclass(frozen=True)
class TierClassification:
    """Result of file tier classification."""

    tier: FileTier
    reason: str


class FileClassifier:
    """Classifies files into Tier 1/2/3 for purity enforcement."""

    def __init__(
        self,
        tier1_patterns: list[str],
        tier3_patterns: list[str],
        project_root: Path,
    ) -> None:
        """Initialize file classifier.

        Args:
            tier1_patterns: Glob patterns for Tier 1 infrastructure files
            tier3_patterns: Glob patterns for Tier 3 effect interpreter files
            project_root: Project root directory path
        """
        self.tier1_patterns = tier1_patterns
        self.tier3_patterns = tier3_patterns
        self.project_root = project_root

    def classify(self, filepath: Path) -> TierClassification | None:
        """Classify file into Tier 1/2/3.

        Args:
            filepath: Absolute path to Python file

        Returns:
            TierClassification or None if file is exempt (not in src/spectralmc/)
        """
        # Convert to relative path from project root
        try:
            rel_path = filepath.relative_to(self.project_root)
        except ValueError:
            # File outside project root - exempt
            return None

        rel_path_str = str(rel_path)

        # Exempt: Not in src/spectralmc/
        if not rel_path_str.startswith("src/spectralmc/"):
            return None

        # Exempt: Generated protobuf code
        if "/proto/" in rel_path_str and rel_path_str.endswith("_pb2.py"):
            return None

        # Exempt: Test files
        if rel_path_str.startswith("tests/"):
            return None

        # Check Tier 1 (Infrastructure)
        for pattern in self.tier1_patterns:
            if fnmatch(rel_path_str, pattern):
                return TierClassification(
                    tier=FileTier.TIER1_INFRASTRUCTURE,
                    reason=f"Infrastructure file (pattern: {pattern})",
                )

        # Check Tier 3 (Effects)
        for pattern in self.tier3_patterns:
            if fnmatch(rel_path_str, pattern):
                return TierClassification(
                    tier=FileTier.TIER3_EFFECTS,
                    reason=f"Effect interpreter file (pattern: {pattern})",
                )

        # Default: Tier 2 (Business Logic) - zero tolerance purity
        return TierClassification(
            tier=FileTier.TIER2_BUSINESS_LOGIC,
            reason="Business logic file (default Tier 2)",
        )
