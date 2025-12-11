# File: tools/purity/config.py
"""Configuration loading for purity checker.

Loads tier classification and whitelist configuration from pyproject.toml.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


class PurityConfig(TypedDict, total=False):
    """Purity checker configuration structure.

    Attributes:
        tier1_infrastructure: File patterns for Tier 1 (infrastructure, exempt)
        tier3_effects: File patterns for Tier 3 (effects/boundaries, exempt)
        whitelist: File-specific line number exceptions with justifications
    """

    tier1_infrastructure: list[str]
    tier3_effects: list[str]
    whitelist: dict[str, dict[int, str]]


def load_purity_config() -> PurityConfig:
    """Load purity configuration from pyproject.toml.

    Returns:
        Configuration dictionary with:
        - tier1_infrastructure: List of infrastructure file patterns (exempt)
        - tier3_effects: List of effect interpreter file patterns (exempt)
        - whitelist: Dict of file:line_number exceptions with justifications

    Raises:
        FileNotFoundError: If pyproject.toml not found
        KeyError: If [tool.purity] section missing
    """
    project_root = Path(__file__).parent.parent.parent
    pyproject_candidates = [
        project_root / "pyproject.toml",
        project_root / "pyproject.source.toml",
        project_root / "pyproject.binary.toml",
    ]

    pyproject_path = next((path for path in pyproject_candidates if path.exists()), None)
    if pyproject_path is None:
        raise FileNotFoundError(
            f"No pyproject configuration found. Searched: {', '.join(str(p) for p in pyproject_candidates)}"
        )

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    if "tool" not in data or "purity" not in data["tool"]:
        # Return default configuration if section doesn't exist yet
        default_config: PurityConfig = {
            "tier1_infrastructure": [
                "src/spectralmc/models/torch.py",
                "src/spectralmc/models/cpu_gpu_transfer.py",
                "src/spectralmc/cvnn.py",
            ],
            "tier3_effects": [
                "src/spectralmc/effects/interpreter.py",
                "src/spectralmc/storage/*.py",
                "src/spectralmc/__main__.py",
            ],
            "whitelist": {
                "src/spectralmc/serialization/tensors.py": {
                    174: "Protobuf requires_grad mutation (I/O boundary)",
                },
                "src/spectralmc/async_normals.py": {
                    311: "RNG advance guard (checkpoint resume)",
                },
            },
        }
        return default_config

    raw_config: dict[str, list[str] | dict[str, dict[str, str]]] = data["tool"]["purity"]

    # Convert whitelist string keys to integers
    # TOML requires string keys, but we need integer line numbers
    result: PurityConfig = {
        "tier1_infrastructure": raw_config.get("tier1_infrastructure", []),  # type: ignore[typeddict-item]
        "tier3_effects": raw_config.get("tier3_effects", []),  # type: ignore[typeddict-item]
    }

    if "whitelist" in raw_config:
        whitelist: dict[str, dict[int, str]] = {}
        raw_whitelist: dict[str, dict[str, str]] = raw_config["whitelist"]  # type: ignore[assignment]
        for filepath, line_dict in raw_whitelist.items():
            whitelist[filepath] = {int(line_num): reason for line_num, reason in line_dict.items()}
        result["whitelist"] = whitelist

    return result


def get_project_root() -> Path:
    """Get SpectralMC project root directory.

    Returns:
        Path to project root (parent of tools/ directory)
    """
    return Path(__file__).parent.parent.parent
