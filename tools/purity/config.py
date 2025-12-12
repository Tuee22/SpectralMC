# File: tools/purity/config.py
"""Configuration loading for purity checker.

Loads tier classification and whitelist configuration from pyproject.toml.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import tomllib


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


def _expect_str_list(value: object, key: str) -> list[str]:
    """Return value as list[str] or raise if type does not match."""
    if not isinstance(value, list):
        raise TypeError(f"[tool.purity.{key}] must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"[tool.purity.{key}] must contain only strings")
    return value


def _parse_whitelist(value: object) -> dict[str, dict[int, str]]:
    """Convert TOML whitelist value to dict with int keys."""
    if not isinstance(value, dict):
        raise TypeError("[tool.purity.whitelist] must be a table")

    result: dict[str, dict[int, str]] = {}
    for filepath, line_dict in value.items():
        if not isinstance(filepath, str):
            raise TypeError("Whitelist file paths must be strings")
        if not isinstance(line_dict, dict):
            raise TypeError(f"Whitelist entry for {filepath} must be a table")

        converted: dict[int, str] = {}
        for line_num, reason in line_dict.items():
            if not isinstance(line_num, str):
                raise TypeError("Whitelist line numbers must be strings in TOML")
            if not isinstance(reason, str):
                raise TypeError("Whitelist reasons must be strings")
            converted[int(line_num)] = reason
        result[filepath] = converted

    return result


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

    raw_config: object = data["tool"]["purity"]
    if not isinstance(raw_config, dict):
        raise KeyError("[tool.purity] must be a table")

    tier1_value = raw_config.get("tier1_infrastructure", [])
    tier3_value = raw_config.get("tier3_effects", [])
    whitelist_value = raw_config.get("whitelist")

    result: PurityConfig = {
        "tier1_infrastructure": _expect_str_list(tier1_value, "tier1_infrastructure"),
        "tier3_effects": _expect_str_list(tier3_value, "tier3_effects"),
    }

    if whitelist_value is not None:
        result["whitelist"] = _parse_whitelist(whitelist_value)

    return result


def get_project_root() -> Path:
    """Get SpectralMC project root directory.

    Returns:
        Path to project root (parent of tools/ directory)
    """
    return Path(__file__).parent.parent.parent
