"""PyProject dual-file synchronization validator."""

from __future__ import annotations

import difflib
import json
import tomllib
from pathlib import Path
from typing import TypeAlias

TomlPrimitive: TypeAlias = str | int | float | bool | None
TomlValue: TypeAlias = TomlPrimitive | dict[str, "TomlValue"] | list["TomlValue"]

SHARED_SECTIONS: list[str] = [
    "tool.poetry.scripts",
    "tool.mypy",
    "tool.black",
    "tool.pytest.ini_options",
    "tool.pydantic-mypy",
    "tool.poetry.group.dev.dependencies",
]


def load_toml(path: Path) -> dict[str, TomlValue]:
    """Load TOML file and return parsed data."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def extract_section(data: dict[str, TomlValue], section_path: str) -> TomlValue | None:
    """Extract a dotted-path section from TOML data."""
    current: TomlValue | None = data
    for part in section_path.split("."):
        if not isinstance(current, dict):
            return None
        if part not in current:
            return None
        current = current[part]
    return current


def format_toml_value(value: TomlValue | None) -> str:
    """Pretty-format TOML value for display."""
    if value is None:
        return "None"
    return json.dumps(value, indent=2, sort_keys=True)


def _diff_values(section: str, left: TomlValue | None, right: TomlValue | None) -> str:
    left_str = format_toml_value(left)
    right_str = format_toml_value(right)
    diff = "\n".join(
        difflib.ndiff(left_str.splitlines(), right_str.splitlines()),
    )
    return f"{section} differs:\n{diff}"


def compare_sections(
    binary_data: dict[str, TomlValue],
    source_data: dict[str, TomlValue],
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Compare shared sections between two pyproject dicts."""
    errors: list[str] = []

    for section in SHARED_SECTIONS:
        binary_section = extract_section(binary_data, section)
        source_section = extract_section(source_data, section)
        if binary_section == source_section:
            continue

        if verbose:
            errors.append(_diff_values(section, binary_section, source_section))
        else:
            errors.append(f"Shared section differs: {section}")

    return (len(errors) == 0, errors)


__all__ = [
    "TomlValue",
    "SHARED_SECTIONS",
    "compare_sections",
    "extract_section",
    "format_toml_value",
    "load_toml",
]
