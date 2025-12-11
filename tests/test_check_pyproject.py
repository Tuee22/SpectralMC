"""Tests for tools/check_pyproject.py dual-file synchronization validator."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from tools import check_pyproject


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test files.

    Args:
        tmp_path: pytest temporary directory

    Returns:
        Path to temporary directory
    """
    return tmp_path


def create_minimal_pyproject(
    path: Path,
    scripts: dict[str, str] | None = None,
    dependencies: dict[str, str] | None = None,
    has_source: bool = False,
) -> None:
    """Create a minimal pyproject.toml file for testing.

    Args:
        path: Path to write file
        scripts: Optional [tool.poetry.scripts] section
        dependencies: Optional [tool.poetry.dependencies] section
        has_source: Whether to include [[tool.poetry.source]] section
    """
    scripts = scripts or {"test-script": "module:main"}
    dependencies = dependencies or {"python": ">=3.12,<3.13"}

    content = f"""[tool.poetry]
name = "test-package"
version = "0.1.0"
description = "Test package"

[tool.poetry.dependencies]
"""
    for key, value in dependencies.items():
        content += f'{key} = "{value}"\n'

    if has_source:
        content += """
[[tool.poetry.source]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
priority = "supplemental"
"""

    content += """
[tool.poetry.scripts]
"""
    for key, value in scripts.items():
        content += f'{key} = "{value}"\n'

    content += """
[tool.mypy]
strict = true
python_version = "3.12"

[tool.black]
line-length = 88
target-version = ["py312"]

[tool.pytest.ini_options]
testpaths = "tests"
addopts = "-ra"

[tool.pydantic-mypy]
init_forbid_extra = true

[tool.poetry.group.dev.dependencies]
pytest = ">=8.0,<9.0"
"""

    path.write_text(content)


def test_shared_sections_synchronized(temp_dir: Path) -> None:
    """Test validator passes when shared sections are identical."""
    binary_path = temp_dir / "pyproject.binary.toml"
    source_path = temp_dir / "pyproject.source.toml"

    # Create identical files (except dependencies - allowed to differ)
    create_minimal_pyproject(
        binary_path,
        scripts={"test": "module:main", "check": "module:check"},
        dependencies={"python": ">=3.12,<3.13", "torch": "2.7.1+cu128"},
        has_source=True,
    )
    create_minimal_pyproject(
        source_path,
        scripts={"test": "module:main", "check": "module:check"},
        dependencies={"python": ">=3.12,<3.13", "torch": "2.4.1"},
        has_source=False,
    )

    # Load and compare
    binary_data = check_pyproject.load_toml(binary_path)
    source_data = check_pyproject.load_toml(source_path)

    all_match, errors = check_pyproject.compare_sections(binary_data, source_data)

    assert all_match, f"Validation should pass, but got errors: {errors}"
    assert len(errors) == 0


def test_detect_script_mismatch(temp_dir: Path) -> None:
    """Test validator fails when [tool.poetry.scripts] differs."""
    binary_path = temp_dir / "pyproject.binary.toml"
    source_path = temp_dir / "pyproject.source.toml"

    create_minimal_pyproject(
        binary_path,
        scripts={"test": "module:main", "new-script": "module:new"},
    )
    create_minimal_pyproject(
        source_path,
        scripts={"test": "module:main"},  # Missing new-script
    )

    binary_data = check_pyproject.load_toml(binary_path)
    source_data = check_pyproject.load_toml(source_path)

    all_match, errors = check_pyproject.compare_sections(binary_data, source_data)

    assert not all_match, "Should detect script mismatch"
    assert len(errors) > 0
    assert any("tool.poetry.scripts" in error for error in errors)


def test_detect_mypy_mismatch(temp_dir: Path) -> None:
    """Test validator fails when [tool.mypy] differs."""
    binary_path = temp_dir / "pyproject.binary.toml"
    source_path = temp_dir / "pyproject.source.toml"

    create_minimal_pyproject(binary_path)
    create_minimal_pyproject(source_path)

    # Modify mypy config in binary file
    binary_content = binary_path.read_text()
    binary_content = binary_content.replace(
        'strict = true', 'strict = true\ndisallow_any_explicit = true'
    )
    binary_path.write_text(binary_content)

    binary_data = check_pyproject.load_toml(binary_path)
    source_data = check_pyproject.load_toml(source_path)

    all_match, errors = check_pyproject.compare_sections(binary_data, source_data)

    assert not all_match, "Should detect mypy mismatch"
    assert len(errors) > 0
    assert any("tool.mypy" in error for error in errors)


def test_detect_black_mismatch(temp_dir: Path) -> None:
    """Test validator fails when [tool.black] differs."""
    binary_path = temp_dir / "pyproject.binary.toml"
    source_path = temp_dir / "pyproject.source.toml"

    create_minimal_pyproject(binary_path)
    create_minimal_pyproject(source_path)

    # Modify black config in source file
    source_content = source_path.read_text()
    source_content = source_content.replace('line-length = 88', 'line-length = 100')
    source_path.write_text(source_content)

    binary_data = check_pyproject.load_toml(binary_path)
    source_data = check_pyproject.load_toml(source_path)

    all_match, errors = check_pyproject.compare_sections(binary_data, source_data)

    assert not all_match, "Should detect black config mismatch"
    assert len(errors) > 0
    assert any("tool.black" in error for error in errors)


def test_detect_pytest_mismatch(temp_dir: Path) -> None:
    """Test validator fails when [tool.pytest.ini_options] differs."""
    binary_path = temp_dir / "pyproject.binary.toml"
    source_path = temp_dir / "pyproject.source.toml"

    create_minimal_pyproject(binary_path)
    create_minimal_pyproject(source_path)

    # Modify pytest config
    source_content = source_path.read_text()
    source_content = source_content.replace('addopts = "-ra"', 'addopts = "-ra -v"')
    source_path.write_text(source_content)

    binary_data = check_pyproject.load_toml(binary_path)
    source_data = check_pyproject.load_toml(source_path)

    all_match, errors = check_pyproject.compare_sections(binary_data, source_data)

    assert not all_match, "Should detect pytest config mismatch"
    assert len(errors) > 0
    assert any("tool.pytest.ini_options" in error for error in errors)


def test_different_sections_allowed(temp_dir: Path) -> None:
    """Test validator allows [tool.poetry.dependencies] to differ."""
    binary_path = temp_dir / "pyproject.binary.toml"
    source_path = temp_dir / "pyproject.source.toml"

    # Different dependencies - this is EXPECTED and ALLOWED
    create_minimal_pyproject(
        binary_path,
        dependencies={"python": ">=3.12,<3.13", "torch": "2.7.1+cu128"},
        has_source=True,
    )
    create_minimal_pyproject(
        source_path,
        dependencies={"python": ">=3.12,<3.13", "torch": "2.4.1"},
        has_source=False,
    )

    binary_data = check_pyproject.load_toml(binary_path)
    source_data = check_pyproject.load_toml(source_path)

    all_match, errors = check_pyproject.compare_sections(binary_data, source_data)

    # Should pass even though dependencies differ (not a shared section)
    assert all_match, "Dependencies are allowed to differ"
    assert len(errors) == 0


def test_missing_file_error(temp_dir: Path) -> None:
    """Test validator errors when one file doesn't exist."""
    binary_path = temp_dir / "pyproject.binary.toml"
    source_path = temp_dir / "nonexistent.toml"

    create_minimal_pyproject(binary_path)

    with pytest.raises(FileNotFoundError):
        check_pyproject.load_toml(source_path)


def test_malformed_toml_error(temp_dir: Path) -> None:
    """Test validator errors on syntax error in TOML."""
    import tomllib

    malformed_path = temp_dir / "malformed.toml"
    malformed_path.write_text("[tool.poetry\n")  # Missing closing bracket

    with pytest.raises(tomllib.TOMLDecodeError):
        check_pyproject.load_toml(malformed_path)


def test_missing_shared_section_warning(temp_dir: Path) -> None:
    """Test validator warnings when shared section missing from one file."""
    binary_path = temp_dir / "pyproject.binary.toml"
    source_path = temp_dir / "pyproject.source.toml"

    # Create binary with all sections
    create_minimal_pyproject(binary_path)

    # Create source missing [tool.black] section
    create_minimal_pyproject(source_path)
    source_content = source_path.read_text()
    # Remove [tool.black] section
    lines = source_content.split("\n")
    filtered_lines = []
    skip = False
    for line in lines:
        if line.strip().startswith("[tool.black]"):
            skip = True
        elif line.strip().startswith("[") and skip:
            skip = False
        if not skip:
            filtered_lines.append(line)
    source_path.write_text("\n".join(filtered_lines))

    binary_data = check_pyproject.load_toml(binary_path)
    source_data = check_pyproject.load_toml(source_path)

    all_match, errors = check_pyproject.compare_sections(binary_data, source_data)

    assert not all_match, "Should detect missing section"
    assert len(errors) > 0
    assert any("tool.black" in error for error in errors)


def test_verbose_mode_output(temp_dir: Path) -> None:
    """Test verbose mode shows detailed diff formatting."""
    binary_path = temp_dir / "pyproject.binary.toml"
    source_path = temp_dir / "pyproject.source.toml"

    create_minimal_pyproject(
        binary_path, scripts={"test": "module:main", "extra": "module:extra"}
    )
    create_minimal_pyproject(source_path, scripts={"test": "module:main"})

    binary_data = check_pyproject.load_toml(binary_path)
    source_data = check_pyproject.load_toml(source_path)

    all_match, errors = check_pyproject.compare_sections(
        binary_data, source_data, verbose=True
    )

    assert not all_match
    assert len(errors) > 0

    # Verbose output should contain diff markers
    error_text = "\n".join(errors)
    assert "diff" in error_text.lower() or "---" in error_text or "+++" in error_text


def test_extract_section() -> None:
    """Test section extraction from nested TOML data."""
    data = {
        "tool": {
            "poetry": {
                "scripts": {"test": "module:main"},
                "dependencies": {"python": ">=3.12"},
            },
            "mypy": {"strict": True},
        }
    }

    # Test valid paths
    assert check_pyproject.extract_section(data, "tool.poetry.scripts") == {
        "test": "module:main"
    }
    assert check_pyproject.extract_section(data, "tool.mypy") == {"strict": True}

    # Test missing section
    assert check_pyproject.extract_section(data, "tool.nonexistent") is None
    assert check_pyproject.extract_section(data, "tool.poetry.missing") is None


def test_format_toml_value() -> None:
    """Test TOML value formatting for display."""
    # Test dict formatting
    dict_val = {"key1": "value1", "key2": 42}
    formatted = check_pyproject.format_toml_value(dict_val)
    assert "key1" in formatted
    assert "value1" in formatted

    # Test list formatting
    list_val = ["item1", "item2"]
    formatted = check_pyproject.format_toml_value(list_val)
    assert "[" in formatted
    assert "item1" in formatted

    # Test nested dict formatting
    nested = {"outer": {"inner": "value"}}
    formatted = check_pyproject.format_toml_value(nested)
    assert "outer" in formatted
    assert "inner" in formatted
