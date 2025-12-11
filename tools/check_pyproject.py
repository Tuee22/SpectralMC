#!/usr/bin/env python3
"""PyProject dual-file synchronization validator.

Ensures shared sections between pyproject.binary.toml and pyproject.source.toml
remain synchronized, preventing configuration drift.

Usage:
    poetry run check-pyproject                  # Check synchronization
    poetry run check-pyproject --verbose        # Show detailed diffs
    poetry run check-pyproject --fix            # Auto-sync (interactive)
    poetry run check-pyproject --fix-from=binary  # Auto-sync binary→source
    poetry run check-pyproject --fix-from=source  # Auto-sync source→binary

Exit codes:
    0 - All shared sections synchronized
    1 - Synchronization failures detected
    2 - File error (missing file, malformed TOML)
"""

from __future__ import annotations

import argparse
import difflib
import sys
import tomllib
from pathlib import Path
from typing import Any

# Shared sections that MUST stay synchronized between both files
SHARED_SECTIONS = [
    "tool.poetry.scripts",
    "tool.mypy",
    "tool.black",
    "tool.pytest.ini_options",
    "tool.pydantic-mypy",
    "tool.poetry.group.dev.dependencies",
]

# Different sections that are EXPECTED to diverge
DIFFERENT_SECTIONS = [
    "tool.poetry.dependencies",
    "tool.poetry.source",  # Binary build only
]


def get_project_root() -> Path:
    """Get project root directory.

    Returns:
        Path to project root (parent of tools/ directory)
    """
    return Path(__file__).parent.parent


def load_toml(filepath: Path) -> dict[str, Any]:
    """Load and parse TOML file.

    Args:
        filepath: Path to TOML file

    Returns:
        Parsed TOML data as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        tomllib.TOMLDecodeError: If file has invalid TOML syntax
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with filepath.open("rb") as f:
        return tomllib.load(f)


def extract_section(data: dict[str, Any], section_path: str) -> Any | None:
    """Extract nested section from TOML data.

    Args:
        data: Parsed TOML data
        section_path: Dot-separated section path (e.g., "tool.poetry.scripts")

    Returns:
        Section value if found, None otherwise

    Examples:
        >>> data = {"tool": {"poetry": {"scripts": {"test": "module:main"}}}}
        >>> extract_section(data, "tool.poetry.scripts")
        {"test": "module:main"}
    """
    parts = section_path.split(".")
    current = data

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]

    return current


def format_toml_value(value: Any, indent: int = 0) -> str:
    """Format TOML value for display.

    Args:
        value: Value to format (dict, list, str, etc.)
        indent: Current indentation level

    Returns:
        Formatted string representation
    """
    prefix = "  " * indent

    if isinstance(value, dict):
        lines = []
        for key, val in value.items():
            if isinstance(val, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(format_toml_value(val, indent + 1))
            elif isinstance(val, list):
                lines.append(f"{prefix}{key} = [")
                for item in val:
                    lines.append(f"{prefix}  {repr(item)},")
                lines.append(f"{prefix}]")
            else:
                lines.append(f"{prefix}{key} = {repr(val)}")
        return "\n".join(lines)
    elif isinstance(value, list):
        lines = [f"{prefix}["]
        for item in value:
            lines.append(f"{prefix}  {repr(item)},")
        lines.append(f"{prefix}]")
        return "\n".join(lines)
    else:
        return f"{prefix}{repr(value)}"


def format_diff(
    section: str, binary_val: Any, source_val: Any, verbose: bool = False
) -> str:
    """Format unified diff for mismatched section.

    Args:
        section: Section name (e.g., "tool.poetry.scripts")
        binary_val: Value from pyproject.binary.toml
        source_val: Value from pyproject.source.toml
        verbose: Show full unified diff

    Returns:
        Formatted diff string
    """
    lines = [f"\n❌ Section [{section}] differs:\n"]

    if verbose:
        # Show unified diff
        binary_str = format_toml_value(binary_val)
        source_str = format_toml_value(source_val)

        diff = difflib.unified_diff(
            binary_str.splitlines(keepends=True),
            source_str.splitlines(keepends=True),
            fromfile="pyproject.binary.toml",
            tofile="pyproject.source.toml",
            lineterm="",
        )
        lines.extend(diff)
    else:
        # Show summary
        lines.append("  pyproject.binary.toml has:")
        lines.append(format_toml_value(binary_val, indent=2))
        lines.append("")
        lines.append("  pyproject.source.toml has:")
        lines.append(format_toml_value(source_val, indent=2))

    lines.append("")
    lines.append(
        "  Fix: Edit both files to match and re-run check-pyproject, or use --fix"
    )

    return "\n".join(lines)


def compare_sections(
    binary_data: dict[str, Any],
    source_data: dict[str, Any],
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Compare shared sections between binary and source files.

    Args:
        binary_data: Parsed pyproject.binary.toml
        source_data: Parsed pyproject.source.toml
        verbose: Show detailed diff output

    Returns:
        Tuple of (success, error_messages)
        - success: True if all shared sections match
        - error_messages: List of formatted error messages
    """
    errors: list[str] = []
    all_match = True

    for section in SHARED_SECTIONS:
        binary_val = extract_section(binary_data, section)
        source_val = extract_section(source_data, section)

        # Check if section exists in both files
        if binary_val is None and source_val is None:
            # Section missing from both - warning but not error
            errors.append(
                f"\n⚠️  Section [{section}] missing from both files (expected?)"
            )
            continue
        elif binary_val is None:
            errors.append(
                f"\n❌ Section [{section}] missing from pyproject.binary.toml"
            )
            all_match = False
            continue
        elif source_val is None:
            errors.append(
                f"\n❌ Section [{section}] missing from pyproject.source.toml"
            )
            all_match = False
            continue

        # Compare values (deep equality)
        if binary_val != source_val:
            errors.append(format_diff(section, binary_val, source_val, verbose))
            all_match = False

    return all_match, errors


def write_toml_section(
    filepath: Path, section_path: str, new_value: Any
) -> None:
    """Write updated section back to TOML file.

    Note: This is a simplified implementation that replaces the entire section.
    For production use, consider using a TOML writer library that preserves
    formatting and comments.

    Args:
        filepath: Path to TOML file
        section_path: Dot-separated section path
        new_value: New value for the section
    """
    # Read current file
    content = filepath.read_text()

    # For now, we'll just warn the user - full implementation would use tomlkit
    # to preserve formatting and comments
    print(
        f"\n⚠️  Auto-fix not fully implemented yet. Please manually sync section [{section_path}]"
    )
    print(f"   in file: {filepath}")
    print("\nExpected value:")
    print(format_toml_value(new_value, indent=1))


def fix_sync(
    binary_path: Path,
    source_path: Path,
    direction: str | None,
    verbose: bool = False,
) -> int:
    """Auto-synchronize shared sections.

    Args:
        binary_path: Path to pyproject.binary.toml
        source_path: Path to pyproject.source.toml
        direction: Sync direction ("binary", "source", or None for interactive)
        verbose: Show detailed output

    Returns:
        Exit code (0 = success, 1 = failures, 2 = error)
    """
    try:
        binary_data = load_toml(binary_path)
        source_data = load_toml(source_path)
    except Exception as e:
        print(f"❌ ERROR: Failed to load TOML files: {e}", file=sys.stderr)
        return 2

    if direction is None:
        print("⚠️  Interactive --fix mode not implemented yet")
        print("   Use --fix-from=binary or --fix-from=source")
        return 2

    # Determine source and target
    if direction == "binary":
        from_data = binary_data
        to_path = source_path
        print("🔄 Syncing from pyproject.binary.toml → pyproject.source.toml")
    elif direction == "source":
        from_data = source_data
        to_path = binary_path
        print("🔄 Syncing from pyproject.source.toml → pyproject.binary.toml")
    else:
        print(f"❌ ERROR: Invalid direction: {direction}", file=sys.stderr)
        return 2

    # Sync each shared section
    for section in SHARED_SECTIONS:
        value = extract_section(from_data, section)
        if value is not None:
            write_toml_section(to_path, section, value)

    print("\n⚠️  Full auto-fix implementation pending - requires tomlkit library")
    print("   Please manually sync the sections shown above")
    return 1


def main() -> int:
    """Run pyproject synchronization validation.

    Returns:
        Exit code (0 = pass, 1 = sync failures, 2 = file error)
    """
    parser = argparse.ArgumentParser(
        description="Validate dual-pyproject synchronization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run check-pyproject                  # Check synchronization
  poetry run check-pyproject --verbose        # Show detailed diffs
  poetry run check-pyproject --fix            # Auto-sync (interactive)
  poetry run check-pyproject --fix-from=binary  # Auto-sync binary→source

Exit codes:
  0 - All shared sections synchronized
  1 - Synchronization failures detected
  2 - File error (missing file, malformed TOML)
        """,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed diff output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show errors, no success messages",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-synchronize shared sections (interactive)",
    )
    parser.add_argument(
        "--fix-from",
        choices=["binary", "source"],
        help="Auto-sync from binary to source or vice versa",
    )

    args = parser.parse_args()

    # Get file paths
    project_root = get_project_root()
    binary_path = project_root / "pyproject.binary.toml"
    source_path = project_root / "pyproject.source.toml"

    # Handle --fix modes
    if args.fix or args.fix_from:
        return fix_sync(
            binary_path,
            source_path,
            direction=args.fix_from,
            verbose=args.verbose,
        )

    # Load TOML files
    try:
        binary_data = load_toml(binary_path)
        source_data = load_toml(source_path)
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        return 2
    except tomllib.TOMLDecodeError as e:
        print(f"❌ ERROR: Invalid TOML syntax: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"❌ ERROR: Failed to load TOML files: {e}", file=sys.stderr)
        return 2

    # Validate file sizes (basic sanity check)
    binary_size = binary_path.stat().st_size
    source_size = source_path.stat().st_size

    if binary_size < 1000 or source_size < 1000:
        print(
            f"⚠️  WARNING: Suspiciously small file size "
            f"(binary: {binary_size}B, source: {source_size}B)",
            file=sys.stderr,
        )

    if not args.quiet:
        print("🔍 Checking PyProject synchronization...")
        print(f"  Binary: {binary_path.name} ({binary_size} bytes)")
        print(f"  Source: {source_path.name} ({source_size} bytes)")
        print()

    # Compare shared sections
    all_match, errors = compare_sections(binary_data, source_data, args.verbose)

    # Report results
    if errors:
        for error in errors:
            print(error)

    if not all_match:
        print("\n❌ PyProject synchronization FAILED")
        print(f"   {len(errors)} issue(s) found")
        print("\nRun with --verbose for detailed diffs:")
        print("  poetry run check-pyproject --verbose")
        print("\nSee documentation:")
        print("  documents/engineering/docker_build_philosophy.md")
        return 1

    if not args.quiet:
        print("✅ All shared sections synchronized")
        print(f"   Checked {len(SHARED_SECTIONS)} shared sections:")
        for section in SHARED_SECTIONS:
            print(f"     • {section}")
        print("\n🎉 PyProject synchronization check passed!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
