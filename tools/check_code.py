#!/usr/bin/env python3
"""
PyProject sync + Immutability + Ruff + Black + Purity + MyPy code quality checker.

Runs pyproject synchronization validation, immutability audit, ruff linting,
black formatting, purity checking, then mypy type checking with fail-fast
behavior. Exits immediately on first failure.
"""

import subprocess
import sys
from pathlib import Path

from tools.check_pyproject import compare_sections, load_toml


def _get_project_root() -> Path:
    return Path(__file__).parent.parent


def _validate_pyproject_sync() -> int:
    """Ensure shared sections match between pyproject files."""
    root = _get_project_root()
    source_path = root / "pyproject.source.toml"
    binary_path = root / "pyproject.binary.toml"

    try:
        source = load_toml(source_path)
        binary = load_toml(binary_path)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ PyProject validation failed: {exc}")
        return 2

    all_match, errors = compare_sections(binary, source)
    if not all_match:
        for error in errors:
            print(f"❌ {error}")
        print("   Update both pyproject files so shared sections match.")
        return 1

    print("✅ PyProject shared sections are synchronized!\n")
    return 0


def main() -> int:
    """Run pyproject sync, ruff, black, purity checker, and mypy with fail-fast."""
    # Step 0: Validate PyProject synchronization
    print("🔍 Running PyProject synchronization check...")
    pyproject_status = _validate_pyproject_sync()
    if pyproject_status != 0:
        return pyproject_status

    # Step 1: Run immutability doctrine audit
    print("🔍 Running Immutability doctrine audit...")
    immutability_result = subprocess.run(
        ["poetry", "run", "python", "-m", "tools.check_immutability"],
        check=False,
    )

    if immutability_result.returncode != 0:
        print(f"❌ Immutability audit failed with exit code {immutability_result.returncode}")
        return immutability_result.returncode

    print("✅ Immutability audit passed!\n")

    # Step 1b: Run Pydantic construction guard
    print("🔍 Running Pydantic construction guard...")
    pydantic_guard_result = subprocess.run(
        ["poetry", "run", "python", "tools/check_pydantic_construction.py", "--root", "src"],
        check=False,
    )
    if pydantic_guard_result.returncode != 0:
        print(
            f"❌ Pydantic construction guard failed with exit code {pydantic_guard_result.returncode}"
        )
        return pydantic_guard_result.returncode

    print("✅ Pydantic construction guard passed!\n")

    # Step 2: Run Ruff
    print("🔍 Running Ruff linter...")
    ruff_result = subprocess.run(
        ["poetry", "run", "ruff", "check", "--fix", "src/spectralmc/", "tools/", "tests/"],
        check=False,
    )

    if ruff_result.returncode != 0:
        print(f"❌ Ruff failed with exit code {ruff_result.returncode}")
        return ruff_result.returncode

    print("✅ Ruff passed!\n")

    # Step 3: Run Black
    print("🔍 Running Black formatter...")
    black_result = subprocess.run(
        ["poetry", "run", "black", "src/spectralmc/", "tools/", "tests/"],
        check=False,
    )

    if black_result.returncode != 0:
        print(f"❌ Black failed with exit code {black_result.returncode}")
        return black_result.returncode

    print("✅ Black passed!\n")

    # Step 4: Run Purity Checker (zero-tolerance for Tier 2 business logic)
    print("🔍 Running Purity checker...")
    purity_result = subprocess.run(
        ["poetry", "run", "check-purity"],
        check=False,
    )

    if purity_result.returncode != 0:
        print(f"❌ Purity check failed with exit code {purity_result.returncode}")
        print("Run with --verbose for details: poetry run check-purity --verbose")
        return purity_result.returncode

    print("✅ Purity check passed!\n")

    # Step 5: Run MyPy
    # Paths match pyproject.toml [tool.mypy] files configuration
    print("🔍 Running MyPy type checker...")
    mypy_result = subprocess.run(
        ["poetry", "run", "mypy", "src/spectralmc", "tests", "tools"],
        check=False,
    )

    if mypy_result.returncode != 0:
        print(f"❌ MyPy failed with exit code {mypy_result.returncode}")
        return mypy_result.returncode

    print("✅ MyPy passed!")
    print("\n🎉 All code quality checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
