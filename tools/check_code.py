#!/usr/bin/env python3
"""
Ruff + Black + Purity + MyPy code quality checker.

Runs ruff linting, black formatting, purity checking, then mypy type checking
with fail-fast behavior. Exits immediately on first failure.
"""

import subprocess
import sys


def main() -> int:
    """Run ruff, black, purity checker, and mypy sequentially with fail-fast."""
    # Step 1: Run Ruff
    print("🔍 Running Ruff linter...")
    ruff_result = subprocess.run(
        ["poetry", "run", "ruff", "check", "--fix", "src/spectralmc/", "tools/", "tests/"],
        check=False,
    )

    if ruff_result.returncode != 0:
        print(f"❌ Ruff failed with exit code {ruff_result.returncode}")
        return ruff_result.returncode

    print("✅ Ruff passed!\n")

    # Step 2: Run Black
    print("🔍 Running Black formatter...")
    black_result = subprocess.run(
        ["poetry", "run", "black", "src/spectralmc/", "tools/", "tests/"],
        check=False,
    )

    if black_result.returncode != 0:
        print(f"❌ Black failed with exit code {black_result.returncode}")
        return black_result.returncode

    print("✅ Black passed!\n")

    # Step 3: Run Purity Checker (zero-tolerance for Tier 2 business logic)
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

    # Step 4: Run MyPy
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
