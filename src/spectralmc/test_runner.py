"""Test runner for poetry scripts.

This module provides poetry-based test execution with flexible argument forwarding.

Usage:
    poetry run test-all                    # Run all tests (CPU + GPU)
    poetry run test-all -v                 # Run all tests with verbose output
    poetry run test-all tests/test_gbm.py  # Run specific test file
    poetry run test-all -k "test_sobol"    # Run tests matching keyword
    poetry run test-all --cov              # Run with coverage

All pytest arguments are forwarded directly to pytest.
"""

import sys

import pytest


def run_all_tests() -> None:
    """
    Run tests via pytest with flexible argument forwarding.

    Behavior:
    - No CLI args: Runs all tests (CPU + GPU) by overriding default "-m 'not gpu'" filter
    - With CLI args: Forwards all arguments to pytest directly

    This allows users to run specific tests while maintaining the poetry-based
    test execution policy documented in CLAUDE.md.

    Examples:
        poetry run test-all                      # All tests
        poetry run test-all tests/test_gbm.py    # Specific file
        poetry run test-all -m gpu               # Only GPU tests
        poetry run test-all -v -k "sobol"        # Verbose + keyword filter
    """
    # Get CLI arguments (excluding the script name)
    cli_args = sys.argv[1:]

    if not cli_args:
        # No arguments: run all tests (override default "-m 'not gpu'" from pyproject.toml)
        pytest_args = ["tests", "-m", ""]
    else:
        # Forward all CLI arguments to pytest
        pytest_args = cli_args

    sys.exit(pytest.main(pytest_args))
