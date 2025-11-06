"""Test runner for poetry scripts."""

import sys

import pytest


def run_all_tests() -> None:
    """
    Run all tests including GPU tests.

    This overrides the default pytest configuration which excludes GPU tests
    via '-m "not gpu"' by passing an empty marker expression.
    """
    # Override the marker filter to include GPU tests
    # The -m "" argument clears the default "-m 'not gpu'" from addopts
    sys.exit(pytest.main(["tests", "-m", ""]))
