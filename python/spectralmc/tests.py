#!/usr/bin/env python
"""
pytest suite with **post‑test printing** of captured stdout / stderr
==================================================================

This is the *simplest* way to see what each `python ‑m <module>` wrote:
we capture its output, run our assertions, **then print the text** once the
process finishes.  No incremental streaming logic needed.

Usage
-----
Run the suite with ``pytest -s`` (or ``--capture=tee-sys``) so pytest
forwards the prints to your terminal:

    $ pytest -s

Replace the placeholder module names (``tool_a``, ``tool_b``) with your
real command‑line packages.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PYTHON = Path(sys.executable)  # interpreter to launch child processes

# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def run_module_cli(module: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run ``python -m <module> args…`` and *capture* its stdout/stderr."""
    cmd = [PYTHON, "-m", module, *args]
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def reverse(s: str) -> str:
    """Sample function to keep our unit‑test section."""
    return s[::-1]

# ---------------------------------------------------------------------------
# integration tests – capture first, print after assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    (
        "module",
        "arg",
        "expect_fragment",
    ),
    [
        ("gbm", None, None),
        ("async_normals", None, None),
    ],
)

def test_cli_smoke(module: str, arg: str, expect_fragment: str | None):
    cp = run_module_cli(module, arg)

    # basic assertions
    assert cp.returncode == 0
    if expect_fragment is not None:
        assert expect_fragment in cp.stdout.lower()

    # now print what the command wrote so the user sees it in the log
    if cp.stdout:
        print(f"\n[stdout from {module}]\n{cp.stdout}")
    if cp.stderr:
        print(f"\n[stderr from {module}]\n{cp.stderr}", file=sys.stderr)


# ---------------------------------------------------------------------------
# ordinary unit test section
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp,expected", [("abc", "cba"), ("racecar", "racecar")])

def test_reverse(inp: str, expected: str):
    assert reverse(inp) == expected
