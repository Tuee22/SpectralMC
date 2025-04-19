#!/usr/bin/env python
"""
pytest suite with **post‑test printing** of captured stdout / stderr
==================================================================

Run it with::

    pytest -s

The ``-s`` (or ``--capture=tee-sys``) flag lets you see the captured output that
each ``python -m <module>`` emits.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PYTHON = Path(sys.executable)  # interpreter used to spawn child processes


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_module_cli(
    module: str, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """
    Launch ``python -m <module> [args…]`` and capture its stdout/stderr.

    Empty strings are ignored so parametrised tests can pass "" instead of None
    without upsetting subprocess.run.
    """
    cmd: list[str] = [str(PYTHON), "-m", module]
    cmd.extend(a for a in args if a)  # drop falsy args ("" / None)

    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def reverse(s: str) -> str:
    """Simple utility to demonstrate a small unit‑test section."""
    return s[::-1]


# ---------------------------------------------------------------------------
# system-level sanity check
# ---------------------------------------------------------------------------


def test_nvidia_smi() -> None:
    """Ensure nvidia-smi is available and print its output."""
    try:
        cp = subprocess.run(
            ["nvidia-smi"],
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        pytest.fail(
            "❌ `nvidia-smi` not found inside the container.\n"
            "This usually means the container is not running with GPU support.\n\n"
            "✅ Make sure you've installed the NVIDIA Container Toolkit and started the container with GPU access:\n"
            "   > sudo docker compose --profile gpu up\n\n"
            "🔧 See the setup instructions in `scripts/install_docker_cuda.sh` or the README."
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"❌ `nvidia-smi` command failed with return code {e.returncode}.\n"
            f"stderr:\n{e.stderr}\n\n"
            "This might mean the container does not have access to the GPU.\n"
            "✅ Ensure the NVIDIA runtime is enabled and the container has GPU access."
        )

    if cp.stdout:
        print(f"\n[stdout from nvidia-smi]\n{cp.stdout}")
    if cp.stderr:
        print(f"\n[stderr from nvidia-smi]\n{cp.stderr}", file=sys.stderr)


# ---------------------------------------------------------------------------
# integration tests – run module CLIs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "spectralmc.gbm",
        "spectralmc.async_normals",
    ],
)
def test_cli_smoke(module: str) -> None:
    """Ensure each ``python -m <module>`` exits successfully and show its output."""
    cp = run_module_cli(module)

    assert cp.returncode == 0

    if cp.stdout:
        print(f"\n[stdout from {module}]\n{cp.stdout}")
    if cp.stderr:
        print(f"\n[stderr from {module}]\n{cp.stderr}", file=sys.stderr)


# ---------------------------------------------------------------------------
# ordinary unit‑test section
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inp", "expected"),
    [
        ("abc", "cba"),
        ("racecar", "racecar"),
    ],
)
def test_reverse(inp: str, expected: str) -> None:
    """Basic example to keep a pure unit test in the suite."""
    assert reverse(inp) == expected
