from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from spectralmc import gbm

# --------------------------------------------------------------------------- #
# CUDA tooling check                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("binary", ["nvidia-smi"])
def test_cuda_tools_available(binary: str) -> None:
    """Require CUDA utilities when a GPU is present."""
    try:
        cp = subprocess.run([binary], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        pytest.skip(f"{binary!r} not found (CPU‑only runner?)")
    else:
        assert cp.returncode == 0


# --------------------------------------------------------------------------- #
# GBM smoke test                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gbm_smoke() -> None:
    sp = gbm.SimulationParams(
        timesteps=4,
        network_size=8,
        batches_per_mc_run=8,
        threads_per_block=32,
        mc_seed=1,
        buffer_size=1,
    )
    engine = gbm.BlackScholes(sp)
    inputs = gbm.BlackScholes.Inputs(X0=100.0, K=101.0, T=1.0, r=0.01, d=0.0, v=0.20)
    res = engine.price(inputs)
    assert res.call_price_intrinsic >= 0.0
    assert res.put_price_intrinsic >= 0.0


# --------------------------------------------------------------------------- #
# Tiny pure‑python example                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("inp", "expected"),
    [("abc", "cba"), ("racecar", "racecar")],
)
def test_reverse(inp: str, expected: str) -> None:
    assert inp[::-1] == expected
