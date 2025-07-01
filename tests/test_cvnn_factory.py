# src/spectralmc/test_cvnn_factory.py
"""Unit tests for ``spectralmc.cvnn_factory``.

All tests run under **pytest** *and* type‑check cleanly with

    mypy --strict src/spectralmc

They assume that a CUDA device is present – absence of a GPU is considered
a **hard failure** because ``cvnn_factory.build_model`` is documented to
deploy to GPU in production.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pytest
import torch
from torch import Tensor, nn

from spectralmc.cvnn_factory import (
    ActivationCfg,
    ActivationKind,
    CVNNConfig,
    CovBNCfg,
    LayerKind,
    LinearCfg,
    NaiveBNCfg,
    ResidualCfg,
    SequentialCfg,
    build_model,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants shared across tests
# ──────────────────────────────────────────────────────────────────────────────
_DEVICE_CUDA: torch.device = torch.device("cuda:0")
assert torch.cuda.is_available(), "These tests require a CUDA‑capable device"

_DTYPES: Tuple[torch.dtype, ...] = (torch.float32, torch.float64)

ATOL: float = 1.0e-5
RTOL: float = 1.0e-4

N_IN: int = 6
N_OUT: int = 4
BATCH: int = 12
OPT_LR: float = 1.0e-3

# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────


def _rand(
    *shape: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = False,
) -> Tensor:
    """Deterministic random tensor that restores the global RNG afterwards."""
    state = torch.random.get_rng_state()
    torch.manual_seed(12345)  # fixed seed → test independence
    out = torch.randn(*shape, dtype=dtype, device=device, requires_grad=requires_grad)
    torch.random.set_rng_state(state)
    return out


def _assert_state_dict_equal(a: Dict[str, Tensor], b: Dict[str, Tensor]) -> None:
    """Strict equality (dtype‑ and bit‑exact) of two *state_dict*s."""
    assert a.keys() == b.keys()
    for k in a:
        t1, t2 = a[k], b[k]
        assert t1.dtype == t2.dtype and t1.shape == t2.shape
        if t1.is_floating_point():
            assert torch.allclose(t1, t2, atol=0.0, rtol=0.0), f"Mismatch in {k}"
        else:  # e.g. integer buffers
            assert torch.equal(t1, t2), f"Mismatch in {k}"


def _example_cfg(dtype: torch.dtype, seed: int = 314159) -> CVNNConfig:
    """A representative, yet compact, CVNN topology."""
    return CVNNConfig(
        dtype=dtype,
        seed=seed,
        layers=[
            # Linear → modReLU
            LinearCfg(
                width=8,
                bias=True,
                activation=ActivationCfg(kind=ActivationKind.MOD_RELU),
            ),
            # Naive BN
            NaiveBNCfg(),
            # Residual (Linear → Cov‑BN) + zReLU
            ResidualCfg(
                body=SequentialCfg(
                    layers=[
                        LinearCfg(width=8, bias=False),
                        CovBNCfg(),
                    ],
                ),
                activation=ActivationCfg(kind=ActivationKind.Z_RELU),
            ),
        ],
        # Final activation after (implicit) output projection
        final_activation=ActivationCfg(kind=ActivationKind.MOD_RELU),
    )


def _single_train_step(
    cfg: CVNNConfig,
    dtype: torch.dtype,
) -> Dict[str, Tensor]:
    """Materialise → forward → backward → optimiser step → return state‐dict."""

    # Build model directly on CUDA with the requested precision.
    model: nn.Module = build_model(
        n_inputs=N_IN,
        n_outputs=N_OUT,
        cfg=cfg,
        device=_DEVICE_CUDA,
    )
    model.train()

    # Simple optimiser – stateless (no parameter momentum) for easier equality.
    opt = torch.optim.SGD(model.parameters(), lr=OPT_LR)

    x_r = _rand(BATCH, N_IN, dtype=dtype, device=_DEVICE_CUDA, requires_grad=True)
    x_i = _rand(BATCH, N_IN, dtype=dtype, device=_DEVICE_CUDA, requires_grad=True)

    y_r: Tensor
    y_i: Tensor
    y_r, y_i = model(x_r, x_i)
    loss: Tensor = y_r.square().mean() + y_i.square().mean()

    loss.backward()
    opt.step()

    # We detach all tensors so that the returned structure contains plain *values*.
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", _DTYPES)
def test_device_and_dtype_placement(dtype: torch.dtype) -> None:
    """Every parameter must respect the requested CUDA device and dtype."""
    cfg = _example_cfg()
    model = build_model(
        n_inputs=N_IN,
        n_outputs=N_OUT,
        cfg=cfg,
        device=_DEVICE_CUDA,
    )

    for p in model.parameters():
        assert p.is_cuda, "Parameter not on CUDA device"
        assert p.dtype is dtype, "Parameter dtype mismatch"


@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward_backward_pass(dtype: torch.dtype) -> None:
    """A complete forward + backward pass should yield finite gradients."""
    cfg = _example_cfg()

    model = build_model(
        n_inputs=N_IN,
        n_outputs=N_OUT,
        cfg=cfg,
        device=_DEVICE_CUDA,
    )
    model.train()

    x_r = _rand(BATCH, N_IN, dtype=dtype, device=_DEVICE_CUDA, requires_grad=True)
    x_i = _rand(BATCH, N_IN, dtype=dtype, device=_DEVICE_CUDA, requires_grad=True)

    y_r, y_i = model(x_r, x_i)
    assert y_r.shape == (BATCH, N_OUT) and y_i.shape == (BATCH, N_OUT)

    loss = (y_r.square().mean() + y_i.square().mean()).to(dtype)
    loss.backward()

    grads: List[Tensor] = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "No gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "Non‑finite gradient detected"


@pytest.mark.parametrize("dtype", _DTYPES)
def test_full_reproducibility(dtype: torch.dtype) -> None:
    """Two *independent* training runs from the same config must be identical."""
    cfg = _example_cfg()

    state_run_1 = _single_train_step(cfg, dtype)
    state_run_2 = _single_train_step(cfg, dtype)

    _assert_state_dict_equal(state_run_1, state_run_2)


def test_config_json_roundtrip() -> None:
    """Serialising and deserialising a config via JSON must be loss‑less."""
    cfg = _example_cfg(seed=271828)
    json_str = cfg.model_dump_json()
    cfg_round = CVNNConfig.model_validate_json(json_str)
    # Pydantic guarantees equality semantics
    assert cfg == cfg_round

    # Sanity – JSON must be valid and contain the top‑level 'layers' key.
    parsed = json.loads(json_str)
    assert "layers" in parsed and parsed["seed"] == 271828


# ──────────────────────────────────────────────────────────────────────────────
# EOF
# ──────────────────────────────────────────────────────────────────────────────
