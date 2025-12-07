# tests/test_cvnn_factory.py
"""
Unit tests for ``spectralmc.cvnn_factory``.

They require a CUDA device - the absence of one is considered a hard failure
because production models are expected to run on the GPU.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import Tensor, nn

from spectralmc.cvnn_factory import (
    ActivationCfg,
    ActivationKind,
    CovBNCfg,
    CVNNConfig,
    ExplicitWidth,
    LinearCfg,
    NaiveBNCfg,
    ResidualCfg,
    SequentialCfg,
    build_model,
)
from spectralmc.models.torch import DType  # ← project's strongly-typed dtype enum
from spectralmc.result import Failure, Result, Success
from typing import TypeVar


# ──────────────────────────────────────────────────────────────────────────────
# Constants shared across tests
# ──────────────────────────────────────────────────────────────────────────────
_DEVICE_CUDA: torch.device = torch.device("cuda:0")
assert torch.cuda.is_available(), "These tests require a CUDA-capable device"

_DTYPES: tuple[DType, ...] = (DType.float32, DType.float64)

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


def _assert_state_dict_equal(a: dict[str, Tensor], b: dict[str, Tensor]) -> None:
    """Strict equality (dtype- and bit-exact) of two *state_dict*s."""
    assert a.keys() == b.keys()
    for k in a:
        t1, t2 = a[k], b[k]
        assert t1.dtype == t2.dtype and t1.shape == t2.shape
        if t1.is_floating_point():
            assert torch.allclose(t1, t2, atol=0.0, rtol=0.0), f"Mismatch in {k}"
        else:  # e.g. integer buffers
            assert torch.equal(t1, t2), f"Mismatch in {k}"


T = TypeVar("T")


def _expect_success(result: Result[T, object]) -> T:
    match result:
        case Success(value):
            return value
        case Failure(error):
            raise AssertionError(f"Unexpected CVNN factory failure: {error}")


def _build_test_model(cfg: CVNNConfig, *, dtype: DType) -> nn.Module:
    return _expect_success(build_model(n_inputs=N_IN, n_outputs=N_OUT, cfg=cfg)).to(
        _DEVICE_CUDA, dtype.to_torch()
    )


def _example_cfg(dtype: DType = DType.float32, seed: int = 314159) -> CVNNConfig:
    """A representative, yet compact, CVNN topology."""
    return CVNNConfig(
        dtype=dtype,
        seed=seed,
        layers=[
            # Linear → modReLU
            LinearCfg(
                width=ExplicitWidth(value=8),
                bias=True,
                activation=ActivationCfg(kind=ActivationKind.MOD_RELU),
            ),
            # Naive BN
            NaiveBNCfg(),
            # Residual (Linear → Cov-BN) + zReLU
            ResidualCfg(
                body=SequentialCfg(
                    layers=[
                        LinearCfg(width=ExplicitWidth(value=8), bias=False),
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
    dtype: DType,
) -> dict[str, Tensor]:
    """Materialise → forward → backward → optimiser step → return state-dict."""
    model: nn.Module = _build_test_model(cfg, dtype=dtype)

    model.train()

    # Simple optimiser - stateless (no parameter momentum) for easier equality.
    opt = torch.optim.SGD(model.parameters(), lr=OPT_LR)

    x_r = _rand(
        BATCH,
        N_IN,
        dtype=dtype.to_torch(),
        device=_DEVICE_CUDA,
        requires_grad=True,
    )
    x_i = _rand(
        BATCH,
        N_IN,
        dtype=dtype.to_torch(),
        device=_DEVICE_CUDA,
        requires_grad=True,
    )

    y_r: Tensor
    y_i: Tensor
    y_r, y_i = model(x_r, x_i)
    loss: Tensor = y_r.square().mean() + y_i.square().mean()

    loss.backward()
    opt.step()

    # Detach & move to CPU so equality checks ignore device placement.
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", _DTYPES)
def test_device_and_dtype_placement(dtype: DType) -> None:
    """Every parameter must respect the requested CUDA device and dtype."""
    cfg = _example_cfg(dtype)
    model = _build_test_model(cfg, dtype=dtype)

    for p in model.parameters():
        assert p.is_cuda, "Parameter not on CUDA device"
        assert p.dtype is dtype.to_torch(), "Parameter dtype mismatch"


@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward_backward_pass(dtype: DType) -> None:
    """A complete forward + backward pass should yield finite gradients."""
    cfg = _example_cfg(dtype)

    model = _build_test_model(cfg, dtype=dtype)
    model.train()

    x_r = _rand(
        BATCH,
        N_IN,
        dtype=dtype.to_torch(),
        device=_DEVICE_CUDA,
        requires_grad=True,
    )
    x_i = _rand(
        BATCH,
        N_IN,
        dtype=dtype.to_torch(),
        device=_DEVICE_CUDA,
        requires_grad=True,
    )

    y_r, y_i = model(x_r, x_i)
    assert y_r.shape == (BATCH, N_OUT) and y_i.shape == (BATCH, N_OUT)

    loss = (y_r.square().mean() + y_i.square().mean()).to(dtype.to_torch())
    loss.backward()

    grads: list[Tensor] = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "No gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradient detected"


@pytest.mark.parametrize("dtype", _DTYPES)
def test_full_reproducibility(dtype: DType) -> None:
    """Two *independent* training runs from the same config must be identical."""
    cfg = _example_cfg(dtype)

    state_run_1 = _single_train_step(cfg, dtype)
    state_run_2 = _single_train_step(cfg, dtype)

    _assert_state_dict_equal(state_run_1, state_run_2)


def test_config_json_roundtrip() -> None:
    """Serialising and deserialising a config via JSON must be loss-less."""
    cfg = _example_cfg(dtype=DType.float64, seed=271828)
    json_str = cfg.model_dump_json()
    cfg_round = CVNNConfig.model_validate_json(json_str)

    # Round-trip must preserve field values exactly
    assert cfg.model_dump(mode="json") == cfg_round.model_dump(mode="json")

    # Sanity - JSON must be valid and contain the top-level 'layers' key.
    parsed = json.loads(json_str)
    assert "layers" in parsed and parsed["seed"] == 271828


# ──────────────────────────────────────────────────────────────────────────────
# EOF
# ──────────────────────────────────────────────────────────────────────────────
