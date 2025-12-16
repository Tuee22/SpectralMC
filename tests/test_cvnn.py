# tests/test_cvnn.py
"""
Precision-parametrised unit tests for :pymod:`spectralmc.cvnn`.

Each test is executed with both ``torch.float32`` and ``torch.float64``
precisions by leveraging the :class:`dtype` enum and
:func:`default_dtype` context-manager from
:pymod:`spectralmc.models.torch`.

The file type-checks under ``mypy --strict`` without suppressions.
"""
from __future__ import annotations

import pytest
from typing import Iterator
from tests.helpers import seed_all_rngs

from spectralmc.cvnn import (
    ComplexLinear,
    ComplexResidual,
    ComplexSequential,
    CovarianceComplexBatchNorm,
    NaiveComplexBatchNorm,
    modReLU,
    zReLU,
)
from spectralmc.models.torch import Device as _dev
from spectralmc.models.torch import FullPrecisionDType as _dt
from spectralmc.models.torch import default_dtype
import torch

Tensor = torch.Tensor


# ─────────────────────────── global constants ────────────────────────────
GPU_DEV: torch.device = _dev.cuda.to_torch()
ATOL: float = 1.0e-5
RTOL: float = 1.0e-4

# ──────────────────────────── pytest fixtures ────────────────────────────


@pytest.fixture(params=[_dt.float32, _dt.float64], ids=["f32", "f64"])
def dt(request: pytest.FixtureRequest) -> Iterator[torch.dtype]:
    """Yield the selected torch.dtype, installing it as the *default*
    inside a :pyfunc:`default_dtype` context so that module construction
    and plain ``torch.tensor`` literals pick up the correct precision.
    """
    enum = request.param  # type: _dt
    with default_dtype(enum.to_torch()):
        yield enum.to_torch()


# ───────────────────────────── utilities ─────────────────────────────────


def _rand(
    *shape: int,
    requires_grad: bool = False,
    dev: torch.device = GPU_DEV,
    dt: torch.dtype | None = None,
) -> Tensor:
    """Return a deterministic random tensor without perturbing the global RNG."""
    state = torch.random.get_rng_state()
    seed_all_rngs(42)
    if dt is None:
        dt = torch.get_default_dtype()
    out = torch.randn(*shape, dtype=dt, device=dev, requires_grad=requires_grad)
    torch.random.set_rng_state(state)
    return out


def assert_close(
    actual: Tensor,
    expected: Tensor,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
    msg: str | None = None,
) -> None:
    """Wrapper around :pyfunc:`torch.allclose` with detailed diagnostics."""
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff_abs = (actual - expected).abs().max().item()
        diff_rel = (((actual - expected).abs()) / (expected.abs() + 1e-12)).max().item()
        raise AssertionError(msg or f"Tensors differ (abs={diff_abs:.2e}, rel={diff_rel:.2e})")


# ──────────────────────────── ComplexLinear ──────────────────────────────


def test_complex_linear_manual(dt: torch.dtype) -> None:
    """Analytical 2 x 2 example with non-trivial bias."""
    layer = ComplexLinear(2, 2, bias=True).to(GPU_DEV)

    with torch.no_grad():
        layer.real_weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=GPU_DEV))
        layer.imag_weight.copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=GPU_DEV))

        assert layer.real_bias is not None, "Real bias should exist when bias=True"
        assert layer.imag_bias is not None, "Imag bias should exist when bias=True"
        assert layer.real_bias.shape == (2,), "Real bias should have shape (out_features,)"
        assert layer.imag_bias.shape == (2,), "Imag bias should have shape (out_features,)"
        layer.real_bias.copy_(torch.tensor([0.1, 0.2], device=GPU_DEV))
        layer.imag_bias.copy_(torch.tensor([0.3, 0.4], device=GPU_DEV))

    x_r = torch.tensor([[1.0, 1.0]], device=GPU_DEV)
    x_i = torch.tensor([[0.5, -0.5]], device=GPU_DEV)

    a = layer.real_weight
    b = layer.imag_weight
    rb = layer.real_bias
    ib = layer.imag_bias
    exp_r = x_r @ a.T - x_i @ b.T + rb
    exp_i = x_r @ b.T + x_i @ a.T + ib

    out_r, out_i = layer(x_r, x_i)
    assert_close(out_r, exp_r)
    assert_close(out_i, exp_i)


@pytest.mark.parametrize("bias", [True, False])
def test_complex_linear_shapes_and_grad(bias: bool, dt: torch.dtype) -> None:
    """Shape checks and gradient propagation."""
    layer = ComplexLinear(5, 3, bias=bias).to(GPU_DEV)
    x_r = _rand(7, 5, requires_grad=True, dt=dt)
    x_i = _rand(7, 5, requires_grad=True, dt=dt)

    out_r, out_i = layer(x_r, x_i)
    assert out_r.shape == (7, 3) and out_i.shape == (7, 3)

    (out_r.pow(2).sum() + out_i.pow(2).sum()).backward()

    for param in layer.parameters():
        grad = param.grad
        assert grad is not None, f"Gradient should be computed for parameter {param.shape}"
        assert torch.isfinite(grad).all(), "Gradient should not contain NaN/Inf"
        assert grad.shape == param.shape, "Gradient should match parameter shape"

    assert x_r.grad is not None, "Real input gradient should be computed"
    assert x_i.grad is not None, "Imag input gradient should be computed"
    assert torch.isfinite(x_r.grad).all(), "Real input gradient should not contain NaN/Inf"
    assert torch.isfinite(x_i.grad).all(), "Imag input gradient should not contain NaN/Inf"


# ───────────────────────────── zReLU ──────────────────────────────────────


def test_zrelu_masking_and_grad(dt: torch.dtype) -> None:
    """zReLU passes first-quadrant values and back-props correct mask."""
    act = zReLU().to(GPU_DEV)
    r_in = torch.tensor([[-1.0, 0.5, 0.2]], requires_grad=True, device=GPU_DEV)
    i_in = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True, device=GPU_DEV)

    out_r, out_i = act(r_in, i_in)
    assert_close(out_r, torch.tensor([[0.0, 0.0, 0.2]], device=GPU_DEV))
    assert_close(out_i, torch.tensor([[0.0, 0.0, 0.1]], device=GPU_DEV))

    (out_r.sum() + out_i.sum()).backward()
    mask = torch.tensor([[0.0, 0.0, 1.0]], device=GPU_DEV)

    assert r_in.grad is not None, "Real input gradient should be computed"
    assert i_in.grad is not None, "Imag input gradient should be computed"
    assert torch.isfinite(r_in.grad).all(), "Real input gradient should not contain NaN/Inf"
    assert torch.isfinite(i_in.grad).all(), "Imag input gradient should not contain NaN/Inf"
    assert_close(r_in.grad, mask)
    assert_close(i_in.grad, mask)


# ───────────────────────────── modReLU ────────────────────────────────────


def test_modrelu_thresholding(dt: torch.dtype) -> None:
    """Below threshold → 0; above threshold → magnitude-scaled."""
    act = modReLU(num_features=1).to(GPU_DEV)
    with torch.no_grad():
        act.bias.fill_(-4.0)  # ensures r+b = 1 for a 3-4-5 input

    hi_r, hi_i = torch.tensor([[3.0]], device=GPU_DEV), torch.tensor([[4.0]], device=GPU_DEV)
    scaling = (5.0 - 4.0) / 5.0  # (r+b)/r

    out_hi_r, out_hi_i = act(hi_r, hi_i)
    assert_close(out_hi_r, hi_r * scaling)
    assert_close(out_hi_i, hi_i * scaling)

    lo_r, lo_i = torch.tensor([[0.1]], device=GPU_DEV), torch.tensor([[0.1]], device=GPU_DEV)
    out_lo_r, out_lo_i = act(lo_r, lo_i)
    assert_close(out_lo_r, torch.zeros_like(lo_r))
    assert_close(out_lo_i, torch.zeros_like(lo_i))


# ─────────────────────── NaiveComplexBatchNorm ───────────────────────────


def test_naive_bn_stats(dt: torch.dtype) -> None:
    """Training mode normalises each component to ≈0 mean and ≈1 var."""
    bn = NaiveComplexBatchNorm(4).to(GPU_DEV)
    bn.train()

    r_in, i_in = _rand(256, 4, dt=dt), _rand(256, 4, dt=dt)
    out_r, out_i = bn(r_in, i_in)

    assert_close(out_r.mean(0), torch.zeros(4, dtype=dt).to(GPU_DEV), atol=1e-2)
    assert_close(out_i.mean(0), torch.zeros(4, dtype=dt).to(GPU_DEV), atol=1e-2)
    assert_close(out_r.var(0, unbiased=False), torch.ones(4, dtype=dt).to(GPU_DEV), atol=1e-2)
    assert_close(out_i.var(0, unbiased=False), torch.ones(4, dtype=dt).to(GPU_DEV), atol=1e-2)


def test_naive_bn_eval_shape(dt: torch.dtype) -> None:
    """Eval mode must preserve input shape."""
    bn = NaiveComplexBatchNorm(3).to(GPU_DEV)
    bn.train()
    _ = bn(_rand(32, 3, dt=dt), _rand(32, 3, dt=dt))  # prime running stats

    bn.eval()
    r_in, i_in = _rand(8, 3, dt=dt), _rand(8, 3, dt=dt)
    out_r, out_i = bn(r_in, i_in)
    assert out_r.shape == r_in.shape and out_i.shape == i_in.shape


# ────────────────── CovarianceComplexBatchNorm ───────────────────────────


def _feature_cov(x_r: Tensor, x_i: Tensor, mean_r: Tensor, mean_i: Tensor) -> Tensor:
    """Return |cov(real, imag)| per feature."""
    return ((x_r - mean_r) * (x_i - mean_i)).mean(0).abs()


def test_cov_bn_whitening(dt: torch.dtype) -> None:
    """Whitening ≈0-means, var≈0.5, and low cross-covariance."""
    bn = CovarianceComplexBatchNorm(6).to(GPU_DEV)
    bn.train()

    r_in, i_in = _rand(512, 6, dt=dt), _rand(512, 6, dt=dt)
    out_r, out_i = bn(r_in, i_in)

    mean_r, mean_i = out_r.mean(0), out_i.mean(0)
    var_r = out_r.var(0, unbiased=False)
    var_i = out_i.var(0, unbiased=False)

    cov_ri = _feature_cov(out_r, out_i, mean_r, mean_i)
    target_var = torch.full_like(var_r, 0.5)

    assert_close(mean_r, torch.zeros_like(mean_r), atol=2e-2)
    assert_close(mean_i, torch.zeros_like(mean_i), atol=2e-2)
    assert_close(var_r, target_var, atol=3e-2)
    assert_close(var_i, target_var, atol=3e-2)
    assert torch.all(cov_ri <= target_var + 3e-2)


def test_cov_bn_eval_shape(dt: torch.dtype) -> None:
    """Inference mode keeps shape unchanged."""
    bn = CovarianceComplexBatchNorm(2).to(GPU_DEV)
    bn.train()
    for _ in range(3):
        _ = bn(_rand(32, 2, dt=dt), _rand(32, 2, dt=dt))

    bn.eval()
    r_in, i_in = _rand(5, 2, dt=dt), _rand(5, 2, dt=dt)
    out_r, out_i = bn(r_in, i_in)
    assert out_r.shape == r_in.shape and out_i.shape == i_in.shape


# ───────────────────────── ComplexSequential ─────────────────────────────


def test_complex_sequential_flow_and_grad(dt: torch.dtype) -> None:
    """Data flows through sequential stack and back-propagates."""
    seq = ComplexSequential(
        ComplexLinear(3, 3),
        zReLU(),
        ComplexLinear(3, 4),
        modReLU(4),
    ).to(GPU_DEV)

    r_in = _rand(10, 3, requires_grad=True, dt=dt)
    i_in = _rand(10, 3, requires_grad=True, dt=dt)

    out_r, out_i = seq(r_in, i_in)
    assert out_r.shape == (10, 4) and out_i.shape == (10, 4)

    (out_r.square().mean() + out_i.square().mean()).backward()
    grads: list[Tensor] = [p.grad for p in seq.parameters() if p.grad is not None]
    assert grads and any(torch.isfinite(g).all() for g in grads)


# ───────────────────────── ComplexResidual ───────────────────────────────


def test_complex_residual_identity_when_body_zero(dt: torch.dtype) -> None:
    """With zeroed body the residual becomes post-activation only."""
    body = ComplexSequential(ComplexLinear(4, 4), modReLU(4))
    res = ComplexResidual(body=body, proj=None, post_act=zReLU()).to(GPU_DEV)

    for m in res.modules():
        if isinstance(m, ComplexLinear):
            with torch.no_grad():
                m.real_weight.zero_()
                m.imag_weight.zero_()
                if m.real_bias is not None:
                    m.real_bias.zero_()
                if m.imag_bias is not None:
                    m.imag_bias.zero_()

    res.eval()
    x_r = torch.tensor([[0.5, -0.5, 0.3, 0.1]], device=GPU_DEV)
    x_i = torch.tensor([[0.4, 0.2, -0.1, 0.0]], device=GPU_DEV)
    out_r, out_i = res(x_r, x_i)

    mask = (x_r >= 0) & (x_i >= 0)
    assert_close(out_r, x_r * mask)
    assert_close(out_i, x_i * mask)


def test_complex_residual_grad_flow(dt: torch.dtype) -> None:
    """Gradients must flow through residual and body paths."""
    body = ComplexSequential(ComplexLinear(4, 4), zReLU())
    res = ComplexResidual(body=body, proj=None, post_act=None).to(GPU_DEV)

    x_r = _rand(6, 4, requires_grad=True, dt=dt)
    x_i = _rand(6, 4, requires_grad=True, dt=dt)
    out_r, out_i = res(x_r, x_i)

    (out_r.square().mean() + out_i.square().mean()).backward()

    assert x_r.grad is not None, "Real input gradient should be computed"
    assert x_i.grad is not None, "Imag input gradient should be computed"
    assert x_r.grad.shape == x_r.shape, "Gradient shape should match input shape"
    assert x_i.grad.shape == x_i.shape, "Gradient shape should match input shape"
    assert torch.isfinite(x_r.grad).all(), "Real input gradient should not contain NaN/Inf"
    assert torch.isfinite(x_i.grad).all(), "Imag input gradient should not contain NaN/Inf"

    grads: list[Tensor] = [
        p.grad for p in res.parameters() if p.grad is not None and p.requires_grad
    ]
    assert grads and any(torch.isfinite(g).all() for g in grads)
