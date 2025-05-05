# tests/test_cvnn.py
"""Unit-tests for ``spectralmc.cvnn`` that satisfy
`pytest` and `mypy --strict` without using ``typing.cast``.
"""
from __future__ import annotations

import math
from typing import List

import pytest
import torch
from torch import Tensor, nn

from spectralmc.cvnn import (
    ComplexLinear,
    zReLU,
    modReLU,
    NaiveComplexBatchNorm,
    CovarianceComplexBatchNorm,
    ResidualBlock,
    SimpleCVNN,
    CVNN,
)

DTYPE: torch.dtype = torch.float32
ATOL: float = 1.0e-5
RTOL: float = 1.0e-4


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#


def _rand(
    *shape: int, requires_grad: bool = False, device: torch.device | str = "cpu"
) -> Tensor:
    """Deterministic random tensor."""
    torch.manual_seed(42)
    return torch.randn(*shape, dtype=DTYPE, device=device, requires_grad=requires_grad)


def assert_close(
    actual: Tensor,
    expected: Tensor,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
    msg: str | None = None,
) -> None:
    """Thin wrapper over :func:`torch.allclose` with helpful diagnostics."""
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff_abs = (actual - expected).abs().max().item()
        diff_rel = (
            (actual - expected).abs() / (expected.abs() + 1e-12)
        ).max().item()
        raise AssertionError(
            msg or f"Tensors differ (abs={diff_abs:.2e}, rel={diff_rel:.2e})"
        )


# -----------------------------------------------------------------------------#
# ComplexLinear
# -----------------------------------------------------------------------------#


def test_complex_linear_manual() -> None:
    """Analytical 2 × 2 check."""
    layer = ComplexLinear(2, 2, bias=True)

    with torch.no_grad():
        layer.real_weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        layer.imag_weight.copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

        # Bias parameters are definitely present when bias=True.
        assert layer.real_bias is not None and layer.imag_bias is not None
        rbias_param: Tensor = layer.real_bias
        ibias_param: Tensor = layer.imag_bias
        rbias_param.copy_(torch.tensor([0.1, 0.2]))
        ibias_param.copy_(torch.tensor([0.3, 0.4]))

    x_r = torch.tensor([[1.0, 1.0]])
    x_i = torch.tensor([[0.5, -0.5]])

    a, b = layer.real_weight, layer.imag_weight
    exp_r = x_r @ a.T - x_i @ b.T + rbias_param
    exp_i = x_r @ b.T + x_i @ a.T + ibias_param

    out_r, out_i = layer(x_r, x_i)
    assert_close(out_r, exp_r)
    assert_close(out_i, exp_i)


@pytest.mark.parametrize("bias", [True, False])
def test_complex_linear_shapes_and_grad(bias: bool) -> None:
    """Shape preservation and gradients."""
    layer = ComplexLinear(5, 3, bias=bias)
    x_r = _rand(7, 5, requires_grad=True)
    x_i = _rand(7, 5, requires_grad=True)

    out_r, out_i = layer(x_r, x_i)
    assert out_r.shape == (7, 3)
    assert out_i.shape == (7, 3)

    (out_r.pow(2).sum() + out_i.pow(2).sum()).backward()

    for p in layer.parameters():
        grad = p.grad
        assert grad is not None and torch.isfinite(grad).all()

    assert x_r.grad is not None and torch.isfinite(x_r.grad).all()
    assert x_i.grad is not None and torch.isfinite(x_i.grad).all()


# -----------------------------------------------------------------------------#
# zReLU
# -----------------------------------------------------------------------------#


def test_zrelu_masking_and_grad() -> None:
    """First-quadrant pass & matching grads."""
    act = zReLU()
    r_in = torch.tensor([[-1.0, 0.5, 0.2]], requires_grad=True)
    i_in = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True)

    out_r, out_i = act(r_in, i_in)
    assert_close(out_r, torch.tensor([[0.0, 0.0, 0.2]]))
    assert_close(out_i, torch.tensor([[0.0, 0.0, 0.1]]))

    (out_r.sum() + out_i.sum()).backward()
    mask = torch.tensor([[0.0, 0.0, 1.0]])
    assert r_in.grad is not None and i_in.grad is not None
    assert_close(r_in.grad, mask)
    assert_close(i_in.grad, mask)


# -----------------------------------------------------------------------------#
# modReLU
# -----------------------------------------------------------------------------#


def test_modrelu_thresholding() -> None:
    """Below threshold → 0; above → rescaled."""
    act = modReLU(num_features=1)
    bias_val = -4.0  # r+b = 1 for 3-4-5 triangle
    with torch.no_grad():
        act.bias.fill_(bias_val)

    hi_r, hi_i = torch.tensor([[3.0]]), torch.tensor([[4.0]])
    scaling = (5.0 + bias_val) / 5.0  # (r+b)/r

    out_r, out_i = act(hi_r, hi_i)
    assert_close(out_r, hi_r * scaling)
    assert_close(out_i, hi_i * scaling)

    lo_r, lo_i = torch.tensor([[0.1]]), torch.tensor([[0.1]])
    z_r, z_i = act(lo_r, lo_i)
    assert_close(z_r, torch.zeros_like(lo_r))
    assert_close(z_i, torch.zeros_like(lo_i))


# -----------------------------------------------------------------------------#
# NaiveComplexBatchNorm
# -----------------------------------------------------------------------------#


def test_naive_bn_stats() -> None:
    """Training → ≈0 mean & ≈1 var per component."""
    bn = NaiveComplexBatchNorm(4)
    bn.train()

    r_in, i_in = _rand(128, 4), _rand(128, 4)
    out_r, out_i = bn(r_in, i_in)

    assert_close(out_r.mean(0), torch.zeros(4), atol=1e-2)
    assert_close(out_i.mean(0), torch.zeros(4), atol=1e-2)
    assert_close(out_r.var(0, unbiased=False), torch.ones(4), atol=1e-2)
    assert_close(out_i.var(0, unbiased=False), torch.ones(4), atol=1e-2)


def test_naive_bn_eval_shape() -> None:
    """Eval-mode keeps shape."""
    bn = NaiveComplexBatchNorm(3)
    bn.train()
    _ = bn(_rand(32, 3), _rand(32, 3))  # prime stats

    bn.eval()
    r_in, i_in = _rand(8, 3), _rand(8, 3)
    out_r, out_i = bn(r_in, i_in)
    assert out_r.shape == r_in.shape
    assert out_i.shape == i_in.shape


# -----------------------------------------------------------------------------#
# CovarianceComplexBatchNorm
# -----------------------------------------------------------------------------#


def test_cov_bn_whitening() -> None:
    """Whitening + γ diag ⇒ var≈0.25, |cov|≤var."""
    bn = CovarianceComplexBatchNorm(6)
    bn.train()

    r_in, i_in = _rand(256, 6), _rand(256, 6)
    out_r, out_i = bn(r_in, i_in)

    mean_r, mean_i = out_r.mean(0), out_i.mean(0)
    var_r = out_r.var(0, unbiased=False)
    var_i = out_i.var(0, unbiased=False)
    cov_ri = ((out_r - mean_r) * (out_i - mean_i)).mean(0)

    assert_close(mean_r, torch.zeros_like(mean_r), atol=1e-2)
    assert_close(mean_i, torch.zeros_like(mean_i), atol=1e-2)

    target_var = torch.full_like(var_r, 0.25)
    assert_close(var_r, target_var, atol=2e-2)
    assert_close(var_i, target_var, atol=2e-2)

    assert torch.all(cov_ri.abs() <= target_var + 2e-2)


def test_cov_bn_eval_shape() -> None:
    """Eval-mode keeps shape."""
    bn = CovarianceComplexBatchNorm(2)
    bn.train()
    for _ in range(3):
        _ = bn(_rand(16, 2), _rand(16, 2))

    bn.eval()
    r_in, i_in = _rand(4, 2), _rand(4, 2)
    out_r, out_i = bn(r_in, i_in)
    assert out_r.shape == r_in.shape
    assert out_i.shape == i_in.shape


# -----------------------------------------------------------------------------#
# ResidualBlock
# -----------------------------------------------------------------------------#


def test_residual_block_identity_when_zeroed() -> None:
    """Zeroed internals ⇒ identity skip + zReLU."""
    block = ResidualBlock(3, activation=zReLU, bn_class=NaiveComplexBatchNorm)

    for mod in block.modules():
        if isinstance(mod, ComplexLinear):
            with torch.no_grad():
                mod.real_weight.zero_()
                mod.imag_weight.zero_()
                if mod.real_bias is not None:
                    mod.real_bias.zero_()
                if mod.imag_bias is not None:
                    mod.imag_bias.zero_()

    block.eval()
    x_r = torch.tensor([[0.2, 0.4, 0.6]])
    x_i = torch.tensor([[0.1, 0.2, 0.3]])
    out_r, out_i = block(x_r, x_i)

    mask = (x_r >= 0) & (x_i >= 0)
    assert_close(out_r, x_r * mask)
    assert_close(out_i, x_i * mask)


# -----------------------------------------------------------------------------#
# End-to-end models
# -----------------------------------------------------------------------------#


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (SimpleCVNN, {"hidden_features": 8}),
        (CVNN, {"hidden_features": 8, "num_residual_blocks": 2}),
    ],
)
def test_models_forward_and_grad(model_cls: type[nn.Module], kwargs: dict[str, int]) -> None:
    """Forward/backward sanity for full models."""
    model = model_cls(input_features=5, output_features=2, **kwargs)

    x_r = _rand(10, 5, requires_grad=True)
    x_i = _rand(10, 5, requires_grad=True)

    o_r, o_i = model(x_r, x_i)
    assert o_r.shape == (10, 2)
    assert o_i.shape == (10, 2)

    (o_r.square().mean() + o_i.square().mean()).backward()

    grads: List[Tensor] = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad)

    assert any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)