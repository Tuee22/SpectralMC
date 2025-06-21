r"""Unit tests for ``spectralmc.cvnn``.

These tests are designed to run under **pytest** *and* type-check cleanly with
``mypy --strict``.  The project ships bespoke Torch stubs, so no
``Any``, ``cast``, or ``type: ignore`` directives are necessary – and none are
present.

Public symbols covered
----------------------
* :class:`spectralmc.cvnn.ComplexLinear`
* :class:`spectralmc.cvnn.zReLU`
* :class:`spectralmc.cvnn.modRELU` (note – case in source is ``modReLU``)
* :class:`spectralmc.cvnn.NaiveComplexBatchNorm`
* :class:`spectralmc.cvnn.CovarianceComplexBatchNorm`
* :class:`spectralmc.cvnn.ComplexSequential`
* :class:`spectralmc.cvnn.ComplexResidual`

Each test focuses on a single behavioural contract – shapes, analytical
expectations, statistical normalisation, or gradient propagation – enabling
rapid diagnostics when regressions appear.
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
    ComplexSequential,
    ComplexResidual,
)

# -----------------------------------------------------------------------------
# Global test-wide constants
# -----------------------------------------------------------------------------

DTYPE: torch.dtype = torch.float32
ATOL: float = 1.0e-5
RTOL: float = 1.0e-4

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _rand(
    *shape: int, requires_grad: bool = False, device: torch.device | str = "cpu"
) -> Tensor:
    """Generate a deterministic random tensor.

    The global RNG state is **not** disturbed – the seed is restored afterwards
    – so tests remain independent.
    """
    state = torch.random.get_rng_state()
    torch.manual_seed(42)
    out = torch.randn(*shape, dtype=DTYPE, device=device, requires_grad=requires_grad)
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
    """Thin wrapper around :pyfunc:`torch.allclose` with diagnostics."""
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff_abs = (actual - expected).abs().max().item()
        diff_rel = (((actual - expected).abs()) / (expected.abs() + 1e-12)).max().item()
        raise AssertionError(
            msg or f"Tensors differ (abs={diff_abs:.2e}, rel={diff_rel:.2e})"
        )


# -----------------------------------------------------------------------------
# ComplexLinear
# -----------------------------------------------------------------------------


def test_complex_linear_manual() -> None:
    """Analytical 2×2 example with non-trivial bias."""
    layer = ComplexLinear(2, 2, bias=True)

    with torch.no_grad():
        layer.real_weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        layer.imag_weight.copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

        # Biases definitely exist because ``bias=True``.
        assert layer.real_bias is not None and layer.imag_bias is not None
        layer.real_bias.copy_(torch.tensor([0.1, 0.2]))
        layer.imag_bias.copy_(torch.tensor([0.3, 0.4]))

    x_r = torch.tensor([[1.0, 1.0]])
    x_i = torch.tensor([[0.5, -0.5]])

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
def test_complex_linear_shapes_and_grad(bias: bool) -> None:
    """Shape checks and gradient propagation."""
    layer = ComplexLinear(5, 3, bias=bias)
    x_r = _rand(7, 5, requires_grad=True)
    x_i = _rand(7, 5, requires_grad=True)

    out_r, out_i = layer(x_r, x_i)
    assert out_r.shape == (7, 3)
    assert out_i.shape == (7, 3)

    (out_r.pow(2).sum() + out_i.pow(2).sum()).backward()

    # All parameter gradients must be finite.
    for param in layer.parameters():
        grad = param.grad
        assert grad is not None and torch.isfinite(grad).all()

    assert x_r.grad is not None and x_i.grad is not None
    assert torch.isfinite(x_r.grad).all() and torch.isfinite(x_i.grad).all()


# -----------------------------------------------------------------------------
# zReLU
# -----------------------------------------------------------------------------


def test_zrelu_masking_and_grad() -> None:
    """zReLU passes first-quadrant values and back-props correct mask."""
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


# -----------------------------------------------------------------------------
# modReLU
# -----------------------------------------------------------------------------


def test_modrelu_thresholding() -> None:
    """Below threshold → 0; above threshold → magnitude-scaled."""
    act = modReLU(num_features=1)
    bias_val = -4.0  # Ensures r+b = 1 for a 3-4-5 triangle input.
    with torch.no_grad():
        act.bias.fill_(bias_val)

    hi_r = torch.tensor([[3.0]])
    hi_i = torch.tensor([[4.0]])
    scaling = (5.0 + bias_val) / 5.0  # (r+b)/r

    out_hi_r, out_hi_i = act(hi_r, hi_i)
    assert_close(out_hi_r, hi_r * scaling)
    assert_close(out_hi_i, hi_i * scaling)

    lo_r = torch.tensor([[0.1]])
    lo_i = torch.tensor([[0.1]])
    out_lo_r, out_lo_i = act(lo_r, lo_i)
    assert_close(out_lo_r, torch.zeros_like(lo_r))
    assert_close(out_lo_i, torch.zeros_like(lo_i))


# -----------------------------------------------------------------------------
# NaiveComplexBatchNorm
# -----------------------------------------------------------------------------


def test_naive_bn_stats() -> None:
    """Training mode normalises each component to ≈0 mean and ≈1 var."""
    bn = NaiveComplexBatchNorm(4)
    bn.train()

    r_in = _rand(256, 4)
    i_in = _rand(256, 4)
    out_r, out_i = bn(r_in, i_in)

    assert_close(out_r.mean(0), torch.zeros(4), atol=1e-2)
    assert_close(out_i.mean(0), torch.zeros(4), atol=1e-2)
    assert_close(out_r.var(0, unbiased=False), torch.ones(4), atol=1e-2)
    assert_close(out_i.var(0, unbiased=False), torch.ones(4), atol=1e-2)


def test_naive_bn_eval_shape() -> None:
    """Eval mode must preserve input shape."""
    bn = NaiveComplexBatchNorm(3)
    bn.train()
    _ = bn(_rand(32, 3), _rand(32, 3))  # prime running statistics

    bn.eval()
    r_in = _rand(8, 3)
    i_in = _rand(8, 3)
    out_r, out_i = bn(r_in, i_in)
    assert out_r.shape == r_in.shape and out_i.shape == i_in.shape


# -----------------------------------------------------------------------------
# CovarianceComplexBatchNorm
# -----------------------------------------------------------------------------


def test_cov_bn_whitening() -> None:
    """Whitening yields ≈0 mean, var≈0.5, and low cross-covariance."""
    bn = CovarianceComplexBatchNorm(6)
    bn.train()

    r_in = _rand(512, 6)
    i_in = _rand(512, 6)
    out_r, out_i = bn(r_in, i_in)

    mean_r = out_r.mean(0)
    mean_i = out_i.mean(0)
    var_r = out_r.var(0, unbiased=False)
    var_i = out_i.var(0, unbiased=False)

    cov_ri = ((out_r - mean_r) * (out_i - mean_i)).mean(0)

    assert_close(mean_r, torch.zeros_like(mean_r), atol=2e-2)
    assert_close(mean_i, torch.zeros_like(mean_i), atol=2e-2)

    target_var = torch.full_like(var_r, 0.5)
    assert_close(var_r, target_var, atol=3e-2)
    assert_close(var_i, target_var, atol=3e-2)

    # Cross-covariance should not exceed variance scale by more than tolerance.
    assert torch.all(cov_ri.abs() <= target_var + 3e-2)


def test_cov_bn_eval_shape() -> None:
    """Inference mode keeps shape unchanged."""
    bn = CovarianceComplexBatchNorm(2)
    bn.train()
    for _ in range(3):
        _ = bn(_rand(32, 2), _rand(32, 2))

    bn.eval()
    r_in = _rand(5, 2)
    i_in = _rand(5, 2)
    out_r, out_i = bn(r_in, i_in)
    assert out_r.shape == r_in.shape and out_i.shape == i_in.shape


# -----------------------------------------------------------------------------
# ComplexSequential
# -----------------------------------------------------------------------------


def test_complex_sequential_flow_and_grad() -> None:
    """Data should flow through sequential stack and back-propagate."""
    seq = ComplexSequential(
        ComplexLinear(3, 3),
        zReLU(),
        ComplexLinear(3, 4),
        modReLU(4),
    )

    r_in = _rand(10, 3, requires_grad=True)
    i_in = _rand(10, 3, requires_grad=True)

    out_r, out_i = seq(r_in, i_in)
    assert out_r.shape == (10, 4) and out_i.shape == (10, 4)

    (out_r.square().mean() + out_i.square().mean()).backward()

    # Ensure some gradient flowed to parameters
    grads: List[Tensor] = [p.grad for p in seq.parameters() if p.grad is not None]
    assert grads and any(torch.isfinite(g).all() for g in grads)


# -----------------------------------------------------------------------------
# ComplexResidual
# -----------------------------------------------------------------------------


def test_complex_residual_identity_when_body_zero() -> None:
    """With zeroed body the residual acts as optional post-activation only."""
    body = ComplexSequential(ComplexLinear(4, 4), modReLU(4))
    res = ComplexResidual(body=body, proj=None, post_act=zReLU())

    # Zero the body parameters.
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
    x_r = torch.tensor([[0.5, -0.5, 0.3, 0.1]])
    x_i = torch.tensor([[0.4, 0.2, -0.1, 0.0]])
    out_r, out_i = res(x_r, x_i)

    # zReLU afterwards keeps only first-quadrant entries.
    mask = (x_r >= 0) & (x_i >= 0)
    assert_close(out_r, x_r * mask)
    assert_close(out_i, x_i * mask)


def test_complex_residual_grad_flow() -> None:
    """Gradients must flow through both residual and body paths."""
    body = ComplexSequential(ComplexLinear(4, 4), zReLU())
    res = ComplexResidual(body=body, proj=None, post_act=None)

    x_r = _rand(6, 4, requires_grad=True)
    x_i = _rand(6, 4, requires_grad=True)
    out_r, out_i = res(x_r, x_i)

    (out_r.square().mean() + out_i.square().mean()).backward()

    # Gradient w.r.t. inputs
    assert x_r.grad is not None and x_i.grad is not None
    assert torch.isfinite(x_r.grad).all() and torch.isfinite(x_i.grad).all()

    # Body parameters must receive gradient as well.
    grads: List[Tensor] = [
        p.grad for p in res.parameters() if p.grad is not None and p.requires_grad
    ]
    assert grads and any(torch.isfinite(g).all() for g in grads)
