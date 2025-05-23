"""Unit-tests for `spectralmc.cvnn` that satisfy *pytest* and *mypy --strict*."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from spectralmc.cvnn import (
    ComplexLinear,
    zReLU,
    modReLU,
    NaiveComplexBatchNorm,
    ResidualBlock,
    CVNN,
    CVNNConfig,
)

DTYPE: torch.dtype = torch.float32
ATOL: float = 1.0e-5
RTOL: float = 1.0e-4


def _rand(*shape: int, requires_grad: bool = False) -> Tensor:
    torch.manual_seed(42)
    return torch.randn(*shape, dtype=DTYPE, requires_grad=requires_grad)


def assert_close(
    a: Tensor, b: Tensor, *, atol: float = ATOL, rtol: float = RTOL
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff_abs = (a - b).abs().max().item()
        diff_rel = ((a - b).abs() / (b.abs() + 1e-12)).max().item()
        raise AssertionError(
            f"Tensor mismatch (abs={diff_abs:.1e}, rel={diff_rel:.1e})"
        )


class ComplexIdentityBN(nn.Module):
    """Fake BN that just returns (r,i), but has a constructor param for `n`."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.n = num_features  # store but never use

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return input_real, input_imag


def test_complex_linear_manual() -> None:
    layer = ComplexLinear(2, 2)
    with torch.no_grad():
        layer.real_weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        layer.imag_weight.copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        assert layer.real_bias is not None and layer.imag_bias is not None
        layer.real_bias.copy_(torch.tensor([0.1, 0.2]))
        layer.imag_bias.copy_(torch.tensor([0.3, 0.4]))

    xr, xi = torch.tensor([[1.0, 1.0]]), torch.tensor([[0.5, -0.5]])
    exp_r = xr @ layer.real_weight.T - xi @ layer.imag_weight.T + layer.real_bias
    exp_i = xr @ layer.imag_weight.T + xi @ layer.real_weight.T + layer.imag_bias
    out_r, out_i = layer(xr, xi)
    assert_close(out_r, exp_r)
    assert_close(out_i, exp_i)


def test_zrelu() -> None:
    act = zReLU()
    xr = torch.tensor([[-1.0, 0.5, 0.2]], requires_grad=True)
    xi = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True)

    or_, oi = act(xr, xi)
    assert_close(or_, torch.tensor([[0.0, 0.0, 0.2]]))
    assert_close(oi, torch.tensor([[0.0, 0.0, 0.1]]))

    (or_.sum() + oi.sum()).backward()
    mask = torch.tensor([[0.0, 0.0, 1.0]])
    xr_grad = xr.grad if xr.grad is not None else torch.zeros_like(mask)
    xi_grad = xi.grad if xi.grad is not None else torch.zeros_like(mask)
    assert_close(xr_grad, mask)
    assert_close(xi_grad, mask)


def test_modrelu() -> None:
    act = modReLU(1)
    with torch.no_grad():
        act.bias.fill_(-4.0)  # 3-4-5 => magnitude=5 => 5 + (-4)=1 => scale=0.2

    or_, oi = act(torch.tensor([[3.0]]), torch.tensor([[4.0]]))
    assert_close(or_, torch.tensor([[0.6]]))
    assert_close(oi, torch.tensor([[0.8]]))


def test_residual_block_identity() -> None:
    # Use a "BN" that is actually an identity so zeroed-out layers + zReLU => partial identity
    block = ResidualBlock(2, activation=zReLU, bn_class=ComplexIdentityBN)
    block.eval()
    with torch.no_grad():
        for mod in block.modules():
            if isinstance(mod, ComplexLinear):
                mod.real_weight.zero_()
                mod.imag_weight.zero_()
                if mod.real_bias is not None:
                    mod.real_bias.zero_()
                if mod.imag_bias is not None:
                    mod.imag_bias.zero_()

    xr = torch.tensor([[0.3, -0.1]])
    xi = torch.tensor([[0.2, 0.2]])
    or_, oi = block(xr, xi)

    mask = (xr >= 0) & (xi >= 0)
    exp_r = xr * mask
    exp_i = xi * mask
    assert_close(or_, exp_r)
    assert_close(oi, exp_i)


@pytest.mark.parametrize(
    "cfg",
    [
        CVNNConfig(input_features=5, output_features=2, hidden_features=8),
        CVNNConfig(
            input_features=5,
            output_features=2,
            hidden_features=8,
            num_residual_blocks=2,
        ),
    ],
)
def test_cvnn_roundtrip_and_grad(cfg: CVNNConfig) -> None:
    model = cfg.build()
    xr = _rand(10, cfg.input_features, requires_grad=True)
    xi = _rand(10, cfg.input_features, requires_grad=True)

    or_, oi = model(xr, xi)
    assert or_.shape == (10, cfg.output_features)
    assert oi.shape == (10, cfg.output_features)

    (or_.square().mean() + oi.square().mean()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)

    clone = model.as_config().build()
    clone.load_state_dict(model.state_dict())
    cr, ci = clone(xr, xi)
    assert_close(cr, or_)
    assert_close(ci, oi)
