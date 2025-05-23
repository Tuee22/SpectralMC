# src/spectralmc/cvnn.py
"""Complex-valued feed-forward network utilities & reference implementation."""

from __future__ import annotations

import math
from typing import (
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
    cast,
)

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict

__all__: Sequence[str] = [
    "ComplexLinear",
    "zReLU",
    "modReLU",
    "NaiveComplexBatchNorm",
    "ResidualBlock",
    "CVNNConfig",
    "CVNN",
]


###############################################################################
# Layers & activations                                                        #
###############################################################################


class ComplexLinear(nn.Module):
    """Fully-connected linear transform for complex inputs."""

    real_weight: nn.Parameter
    imag_weight: nn.Parameter
    real_bias: Optional[nn.Parameter]
    imag_bias: Optional[nn.Parameter]

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.real_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.imag_weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.real_bias = nn.Parameter(torch.empty(out_features))
            self.imag_bias = nn.Parameter(torch.empty(out_features))
        else:
            self.real_bias = None
            self.imag_bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.real_weight)
        nn.init.xavier_uniform_(self.imag_weight)
        if self.real_bias is not None:
            nn.init.zeros_(self.real_bias)
        if self.imag_bias is not None:
            nn.init.zeros_(self.imag_bias)

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r = input_real @ self.real_weight.T - input_imag @ self.imag_weight.T
        i = input_real @ self.imag_weight.T + input_imag @ self.real_weight.T
        if self.real_bias is not None and self.imag_bias is not None:
            r = r + self.real_bias
            i = i + self.imag_bias
        return r, i


class zReLU(nn.Module):
    """Pass values in the *first quadrant* only."""

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (input_real >= 0) & (input_imag >= 0)
        return input_real * mask, input_imag * mask


class modReLU(nn.Module):
    """Magnitude-threshold activation."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.sqrt(input_real**2 + input_imag**2 + 1e-9)
        thr = torch.relu(mag + self.bias.unsqueeze(0))
        scale = thr / mag
        return scale * input_real, scale * input_imag


class NaiveComplexBatchNorm(nn.Module):
    """Independent BN on real & imag parts."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features)
        self.bn_imag = nn.BatchNorm1d(num_features)

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.bn_real(input_real), self.bn_imag(input_imag)


###############################################################################
# Helper factories (single source of truth)                                   #
###############################################################################


@runtime_checkable
class _ActProto(Protocol):
    def __call__(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...


def _make_act(
    activation: Union[Type[nn.Module], nn.Module], num_features: int
) -> _ActProto:
    """Make an activation instance that takes (real, imag) -> (real, imag)."""
    if isinstance(activation, type):
        try:
            act: nn.Module = activation(num_features)
        except TypeError:
            act = activation()
    else:
        act = activation
    return cast(_ActProto, act)


def _make_bn(bn_cls: Union[Type[nn.Module], nn.Module], n: int) -> nn.Module:
    if isinstance(bn_cls, type):
        return bn_cls(n)
    else:
        return bn_cls


###############################################################################
# Residual block & full network                                               #
###############################################################################


class ResidualBlock(nn.Module):
    """Two-layer residual block for complex data."""

    def __init__(
        self,
        num_features: int,
        *,
        activation: Union[Type[nn.Module], nn.Module] = modReLU,
        bn_class: Union[Type[nn.Module], nn.Module] = NaiveComplexBatchNorm,
    ) -> None:
        super().__init__()
        self.linear1 = ComplexLinear(num_features, num_features)
        self.bn1 = _make_bn(bn_class, num_features)
        self.act1 = _make_act(activation, num_features)

        self.linear2 = ComplexLinear(num_features, num_features)
        self.bn2 = _make_bn(bn_class, num_features)
        self.act2 = _make_act(activation, num_features)

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res_r, res_i = input_real, input_imag
        r, i = self.linear1(input_real, input_imag)
        r, i = self.bn1(r, i)
        r, i = self.act1(r, i)

        r2, i2 = self.linear2(r, i)
        r2, i2 = self.bn2(r2, i2)
        r2, i2 = r2 + r, i2 + i
        return self.act2(r2, i2)


###############################################################################
# Config & CVNN                                                               #
###############################################################################


class CVNNConfig(BaseModel):
    """Frozen architectural description of a `CVNN`."""

    input_features: int
    output_features: int
    hidden_features: int
    num_residual_blocks: int = 1

    model_config = ConfigDict(frozen=True)

    def build(self) -> CVNN:
        """Instantiate a fresh network from this config."""
        return CVNN(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_features=self.hidden_features,
            num_residual_blocks=self.num_residual_blocks,
        )


class CVNN(nn.Module):
    """Simple residual CVNN used by `spectralmc.gbm_trainer`."""

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_features: int,
        *,
        num_residual_blocks: int = 1,
        activation: Union[Type[nn.Module], nn.Module] = modReLU,
        bn_class: Union[Type[nn.Module], nn.Module] = NaiveComplexBatchNorm,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_features = hidden_features

        self.input_linear = ComplexLinear(input_features, hidden_features)
        self.input_bn = _make_bn(bn_class, hidden_features)
        self.input_act = _make_act(activation, hidden_features)

        self.residuals = nn.ModuleList(
            [
                ResidualBlock(hidden_features, activation=activation, bn_class=bn_class)
                for _ in range(num_residual_blocks)
            ]
        )
        self.output_linear = ComplexLinear(hidden_features, output_features)

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r, i = self.input_linear(input_real, input_imag)
        r, i = self.input_bn(r, i)
        r, i = self.input_act(r, i)
        for block in self.residuals:
            r, i = block(r, i)
        return self.output_linear(r, i)

    def as_config(self) -> CVNNConfig:
        """Return a config that can rebuild **this** architecture."""
        return CVNNConfig(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_features=self.hidden_features,
            num_residual_blocks=len(self.residuals),
        )
