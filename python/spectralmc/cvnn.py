# python/spectralmc/cvnn.py

"""
Complex-valued neural networks (CVNNs) using PyTorch. Includes zReLU, modReLU,
batchnorm, residual blocks, etc. We cast layer(...) to torch.Tensor in forward
so mypy doesn't complain about returning Any.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, cast


class zReLU(nn.Module):
    """
    zReLU activation for complex inputs: zero out negative real or imag.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x.real >= 0) & (x.imag >= 0)
        return x * mask.float()


class modReLU(nn.Module):
    """
    modReLU activation for complex inputs. A bias is applied to the modulus.
    """

    def __init__(self, bias_init: float = 0.1) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([bias_init], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        activated_modulus = F.relu(magnitude + self.bias)
        phase = torch.angle(x)
        return torch.polar(activated_modulus, phase)


class ComplexBatchNorm(nn.Module):
    """
    Naive complex batch normalization, BN on magnitude + BN on phase.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.mag_bn = nn.BatchNorm1d(num_features)
        self.phase_bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        nm = self.mag_bn(magnitude)
        np_ = self.phase_bn(phase)
        return torch.polar(nm, np_)


class ResidualBlock(nn.Module):
    """
    A residual block for complex-valued networks.
    """

    def __init__(
        self, in_features: int, out_features: int, bias_init: float = 0.1
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = ComplexBatchNorm(out_features)
        self.modrelu1 = modReLU(bias_init)

        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = ComplexBatchNorm(out_features)
        self.modrelu2 = modReLU(bias_init)

        # match_dimensions can be None or a Sequential
        self.match_dimensions: Optional[nn.Sequential] = None
        if in_features != out_features:
            self.match_dimensions = nn.Sequential(
                nn.Linear(in_features, out_features), ComplexBatchNorm(out_features)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.modrelu1(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.modrelu2(out)

        if self.match_dimensions is not None:
            residual = self.match_dimensions(residual)
        return out + residual


class CVNN(nn.Module):
    """
    A complex-valued neural network (CVNN) with residual blocks.
    We cast each layer(...) call to torch.Tensor so Mypy sees a definite return.
    """

    def __init__(
        self, input_size: int, hidden_sizes: List[int], output_size: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for h in hidden_sizes:
            self.layers.append(ResidualBlock(prev_size, h))
            prev_size = h
        self.final_linear = nn.Linear(prev_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = cast(torch.Tensor, layer(x))
        x = cast(torch.Tensor, self.final_linear(x))
        return x


class SimpleCVNN(nn.Module):
    """
    A simpler CVNN example, not truly complex-appropriate, uses normal ReLU.
    Also cast(...) so Mypy sees a torch.Tensor return, not Any.
    """

    def __init__(
        self, input_size: int, hidden_sizes: List[int], output_size: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for h in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, h))
            self.layers.append(nn.ReLU())
            prev_size = h
        self.final_linear = nn.Linear(prev_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = cast(torch.Tensor, layer(x))
        x = cast(torch.Tensor, self.final_linear(x))
        return x
