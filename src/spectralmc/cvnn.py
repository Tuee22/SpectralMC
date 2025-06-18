# src/spectralmc/cvnn.py
import math
from typing import Tuple, Union, Type, Optional
import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    """A fully-connected linear layer for complex-valued inputs.

    This layer supports complex weights and biases, and performs a linear transformation
    on a complex input vector. If the input is represented as (real, imag) components,
    the weight and bias are likewise represented by real and imaginary parts.

    The forward operation for a complex input z = x + i*y (with real part x and imaginary part y)
    and complex weight W = A + i*B (with real part A and imaginary part B) is:

        W * z + b = (A x - B y) + i(B x + A y) + (b_real + i*b_imag)

    where b_real and b_imag are the real and imaginary parts of the complex bias.

    Args:
        in_features (int): Number of input features (per complex component).
        out_features (int): Number of output features (per complex component).
        bias (bool): If True, include a complex bias term. Default: True.
    """

    real_weight: nn.Parameter
    imag_weight: nn.Parameter
    real_bias: Optional[nn.Parameter]
    imag_bias: Optional[nn.Parameter]

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features

        # Real and imaginary weight matrices
        self.real_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.imag_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        # Real and imaginary bias if needed
        if bias:
            self.real_bias = nn.Parameter(torch.Tensor(out_features))
            self.imag_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.real_bias = None
            self.imag_bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset weights and biases using Xavier initialization for weights and zeros for biases."""
        nn.init.xavier_uniform_(self.real_weight)
        nn.init.xavier_uniform_(self.imag_weight)
        if self.real_bias is not None:
            nn.init.zeros_(self.real_bias)
        if self.imag_bias is not None:
            nn.init.zeros_(self.imag_bias)

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that computes W * (input_real + i*input_imag) + bias.

        Args:
            input_real (torch.Tensor): Real part of input, shape (batch, in_features).
            input_imag (torch.Tensor): Imaginary part of input, shape (batch, in_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Real and imaginary parts of the output,
            each of shape (batch, out_features).
        """
        output_real = torch.matmul(input_real, self.real_weight.T) - torch.matmul(
            input_imag, self.imag_weight.T
        )
        output_imag = torch.matmul(input_real, self.imag_weight.T) + torch.matmul(
            input_imag, self.real_weight.T
        )

        if self.real_bias is not None:
            output_real = output_real + self.real_bias
        if self.imag_bias is not None:
            output_imag = output_imag + self.imag_bias
        return output_real, output_imag


class zReLU(nn.Module):
    """Complex-valued ReLU activation that passes values in the first quadrant only.

    The zReLU activation zeroes out any complex input whose real or imaginary part is negative.
    In other words, it outputs z if Re(z) >= 0 and Im(z) >= 0, and 0 otherwise. This restricts the
    output to the first quadrant of the complex plane. Proposed by Guberman (2016) and used in
    Trabelsi et al. (2018) as a non-holomorphic activation for complex networks.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for zReLU activation.

        Args:
            input_real (torch.Tensor): Real part of input.
            input_imag (torch.Tensor): Imaginary part of input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Real and imaginary parts after applying zReLU.
        """
        mask = (input_real >= 0) & (input_imag >= 0)
        output_real = input_real * mask
        output_imag = input_imag * mask
        return output_real, output_imag


class modReLU(nn.Module):
    """Modulus-based complex ReLU activation (modReLU).

    modReLU was introduced by Arjovsky et al. (2016) for unitary RNNs and applied in deep complex networks (Trabelsi et al., 2018).
    It operates on the magnitude of the complex input and preserves the phase. For a complex input z = x + i*y with magnitude r = |z|
    and phase θ, modReLU computes:

        if r + b > 0: output = (r + b) * (z / r)
        else: output = 0

    where b is a learned bias (per feature) that defines the magnitude threshold. This effectively zeros out inputs whose magnitude is
    below a learnable threshold, while scaling others by (r + b)/r (preserving the phase θ).

    Note: modReLU is not holomorphic (does not satisfy the Cauchy-Riemann equations), but it provides a useful non-linearity in the complex domain.

    Args:
        num_features (int): Number of features (channels) of the input. One bias parameter per feature will be learned.
    """

    bias: nn.Parameter

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for modReLU activation.

        Args:
            input_real (torch.Tensor): Real part of input, shape (batch, num_features).
            input_imag (torch.Tensor): Imaginary part of input, shape (batch, num_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Real and imaginary parts after applying modReLU.
        """
        magnitude = torch.sqrt(input_real * input_real + input_imag * input_imag + 1e-9)
        threshold = torch.relu(magnitude + self.bias.unsqueeze(0))
        scaling = threshold / magnitude
        output_real = scaling * input_real
        output_imag = scaling * input_imag
        return output_real, output_imag


class NaiveComplexBatchNorm(nn.Module):
    """Naive complex batch normalization applying BN separately to real and imag parts.

    This batch normalization treats the real and imaginary components independently. It applies standard
    BatchNorm1d to the real part and to the imaginary part separately. This approach does not account for
    correlations between real and imaginary components (thus "naive"), but is computationally simpler.

    Args:
        num_features (int): Number of features (channels) in the input.
        eps (float): A small value added for numerical stability. Default: 1e-5.
        momentum (float): Momentum for running statistics. Default: 0.1.
        affine (bool): If True, apply learnable affine parameters (scale and shift) to both real and imag parts. Default: True.
        track_running_stats (bool): If True, track running mean and variance. Default: True.
    """

    bn_real: nn.BatchNorm1d
    bn_imag: nn.BatchNorm1d

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.bn_real = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.bn_imag = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for naive complex batch normalization.

        Normalizes the real and imaginary parts independently to zero mean and unit variance (per feature),
        then applies separate affine transformations for each part if enabled.

        Args:
            input_real (torch.Tensor): Real part of input, shape (batch, num_features).
            input_imag (torch.Tensor): Imaginary part of input, shape (batch, num_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Normalized real and imaginary parts.
        """
        out_real = self.bn_real(input_real)
        out_imag = self.bn_imag(input_imag)
        return out_real, out_imag


class CovarianceComplexBatchNorm(nn.Module):
    """Complex batch normalization using covariance matrix whitening (Trabelsi et al., 2018).

    This implementation follows the formulation of complex batch normalization introduced in
    "Deep Complex Networks" by Trabelsi et al. (2018). Instead of treating real and imaginary parts
    separately, it whitens each complex feature by using the 2x2 covariance matrix of its real and imaginary parts.

    For each feature (channel) across the batch, we compute the mean (μ_r, μ_i) and the 2x2 covariance matrix:

        V = [[Var(r), Cov(r,i)],
             [Cov(i,r), Var(i)]],

    where Var(r) is the variance of the real part, Var(i) the variance of the imaginary part,
    and Cov(r,i) the covariance between real and imaginary parts of that feature. We then whiten
    the data by multiplying the zero-mean input by V^{-1/2}, the inverse square root of this covariance matrix.
    This operation ensures that the real and imaginary components are decorrelated and have equal variances,
    yielding a "circular" (isotropic) unit variance complex distribution.

    After whitening, a learnable affine transform is applied:
    a complex shift β = β_r + i β_i (two parameters per feature for centering) and
    a complex scaling matrix Γ (2x2 per feature, with 3 degrees of freedom since Γ is Hermitian/real symmetric).
    We parameterize Γ as:

        Γ = [[γ_rr, γ_ri],
             [γ_ri, γ_ii]],

    which is a positive semi-definite matrix. Initially, γ_rr = γ_ii = 1/√2 and γ_ri = 0,
    to normalize each component to variance 0.5 (so that overall complex variance is 1).
    β is initialized to 0.

    During training, this module keeps running estimates of the mean and covariance for use in evaluation (inference) mode,
    similar to how standard BatchNorm tracks running mean and variance.

    Args:
        num_features (int): Number of features (channels).
        eps (float): Small constant to add to covariance for numerical stability. Default: 1e-5.
        momentum (float): Momentum for running statistics. Default: 0.1.
        affine (bool): If True, includes learnable affine parameters (β and Γ). Default: True.
        track_running_stats (bool): If True, tracks running mean and covariance. Default: True.
    """

    beta_real: Optional[nn.Parameter]
    beta_imag: Optional[nn.Parameter]
    gamma_rr: Optional[nn.Parameter]
    gamma_ri: Optional[nn.Parameter]
    gamma_ii: Optional[nn.Parameter]

    # We'll store typed references to the actual buffer Tensors:
    running_mean_real: torch.Tensor
    running_mean_imag: torch.Tensor
    running_C_rr: torch.Tensor
    running_C_ri: torch.Tensor
    running_C_ii: torch.Tensor

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # 1) Create a local Tensor
        # 2) Register it as a buffer
        # 3) Also store it in a typed attribute so mypy recognizes it as a torch.Tensor
        rm_real = torch.zeros(num_features, dtype=torch.float)
        self.register_buffer("running_mean_real", rm_real)
        self.running_mean_real = rm_real

        rm_imag = torch.zeros(num_features, dtype=torch.float)
        self.register_buffer("running_mean_imag", rm_imag)
        self.running_mean_imag = rm_imag

        c_rr = torch.full((num_features,), 0.5, dtype=torch.float)
        self.register_buffer("running_C_rr", c_rr)
        self.running_C_rr = c_rr

        c_ri = torch.zeros(num_features, dtype=torch.float)
        self.register_buffer("running_C_ri", c_ri)
        self.running_C_ri = c_ri

        c_ii = torch.full((num_features,), 0.5, dtype=torch.float)
        self.register_buffer("running_C_ii", c_ii)
        self.running_C_ii = c_ii

        # Learnable affine parameters
        if affine:
            init_val = 1.0 / math.sqrt(2.0)
            self.beta_real = nn.Parameter(torch.zeros(num_features))
            self.beta_imag = nn.Parameter(torch.zeros(num_features))
            self.gamma_rr = nn.Parameter(torch.full((num_features,), init_val))
            self.gamma_ri = nn.Parameter(torch.zeros(num_features))
            self.gamma_ii = nn.Parameter(torch.full((num_features,), init_val))
        else:
            self.beta_real = None
            self.beta_imag = None
            self.gamma_rr = None
            self.gamma_ri = None
            self.gamma_ii = None

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for covariance-based complex batch normalization.

        Normalizes a complex input by subtracting the mean and whitening using the covariance matrix,
        then applies a learnable affine transformation (scale and shift).

        Args:
            input_real (torch.Tensor): Real part of input, shape (batch, num_features).
            input_imag (torch.Tensor): Imaginary part of input, shape (batch, num_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The normalized and transformed real and imaginary parts.
        """
        if self.training or not self.track_running_stats:
            batch_mean_real = input_real.mean(dim=0)
            batch_mean_imag = input_imag.mean(dim=0)
            centered_real = input_real - batch_mean_real
            centered_imag = input_imag - batch_mean_imag
            batch_C_rr = (centered_real * centered_real).mean(dim=0)
            batch_C_ii = (centered_imag * centered_imag).mean(dim=0)
            batch_C_ri = (centered_real * centered_imag).mean(dim=0)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean_real.mul_(1 - self.momentum).add_(
                        batch_mean_real * self.momentum
                    )
                    self.running_mean_imag.mul_(1 - self.momentum).add_(
                        batch_mean_imag * self.momentum
                    )
                    self.running_C_rr.mul_(1 - self.momentum).add_(
                        batch_C_rr * self.momentum
                    )
                    self.running_C_ri.mul_(1 - self.momentum).add_(
                        batch_C_ri * self.momentum
                    )
                    self.running_C_ii.mul_(1 - self.momentum).add_(
                        batch_C_ii * self.momentum
                    )
            mean_real = batch_mean_real
            mean_imag = batch_mean_imag
            C_rr = batch_C_rr
            C_ri = batch_C_ri
            C_ii = batch_C_ii
        else:
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag
            C_rr = self.running_C_rr
            C_ri = self.running_C_ri
            C_ii = self.running_C_ii
            centered_real = input_real - mean_real
            centered_imag = input_imag - mean_imag

        cov_mats = torch.stack(
            [
                torch.stack([C_rr + self.eps, C_ri], dim=1),
                torch.stack([C_ri, C_ii + self.eps], dim=1),
            ],
            dim=1,
        )
        eigvals, eigvecs = torch.linalg.eigh(cov_mats)
        inv_sqrt_eigvals = 1.0 / torch.sqrt(torch.clamp(eigvals, min=self.eps))
        vecs_scaled = eigvecs * inv_sqrt_eigvals.unsqueeze(1)
        whitening_mats = torch.matmul(vecs_scaled, eigvecs.transpose(1, 2))

        centered = torch.stack([centered_real, centered_imag], dim=2).unsqueeze(-1)
        whitened = torch.matmul(whitening_mats.unsqueeze(0), centered).squeeze(-1)
        whitened_real = whitened[..., 0]
        whitened_imag = whitened[..., 1]

        if self.affine:
            if (
                self.gamma_rr is None
                or self.gamma_ri is None
                or self.gamma_ii is None
                or self.beta_real is None
                or self.beta_imag is None
            ):
                raise RuntimeError("Affine parameters not initialized.")
            out_real = self.gamma_rr * whitened_real + self.gamma_ri * whitened_imag
            out_imag = self.gamma_ri * whitened_real + self.gamma_ii * whitened_imag
            out_real = out_real + self.beta_real
            out_imag = out_imag + self.beta_imag
        else:
            out_real = whitened_real
            out_imag = whitened_imag

        return out_real, out_imag
