"""
Complex-valued neural-network blocks for SpectralMC's test-suite
================================================================

Overview
--------
This module collects **fully-typed**, *dependency-free* PyTorch layers
that operate on pairs of real tensors representing the real and
imaginary parts of a complex signal.  The public API mirrors
:pyclass:`torch.nn` as closely as possible so that the layers plug
seamlessly into existing training code-the only difference is that each
:py:meth:`forward` takes **two** tensors `(real, imag)` and returns the
same pair.

Compared with the reference implementation shipped with SpectralMC this
version merely *renames internal variables for clarity* and *expands the
explanatory doc-strings*.  **All numerical behaviour is bit-for-bit
identical**.

Precision policy
----------------
The SpectralMC test-suite executes every case twice-once with
``torch.float32`` and once with ``torch.float64``.  For that reason **all
persistent state** (parameters *and* running statistics) is initialised
in the dtype returned by :pyfunc:`torch.get_default_dtype` *at
construction time*.

Layer catalogue
---------------

| Class name                        | Purpose                                                 |
| --------------------------------- | ------------------------------------------------------- |
| :class:`ComplexLinear`            | Dense ``ℂⁿ → ℂᵐ`` linear projection                      |
| :class:`zReLU`                    | First-quadrant rectifier (Guberman 2016)                |
| :class:`modReLU`                  | Magnitude gate with learned threshold (Arjovsky 2016)   |
| :class:`NaiveComplexBatchNorm`    | Plain BN run on Re/Im **independently**                 |
| :class:`CovarianceComplexBatchNorm`| Whitening BN using the full 2x2 covariance (Trabelsi 2018) |
| :class:`ComplexSequential`        | ``nn.Sequential`` analogue for complex blocks           |
| :class:`ComplexResidual`          | Residual wrapper with optional projection & activation  |

Each building block obeys the same *public* signature as its real-valued
cousin so you can freely compose them.
"""

from __future__ import annotations

from functools import reduce

import torch
from torch import nn


# CRITICAL: Import facade BEFORE torch for deterministic algorithms


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Tensor = torch.Tensor  # Keep signatures concise


# ---------------------------------------------------------------------------
# ComplexLinear
# ---------------------------------------------------------------------------
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
    real_bias: nn.Parameter | None
    imag_bias: nn.Parameter | None

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

    # .....................................................................
    def reset_parameters(self) -> None:
        """Xavier-uniform initialisation for weights and zeros for biases."""
        nn.init.xavier_uniform_(self.real_weight)
        nn.init.xavier_uniform_(self.imag_weight)
        if self.real_bias is not None:
            nn.init.zeros_(self.real_bias)
        if self.imag_bias is not None:
            nn.init.zeros_(self.imag_bias)

    # .....................................................................
    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the affine projection.

        Parameters
        ----------
        real, imag:
            Real and imaginary parts of shape ``(batch, in_features)``.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Output real and imaginary parts of shape
            ``(batch, out_features)``.
        """
        weight_real, weight_imag = self.real_weight, self.imag_weight

        output_real = real @ weight_real.T - imag @ weight_imag.T
        output_imag = real @ weight_imag.T + imag @ weight_real.T

        if self.real_bias is not None:
            output_real = output_real + self.real_bias
        if self.imag_bias is not None:
            output_imag = output_imag + self.imag_bias
        return output_real, output_imag


# ---------------------------------------------------------------------------
# zReLU
# ---------------------------------------------------------------------------
class zReLU(nn.Module):
    r"""First-quadrant rectifier (``Re ≥ 0`` *and* ``Im ≥ 0``).

    The activation passes through a complex number unchanged if *both*
    its real and imaginary components are non-negative and zeroes it out
    otherwise.

    This is the complex analogue of ReLU proposed by
    :cite:`Guberman2016`.
    """

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        mask = (real >= 0) & (imag >= 0)
        return real * mask, imag * mask


# ---------------------------------------------------------------------------
# modReLU
# ---------------------------------------------------------------------------
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        *,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        Γ = [[gamma_rr, gamma_ri],
             [gamma_ri, gamma_ii]],

    which is a positive semi-definite matrix. Initially, gamma_rr = gamma_ii = 1/√2 and gamma_ri = 0,
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

    beta_real: nn.Parameter | None
    beta_imag: nn.Parameter | None
    gamma_rr: nn.Parameter | None
    gamma_ri: nn.Parameter | None
    gamma_ii: nn.Parameter | None

    running_mean_real: Tensor
    running_mean_imag: Tensor
    running_C_rr: Tensor
    running_C_ri: Tensor
    running_C_ii: Tensor

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        dtype = torch.get_default_dtype()

        # Running estimates (initialised to identity-like statistics)
        self.register_buffer("running_mean_real", torch.zeros(num_features, dtype=dtype))
        self.register_buffer("running_mean_imag", torch.zeros(num_features, dtype=dtype))
        self.register_buffer("running_C_rr", torch.full((num_features,), 0.5, dtype=dtype))
        self.register_buffer("running_C_ri", torch.zeros(num_features, dtype=dtype))
        self.register_buffer("running_C_ii", torch.full((num_features,), 0.5, dtype=dtype))

        # Small constants / flags
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Optional affine parameters (β, Γ)
        if affine:
            self.beta_real = nn.Parameter(torch.zeros(num_features, dtype=dtype))
            self.beta_imag = nn.Parameter(torch.zeros(num_features, dtype=dtype))
            self.gamma_rr = nn.Parameter(torch.ones(num_features, dtype=dtype))
            self.gamma_ri = nn.Parameter(torch.zeros(num_features, dtype=dtype))
            self.gamma_ii = nn.Parameter(torch.ones(num_features, dtype=dtype))
        else:
            self.beta_real = None
            self.beta_imag = None
            self.gamma_rr = None
            self.gamma_ri = None
            self.gamma_ii = None

    # .....................................................................
    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        if self.training or not self.track_running_stats:
            # ----------------------------- Batch statistics -----------------------------
            mean_real = real.mean(dim=0)
            mean_imag = imag.mean(dim=0)

            centered_real = real - mean_real
            centered_imag = imag - mean_imag

            C_rr = (centered_real * centered_real).mean(dim=0)
            C_ii = (centered_imag * centered_imag).mean(dim=0)
            C_ri = (centered_real * centered_imag).mean(dim=0)

            if self.track_running_stats:
                m = self.momentum
                with torch.no_grad():
                    self.running_mean_real.mul_(1 - m).add_(mean_real * m)
                    self.running_mean_imag.mul_(1 - m).add_(mean_imag * m)
                    self.running_C_rr.mul_(1 - m).add_(C_rr * m)
                    self.running_C_ri.mul_(1 - m).add_(C_ri * m)
                    self.running_C_ii.mul_(1 - m).add_(C_ii * m)
        else:
            # ----------------------------- Running statistics ---------------------------
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag

            centered_real = real - mean_real
            centered_imag = imag - mean_imag

            C_rr = self.running_C_rr
            C_ri = self.running_C_ri
            C_ii = self.running_C_ii

        # ----------------------------- Whitening transform -----------------------------
        covariance = torch.stack(
            [
                torch.stack([C_rr + self.eps, C_ri], dim=1),
                torch.stack([C_ri, C_ii + self.eps], dim=1),
            ],
            dim=1,
        )  # shape: (C, 2, 2)

        eigvals, eigvecs = torch.linalg.eigh(covariance)  # stable for 2x2
        inv_sqrt = (1.0 / eigvals.clamp_min(self.eps).sqrt()).unsqueeze(1)
        whitening = (eigvecs * inv_sqrt) @ eigvecs.transpose(1, 2)  # (C,2,2)

        stacked = torch.stack([centered_real, centered_imag], dim=2).unsqueeze(-1)  # (N,C,2,1)
        whitened = (whitening.unsqueeze(0) @ stacked).squeeze(-1)  # (N,C,2)
        white_real, white_imag = whitened[..., 0], whitened[..., 1]

        if not self.affine:
            return white_real, white_imag

        # mypy: prove that parameters are present
        assert (
            self.gamma_rr is not None
            and self.gamma_ri is not None
            and self.gamma_ii is not None
            and self.beta_real is not None
            and self.beta_imag is not None
        )

        out_real = self.gamma_rr * white_real + self.gamma_ri * white_imag + self.beta_real
        out_imag = self.gamma_ri * white_real + self.gamma_ii * white_imag + self.beta_imag
        return out_real, out_imag


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------
class ComplexSequential(nn.Module):
    """Drop-in replacement for :class:`torch.nn.Sequential`.

    The container expects each sub-module to consume and return a pair of
    tensors ``(real, imag)``.
    """

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(modules)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return reduce(lambda state, layer: layer(*state), self.layers, (real, imag))


class ComplexResidual(nn.Module):
    """Residual wrapper with optional projection and post-activation."""

    def __init__(
        self,
        body: nn.Module,
        proj: nn.Module | None = None,
        post_act: nn.Module | None = None,
    ) -> None:
        """Create a residual block ``x + body(x)``.

        Parameters
        ----------
        body:
            The main transformation :math:`f(x)`.
        proj:
            Optional projection so that *x* and *f(x)* have matching
            widths before addition.
        post_act:
            Optional activation applied *after* the residual sum.
        """
        super().__init__()
        self.body = body
        self.proj = proj
        self.post_act = post_act

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        residual_real, residual_imag = real, imag
        body_real, body_imag = self.body(real, imag)

        if self.proj is not None:
            residual_real, residual_imag = self.proj(residual_real, residual_imag)

        summed_real = body_real + residual_real
        summed_imag = body_imag + residual_imag

        if self.post_act is not None:
            summed_real, summed_imag = self.post_act(summed_real, summed_imag)

        return summed_real, summed_imag
