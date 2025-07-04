"""
Complex-valued neural-network blocks for SpectralMC’s test-suite
================================================================

Overview
--------
This module collects **fully-typed**, *dependency-free* PyTorch layers
that operate on pairs of real tensors representing the real and
imaginary parts of a complex signal.  The public API mirrors
:pyclass:`torch.nn` as closely as possible so that the layers plug
seamlessly into existing training code—the only difference is that each
:py:meth:`forward` takes **two** tensors `(real, imag)` and returns the
same pair.

Compared with the reference implementation shipped with SpectralMC this
version merely *renames internal variables for clarity* and *expands the
explanatory doc-strings*.  **All numerical behaviour is bit-for-bit
identical**.

Precision policy
----------------
The SpectralMC test-suite executes every case twice—once with
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
| :class:`CovarianceComplexBatchNorm`| Whitening BN using the full 2×2 covariance (Trabelsi 2018) |
| :class:`ComplexSequential`        | ``nn.Sequential`` analogue for complex blocks           |
| :class:`ComplexResidual`          | Residual wrapper with optional projection & activation  |

Each building block obeys the same *public* signature as its real-valued
cousin so you can freely compose them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Tensor = torch.Tensor  # Keep signatures concise


# ---------------------------------------------------------------------------
# ComplexLinear
# ---------------------------------------------------------------------------
class ComplexLinear(nn.Module):
    r"""Fully-connected linear layer for complex inputs ``ℂⁿ → ℂᵐ``.

    The layer stores the real and imaginary parts of the weight matrix in
    *separate* real tensors and applies the split-formula

    .. math::

        (A + iB)(x + iy) + (b_r + ib_i)
        = (Ax - By + b_r) + i(Bx + Ay + b_i),

    where :math:`A,B∈ℝ^{m×n}` and ``x``, ``y`` are the real and imaginary
    components of the mini-batch.

    Parameters
    ----------
    in_features:
        Number of *complex* input channels.  (Both the real and imaginary
        tensors therefore have ``in_features`` columns.)
    out_features:
        Number of *complex* output channels.
    bias:
        If ``True`` (default) a complex bias vector is learned.
    """

    real_weight: nn.Parameter
    imag_weight: nn.Parameter
    real_bias: Optional[nn.Parameter]
    imag_bias: Optional[nn.Parameter]

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        dtype = torch.get_default_dtype()

        # Weight matrices A (real) and B (imag) as per equation above
        self.real_weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )
        self.imag_weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )

        if bias:
            self.real_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
            self.imag_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
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
    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
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

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        mask = (real >= 0) & (imag >= 0)
        return real * mask, imag * mask


# ---------------------------------------------------------------------------
# modReLU
# ---------------------------------------------------------------------------
class modReLU(nn.Module):
    r"""Phase-preserving magnitude gating.

    For each complex element :math:`z` with magnitude :math:`r = |z|` and
    learnable bias :math:`b`, the activation is

    .. math::

        \operatorname{modReLU}(z) =
        \begin{cases}
            0, & r + b \le 0,\\
            (r + b)\;z / r, & \text{otherwise.}
        \end{cases}

    Originally introduced by :cite:`Arjovsky2016`.
    """

    bias: nn.Parameter

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(
            torch.zeros(num_features, dtype=torch.get_default_dtype())
        )

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        magnitude = torch.sqrt(real * real + imag * imag + 1e-9)
        thresholded = torch.relu(magnitude + self.bias.unsqueeze(0))
        scaling = thresholded / magnitude
        return scaling * real, scaling * imag


# ---------------------------------------------------------------------------
# Naive component-wise BatchNorm
# ---------------------------------------------------------------------------
class NaiveComplexBatchNorm(nn.Module):
    """BatchNorm applied **independently** to real and imaginary parts."""

    real_bn: nn.BatchNorm1d
    imag_bn: nn.BatchNorm1d

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
        self.real_bn = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.imag_bn = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        """Normalise each component to zero mean / unit variance."""
        return self.real_bn(real), self.imag_bn(imag)


# ---------------------------------------------------------------------------
# Covariance-based Complex BatchNorm
# ---------------------------------------------------------------------------
class CovarianceComplexBatchNorm(nn.Module):
    r"""Whitening BN using the 2×2 covariance matrix (Trabelsi 2018).

    **Training pass**
    -----------------
    1. Compute the batch means :math:`\mu_r`, :math:`\mu_i` and centre
       the data.
    2. Estimate the per-feature covariance matrix

       .. math::

           \Sigma =
               \begin{pmatrix}
                   \operatorname{Var}(r) & \operatorname{Cov}(r,i)\\
                   \operatorname{Cov}(r,i) & \operatorname{Var}(i)
               \end{pmatrix}.

    3. Whiten via :math:`\Sigma^{-1/2}` so that each output has variance
       **0.5** on both axes and zero cross-covariance.
    4. Optionally apply a learned affine transform ``Γ`` (symmetric 2×2)
       and complex bias ``β``.

    Running statistics are updated with momentum *m* and reused during
    evaluation if ``track_running_stats`` is *True*.
    """

    beta_real: Optional[nn.Parameter]
    beta_imag: Optional[nn.Parameter]
    gamma_rr: Optional[nn.Parameter]
    gamma_ri: Optional[nn.Parameter]
    gamma_ii: Optional[nn.Parameter]

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
        self.register_buffer(
            "running_mean_real", torch.zeros(num_features, dtype=dtype)
        )
        self.register_buffer(
            "running_mean_imag", torch.zeros(num_features, dtype=dtype)
        )
        self.register_buffer(
            "running_C_rr", torch.full((num_features,), 0.5, dtype=dtype)
        )
        self.register_buffer("running_C_ri", torch.zeros(num_features, dtype=dtype))
        self.register_buffer(
            "running_C_ii", torch.full((num_features,), 0.5, dtype=dtype)
        )

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
    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
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

        eigvals, eigvecs = torch.linalg.eigh(covariance)  # stable for 2×2
        inv_sqrt = (1.0 / eigvals.clamp_min(self.eps).sqrt()).unsqueeze(1)
        whitening = (eigvecs * inv_sqrt) @ eigvecs.transpose(1, 2)  # (C,2,2)

        stacked = torch.stack([centered_real, centered_imag], dim=2).unsqueeze(
            -1
        )  # (N,C,2,1)
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

        out_real = (
            self.gamma_rr * white_real + self.gamma_ri * white_imag + self.beta_real
        )
        out_imag = (
            self.gamma_ri * white_real + self.gamma_ii * white_imag + self.beta_imag
        )
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

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in self.layers:
            real, imag = layer(real, imag)
        return real, imag


class ComplexResidual(nn.Module):
    """Residual wrapper with optional projection and post-activation."""

    def __init__(
        self,
        body: nn.Module,
        proj: Optional[nn.Module] = None,
        post_act: Optional[nn.Module] = None,
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

    def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
        residual_real, residual_imag = real, imag
        body_real, body_imag = self.body(real, imag)

        if self.proj is not None:
            residual_real, residual_imag = self.proj(residual_real, residual_imag)

        summed_real = body_real + residual_real
        summed_imag = body_imag + residual_imag

        if self.post_act is not None:
            summed_real, summed_imag = self.post_act(summed_real, summed_imag)

        return summed_real, summed_imag
