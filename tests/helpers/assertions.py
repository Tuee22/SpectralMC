# tests/helpers/assertions.py
"""Custom assertion helpers for SpectralMC tests.

Provides enhanced assertion helpers with better error messages than standard
assert statements. These helpers are particularly useful for numerical tests
where understanding the magnitude of differences is critical.
"""

from __future__ import annotations

import torch


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str | None = None,
) -> None:
    """Assert tensors are close with enhanced error messages.

    Wraps torch.allclose with better error reporting. When tensors differ,
    reports both absolute and relative differences to help debug numerical
    issues.

    Args:
        actual: Actual tensor from test
        expected: Expected tensor
        rtol: Relative tolerance (default: 1e-5)
        atol: Absolute tolerance (default: 1e-8)
        msg: Optional custom error message prefix

    Raises:
        AssertionError: If tensors differ beyond tolerance, with detailed
            error message showing max absolute/relative differences

    Example:
        >>> result = model(input)
        >>> expected = torch.tensor([1.0, 2.0, 3.0])
        >>> assert_tensors_close(result, expected, rtol=1e-4)
    """
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff_abs = (actual - expected).abs().max().item()
        diff_rel = ((actual - expected).abs() / (expected.abs() + 1e-8)).max().item()
        error_msg = (
            f"Tensors differ: max_abs_diff={diff_abs:.2e}, "
            f"max_rel_diff={diff_rel:.2e}, rtol={rtol:.2e}, atol={atol:.2e}"
        )
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def assert_no_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert tensor contains no NaN or Inf values.

    Critical check for numerical stability. Many bugs in GPU kernels or
    training loops manifest as NaN/Inf propagation.

    Args:
        tensor: Tensor to validate
        name: Name for error message (default: "tensor")

    Raises:
        AssertionError: If NaN or Inf found, with counts of each

    Example:
        >>> loss = compute_loss(model, data)
        >>> assert_no_nan_inf(loss, "loss")
        >>> gradients = compute_gradients(model)
        >>> assert_no_nan_inf(gradients, "gradients")
    """
    finite_mask = torch.isfinite(tensor)
    if finite_mask.all():
        return

    total = finite_mask.reshape(-1).shape[0]
    non_finite_count = total - int(finite_mask.sum().item())
    raise AssertionError(
        f"{name} contains non-finite values: {non_finite_count} non-finite entries"
    )


def assert_converged(
    loss: float,
    threshold: float,
    variance: float | None = None,
    max_variance: float | None = None,
) -> None:
    """Assert convergence criteria met.

    Validates training convergence by checking loss below threshold and
    optionally variance below maximum. Useful for Monte Carlo training tests.

    Args:
        loss: Current loss value
        threshold: Maximum acceptable loss
        variance: Optional variance value to check
        max_variance: Optional maximum acceptable variance

    Raises:
        AssertionError: If loss exceeds threshold or variance exceeds maximum

    Example:
        >>> final_loss = train_model(model, data)
        >>> assert_converged(final_loss, threshold=1e-3)
        >>>
        >>> # With variance check
        >>> loss, var = train_with_uncertainty(model, data)
        >>> assert_converged(loss, threshold=1e-3, variance=var, max_variance=1e-4)
    """
    if loss >= threshold:
        raise AssertionError(f"Loss {loss:.4e} did not converge below threshold {threshold:.4e}")

    if variance is not None and max_variance is not None:
        if variance >= max_variance:
            raise AssertionError(f"Variance {variance:.4e} exceeds maximum {max_variance:.4e}")
