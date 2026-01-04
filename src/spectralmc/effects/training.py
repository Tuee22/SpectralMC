"""
Training Effect ADTs for forward/backward passes and optimizer steps.

This module defines frozen dataclasses representing all training-related side effects,
enabling exhaustive pattern matching and type-safe effect composition.

Type Safety:
    - All effect types are frozen dataclasses (immutable)
    - Literal discriminators enable exhaustive pattern matching
    - Union types define closed sets of effect variants

See Also:
    - effect_interpreter.md - Effect Interpreter doctrine
    - coding_standards.md - ADT patterns and Result types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ForwardPass:
    """Request to execute forward pass through a model.

    Attributes:
        kind: Discriminator for pattern matching. Always "ForwardPass".
        model_id: Identifier for the model to use.
        input_tensor_id: Identifier for the input tensor.
        output_tensor_id: Identifier for storing the output tensor.
    """

    kind: Literal["ForwardPass"] = "ForwardPass"
    model_id: str = ""
    input_tensor_id: str = ""
    output_tensor_id: str = "output"


@dataclass(frozen=True)
class BackwardPass:
    """Request to compute gradients via backpropagation.

    Attributes:
        kind: Discriminator for pattern matching. Always "BackwardPass".
        loss_tensor_id: Identifier for the loss tensor to backpropagate from.
    """

    kind: Literal["BackwardPass"] = "BackwardPass"
    loss_tensor_id: str = ""


@dataclass(frozen=True)
class OptimizerStep:
    """Request to update model parameters using optimizer.

    Attributes:
        kind: Discriminator for pattern matching. Always "OptimizerStep".
        optimizer_id: Identifier for the optimizer to step.
    """

    kind: Literal["OptimizerStep"] = "OptimizerStep"
    optimizer_id: str = ""


@dataclass(frozen=True)
class ComputeLoss:
    """Request to compute loss between predictions and targets.

    Attributes:
        kind: Discriminator for pattern matching. Always "ComputeLoss".
        pred_tensor_id: Identifier for the prediction tensor.
        target_tensor_id: Identifier for the target tensor.
        loss_type: Type of loss function to compute.
        output_tensor_id: Identifier for storing the computed loss.
    """

    kind: Literal["ComputeLoss"] = "ComputeLoss"
    pred_tensor_id: str = ""
    target_tensor_id: str = ""
    loss_type: Literal["mse", "mae", "huber"] = "mse"
    output_tensor_id: str = "loss"


@dataclass(frozen=True)
class LogMetrics:
    """Request to log training metrics to TensorBoard.

    Attributes:
        kind: Discriminator for pattern matching. Always "LogMetrics".
        metrics: Tuple of (metric_name, metric_value) pairs.
        step: Training step number for logging.
    """

    kind: Literal["LogMetrics"] = "LogMetrics"
    metrics: tuple[tuple[str, float | str], ...] = ()
    step: int = 0


# Training Effect Union
@dataclass(frozen=True)
class ForwardPassComplex:
    """Request to execute complex-valued forward pass (real + imaginary inputs)."""

    kind: Literal["ForwardPassComplex"] = "ForwardPassComplex"
    model_id: str = ""
    real_input_tensor_id: str = ""
    imag_input_tensor_id: str = ""
    real_output_tensor_id: str = "pred_real"
    imag_output_tensor_id: str = "pred_imag"


@dataclass(frozen=True)
class ComputeComplexLoss:
    """Request to compute loss for complex-valued predictions."""

    kind: Literal["ComputeComplexLoss"] = "ComputeComplexLoss"
    pred_real_tensor_id: str = ""
    pred_imag_tensor_id: str = ""
    target_tensor_id: str = ""
    loss_type: Literal["mse", "mae", "huber"] = "mse"
    output_tensor_id: str = "loss"


@dataclass(frozen=True)
class ZeroGrad:
    """Request to zero gradients before forward pass."""

    kind: Literal["ZeroGrad"] = "ZeroGrad"
    optimizer_id: str = ""
    set_to_none: bool = True


@dataclass(frozen=True)
class ComputeGradNorm:
    """Request to compute (and optionally clip) gradient norm."""

    kind: Literal["ComputeGradNorm"] = "ComputeGradNorm"
    model_id: str = ""
    max_norm: float = float("inf")
    output_tensor_id: str = "grad_norm"


@dataclass(frozen=True)
class UpdateLearningRate:
    """Optional effect to adjust optimizer learning rate."""

    kind: Literal["UpdateLearningRate"] = "UpdateLearningRate"
    optimizer_id: str = ""
    lr: float = 0.0


# Training Effect Union
TrainingEffect = (
    ForwardPass
    | BackwardPass
    | OptimizerStep
    | ComputeLoss
    | LogMetrics
    | ForwardPassComplex
    | ComputeComplexLoss
    | ZeroGrad
    | ComputeGradNorm
    | UpdateLearningRate
)
