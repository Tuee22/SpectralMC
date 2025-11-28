# Project-Specific Patterns

## Overview

SpectralMC has established patterns for complex-valued neural networks, type-safe interfaces, GPU/CPU data movement, and error handling. These patterns ensure consistency, performance, and maintainability.

**Related Standards**: [PyTorch Facade](pytorch_facade.md), [Type Safety](type_safety.md), [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md)

---

## Complex-Valued Neural Networks

SpectralMC represents complex numbers as **pairs of real tensors** (real, imag). All complex-valued layers follow this pattern:

### Base Pattern

```python
class ComplexLayer(nn.Module):
    """Base pattern for complex-valued layers."""

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process complex input as (real, imag) pair.

        Args:
            real: Real component of complex input.
            imag: Imaginary component of complex input.

        Returns:
            Tuple of (real_output, imag_output) representing complex output.
        """
        # Implementation here
        return real_output, imag_output
```

**Key principles**:
- Input: `(real, imag)` tensor pair
- Output: `(real, imag)` tensor pair
- Shapes: `real.shape == imag.shape` always
- Device: Both tensors on same device
- Dtype: Both tensors same dtype

### Example: ComplexLinear

```python
class ComplexLinear(nn.Module):
    """Complex-valued linear layer: W*z + b where W, z, b are complex."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Complex weight = real_weight + i*imag_weight
        self.real_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.imag_weight = nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.real_bias = nn.Parameter(torch.zeros(out_features))
            self.imag_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.real_bias = None
            self.imag_bias = None

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply complex linear transformation: (A + iB)(x + iy) + (c + id).

        Formula: (Ax - By) + i(Bx + Ay) + (c + id)

        Args:
            real: Real part of input, shape (batch, in_features).
            imag: Imaginary part of input, shape (batch, in_features).

        Returns:
            Tuple of (real_output, imag_output), shape (batch, out_features).
        """
        # Real part: Ax - By + c
        real_out = torch.nn.functional.linear(real, self.real_weight, self.real_bias)
        real_out = real_out - torch.nn.functional.linear(imag, self.imag_weight, None)

        # Imaginary part: Bx + Ay + d
        imag_out = torch.nn.functional.linear(real, self.imag_weight, self.imag_bias)
        imag_out = imag_out + torch.nn.functional.linear(imag, self.real_weight, None)

        return real_out, imag_out
```

---

## Protocol Usage

Define interfaces using `Protocol` for type-safe duck typing:

### Example: ComplexValuedModel Protocol

```python
from typing import Protocol, runtime_checkable, Iterable
import torch.nn as nn

@runtime_checkable
class ComplexValuedModel(Protocol):
    """Protocol for complex-valued neural network models."""

    def __call__(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with complex input."""
        ...

    def parameters(self) -> Iterable[nn.Parameter]:
        """Return model parameters for optimization."""
        ...
```

**Usage**:

```python
def train_complex_model(
    model: ComplexValuedModel,
    real_data: torch.Tensor,
    imag_data: torch.Tensor,
) -> None:
    """Train any model implementing ComplexValuedModel protocol."""
    # Type checker ensures model has __call__ and parameters()
    real_out, imag_out = model(real_data, imag_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # ... training loop
```

**Benefits**:
- No inheritance required
- Type-safe duck typing
- Runtime checking with `isinstance(model, ComplexValuedModel)`
- mypy validates protocol compliance

---

## GPU/CPU Transfer Patterns

SpectralMC **forbids** implicit GPU/CPU transfers (`.cuda()`, `.cpu()`, `.to(device)`). Use explicit, batched transfers via `move_tensor_tree`.

### The PCIe Bottleneck Problem

**Performance context**:
- GPU memory bandwidth: 900+ GB/s (HBM2/HBM3)
- PCIe 4.0 x16: ~25 GB/s practical
- PCIe 3.0 x16: ~12 GB/s practical

**Result**: GPU↔CPU transfers are **30-75x slower** than GPU-internal operations. A single implicit `.cuda()` call can dominate your entire training step.

### Forbidden Patterns

```python
# ❌ FORBIDDEN - Implicit GPU transfer
tensor.cuda()

# ❌ FORBIDDEN - Implicit CPU transfer
tensor.cpu()

# ❌ FORBIDDEN - Implicit transfer via .to()
tensor.to(device)
tensor.to("cuda")

# ❌ FORBIDDEN - Individual tensor transfers in loop
for tensor in tensor_list:
    tensor = tensor.cuda()  # Slow! Multiple PCIe transfers
```

### Required Pattern: TensorTree

Use `move_tensor_tree()` for explicit, batched transfers:

```python
from spectralmc.models.cpu_gpu_transfer import move_tensor_tree, Device

# ✅ CORRECT - Explicit, batched transfer
gpu_state = move_tensor_tree(
    cpu_state,
    dest=Device.cuda,
    pin_memory=True,  # Enables faster host→device transfers
)
```

### TensorTree Type

`TensorTree` handles arbitrarily nested data structures:

```python
from typing import Union, List, Tuple, Mapping, Any

TensorTree = Union[
    torch.Tensor,               # Leaf tensors
    List["TensorTree"],         # Lists of tensors/structures
    Tuple["TensorTree", ...],   # Tuples of tensors/structures
    Mapping[str, "TensorTree"], # Dicts of tensors/structures
    int, float, str, None,      # Scalar data (passed through)
]
```

**Example**:

```python
# Complex nested structure
optimizer_state = {
    "step": 1000,  # Scalar (passed through)
    "params": [
        {"momentum": torch.randn(100, 50), "variance": torch.randn(100, 50)},
        {"momentum": torch.randn(50, 10), "variance": torch.randn(50, 10)},
    ],
    "lr": 0.001,  # Scalar (passed through)
}

# Single batched transfer
gpu_optimizer_state = move_tensor_tree(
    optimizer_state,
    dest=Device.cuda,
    pin_memory=True,
)
```

### Performance Best Practices

1. **Batch transfers**: Move entire state dictionaries at once
2. **Use pinned memory**: Set `pin_memory=True` for faster host→device
3. **Minimize frequency**: Transfer only when necessary (checkpointing, loading)
4. **Stream synchronization**: Transfer functions handle CUDA stream sync automatically

### Example: Optimizer State Transfer

```python
from spectralmc.models.cpu_gpu_transfer import move_tensor_tree, Device

def save_optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    """Save optimizer state to CPU for checkpointing."""
    gpu_state = optimizer.state_dict()

    # Single batched transfer to CPU
    cpu_state = move_tensor_tree(
        gpu_state,
        dest=Device.cpu,
        pin_memory=False,  # Already on CPU after transfer
    )

    return cpu_state

def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: dict,
) -> None:
    """Load optimizer state from CPU checkpoint to GPU."""
    # Single batched transfer to GPU
    gpu_state = move_tensor_tree(
        checkpoint,
        dest=Device.cuda,
        pin_memory=True,  # Faster host→device transfer
    )

    optimizer.load_state_dict(gpu_state)
```

---

## Error Handling

Use expression-level exception raising for functional style:

### Helper Function

```python
from typing import NoReturn

def _raise(exc: Exception) -> NoReturn:
    """Expression-level exception raising.

    Allows raising exceptions in expression context:
    value = some_value if condition else _raise(ValueError("message"))
    """
    raise exc
```

### Usage Example

```python
def validate_positive(value: float, name: str) -> float:
    """Validate that a value is positive.

    Args:
        value: Value to validate.
        name: Name of parameter (for error message).

    Returns:
        Value if valid.

    Raises:
        ValueError: If value is not positive.
    """
    return value if value > 0 else _raise(
        ValueError(f"{name} must be positive, got {value}")
    )

# Usage in function
def simulate(spot: float, rate: float, volatility: float) -> np.ndarray:
    """Simulate GBM with validated parameters."""
    spot = validate_positive(spot, "spot")
    volatility = validate_positive(volatility, "volatility")

    # Safe to proceed - parameters validated
    ...
```

### Custom Exceptions

Define domain-specific exceptions:

```python
class SimulationError(Exception):
    """Base exception for simulation errors."""
    pass

class ConvergenceError(SimulationError):
    """Raised when simulation fails to converge."""
    pass

class NumericalInstabilityError(SimulationError):
    """Raised when numerical instability detected (NaN, Inf)."""
    pass

# Usage
def train_model(model: nn.Module, data: torch.Tensor) -> None:
    """Train model, raising on numerical issues."""
    for epoch in range(100):
        loss = train_step(model, data)

        if not torch.isfinite(loss):
            raise NumericalInstabilityError(
                f"Loss became {loss} at epoch {epoch}"
            )
```

---

## Summary

- **Complex-valued layers**: `(real, imag)` tensor pair pattern
- **Protocols**: Use `Protocol` for type-safe duck typing
- **GPU/CPU transfers**: Forbidden implicit transfers, required `move_tensor_tree()`
- **TensorTree**: Handles nested structures efficiently
- **Error handling**: Expression-level `_raise()` and custom exceptions

See also: [PyTorch Facade](pytorch_facade.md), [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md), [Type Safety](type_safety.md)
