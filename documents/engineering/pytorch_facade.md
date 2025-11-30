# PyTorch Facade Pattern

## Overview

SpectralMC uses a custom PyTorch facade (`spectralmc.models.torch`) to ensure guaranteed reproducibility of training across different environments. This facade **must** be imported before any direct PyTorch imports.

**Related Standards**: [Coding Standards](coding_standards.md), [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md), [Reproducibility Proofs](reproducibility_proofs.md), [Effect Interpreter](effect_interpreter.md)

---

## Rationale

The PyTorch facade ensures:

1. **Guaranteed reproducibility** of training across different environments
2. **Deterministic execution** on both CPU and GPU
3. **Thread safety** for multi-threaded applications
4. **Controlled import order** to set global flags before PyTorch initialization

Without the facade, PyTorch's default behavior is **non-deterministic**:
- CUDA operations may use non-deterministic algorithms
- cuDNN auto-tuning selects fastest (not deterministic) kernels
- TensorFloat-32 (TF32) operations lose precision
- Thread-local state can cause race conditions

The facade fixes all of these issues **before** PyTorch initializes.

**Formal Proofs**: For the formal proof of how the facade guarantees deterministic execution, see [Reproducibility Proofs](reproducibility_proofs.md#reproducibility-via-effect-sequencing).

**Effect Interpreter Role**: The facade serves as an effect interpreter for determinism. See [Effect Interpreter](effect_interpreter.md) for how this fits into the broader effect system.

---

## Import Requirements

**CRITICAL**: Always import the facade before any direct PyTorch imports:

```python
# ✅ CORRECT - Facade first
import spectralmc.models.torch as sm_torch
import torch  # This will be the configured version
import torch.nn as nn

# Now safe to use PyTorch
model = nn.Linear(10, 5)
```

```python
# ❌ INCORRECT - Direct import first
import torch  # This will raise ImportError
import spectralmc.models.torch as sm_torch
```

**Why this matters**:

The facade sets global PyTorch flags during import. If PyTorch is imported first, these flags cannot be set reliably.

### Error Message

If you import PyTorch before the facade, you'll see:

```
ImportError: PyTorch must be imported via spectralmc.models.torch facade first.
Import spectralmc.models.torch before importing torch directly.
```

---

## Deterministic Settings

The facade automatically configures PyTorch for reproducibility:

```python
# Automatically set by the facade:
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
```

**What each setting does**:

- `use_deterministic_algorithms(True)` - Forces deterministic operations, raises error if non-deterministic op used
- `cudnn.deterministic = True` - Use deterministic cuDNN algorithms (slower but reproducible)
- `cudnn.benchmark = False` - Disable cuDNN auto-tuning (would select fastest, not deterministic, kernel)
- `cudnn.allow_tf32 = False` - Disable TensorFloat-32 (maintains full float32 precision)
- `cuda.matmul.allow_tf32 = False` - Disable TF32 in matrix multiplications
- `CUBLAS_WORKSPACE_CONFIG` - Required for deterministic CUDA operations

### Performance Impact

Deterministic mode is **slower** than default PyTorch:

- cuDNN deterministic algorithms: ~5-15% slower
- No auto-tuning: First batch slower (no kernel selection overhead later)
- No TF32: ~2x slower on Ampere+ GPUs (but maintains precision)

**This is acceptable**: Reproducibility is more important than raw speed for scientific computing.

---

## Thread Safety

The facade enforces thread-safety requirements:

```python
# ✅ CORRECT - Import in main thread before spawning workers
import spectralmc.models.torch as sm_torch

def worker_function():
    # Workers can use the pre-imported facade safely
    tensor = sm_torch.torch.randn(10)

# Spawn workers (Ray, Dask, multiprocessing)
workers = [spawn_worker(worker_function) for _ in range(4)]
```

```python
# ❌ INCORRECT - Importing facade in worker thread
def worker_function():
    import spectralmc.models.torch as sm_torch  # RuntimeError!
    tensor = sm_torch.torch.randn(10)
```

**Why**: PyTorch's global state is not thread-safe. The facade must be initialized once in the main thread before any workers start.

### Error Message

If you import the facade in a worker thread:

```
RuntimeError: spectralmc.models.torch facade must be imported in main thread
before spawning workers. Import it at module level in your main script.
```

---

## Device and Dtype Management

The facade provides type-safe helpers for device and dtype management:

### Setting Default Dtype

```python
import spectralmc.models.torch as sm_torch
import torch

# ✅ CORRECT - Type-safe context manager
with sm_torch.default_dtype(torch.float64):
    model = ComplexLinear(128, 64)  # Parameters created in float64

# ✅ CORRECT - Explicit dtype argument
model = ComplexLinear(128, 64)
model = model.to(dtype=torch.float64)

# ❌ INCORRECT - Direct PyTorch global state (not thread-safe)
torch.set_default_dtype(torch.float64)
```

### Setting Default Device

```python
# ✅ CORRECT - Type-safe device helper
with sm_torch.default_device(sm_torch.Device.cuda.to_torch()):
    model = ComplexLinear(128, 64)  # Parameters created on CUDA

# ❌ INCORRECT - Direct PyTorch calls (no type checking)
torch.set_default_device("cuda")  # Accepts any string, error-prone
```

### Device Enum

The facade provides a type-safe `Device` enum:

```python
from spectralmc.models.torch import Device

# Type-safe device specification
cpu_dev = Device.cpu.to_torch()  # Returns torch.device("cpu")
gpu_dev = Device.cuda.to_torch()  # Returns torch.device("cuda:0")

# Use in model initialization
model = model.to(device=gpu_dev)
```

---

## Complete Example

Typical usage pattern in SpectralMC:

```python
"""Training script with deterministic execution."""

# STEP 1: Import facade FIRST
import spectralmc.models.torch as sm_torch

# STEP 2: Now safe to import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# STEP 3: Import SpectralMC models (they use PyTorch internally)
from spectralmc.cvnn import ComplexSequential
from spectralmc.gbm import simulate_gbm

# STEP 4: Set random seed for reproducibility
torch.manual_seed(42)

# STEP 5: Use type-safe device and dtype helpers
def train():
    """Train model with deterministic execution."""
    device = sm_torch.Device.cuda.to_torch()
    dtype = torch.float64

    with sm_torch.default_dtype(dtype):
        with sm_torch.default_device(device):
            # Model parameters created on GPU in float64
            model = ComplexSequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

    # Training loop (deterministic)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        # Forward pass (deterministic)
        output = model(data)
        loss = loss_fn(output, target)

        # Backward pass (deterministic)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train()
```

---

## Testing with Facade

Tests must also import the facade first:

```python
"""Test module for ComplexLinear."""

# Import facade before pytest
import spectralmc.models.torch as sm_torch

# Now safe to import PyTorch and pytest
import torch
import pytest

from spectralmc.models.torch import ComplexLinear

def test_complex_linear_forward():
    """Test ComplexLinear forward pass."""
    torch.manual_seed(42)  # Deterministic test

    layer = ComplexLinear(in_features=4, out_features=2)
    real_input = torch.randn(3, 4)
    imag_input = torch.randn(3, 4)

    real_output, imag_output = layer(real_input, imag_input)

    assert real_output.shape == (3, 2)
    assert imag_output.shape == (3, 2)
```

---

## Facade Implementation Details

The facade is implemented in `src/spectralmc/models/torch.py`:

```python
"""PyTorch facade for deterministic execution."""

import os
import sys
import threading

# Check if imported in main thread
_MAIN_THREAD_ID = threading.get_ident()
if threading.get_ident() != _MAIN_THREAD_ID:
    raise RuntimeError(
        "spectralmc.models.torch facade must be imported in main thread"
    )

# Check if PyTorch already imported
if "torch" in sys.modules:
    raise ImportError(
        "PyTorch must be imported via spectralmc.models.torch facade first"
    )

# Import PyTorch
import torch

# Set deterministic mode BEFORE any CUDA operations
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

# Re-export torch for convenience
torch = torch
```

**Note**: The facade is a thin wrapper that sets global flags, then re-exports PyTorch.

---

## Common Questions

### Q: Can I use `import torch` without the facade in scripts?

**A: No.** All scripts, tests, and modules must import the facade first. This ensures deterministic execution everywhere.

### Q: What if I need to import PyTorch in a library?

**A: Import facade at the top of the library module.** This ensures the facade is initialized when the library is first imported.

```python
# In src/spectralmc/my_module.py
import spectralmc.models.torch as sm_torch  # Facade first
import torch  # Now safe

def my_function():
    return torch.randn(10)
```

### Q: Does the facade work with Ray/Dask/multiprocessing?

**A: Yes, if imported in main thread first.** Import the facade before spawning workers.

### Q: Can I disable deterministic mode for faster training?

**A: No.** Reproducibility is a core requirement of SpectralMC. Use the facade as-is.

---

## Complex-Valued Neural Networks

SpectralMC represents complex numbers as **pairs of real tensors** (real, imag). All complex-valued layers follow this pattern.

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

### ComplexValuedModel Protocol

Define interfaces using `Protocol` for type-safe duck typing:

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

## Summary

- **Import facade FIRST**: `import spectralmc.models.torch as sm_torch`
- **Deterministic by default**: All operations reproducible
- **Thread-safe**: Import in main thread before spawning workers
- **Type-safe helpers**: Use `Device` enum and context managers
- **Complex-valued layers**: Use `(real, imag)` tensor pair pattern
- **Protocol usage**: Type-safe duck typing with `ComplexValuedModel` protocol
- **Performance cost**: ~5-15% slower, but reproducible
- **Required everywhere**: Tests, scripts, libraries all use facade

See also: [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md), [Testing Requirements](testing_requirements.md), [Coding Standards](coding_standards.md), [Reproducibility Proofs](reproducibility_proofs.md), [Effect Interpreter](effect_interpreter.md)
