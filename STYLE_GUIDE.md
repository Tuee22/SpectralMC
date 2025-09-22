# SpectralMC Python Style Guide

## Introduction

This document outlines the coding standards and best practices for the SpectralMC project. These rules are requirements for maintaining the scientific reproducibility and performance guarantees that make SpectralMC reliable for GPU-accelerated Monte Carlo learning.

### Why These Rules Matter

SpectralMC operates in an environment where reproducibility is paramount:

- **Financial Quantitative Models**: Results must be bit-for-bit identical across different environments
- **GPU-Accelerated Computation**: Performance bottlenecks can occur silently if data movement isn't carefully controlled
- **Scientific Computing**: Reproducible results are essential for validating mathematical models
- **Complex Systems**: Strict typing prevents runtime errors in computationally expensive operations

Every rule in this guide serves one of these critical goals:

1. **Reproducibility**: Deterministic behavior across platforms, environments, and time
2. **Performance**: Explicit control over expensive operations like GPU/CPU transfers
3. **Reliability**: Catch errors at compile-time rather than during expensive computations
4. **Maintainability**: Clear, self-documenting code that can be understood and modified safely

### The Cost of Non-Compliance

Violating these guidelines can result in:
- Non-reproducible training runs costing hours of GPU time
- Silent performance degradation due to unexpected PCIe transfers
- Runtime type errors in production quantitative models
- Inability to validate scientific results across different environments

**All code must adhere to these guidelines without exception.**

## Table of Contents

1. [Code Formatting with Black](#code-formatting-with-black)
2. [Strict Type Safety with mypy](#strict-type-safety-with-mypy)
3. [Custom Type Stubs (.pyi files)](#custom-type-stubs-pyi-files)
4. [Pydantic Best Practices](#pydantic-best-practices)
5. [Documentation Standards](#documentation-standards)
6. [PyTorch Facade Pattern](#pytorch-facade-pattern)
7. [Project-Specific Patterns](#project-specific-patterns)
8. [Testing Requirements](#testing-requirements)

---

## Code Formatting with Black

### Running Black

Format all Python code from the project root:

```bash
black .
```

This command will automatically format all Python files in the project, including source code, tests, and scripts.

### Configuration

All Python code must be formatted with **Black 25.1+**. The configuration is defined in `pyproject.toml`:

```toml
[tool.poetry.group.dev.dependencies]
black = ">=25.1,<26.0"
```

### Key Principles

- **Zero configuration**: Black's defaults are non-negotiable
- **Consistent formatting**: No manual formatting decisions
- **Automated enforcement**: Integrate with pre-commit hooks

---

## Strict Type Safety with mypy

### Running mypy

Type-check the entire codebase from the project root:

```bash
mypy
```

This command uses the configuration in `pyproject.toml` to check all source code and tests. **All code must pass mypy --strict with zero errors.**

### Configuration

mypy is configured with **strict mode** in `pyproject.toml`:

```toml
[tool.mypy]
mypy_path = "typings"
python_version = "3.12"
strict = true
files = ["src/spectralmc", "tests"]
```

The `typings` directory provides project-specific stubs for all third-party dependencies, ensuring complete type coverage.

### Zero Tolerance Policy

The following are **NEVER** allowed:

- `Any` type annotations
- `cast()` function calls
- `# type: ignore` comments

### Required Annotations

All functions, methods, and class attributes must have explicit type annotations:

```python
# ✅ CORRECT
def complex_multiply(
    real_a: float, imag_a: float, real_b: float, imag_b: float
) -> Tuple[float, float]:
    """Multiply two complex numbers represented as (real, imag) pairs."""
    real_result: float = real_a * real_b - imag_a * imag_b
    imag_result: float = real_a * imag_b + imag_a * real_b
    return real_result, imag_result

# ❌ INCORRECT - Missing type annotations
def complex_multiply(real_a, imag_a, real_b, imag_b):
    real_result = real_a * real_b - imag_a * imag_b
    imag_result = real_a * imag_b + imag_a * real_b
    return real_result, imag_result
```

### Class Attribute Annotations

All class attributes must be explicitly typed:

```python
# ✅ CORRECT
class ComplexLinear(nn.Module):
    real_weight: nn.Parameter
    imag_weight: nn.Parameter
    real_bias: Optional[nn.Parameter]
    imag_bias: Optional[nn.Parameter]

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        # ... implementation
```

---

## Custom Type Stubs (.pyi files)

### Rationale

SpectralMC maintains custom type stubs for all third-party libraries to ensure:

- **Complete type coverage** for external dependencies
- **Project-specific typing** that may be stricter than upstream
- **Reproducible type checking** independent of external stub changes

### Organization

Type stubs are organized in the `typings/` directory:

```
typings/
├── torch/
│   ├── __init__.pyi
│   ├── nn/
│   │   ├── __init__.pyi
│   │   └── functional.pyi
│   └── cuda/
│       └── __init__.pyi
├── scipy/
│   ├── __init__.pyi
│   └── stats/
│       └── qmc/__init__.pyi
├── numba/
│   ├── __init__.pyi
│   └── cuda/__init__.pyi
├── cupy/
│   ├── __init__.pyi
│   └── cuda/__init__.pyi
├── QuantLib/__init__.pyi
└── safetensors/
    ├── __init__.pyi
    └── torch.pyi
```

### Stub Requirements

All stubs must be:

1. **Type-pure**: No `Any`, `cast`, or `type: ignore`
2. **Minimal**: Only include APIs actually used by SpectralMC
3. **Documented**: Include docstrings explaining purpose

Example stub structure:

```python
# typings/torch/__init__.pyi
"""
Strict, project‑specific stub for the **top‑level** :pymod:`torch` namespace.

Only the public surface exercised by SpectralMC is declared.  The stub must
remain *type‑pure*: **no** ``Any``, **no** ``cast``, **no** ``type: ignore``.
"""

from __future__ import annotations

from typing import Protocol, overload

class dtype: ...

float32: dtype
float64: dtype
complex64: dtype
complex128: dtype

def get_default_dtype() -> dtype: ...
def set_default_dtype(d: dtype) -> None: ...
```

### Adding New Dependencies

When adding a new third-party dependency:

1. Create corresponding stub files in `typings/`
2. Include only the minimal API surface used by SpectralMC
3. Ensure all stubs pass `mypy --strict`
4. Document the stub's purpose and scope

---

## Pydantic Best Practices

### Model Definition

Use Pydantic for all configuration and data validation:

```python
from pydantic import BaseModel, ConfigDict, model_validator

class BoundSpec(BaseModel):
    """Inclusive numeric bounds for a single coordinate axis."""

    lower: float
    upper: float

    @model_validator(mode="after")
    def _validate(self) -> "BoundSpec":
        """Ensure the lower bound is strictly less than the upper bound."""
        if self.lower >= self.upper:
            raise ValueError("`lower` must be strictly less than `upper`.")
        return self
```

### Configuration Classes

Use Pydantic for complex configuration with nested validation:

```python
class BlackScholesConfig(BaseModel):
    """Configuration for Black-Scholes Monte Carlo simulation."""

    model_config = ConfigDict(frozen=True)

    spot: float
    rate: float
    volatility: float
    maturity: float

    @model_validator(mode="after")
    def _validate_positive_values(self) -> "BlackScholesConfig":
        """Ensure all financial parameters are positive."""
        for field_name, value in [
            ("spot", self.spot),
            ("volatility", self.volatility),
            ("maturity", self.maturity)
        ]:
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        return self
```

### Type Safety with Pydantic

Always use proper typing with Pydantic models:

```python
# ✅ CORRECT - Explicit typing
def create_sampler(bounds: Dict[str, BoundSpec], model_class: Type[PointT]) -> SobolSampler[PointT]:
    return SobolSampler(bounds=bounds, model_class=model_class)

# ❌ INCORRECT - Implicit typing
def create_sampler(bounds, model_class):
    return SobolSampler(bounds=bounds, model_class=model_class)
```

---

## Documentation Standards

### Module Documentation

Every module must include comprehensive documentation:

```python
"""
Complex-valued neural-network blocks for SpectralMC's test-suite
================================================================

Overview
--------
This module collects **fully-typed**, *dependency-free* PyTorch layers
that operate on pairs of real tensors representing the real and
imaginary parts of a complex signal.

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
"""
```

### Function Documentation

Use **Google-style docstrings** for all public functions and methods:

#### Google-Style Docstring Format

Google-style docstrings use section headers followed by content. The key sections are:

- **Args**: Function arguments (one per line)
- **Returns**: Return value description
- **Raises**: Exceptions that may be raised
- **Yields**: For generator functions
- **Note**: Additional implementation details

#### Example from SpectralMC Codebase

```python
def forward(self, real: Tensor, imag: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply the affine projection.

    Args:
        real: Real part of input tensor, shape (batch, in_features).
        imag: Imaginary part of input tensor, shape (batch, in_features).

    Returns:
        Tuple of (real_output, imag_output) tensors with shape
        (batch, out_features).
    """
```

#### Complete Function Example

```python
def complex_linear_forward(
    real_input: torch.Tensor,
    imag_input: torch.Tensor,
    real_weight: torch.Tensor,
    imag_weight: torch.Tensor,
    real_bias: Optional[torch.Tensor] = None,
    imag_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform complex linear transformation.

    This function computes the complex matrix multiplication:
    W * z + b = (A x - B y) + i(B x + A y) + (b_real + i*b_imag)

    Args:
        real_input: Real part of input tensor, shape (batch_size, in_features).
        imag_input: Imaginary part of input tensor, shape (batch_size, in_features).
        real_weight: Real part of weight matrix, shape (out_features, in_features).
        imag_weight: Imaginary part of weight matrix, shape (out_features, in_features).
        real_bias: Real part of bias vector, shape (out_features,). Optional.
        imag_bias: Imaginary part of bias vector, shape (out_features,). Optional.

    Returns:
        Tuple of (real_output, imag_output) where each tensor has shape
        (batch_size, out_features).

    Raises:
        ValueError: If input tensor shapes are incompatible with weight matrices.
    """
```

#### Class Documentation Example

```python
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
        in_features: Number of input features (per complex component).
        out_features: Number of output features (per complex component).
        bias: If True, include a complex bias term. Default: True.
    """
```

### Technical Documentation

Maintain the `docs/` folder with technical papers and implementation details:

- `docs/whitepaper.md` - Core mathematical foundations
- `docs/characteristic_function_for_stochastic_processes.md` - Theory background
- `docs/deep_complex_valued_set_encoders.md` - Architecture details
- `docs/variable_length_cvnn_inputs.md` - Input handling
- `docs/imaginary_numbers_unified_intuition.md` - Mathematical intuition

---

## PyTorch Facade Pattern

### Rationale

SpectralMC uses a custom PyTorch facade (`spectralmc.models.torch`) to ensure:

1. **Guaranteed reproducibility** of training across different environments
2. **Deterministic execution** on both CPU and GPU
3. **Thread safety** for multi-threaded applications
4. **Controlled import order** to set global flags before PyTorch initialization

### Import Requirements

**CRITICAL**: Always import the facade before any direct PyTorch imports:

```python
# ✅ CORRECT - Facade first
import spectralmc.models.torch as sm_torch
import torch  # This will be the configured version

# ❌ INCORRECT - Direct import first
import torch  # This will raise ImportError
import spectralmc.models.torch as sm_torch
```

### Deterministic Settings

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

### Thread Safety

The facade enforces thread safety requirements:

```python
# ✅ CORRECT - Import in main thread before spawning workers
import spectralmc.models.torch as sm_torch

def worker_function():
    # Workers can use the pre-imported facade safely
    tensor = sm_torch.torch.randn(10)

# ❌ INCORRECT - Importing facade in worker thread
def worker_function():
    import spectralmc.models.torch as sm_torch  # RuntimeError!
```

### Device and Dtype Management

Use the facade's type-safe device and dtype helpers:

```python
# ✅ CORRECT - Type-safe helpers
with sm_torch.default_dtype(torch.float64):
    with sm_torch.default_device(sm_torch.Device.cuda.to_torch()):
        model = ComplexLinear(128, 64)

# ❌ INCORRECT - Direct PyTorch calls
torch.set_default_dtype(torch.float64)  # Not thread-safe
torch.set_default_device("cuda")  # No type checking
```

---

## Project-Specific Patterns

### Complex-Valued Neural Networks

Use the established pattern for complex-valued layers:

```python
class ComplexLayer(nn.Module):
    """Base pattern for complex-valued layers."""

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process complex input as (real, imag) pair.

        Parameters
        ----------
        real : torch.Tensor
            Real component of complex input.
        imag : torch.Tensor
            Imaginary component of complex input.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Real and imaginary components of complex output.
        """
        # Implementation here
        return real_output, imag_output
```

### Protocol Usage

Define interfaces using `Protocol` for type safety:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ComplexValuedModel(Protocol):
    """Protocol for complex-valued neural network models."""

    def __call__(
        self, __real: torch.Tensor, __imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def parameters(self) -> Iterable[nn.Parameter]: ...
```

### GPU/CPU Transfer Patterns

#### The PCIe Bottleneck Problem

SpectralMC enforces explicit, controlled data movement between CPU and GPU to prevent silent performance degradation. Understanding why this matters:

**Performance Context:**
- **GPU Memory Bandwidth**: 900+ GB/s (HBM2/HBM3)
- **PCIe 4.0 x16 Bandwidth**: ~32 GB/s theoretical, ~25 GB/s practical
- **PCIe 3.0 x16 Bandwidth**: ~16 GB/s theoretical, ~12 GB/s practical

This means GPU↔CPU transfers are **30-75x slower** than GPU-internal operations. A single implicit `.cuda()` call can dominate your entire training step.

#### Preventing Implicit Transfers

SpectralMC makes implicit transfers **impossible** by:

1. **Avoiding PyTorch's implicit transfer methods**:
   ```python
   # ❌ FORBIDDEN - Silent performance killers
   tensor.cuda()           # Implicit GPU transfer
   tensor.cpu()            # Implicit CPU transfer
   tensor.to(device)       # Implicit transfer
   tensor.to("cuda")       # Implicit transfer
   ```

2. **Using explicit, batched transfers**:
   ```python
   # ✅ CORRECT - Explicit, controlled transfers
   from spectralmc.models.cpu_gpu_transfer import move_tensor_tree

   gpu_state = move_tensor_tree(
       cpu_state,
       dest=Device.cuda,
       pin_memory=True  # Enables faster host→device transfers
   )
   ```

#### The TensorTree Pattern

SpectralMC uses `TensorTree` to handle arbitrarily nested data structures:

```python
from spectralmc.models.cpu_gpu_transfer import move_tensor_tree, TensorTree

# TensorTree can handle any nesting:
TensorTree = Union[
    torch.Tensor,           # Leaf tensors
    List["TensorTree"],     # Lists of tensors/structures
    Tuple["TensorTree"],    # Tuples of tensors/structures
    Mapping[str, "TensorTree"],  # Dicts of tensors/structures
    Scalar                  # Non-tensor data (passed through)
]

# ✅ CORRECT - Batched transfer of complex state
def move_optimizer_state_to_gpu(opt_state: Dict[str, Any]) -> Dict[str, Any]:
    """Move entire optimizer state to GPU in a single batched operation."""
    return move_tensor_tree(
        opt_state,
        dest=Device.cuda,
        pin_memory=True
    )
```

#### Performance Best Practices

1. **Batch transfers**: Move entire state dictionaries at once, not individual tensors
2. **Use pinned memory**: Set `pin_memory=True` for faster host→device transfers
3. **Minimize frequency**: Transfer only when necessary (e.g., checkpointing)
4. **Stream synchronization**: The transfer functions handle CUDA stream synchronization automatically

#### Non-Blocking Transfers

SpectralMC's transfer functions use non-blocking operations with automatic synchronization:

```python
def _copy_tensor(src: torch.Tensor, *, target_dev: torch.device, pin_memory: bool) -> torch.Tensor:
    """Clone src onto target_dev using non-blocking transfer."""
    # Uses dedicated CUDA stream for transfers
    with torch.cuda.stream(_CUDA_STREAM):
        dst.copy_(src.detach(), non_blocking=True)
    # Automatic synchronization ensures completion
    return dst
```

#### Why This Matters for SpectralMC

In quantitative finance applications:
- **Training datasets** can be 10+ GB (millions of Monte Carlo paths)
- **Model checkpoints** contain complex optimizer states (Adam momentum, variance)
- **Batch inference** processes thousands of contracts simultaneously

Unexpected PCIe transfers can:
- Turn a 10ms GPU operation into a 1000ms CPU↔GPU transfer
- Make training 100x slower than necessary
- Create non-deterministic performance based on tensor placement

**The explicit transfer pattern ensures predictable, optimal performance.**

### Error Handling

Use expression-level exception raising for functional style:

```python
def _raise(exc: Exception) -> NoReturn:
    """Expression-level exception raising."""
    raise exc

def validate_positive(value: float, name: str) -> float:
    """Validate that a value is positive."""
    return value if value > 0 else _raise(ValueError(f"{name} must be positive"))
```

---

## Testing Requirements

### Test Structure

All tests must be fully typed and mypy-strict clean:

```python
"""
End‑to‑end tests for ``spectralmc.models.torch``.

All tests are fully typed and mypy‑strict‑clean.
"""

from __future__ import annotations

from typing import List, Tuple
import pytest
import torch

def test_complex_linear_forward() -> None:
    """Test complex linear layer forward pass."""
    layer = ComplexLinear(in_features=4, out_features=2)
    real_input: torch.Tensor = torch.randn(3, 4)
    imag_input: torch.Tensor = torch.randn(3, 4)

    real_output, imag_output = layer(real_input, imag_input)

    assert real_output.shape == (3, 2)
    assert imag_output.shape == (3, 2)
```

### GPU Testing

Mark GPU-specific tests appropriately:

```python
@pytest.mark.gpu
def test_cuda_transfer() -> None:
    """Test tensor transfer to CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cpu_tensor: torch.Tensor = torch.randn(10)
    gpu_tensor: torch.Tensor = cpu_tensor.cuda()

    assert gpu_tensor.device.type == "cuda"
```

### Deterministic Testing

Ensure all tests are deterministic:

```python
def test_deterministic_generation() -> None:
    """Test that random generation is deterministic."""
    torch.manual_seed(42)
    tensor1: torch.Tensor = torch.randn(10)

    torch.manual_seed(42)
    tensor2: torch.Tensor = torch.randn(10)

    assert torch.equal(tensor1, tensor2)
```

---

## Summary

This style guide ensures that SpectralMC maintains:

- **Strict type safety** with zero tolerance for `Any`, `cast`, or `type: ignore`
- **Reproducible execution** through the PyTorch facade pattern
- **Comprehensive documentation** for all modules and functions
- **Consistent formatting** with Black
- **Robust testing** with full type coverage

All code must pass `mypy --strict` and `black --check` before being committed. The custom type stubs in `typings/` ensure complete type coverage for all dependencies, while the PyTorch facade guarantees reproducible, deterministic execution across environments.

Following these guidelines ensures that SpectralMC remains a high-quality, maintainable, and reliable codebase for GPU-accelerated Monte Carlo learning.