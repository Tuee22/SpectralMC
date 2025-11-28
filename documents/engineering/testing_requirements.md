# Testing Requirements

## Overview

All tests in SpectralMC must be fully typed, deterministic, and pass mypy strict mode. Tests are executed via `poetry run test-all` and are separated into CPU and GPU categories.

**Related Standards**: [Type Safety](type_safety.md), [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md), [PyTorch Facade](pytorch_facade.md)

---

## Test Structure

All tests must be **fully typed** and **mypy-strict-clean**:

```python
"""
End-to-end tests for spectralmc.models.torch.

All tests are fully typed and mypy-strict-clean.
"""

from __future__ import annotations

from typing import List, Tuple
import pytest
import torch

# Import facade before PyTorch
import spectralmc.models.torch as sm_torch

def test_complex_linear_forward() -> None:
    """Test complex linear layer forward pass."""
    layer = ComplexLinear(in_features=4, out_features=2)
    real_input: torch.Tensor = torch.randn(3, 4)
    imag_input: torch.Tensor = torch.randn(3, 4)

    real_output, imag_output = layer(real_input, imag_input)

    assert real_output.shape == (3, 2)
    assert imag_output.shape == (3, 2)
```

**Required elements**:
- Module-level docstring
- Type hints on all test functions (`-> None`)
- Type hints on all test variables
- Explicit imports (no `from module import *`)
- Facade imported before PyTorch

---

## Running Tests

**CRITICAL**: Tests must be run via `poetry run test-all`, **never** via direct `pytest`:

```bash
# ✅ CORRECT - Via Poetry script
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all

# ✅ CORRECT - With arguments
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all -v
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all -m gpu
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_gbm.py

# ❌ FORBIDDEN - Direct pytest (enforced by Dockerfile)
docker compose -f docker/docker-compose.yml exec spectralmc pytest tests/
```

**Why**: `poetry run test-all` enforces test output redirection and ensures complete output capture (see CLAUDE.md testing policies).

---

## GPU Testing

Mark GPU-specific tests with `@pytest.mark.gpu`:

```python
import pytest
import torch

@pytest.mark.gpu
def test_cuda_transfer() -> None:
    """Test tensor transfer to CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cpu_tensor: torch.Tensor = torch.randn(10)
    gpu_tensor: torch.Tensor = cpu_tensor.cuda()

    assert gpu_tensor.device.type == "cuda"
```

### Running GPU Tests

```bash
# Run only GPU tests
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all -m gpu

# Run all tests except GPU
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all -m 'not gpu'

# Run all tests (CPU + GPU)
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all
```

### Default Behavior

By default, `pytest` excludes GPU tests (configured in `pyproject.toml`):

```toml
[tool.pytest.ini_options]
addopts = "-ra -q -m 'not gpu'"
markers = [
    "gpu: needs CUDA",
]
```

Use `poetry run test-all` to include GPU tests.

---

## Deterministic Testing

Ensure all tests are **deterministic** by setting random seeds:

```python
import torch

def test_deterministic_generation() -> None:
    """Test that random generation is deterministic."""
    torch.manual_seed(42)
    tensor1: torch.Tensor = torch.randn(10)

    torch.manual_seed(42)
    tensor2: torch.Tensor = torch.randn(10)

    assert torch.equal(tensor1, tensor2)  # Exact equality
```

**Best practices**:
- Set `torch.manual_seed(42)` at start of each test
- Use `torch.equal()` for exact tensor equality
- Use `torch.allclose()` for floating-point comparisons with tolerance

### Floating-Point Comparisons

```python
import torch

def test_numerical_computation() -> None:
    """Test numerical computation with tolerance."""
    torch.manual_seed(42)

    result = compute_something()
    expected = torch.tensor([1.0, 2.0, 3.0])

    # ✅ CORRECT - Use allclose with appropriate tolerance
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-8)

    # ❌ INCORRECT - Exact equality for floating point
    # assert torch.equal(result, expected)  # May fail due to rounding
```

---

## Test Fixtures

Use pytest fixtures for common test setup:

```python
import pytest
import torch

@pytest.fixture
def device() -> torch.device:
    """Get CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def complex_model() -> ComplexLinear:
    """Create a complex linear layer for testing."""
    torch.manual_seed(42)
    return ComplexLinear(in_features=10, out_features=5)

def test_with_fixture(complex_model: ComplexLinear, device: torch.device) -> None:
    """Test using fixtures."""
    complex_model = complex_model.to(device)
    real_in = torch.randn(32, 10, device=device)
    imag_in = torch.randn(32, 10, device=device)

    real_out, imag_out = complex_model(real_in, imag_in)

    assert real_out.shape == (32, 5)
    assert imag_out.device == device
```

---

## Test Coverage

Tests should cover:

1. **Happy path**: Normal usage with valid inputs
2. **Edge cases**: Boundary conditions (empty tensors, single element, etc.)
3. **Error cases**: Invalid inputs should raise expected exceptions
4. **Determinism**: Same input → same output with fixed seed
5. **Type safety**: All code paths type-checked by mypy

### Example: Comprehensive Test

```python
import pytest
import torch
from pydantic import ValidationError

def test_bound_spec_happy_path() -> None:
    """Test BoundSpec with valid bounds."""
    bounds = BoundSpec(lower=0.0, upper=1.0)
    assert bounds.lower == 0.0
    assert bounds.upper == 1.0

def test_bound_spec_edge_case() -> None:
    """Test BoundSpec with very small range."""
    bounds = BoundSpec(lower=0.0, upper=1e-10)
    assert bounds.upper > bounds.lower

def test_bound_spec_invalid() -> None:
    """Test BoundSpec rejects invalid bounds."""
    with pytest.raises(ValidationError, match="lower.*less than.*upper"):
        BoundSpec(lower=1.0, upper=0.0)

def test_bound_spec_deterministic() -> None:
    """Test BoundSpec produces same result."""
    bounds1 = BoundSpec(lower=0.0, upper=1.0)
    bounds2 = BoundSpec(lower=0.0, upper=1.0)
    assert bounds1.lower == bounds2.lower
    assert bounds1.upper == bounds2.upper
```

---

## Async Testing

For async code, use `pytest-asyncio`:

```python
import pytest

@pytest.mark.asyncio
async def test_async_function() -> None:
    """Test asynchronous function."""
    result = await async_compute()
    assert result == expected_value
```

Configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "asyncio: async test",
]
```

---

## Test Organization

Tests are organized by module:

```
tests/
├── conftest.py                # Global fixtures
├── test_async_normals.py      # Async normal distribution tests
├── test_cvnn.py               # Complex-valued NN tests
├── test_cvnn_factory.py       # CVNN factory tests
├── test_gbm.py                # GBM simulation tests (GPU-heavy)
├── test_gbm_trainer.py        # GBM trainer tests
├── test_models_torch.py       # PyTorch model tests
├── test_sobol_sampler.py      # Sobol sampler tests
└── test_storage/              # Storage system tests
    ├── test_chain.py
    ├── test_store.py
    └── test_verification.py
```

**File naming**: `test_<module_name>.py` matches `src/spectralmc/<module_name>.py`

---

## Common Pitfalls

### Pitfall 1: Implicit Device Fallback

```python
# ❌ WRONG - Silent CPU fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ CORRECT - Explicit GPU requirement
@pytest.mark.gpu
def test_gpu_operation() -> None:
    """Test GPU operation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda:0")  # Explicit device
```

### Pitfall 2: Missing Type Hints

```python
# ❌ WRONG - Missing type hints
def test_something():
    result = compute()
    assert result == expected

# ✅ CORRECT - Full type hints
def test_something() -> None:
    result: torch.Tensor = compute()
    expected: torch.Tensor = torch.tensor([1.0, 2.0])
    assert torch.allclose(result, expected)
```

### Pitfall 3: Non-Deterministic Tests

```python
# ❌ WRONG - No seed
def test_random_generation() -> None:
    tensor = torch.randn(10)  # Different every time
    assert tensor.shape == (10,)

# ✅ CORRECT - Deterministic with seed
def test_random_generation() -> None:
    torch.manual_seed(42)
    tensor = torch.randn(10)
    expected = torch.randn(10)  # Won't match! Need to re-seed

# ✅ CORRECT - Proper deterministic test
def test_random_generation() -> None:
    torch.manual_seed(42)
    tensor1 = torch.randn(10)

    torch.manual_seed(42)
    tensor2 = torch.randn(10)

    assert torch.equal(tensor1, tensor2)
```

---

## Summary

- **Fully typed**: All tests pass `mypy --strict`
- **Run via Poetry**: Use `poetry run test-all`, never direct `pytest`
- **GPU tests**: Mark with `@pytest.mark.gpu`
- **Deterministic**: Set `torch.manual_seed()` in every test
- **Comprehensive**: Cover happy path, edge cases, errors
- **Async**: Use `pytest-asyncio` for async code

See also: [Type Safety](type_safety.md), [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md), [PyTorch Facade](pytorch_facade.md)
