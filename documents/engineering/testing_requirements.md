# File: documents/engineering/testing_requirements.md
# Testing Requirements

**Status**: Authoritative source  
**Supersedes**: Prior testing requirements drafts  
**Referenced by**: documents/documentation_standards.md; documents/testing_architecture.md

> **Purpose**: SSoT for SpectralMC testing expectations, determinism, and GPU constraints.

## Cross-References
- [Purity Doctrine](purity_doctrine.md)
- [Coding Standards](coding_standards.md)
- [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md)
- [PyTorch Facade](pytorch_facade.md)
- [Documentation Standards](../documentation_standards.md)

## Overview

All tests in SpectralMC must be fully typed, deterministic, and pass mypy strict mode. Tests are executed via `poetry run test-all`. All tests require GPU - silent CPU fallbacks are strictly forbidden.

### Test Code Purity Exception

Test code is **exempt** from certain purity requirements defined in the [Purity Doctrine](purity_doctrine.md):

| Allowed in Tests | Why |
|-----------------|-----|
| `assert` statements | pytest requires them |
| `pytest.raises()` | Testing exception behavior |
| `for` loops in setup/teardown | Test infrastructure |

Test code should still follow purity for the **code under test** - only test infrastructure may use these constructs.

---

## Test Structure

All tests must be **fully typed** and **mypy-strict-clean**:

```python
# File: documents/engineering/testing_requirements.md
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
# File: documents/engineering/testing_requirements.md
# ✅ CORRECT - Via Poetry script
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all

# ✅ CORRECT - With arguments
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all -v
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_gbm.py

# ❌ FORBIDDEN - Direct pytest (enforced by Dockerfile)
docker compose -f docker/docker-compose.yml exec spectralmc pytest tests/
```

**Why**: `poetry run test-all` enforces test output redirection and ensures complete output capture (see CLAUDE.md testing policies).

---

## Timeout Policy

- **Per-test guard**: 60s autouse timeout in `tests/conftest.py`; applies to setup/teardown,
  async tests, storage CLI calls, and GPU kernels across unit/integration/e2e suites.
- **Overrides**: Use `@pytest.mark.timeout(seconds=...)` only when a test is expected to exceed
  60s; keep values minimal, positive, and justified inline.
- **Suite execution/logs**: Do not wrap `poetry run test-all` with shell-level timeouts; rely on
  the per-test guard and preserve full logs (already enforced by `test-all` redirection).

---

## Code Quality Checks

**CRITICAL**: Run code quality checks via `poetry run check-code`:

```bash
# File: documents/engineering/testing_requirements.md
# ✅ CORRECT - Via Poetry script (runs Ruff → Black → MyPy)
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code

# Additional type safety variants:
docker compose -f docker/docker-compose.yml exec spectralmc poetry run typecheck      # MyPy only
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-types    # AST checker
docker compose -f docker/docker-compose.yml exec spectralmc poetry run type-check-all # MyPy + AST
```

### check-code Pipeline

The `check-code` command runs three tools in sequence with fail-fast behavior:

1. **Ruff** - Linting with auto-fix (`ruff check --fix`)
2. **Black** - Code formatting (modifies files in place)
3. **MyPy** - Static type checking

If any tool fails, the pipeline exits immediately with the error code.

### AST-Based Type Safety

Beyond mypy's `strict` mode, SpectralMC enforces additional type safety via AST analysis:

- **TYP001**: No `Any` type - imports or usage forbidden
- **TYP002**: No `cast()` - type narrowing must use isinstance or protocols
- **TYP003**: No `type: ignore` - all type errors must be fixed, not suppressed

Run AST checker:
```bash
# File: documents/engineering/testing_requirements.md
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-types
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-types --fix  # Auto-fix
```

---

## GPU Requirements

**All tests assume GPU is available. Tests MUST fail if CUDA is unavailable.**

SpectralMC is a GPU-accelerated library. Silent fallbacks from GPU to CPU are strictly forbidden because they mask performance regressions and can hide real bugs in GPU code paths.

### Required Pattern

Use module-level assertions to fail immediately without GPU:

```python
# File: documents/engineering/testing_requirements.md
import torch

# Module-level GPU requirement - test file fails immediately without GPU
assert torch.cuda.is_available(), "CUDA required for SpectralMC tests"

GPU_DEV: torch.device = torch.device("cuda:0")

def test_gpu_operation() -> None:
    """All tests run on GPU by default."""
    tensor = torch.randn(10, device=GPU_DEV)
    assert tensor.device.type == "cuda"
```

### Forbidden Patterns

```python
# File: documents/engineering/testing_requirements.md
# ❌ FORBIDDEN - Silent CPU fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ❌ FORBIDDEN - pytest.mark.gpu markers (GPU is assumed, not optional)
@pytest.mark.gpu
def test_something() -> None:
    ...

# ❌ FORBIDDEN - pytest.skip for missing GPU (missing GPU is a failure)
if not torch.cuda.is_available():
    pytest.skip("CUDA not available")

# ❌ FORBIDDEN - Conditional device fixtures
@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Correct Patterns

```python
# File: documents/engineering/testing_requirements.md
# ✅ CORRECT - Module-level GPU assertion
assert torch.cuda.is_available(), "CUDA required"
GPU_DEV: torch.device = torch.device("cuda:0")

# ✅ CORRECT - Explicit GPU device
def test_training() -> None:
    model = MyModel().to(GPU_DEV)
    tensor = torch.randn(10, device=GPU_DEV)
    ...

# ✅ CORRECT - GPU device fixture (no fallback)
@pytest.fixture
def gpu_device() -> torch.device:
    return torch.device("cuda:0")
```

---

## Deterministic Testing

Ensure all tests are **deterministic** by setting random seeds:

```python
# File: documents/engineering/testing_requirements.md
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
# File: documents/engineering/testing_requirements.md
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

Use pytest fixtures for common test setup. **Never use conditional device fallback in fixtures.**

```python
# File: documents/engineering/testing_requirements.md
import pytest
import torch

# Module-level GPU requirement
assert torch.cuda.is_available(), "CUDA required"
GPU_DEV: torch.device = torch.device("cuda:0")

@pytest.fixture
def gpu_device() -> torch.device:
    """Get CUDA device (no fallback)."""
    return GPU_DEV

@pytest.fixture
def complex_model() -> ComplexLinear:
    """Create a complex linear layer for testing."""
    torch.manual_seed(42)
    return ComplexLinear(in_features=10, out_features=5)

def test_with_fixture(complex_model: ComplexLinear, gpu_device: torch.device) -> None:
    """Test using fixtures."""
    complex_model = complex_model.to(gpu_device)
    real_in = torch.randn(32, 10, device=gpu_device)
    imag_in = torch.randn(32, 10, device=gpu_device)

    real_out, imag_out = complex_model(real_in, imag_in)

    assert real_out.shape == (32, 5)
    assert imag_out.device == gpu_device
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
# File: documents/engineering/testing_requirements.md
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
# File: documents/engineering/testing_requirements.md
import pytest

@pytest.mark.asyncio
async def test_async_function() -> None:
    """Test asynchronous function."""
    result = await async_compute()
    assert result == expected_value
```

Configuration in `pyproject.toml`:

```toml
# File: documents/engineering/testing_requirements.md
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = ["asyncio: async test"]
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
├── test_storage/              # Storage system tests (14 files)
│   ├── test_atomic_cas.py         # Atomic CAS operations
│   ├── test_audit_log.py          # Audit logging
│   ├── test_chain.py              # Blockchain chain operations
│   ├── test_cli.py                # CLI commands
│   ├── test_context_manager.py    # Async context manager
│   ├── test_e2e_storage.py        # End-to-end storage tests
│   ├── test_fast_forward.py       # Fast-forward merges
│   ├── test_gc.py                 # Garbage collection
│   ├── test_inference_client.py   # Inference client modes
│   ├── test_retry_logic.py        # S3 retry logic
│   ├── test_rollback.py           # Rollback operations
│   ├── test_store.py              # Core store operations
│   ├── test_tensorboard.py        # TensorBoard logging
│   └── test_training_integration.py  # Training integration
└── test_effects/              # Effect system tests
    ├── test_effect_types.py       # Effect ADT validation
    ├── test_mock_interpreter.py   # MockInterpreter tests
    └── test_composition.py        # Effect composition tests
```

**File naming**: `test_<module_name>.py` matches `src/spectralmc/<module_name>.py`

---

## Common Pitfalls

### Pitfall 1: Implicit Device Fallback

```python
# File: documents/engineering/testing_requirements.md
# ❌ WRONG - Silent CPU fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ❌ WRONG - pytest.skip for missing GPU
if not torch.cuda.is_available():
    pytest.skip("CUDA required")

# ✅ CORRECT - Module-level assertion (fails immediately)
assert torch.cuda.is_available(), "CUDA required"
GPU_DEV: torch.device = torch.device("cuda:0")

def test_gpu_operation() -> None:
    """Test GPU operation."""
    tensor = torch.randn(10, device=GPU_DEV)
    assert tensor.device.type == "cuda"
```

### Pitfall 2: Missing Type Hints

```python
# File: documents/engineering/testing_requirements.md
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
# File: documents/engineering/testing_requirements.md
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

---

## Test Output Handling

### CRITICAL - Output Truncation Issue

The Bash tool truncates output at 30,000 characters. Test suites may produce large output that WILL BE TRUNCATED, making it impossible to properly analyze failures.

### Required Test Execution Workflow

**ALWAYS** redirect test output to files in /tmp/, then read the complete output:

```bash
# File: documents/engineering/testing_requirements.md
# Step 1: Run tests with output redirection
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all > /tmp/test-output.txt 2>&1

# Step 2: Read complete output using Read tool
# Read /tmp/test-output.txt

# Step 3: Analyze ALL failures, not just visible ones
```

### For Specific Test Categories

```bash
# File: documents/engineering/testing_requirements.md
# Run specific test file
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_gbm.py > /tmp/test-gbm.txt 2>&1

# Run storage tests
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_storage/ > /tmp/test-storage.txt 2>&1

# Type checking
docker compose -f docker/docker-compose.yml exec spectralmc mypy src/spectralmc --strict > /tmp/mypy-output.txt 2>&1
```

### Why This Matters

- Bash tool output truncation at 30K chars is HARD LIMIT
- Test suites can produce 100KB+ of output
- Truncated output hides most failures, making diagnosis impossible
- File-based approach ensures complete output is always available
- Read tool has no size limits for files

### Test Execution Requirements

**Forbidden**:
- ❌ `Bash(command="...test...", timeout=60000)` - Short timeouts truncate output and kill tests
- ❌ `Bash(command="...test...", run_in_background=True)` - Can't see failures in real-time
- ❌ Reading only partial output with `head -n 100` or similar truncation
- ❌ Checking test status before completion (polling BashOutput prematurely)

**Required**:
- ✅ No timeout parameter on test commands for local runs (per-test guard handles hangs)
- ✅ Wait for complete test execution (GPU tests can take several minutes)
- ✅ Review ALL stdout/stderr output before drawing conclusions
- ✅ Let tests complete naturally, read full results

---

## Testing Anti-Patterns

### 1. Tests Pass When Features Are Broken

**Problem**: Test validates that code runs, not that it produces correct results

- ❌ `assert result is not None` - accepts any output
- ❌ `assert len(simulations) > 0` - passes even with wrong values
- ✅ `assert torch.allclose(result, expected, rtol=1e-5)` - validates numerical accuracy
- ✅ `assert torch.isfinite(result).all()` - ensures no NaN/Inf values

**Impact**: Broken numerical computations go undetected, corrupting downstream calculations

### 2. Accepting NotImplementedError as Success

**Problem**: Treating placeholder implementations as working features

- ❌ Accepting `NotImplementedError` or returning `None` in convergence checks
- ❌ Methods that return empty tensors when they should compute values
- ✅ All methods must have complete implementations before merging
- ✅ Use abstract base classes to enforce interface contracts

**Example**: A pricer method that returns `torch.zeros()` instead of computing actual prices

### 3. Using pytest.skip()

**Problem**: Hides test failures instead of fixing them

- ❌ `pytest.skip("TODO: fix later")` - technical debt grows unbounded
- ❌ `pytest.skip("flaky test")` - masks real bugs
- ❌ `pytest.skip("CUDA not available")` - missing GPU is a failure, not a skip
- ✅ Fix the test or remove it entirely
- ✅ Use module-level `assert torch.cuda.is_available()` to fail fast

**Impact**: Critical bugs remain hidden, test suite provides false confidence

### 4. Testing Actions Without Validating Results

**Problem**: Running simulations without verifying convergence or correctness

- ❌ Run Monte Carlo simulation, only check that it completes
- ❌ Train model, only verify training loop finishes
- ✅ Validate convergence metrics (loss, variance reduction)
- ✅ Check statistical properties of outputs (mean, std, distribution shape)
- ✅ Compare against analytical solutions when available

**Example**: Test runs 10,000 simulations but never checks if the estimated mean converges to expected value

### 5. Hardcoded Success Tests

**Problem**: Tests that always pass regardless of implementation

- ❌ `assert True` - meaningless validation
- ❌ `assert result or not result` - tautology
- ❌ `assert gradient is not None` - doesn't validate gradient correctness
- ✅ `assert torch.autograd.gradcheck(func, inputs)` - validates gradient computation
- ✅ Use property-based testing with hypothesis for numerical properties

### 6. Overly Permissive Convergence Criteria

**Problem**: Accepting numerical instability or divergence as "success"

- ❌ Not checking for NaN/Inf in outputs
- ❌ Accepting any loss decrease as "converged"
- ❌ `assert results.shape == expected_shape` - shape correct but values wrong
- ✅ `assert torch.isfinite(results).all()` - reject NaN/Inf
- ✅ `assert loss < threshold and variance < max_variance` - validate convergence quality
- ✅ Check condition numbers for numerical stability

**Impact**: Silently propagates numerical instability through pipelines

### 7. Lowered Standards for Flaky Tests

**Problem**: Weakening assertions to make tests pass instead of fixing root cause

- ❌ Changing `rtol=1e-5` to `rtol=1e-1` to make test pass
- ❌ Using `assert len(results) > 0` instead of validating statistical properties
- ❌ Adding broad exception handlers to suppress errors
- ✅ Investigate why tolerance needs to be so loose
- ✅ Use statistical tests (e.g., Kolmogorov-Smirnov) for distribution validation
- ✅ Set random seeds for reproducibility, investigate variance sources

### 8. Test Timeouts for Long Simulations

**Problem**: Artificially limiting execution time instead of optimizing or validating correctly

- ❌ Adding `pytest.timeout(10)` to skip slow tests
- ❌ Reducing simulation count to avoid timeout
- ✅ Optimize the algorithm if it's too slow
- ✅ Use @pytest.mark.slow for long-running tests, run in CI
- ✅ Profile to identify bottlenecks before adding timeouts

**Note**: Some Monte Carlo methods are inherently expensive - validate, don't skip

### 9. Masking Root Causes with Increased Iterations

**Problem**: Increasing iteration limits or simulation counts to hide convergence issues

- ❌ Increasing max_iter from 1000 to 100000 to make test pass
- ❌ Doubling sample size instead of investigating why convergence is slow
- ✅ Investigate why convergence is slow (bad initialization, poor hyperparameters)
- ✅ Fix the algorithm, not the iteration count
- ✅ Set realistic iteration limits and validate convergence criteria

**Example**: GBM trainer takes 10x more iterations than expected - investigate learning rate, loss function, not max_iter

### 10. Trial-and-Error Debugging

**Problem**: Randomly changing code without understanding the root cause

- ❌ Changing learning rates, batch sizes, or model architecture randomly
- ❌ Adding `.detach()`, `.clone()`, or `.cpu()` calls without understanding memory flow

**Systematic debugging process**:
1. **Establish baseline**: Document current behavior (loss curves, gradients, outputs)
2. **Identify root cause**: Use debugger, print intermediate tensors, check gradients
3. **Form hypothesis**: What specific change will fix the issue and why?
4. **Make targeted change**: Implement one fix at a time
5. **Validate fix**: Confirm it resolves the issue without breaking other tests

### 11. Adding Unvalidated Features During Test Fixing

**Problem**: Adding new functionality while debugging test failures

- ❌ Adding new training features while fixing convergence test
- ❌ Refactoring model architecture while debugging gradient test
- ✅ Fix the test first, add features later
- ✅ Keep debugging changes minimal and focused
- ✅ Create separate PR for new features after tests are green

### 12. Analyzing Truncated Test Output

**Problem**: Making decisions based on incomplete test output

- ❌ Terminal truncates 10,000 line stack trace, missing root cause
- ❌ Truncated tensor values hide NaN/Inf in middle of array

**Always redirect to file for analysis**:
```bash
# File: documents/engineering/testing_requirements.md
pytest tests/test_gbm.py -v > test_output.txt 2>&1
# Then read complete output
```

- ✅ Use `torch.set_printoptions(profile="full")` to see all tensor values
- ✅ Save intermediate results to disk for post-mortem analysis

### 13. Disabling Safety Checks for Performance

**Problem**: Removing validation to speed up tests

- ❌ Commenting out `assert torch.isfinite()` checks
- ❌ Disabling gradient checking in tests
- ❌ Skipping convergence validation
- ✅ Keep safety checks in tests, optimize implementation instead
- ✅ Use separate performance benchmarks without safety overhead
- ✅ In production, make safety checks configurable but default to enabled

---

## Blockchain Storage Test Coverage

All storage features have comprehensive test coverage:

- **CLI commands**: 22 tests (83% coverage of `__main__.py`)
  - verify, find-corruption, list-versions, inspect commands
  - gc-preview, gc-run with protected tags
  - tensorboard-log, error handling
- **InferenceClient**: 8 tests (pinned mode, tracking mode, lifecycle)
- **Chain verification**: 15 tests (genesis, merkle chain, corruption detection)
- **Garbage collection**: 15 tests (retention policies, safety checks)
- **TensorBoard**: 12 tests (logging, metadata, error handling)
- **Training integration**: 7 tests (auto_commit, periodic commits, optimizer state preservation)

**Total: 86 storage tests, 73% overall coverage**

Run storage tests:
```bash
# File: documents/engineering/testing_requirements.md
docker compose -f docker/docker-compose.yml exec spectralmc pytest tests/test_storage/ -v
docker compose -f docker/docker-compose.yml exec spectralmc pytest tests/test_integrity/ -v
```

See also: [Blockchain Storage](blockchain_storage.md) for complete storage documentation.

---

## Summary

- **Fully typed**: All tests pass `mypy --strict`
- **Run via Poetry**: Use `poetry run test-all`, never direct `pytest`
- **GPU required**: All tests require GPU - use `assert torch.cuda.is_available()` at module level
- **No fallbacks**: Silent CPU fallbacks are strictly forbidden
- **No skips**: `pytest.skip("CUDA not available")` is forbidden - missing GPU is a failure
- **Deterministic**: Set `torch.manual_seed()` in every test
- **Comprehensive**: Cover happy path, edge cases, errors
- **Async**: Use `pytest-asyncio` for async code
- **Output handling**: Always redirect to files, read complete output
- **Avoid anti-patterns**: See 13 testing anti-patterns above

See also: [Coding Standards](coding_standards.md), [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md), [PyTorch Facade](pytorch_facade.md), [Blockchain Storage](blockchain_storage.md)
