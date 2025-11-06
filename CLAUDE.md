# SpectralMC Development Guide

## Project Overview

SpectralMC is a GPU-accelerated library for online machine learning using Monte Carlo simulation. It trains complex-valued neural networks (CVNNs) using Monte Carlo data, drawing on techniques from Reinforcement Learning, particularly policy gradient methods.

**Primary Use Case**: Quantitative Finance - stochastic process modeling and derivative pricing with significantly reduced computational requirements compared to traditional Monte Carlo methods.

**Key Technology Stack**:
- Python 3.12
- PyTorch 2.7.0 (CUDA 12.8)
- CuPy (CUDA 12.x)
- NumPy, SciPy, scikit-learn
- Distributed: Dask, Ray
- Testing: pytest, hypothesis
- Type checking: mypy (strict mode)

## Architecture

### Core Components

**src/spectralmc/**
- `async_normals.py` - Asynchronous normal distribution generation on GPU
- `cvnn.py` - Complex-valued neural network implementation
- `cvnn_factory.py` - Factory for creating CVNN instances
- `gbm.py` - Geometric Brownian Motion simulation
- `gbm_trainer.py` - Training loop for GBM models
- `sobol_sampler.py` - Quasi-Monte Carlo sampling using Sobol sequences
- `quantlib.py` - QuantLib integration utilities
- `models/` - Model implementations
  - `torch.py` - PyTorch-based model definitions
  - `numerical.py` - Numerical model utilities
  - `cpu_gpu_transfer.py` - CPU/GPU memory transfer utilities

### Workflow

1. **Monte Carlo Simulation**: Generate finite samples from parametric distributions directly on GPU
2. **Fourier Transform**: Use FFT to estimate the sample's characteristic function
3. **CVNN Training**: Update complex-valued neural network parameters to approximate the characteristic function
4. **CVNN Inference**: Produce estimated distributions for computing means, moments, quantiles, and other metrics

## Testing

### Running Tests

SpectralMC uses pytest with GPU and CPU test separation:

```bash
# Run all tests (CPU + GPU)
poetry run test-all

# Run default tests (CPU only, excludes @pytest.mark.gpu)
pytest tests

# Run only GPU tests
pytest tests -m gpu

# Run with coverage
pytest tests --cov=spectralmc --cov-report=term-missing
```

### Test Configuration

- **Test directory**: `tests/`
- **Default behavior**: Excludes GPU tests (`-m 'not gpu'`)
- **GPU tests**: Marked with `@pytest.mark.gpu` decorator
- **Fixtures**: Global GPU memory cleanup in `tests/conftest.py`

### Test Files

- `test_async_normals.py` - Async normal distribution generation
- `test_cvnn.py` - Complex-valued neural network tests
- `test_cvnn_factory.py` - CVNN factory tests
- `test_gbm.py` - Geometric Brownian Motion tests
- `test_gbm_trainer.py` - GBM trainer tests
- `test_models_cpu_gpu_transfer.py` - CPU/GPU transfer tests
- `test_models_torch.py` - PyTorch model tests
- `test_sobol_sampler.py` - Sobol sampler tests

## 🚫 Testing Anti-Patterns (1-13)

### 1. Tests Pass When Features Are Broken
- **Problem**: Test validates that code runs, not that it produces correct results
- ❌ `assert result is not None` - accepts any output
- ❌ `assert len(simulations) > 0` - passes even with wrong values
- ✅ `assert torch.allclose(result, expected, rtol=1e-5)` - validates numerical accuracy
- ✅ `assert torch.isfinite(result).all()` - ensures no NaN/Inf values
- **Impact**: Broken numerical computations go undetected, corrupting downstream calculations

### 2. Accepting NotImplementedError as Success
- **Problem**: Treating placeholder implementations as working features
- ❌ Accepting `NotImplementedError` or returning `None` in convergence checks
- ❌ Methods that return empty tensors when they should compute values
- ✅ All methods must have complete implementations before merging
- ✅ Use abstract base classes to enforce interface contracts
- **Example**: A pricer method that returns `torch.zeros()` instead of computing actual prices

### 3. Using pytest.skip()
- **Problem**: Hides test failures instead of fixing them
- ❌ `pytest.skip("TODO: fix later")` - technical debt grows unbounded
- ❌ `pytest.skip("flaky test")` - masks real bugs
- ✅ Fix the test or remove it entirely
- ✅ Use `@pytest.mark.gpu` for hardware requirements, not skip
- **Impact**: Critical bugs remain hidden, test suite provides false confidence

### 4. Testing Actions Without Validating Results
- **Problem**: Running simulations without verifying convergence or correctness
- ❌ Run Monte Carlo simulation, only check that it completes
- ❌ Train model, only verify training loop finishes
- ✅ Validate convergence metrics (loss, variance reduction)
- ✅ Check statistical properties of outputs (mean, std, distribution shape)
- ✅ Compare against analytical solutions when available
- **Example**: Test runs 10,000 simulations but never checks if the estimated mean converges to expected value

### 5. Hardcoded Success Tests
- **Problem**: Tests that always pass regardless of implementation
- ❌ `assert True` - meaningless validation
- ❌ `assert result or not result` - tautology
- ❌ `assert gradient is not None` - doesn't validate gradient correctness
- ✅ `assert torch.autograd.gradcheck(func, inputs)` - validates gradient computation
- ✅ Use property-based testing with hypothesis for numerical properties

### 6. Overly Permissive Convergence Criteria
- **Problem**: Accepting numerical instability or divergence as "success"
- ❌ Not checking for NaN/Inf in outputs
- ❌ Accepting any loss decrease as "converged"
- ❌ `assert results.shape == expected_shape` - shape correct but values wrong
- ✅ `assert torch.isfinite(results).all()` - reject NaN/Inf
- ✅ `assert loss < threshold and variance < max_variance` - validate convergence quality
- ✅ Check condition numbers for numerical stability
- **Impact**: Silently propagates numerical instability through pipelines

### 7. Lowered Standards for Flaky Tests
- **Problem**: Weakening assertions to make tests pass instead of fixing root cause
- ❌ Changing `rtol=1e-5` to `rtol=1e-1` to make test pass
- ❌ Using `assert len(results) > 0` instead of validating statistical properties
- ❌ Adding broad exception handlers to suppress errors
- ✅ Investigate why tolerance needs to be so loose
- ✅ Use statistical tests (e.g., Kolmogorov-Smirnov) for distribution validation
- ✅ Set random seeds for reproducibility, investigate variance sources

### 8. Test Timeouts for Long Simulations
- **Problem**: Artificially limiting execution time instead of optimizing or validating correctly
- ❌ Adding `pytest.timeout(10)` to skip slow tests
- ❌ Reducing simulation count to avoid timeout
- ✅ Optimize the algorithm if it's too slow
- ✅ Use @pytest.mark.slow for long-running tests, run in CI
- ✅ Profile to identify bottlenecks before adding timeouts
- **Note**: Some Monte Carlo methods are inherently expensive - validate, don't skip

### 9. Masking Root Causes with Increased Iterations
- **Problem**: Increasing iteration limits or simulation counts to hide convergence issues
- ❌ Increasing max_iter from 1000 to 100000 to make test pass
- ❌ Doubling sample size instead of investigating why convergence is slow
- ✅ Investigate why convergence is slow (bad initialization, poor hyperparameters)
- ✅ Fix the algorithm, not the iteration count
- ✅ Set realistic iteration limits and validate convergence criteria
- **Example**: GBM trainer takes 10x more iterations than expected - investigate learning rate, loss function, not max_iter

### 10. Trial-and-Error Debugging
- **Problem**: Randomly changing code without understanding the root cause
- ❌ Changing learning rates, batch sizes, or model architecture randomly
- ❌ Adding `.detach()`, `.clone()`, or `.cpu()` calls without understanding memory flow
- ✅ **Systematic debugging process**:
  1. **Establish baseline**: Document current behavior (loss curves, gradients, outputs)
  2. **Identify root cause**: Use debugger, print intermediate tensors, check gradients
  3. **Form hypothesis**: What specific change will fix the issue and why?
  4. **Make targeted change**: Implement one fix at a time
  5. **Validate fix**: Confirm it resolves the issue without breaking other tests

### 11. Adding Unvalidated Features During Test Fixing
- **Problem**: Adding new functionality while debugging test failures
- ❌ Adding new training features while fixing convergence test
- ❌ Refactoring model architecture while debugging gradient test
- ✅ Fix the test first, add features later
- ✅ Keep debugging changes minimal and focused
- ✅ Create separate PR for new features after tests are green

### 12. Analyzing Truncated Test Output
- **Problem**: Making decisions based on incomplete test output
- ❌ Terminal truncates 10,000 line stack trace, missing root cause
- ❌ Truncated tensor values hide NaN/Inf in middle of array
- ✅ **Always redirect to file for analysis**:
  ```bash
  pytest tests/test_gbm.py -v > test_output.txt 2>&1
  # Then read complete output
  ```
- ✅ Use `torch.set_printoptions(profile="full")` to see all tensor values
- ✅ Save intermediate results to disk for post-mortem analysis

### 13. Disabling Safety Checks for Performance
- **Problem**: Removing validation to speed up tests
- ❌ Commenting out `assert torch.isfinite()` checks
- ❌ Disabling gradient checking in tests
- ❌ Skipping convergence validation
- ✅ Keep safety checks in tests, optimize implementation instead
- ✅ Use separate performance benchmarks without safety overhead
- ✅ In production, make safety checks configurable but default to enabled

## 🚫 Implementation Anti-Patterns (1-5)

### 1. Silent Failure Handling
- **Problem**: Catching exceptions without proper handling or logging
- ❌ `try: result = simulate() except: return default_value` - hides errors
- ❌ `if torch.isnan(loss).any(): loss = torch.tensor(0.0)` - masks numerical issues
- ❌ Broad exception handlers: `except Exception: pass`
- ✅ Let exceptions propagate unless you can meaningfully handle them
- ✅ Log errors with context before re-raising
- ✅ Use specific exception types: `except ValueError as e:`
- **Impact**: Silent failures in numerical code lead to incorrect results downstream

**Example**:
```python
# ❌ Silent failure
try:
    result = monte_carlo_simulation(params)
except Exception:
    result = torch.zeros(batch_size)  # Wrong! Hides the error

# ✅ Proper handling
try:
    result = monte_carlo_simulation(params)
except NumericalInstabilityError as e:
    logger.error(f"Simulation failed: {e}, params: {params}")
    raise  # Re-raise to fail fast
```

### 2. False Success Patterns
- **Problem**: Tests or implementations that report success without validation
- ❌ Training loop returns success even when loss diverged
- ❌ `status = "converged"` without checking convergence criteria
- ❌ Function returns successfully with NaN/Inf values
- ✅ Always validate outputs before returning success status
- ✅ Use type hints and runtime validation (pydantic)
- ✅ Raise exceptions for invalid states rather than returning error codes

**Example**:
```python
# ❌ False success
def train_model(model, data):
    for epoch in range(100):
        loss = train_step(model, data)
    return {"status": "success", "loss": loss}  # Could be NaN!

# ✅ Validated success
def train_model(model, data):
    for epoch in range(100):
        loss = train_step(model, data)
        if not torch.isfinite(loss):
            raise TrainingDivergenceError(f"Loss became {loss} at epoch {epoch}")
    return {"status": "converged", "final_loss": float(loss)}
```

### 3. Ignoring Numerical Warnings
- **Problem**: Treating warnings as noise instead of signals
- ❌ Suppressing "divide by zero" warnings
- ❌ Ignoring "invalid value encountered" from NumPy/PyTorch
- ❌ Filtering out all warnings with `warnings.filterwarnings("ignore")`
- ✅ Investigate and fix root cause of warnings
- ✅ Only filter specific expected warnings (e.g., QuantLib deprecation warnings)
- ✅ Convert warnings to errors during testing: `warnings.simplefilter("error")`

### 4. Mutable Default Arguments
- **Problem**: Using mutable objects as default arguments
- ❌ `def simulate(config={}):` - shared across calls
- ❌ `def run_batch(params=[]):` - accumulates across calls
- ✅ `def simulate(config=None): config = config or {}`
- ✅ Use immutable defaults or None
- **Impact**: Especially dangerous in parallel/distributed computing with Ray/Dask

### 5. Inconsistent Device Handling
- **Problem**: Not managing CPU/GPU device placement consistently
- ❌ Assuming tensors are on CUDA without checking
- ❌ Moving tensors between devices unnecessarily
- ❌ Not handling device in function signatures
- ✅ Explicit device management: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- ✅ Keep tensors on same device throughout computation
- ✅ Use `models/cpu_gpu_transfer.py` utilities for controlled transfers
- **Example**: Model on GPU, input data on CPU - causes cryptic errors

## Recovery Checklist

When you encounter test failures or bugs:

1. **Read the complete error output** (redirect to file if truncated)
2. **Reproduce the failure** reliably with minimal example
3. **Check for numerical issues** (NaN, Inf, loss divergence)
4. **Validate inputs and outputs** at each stage
5. **Use debugger** to inspect tensor values, gradients, device placement
6. **Check git diff** - what changed since tests last passed?
7. **Run single test in isolation** - rule out test interaction
8. **Verify random seeds** - ensure reproducibility
9. **Check GPU memory** - OOM can cause silent failures
10. **Profile if slow** - don't guess at bottlenecks

## Prevention: Pre-commit Checklist

Before committing:

- [ ] All tests pass (CPU and GPU)
- [ ] mypy type checking passes (strict mode)
- [ ] No NaN/Inf in test outputs
- [ ] Convergence validated, not just "runs without error"
- [ ] No pytest.skip() added
- [ ] No overly broad exception handlers added
- [ ] No hardcoded magic numbers without comments
- [ ] Type hints on all functions
- [ ] Docstrings on public functions
- [ ] No TODO comments without GitHub issues

## 🔒 Git Workflow Policy

**Critical Rule**: Claude Code is NOT authorized to commit or push changes.

### Forbidden Git Operations
- ❌ **NEVER** run `git commit` (including `--amend`, `--no-verify`, etc.)
- ❌ **NEVER** run `git push` (including `--force`, `--force-with-lease`, etc.)
- ❌ **NEVER** run `git add` followed by commit operations
- ❌ **NEVER** create commits under any circumstances

### Required Workflow
- ✅ Make all code changes as requested
- ✅ Run tests and validation (`poetry run test-all`, `mypy`)
- ✅ Leave ALL changes as **uncommitted** working directory changes
- ✅ User reviews changes using `git status` and `git diff`
- ✅ User manually commits and pushes when satisfied

**Rationale**: All changes must be human-reviewed before entering version control. This ensures code quality, prevents automated commit mistakes, and maintains clear authorship.

## Style Guide

See `STYLE_GUIDE.md` for code style conventions.

## Contact

Author: matt@resolvefintech.com
