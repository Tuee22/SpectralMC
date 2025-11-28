# Strict Type Safety with mypy

## Overview

SpectralMC enforces **strict static typing** with mypy. All code must pass `mypy --strict` with **zero errors** and **zero warnings**. This non-negotiable requirement ensures type correctness, prevents runtime errors, and serves as executable documentation.

**Related Standards**: [Type Stubs](type_stubs.md), [Code Formatting](code_formatting.md)

---

## Running mypy

Type-check the entire codebase from the project root:

```bash
# Inside Docker container
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```

This command uses the configuration in `pyproject.toml` to check all source code, tests, and examples. **All code must pass with zero errors.**

**CRITICAL**: Always run mypy from the **repository root** with **no path arguments**:

```bash
# ✅ CORRECT: Run from repo root, no path argument
docker compose -f docker/docker-compose.yml exec spectralmc mypy

# ❌ WRONG: Don't specify paths or run from subdirectories
# docker compose -f docker/docker-compose.yml exec spectralmc mypy src/spectralmc
# cd src && mypy spectralmc
```

**Why no path argument?**
- Configuration in `pyproject.toml` controls what gets checked via `files = [...]`
- Currently checks: `src/spectralmc/`, `tests/`, `examples/`
- Running `mypy` with no args uses the config automatically
- Specifying paths bypasses config and can miss folders

---

## Configuration

mypy is configured with **strict mode** in `pyproject.toml`:

```toml
[tool.mypy]
mypy_path = "stubs"
python_version = "3.12"
files = ["src/spectralmc", "tests", "examples"]
exclude = ["src/spectralmc/proto/.*_pb2\\.py$"]
plugins = ["pydantic.mypy"]

# Core strictness (equivalent to --strict --disallow-any-explicit)
strict = true
disallow_any_explicit = true

# Additional rigorous checks beyond --strict
disallow_any_unimported = true       # No Any from missing import stubs
disallow_any_decorated = true        # No Any from untyped decorators
warn_unreachable = true              # Warn about unreachable code
extra_checks = true                  # Enable newer experimental checks
```

The `stubs` directory provides project-specific stubs for all third-party dependencies, ensuring complete type coverage. See [Type Stubs](type_stubs.md) for details.

---

## Zero Tolerance Policy

The following are **NEVER** allowed in `src/spectralmc/`:

### ❌ Forbidden: `Any` Type Annotations

```python
# ❌ INCORRECT - Explicit Any
from typing import Any

def process_data(data: Any) -> Any:
    return data

# ❌ INCORRECT - Implicit Any from missing type hints
def process_data(data):  # Implicitly Any
    return data  # Implicitly Any
```

### ❌ Forbidden: `cast()` Function

```python
# ❌ INCORRECT - Using cast to bypass type checking
from typing import cast

def get_tensor(data: object) -> torch.Tensor:
    return cast(torch.Tensor, data)  # Forbidden!
```

### ❌ Forbidden: `# type: ignore` Comments

```python
# ❌ INCORRECT - Suppressing type errors
result = some_function()  # type: ignore
```

### Why Zero Tolerance?

In scientific computing and quantitative finance:

- **`Any` defeats the purpose** - Type errors can silently propagate, causing wrong results
- **`cast()` is a lie** - Runtime type errors can occur after expensive GPU computations
- **`type: ignore` hides bugs** - Silent failures in numerical code produce incorrect results

SpectralMC's mission is **bit-exact reproducibility**. Type safety is essential for this guarantee.

---

## Required Annotations

### All Functions Must Have Type Hints

Every function and method must have explicit type annotations for **all parameters** and the **return type**:

```python
# ✅ CORRECT - Complete type annotations
def complex_multiply(
    real_a: float, imag_a: float, real_b: float, imag_b: float
) -> tuple[float, float]:
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

All class attributes must be explicitly typed, including those assigned in `__init__`:

```python
# ✅ CORRECT - All attributes typed
class ComplexLinear(nn.Module):
    real_weight: nn.Parameter
    imag_weight: nn.Parameter
    real_bias: nn.Parameter | None
    imag_bias: nn.Parameter | None
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        # ... parameter initialization

# ❌ INCORRECT - Missing attribute annotations
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features  # Type unknown
        self.out_features = out_features  # Type unknown
```

**Note**: Attributes can be annotated at class level (without assignment) OR at assignment site in `__init__`. Both are acceptable:

```python
# Style 1: Class-level annotations
class MyClass:
    value: int

    def __init__(self, value: int) -> None:
        self.value = value

# Style 2: Assignment-site annotations
class MyClass:
    def __init__(self, value: int) -> None:
        self.value: int = value
```

---

## Verifying No Forbidden Constructs

Use grep to verify that forbidden patterns are not present:

```bash
# Check for type: ignore (should find nothing in src/)
docker compose -f docker/docker-compose.yml exec spectralmc \
  grep -r "# type: ignore" src/spectralmc/ && echo "FOUND type:ignore" || echo "OK"

# Check for cast() (should find nothing in src/)
docker compose -f docker/docker-compose.yml exec spectralmc \
  grep -r "cast(" src/spectralmc/ && echo "FOUND cast()" || echo "OK"

# Check for explicit Any (should find nothing in src/)
docker compose -f docker/docker-compose.yml exec spectralmc \
  grep -r ": Any" src/spectralmc/ && echo "FOUND Any" || echo "OK"
```

**Expected output**: All three commands should output "OK", indicating zero occurrences.

---

## Success Criteria

Before committing, verify:

1. **mypy exits with code 0** (zero errors, zero warnings)
2. **No `# type: ignore`** in `src/spectralmc/`
3. **No `cast()`** in `src/spectralmc/`
4. **No explicit `Any`** in `src/spectralmc/`

```bash
# Full validation
docker compose -f docker/docker-compose.yml exec spectralmc mypy && \
  ! grep -r "# type: ignore" src/spectralmc/ && \
  ! grep -r "cast(" src/spectralmc/ && \
  ! grep -r ": Any" src/spectralmc/ && \
  echo "✅ Type safety verified"
```

---

## Rationale: Why Strict Typing Matters

Type safety is **non-negotiable** for SpectralMC because:

### 1. Reproducibility

Type errors can silently break deterministic guarantees:

```python
# Without strict typing, this could silently accept wrong types
def simulate(seed: int, samples: int) -> np.ndarray:
    ...

# Accidental call with float seed (would give different results!)
result = simulate(seed=42.5, samples=1000)  # mypy catches this!
```

### 2. Correctness

Numerical code with `Any` can produce **wrong results** at runtime:

```python
# Without type hints, numpy array of wrong shape could slip through
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # mypy + stubs ensure shapes are compatible
    ...
```

### 3. Maintainability

Complete types serve as **executable documentation**:

```python
# Type signature documents inputs, outputs, and constraints
def gbm_path(
    spot: float,
    rate: float,
    volatility: float,
    maturity: float,
    steps: int,
    seed: int
) -> np.ndarray:
    """Generate GBM path - types document all parameters."""
    ...
```

### 4. Refactoring Confidence

Strong types enable **confident large-scale changes**:

- Rename a function → mypy finds all call sites
- Change a return type → mypy finds all affected code
- Refactor data structures → mypy ensures consistency

---

## Common Patterns

### Optional Values

Use `| None` (Python 3.10+ union syntax):

```python
# ✅ CORRECT - Python 3.10+ syntax
def get_bias(model: nn.Module) -> torch.Tensor | None:
    ...

# ❌ DEPRECATED - Old-style Optional (still works but discouraged)
from typing import Optional
def get_bias(model: nn.Module) -> Optional[torch.Tensor]:
    ...
```

### Generic Types

Use built-in generics (Python 3.9+):

```python
# ✅ CORRECT - Built-in generics
def process_batch(data: list[torch.Tensor]) -> dict[str, float]:
    ...

# ❌ DEPRECATED - typing module generics
from typing import List, Dict
def process_batch(data: List[torch.Tensor]) -> Dict[str, float]:
    ...
```

### TypeVar for Generics

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T | None:
    return items[0] if items else None
```

---

## Troubleshooting

### "Missing library stubs" Error

If mypy reports missing stubs for a third-party library:

1. Check if stubs exist in `stubs/` directory
2. If not, create them (see [Type Stubs](type_stubs.md))
3. Never use `# type: ignore` to suppress the error

### "Incompatible types" Error

If mypy reports incompatible types:

1. **Do not use `cast()`** - Fix the actual type
2. Check if function signature is correct
3. Use type narrowing (isinstance checks) if needed

```python
# ✅ CORRECT - Type narrowing
def process(value: int | str) -> int:
    if isinstance(value, str):
        return int(value)  # mypy knows value is str here
    return value  # mypy knows value is int here

# ❌ INCORRECT - Using cast
from typing import cast
def process(value: int | str) -> int:
    return cast(int, value)  # Bypasses type checking!
```

---

## Summary

- **All code must pass `mypy --strict`** with zero errors
- **Zero tolerance** for `Any`, `cast()`, or `type: ignore`
- **All functions** must have type hints (parameters and return)
- **All class attributes** must be typed
- **Run from repo root** without path arguments
- **Verify forbidden patterns** with grep before committing

See also: [Type Stubs](type_stubs.md), [Pydantic Patterns](pydantic_patterns.md)
