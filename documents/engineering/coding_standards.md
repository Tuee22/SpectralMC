# Coding Standards

## Overview

SpectralMC enforces strict coding standards to ensure reproducibility, correctness, and maintainability. These standards cover **code formatting**, **type safety**, and **custom type stubs** for third-party libraries.

**Related Standards**: [Functional Programming](./functional_programming.md), [Immutability Doctrine](./immutability_doctrine.md), [Pydantic Patterns](./pydantic_patterns.md)

---

## Code Formatting with Black

### Overview

All Python code in SpectralMC must be formatted with **Black 25.1+**. Black is an opinionated code formatter that enforces a consistent style across the entire codebase, eliminating debates about formatting and ensuring readability.

### Running Black

Format all Python code from the project root:

```bash
# Inside Docker container
docker compose -f docker/docker-compose.yml exec spectralmc black .
```

This command will automatically format all Python files in the project, including source code, tests, and scripts.

#### Checking Without Modifying

To check if files would be reformatted without actually changing them:

```bash
docker compose -f docker/docker-compose.yml exec spectralmc black --check .
```

This is useful for CI/CD pipelines to verify that all code is properly formatted before merging.

### Configuration

The Black version and settings are defined in `pyproject.toml`:

```toml
[tool.poetry.group.dev.dependencies]
black = ">=25.1,<26.0"
```

Black uses its default configuration with **no customization**. This is intentional:

- **Line length**: 88 characters (Black's default)
- **String quotes**: Double quotes (Black's default)
- **Trailing commas**: Added where appropriate (Black's default)
- **Import sorting**: Not handled by Black (use isort or similar if needed)

### Key Principles

#### 1. Zero Configuration

Black's defaults are **non-negotiable**. The project does not override Black's built-in settings because:

- **Consistency**: Everyone uses the same formatting, no exceptions
- **Simplicity**: No time wasted debating formatting rules
- **Compatibility**: Works seamlessly across different environments
- **Future-proof**: Updates to Black automatically apply consistent improvements

#### 2. Consistent Formatting

All code must be Black-formatted before committing. This ensures:

- **Uniform style**: All code looks like it was written by the same person
- **Minimal diffs**: Changes focus on logic, not whitespace
- **No bikeshedding**: Formatting is automated, not debated

#### 3. Automated Enforcement

##### Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically format code before each commit:

```bash
# Install pre-commit (one-time setup)
docker compose -f docker/docker-compose.yml exec spectralmc pip install pre-commit

# Install git hooks (one-time setup)
docker compose -f docker/docker-compose.yml exec spectralmc pre-commit install
```

Create `.pre-commit-config.yaml` in the repository root:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 25.1.0
    hooks:
      - id: black
        language_version: python3.12
```

Now Black will automatically format files when you run `git commit`.

##### CI/CD Integration

In continuous integration, verify formatting with:

```bash
docker compose -f docker/docker-compose.yml exec spectralmc black --check .
```

This command exits with a non-zero code if any files need formatting, failing the build.

### Example: Before and After

#### Before Black

```python
def complex_multiply(real_a,imag_a,real_b,imag_b):
    real_result=real_a*real_b-imag_a*imag_b
    imag_result=real_a*imag_b+imag_a*real_b
    return real_result,imag_result
```

#### After Black

```python
def complex_multiply(real_a, imag_a, real_b, imag_b):
    real_result = real_a * real_b - imag_a * imag_b
    imag_result = real_a * imag_b + imag_a * real_b
    return real_result, imag_result
```

Black adds:
- Spaces around operators
- Spaces after commas
- Consistent spacing

### Common Questions

#### Q: Can I disable Black for specific lines?

**A: No.** Black does not support disabling formatting for specific code sections. This is intentional to maintain absolute consistency.

If Black's formatting seems problematic for a specific case, it's usually a sign that the code structure should be refactored for clarity.

#### Q: What about import sorting?

**A: Black does not sort imports.** For import sorting, consider using `isort` with Black-compatible settings:

```toml
[tool.isort]
profile = "black"
```

However, import sorting is **not currently enforced** in SpectralMC. Black handles only whitespace and line breaks.

#### Q: Can I use a different line length?

**A: No.** SpectralMC uses Black's default 88-character line length. This is Black's recommended default and works well for most code.

---

## Strict Type Safety with mypy

### Overview

SpectralMC enforces **strict static typing** with mypy. All code must pass `mypy --strict` with **zero errors** and **zero warnings**. This non-negotiable requirement ensures type correctness, prevents runtime errors, and serves as executable documentation.

### Running mypy

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

### Configuration

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

The `stubs` directory provides project-specific stubs for all third-party dependencies, ensuring complete type coverage. See the [Custom Type Stubs](#custom-type-stubs-pyi-files) section for details.

### Zero Tolerance Policy

The following are **NEVER** allowed in `src/spectralmc/`:

#### ❌ Forbidden: `Any` Type Annotations

```python
# ❌ INCORRECT - Explicit Any
from typing import Any

def process_data(data: Any) -> Any:
    return data

# ❌ INCORRECT - Implicit Any from missing type hints
def process_data(data):  # Implicitly Any
    return data  # Implicitly Any
```

#### ❌ Forbidden: `cast()` Function

```python
# ❌ INCORRECT - Using cast to bypass type checking
from typing import cast

def get_tensor(data: object) -> torch.Tensor:
    return cast(torch.Tensor, data)  # Forbidden!
```

#### ❌ Forbidden: `# type: ignore` Comments

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

### Required Annotations

#### All Functions Must Have Type Hints

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

#### Class Attribute Annotations

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

### Verifying No Forbidden Constructs

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

### Success Criteria

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

### Rationale: Why Strict Typing Matters

Type safety is **non-negotiable** for SpectralMC because:

#### 1. Reproducibility

Type errors can silently break deterministic guarantees:

```python
# Without strict typing, this could silently accept wrong types
def simulate(seed: int, samples: int) -> np.ndarray:
    ...

# Accidental call with float seed (would give different results!)
result = simulate(seed=42.5, samples=1000)  # mypy catches this!
```

#### 2. Correctness

Numerical code with `Any` can produce **wrong results** at runtime:

```python
# Without type hints, numpy array of wrong shape could slip through
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # mypy + stubs ensure shapes are compatible
    ...
```

#### 3. Maintainability

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

#### 4. Refactoring Confidence

Strong types enable **confident large-scale changes**:

- Rename a function → mypy finds all call sites
- Change a return type → mypy finds all affected code
- Refactor data structures → mypy ensures consistency

### Common Patterns

#### Optional Values

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

#### Generic Types

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

#### TypeVar for Generics

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T | None:
    return items[0] if items else None
```

### Troubleshooting

#### "Missing library stubs" Error

If mypy reports missing stubs for a third-party library:

1. Check if stubs exist in `stubs/` directory
2. If not, create them (see [Custom Type Stubs](#custom-type-stubs-pyi-files) section)
3. Never use `# type: ignore` to suppress the error

#### "Incompatible types" Error

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

## Custom Type Stubs (.pyi files)

### Overview

SpectralMC maintains **custom type stubs** for all third-party libraries in the `stubs/` directory. These `.pyi` files provide complete, project-specific type information that may be stricter than upstream library stubs.

### Rationale

Custom type stubs ensure:

1. **Complete type coverage** - All external dependencies have full type information
2. **Project-specific typing** - Stricter than upstream (e.g., no `Any` where we know the actual type)
3. **Reproducible type checking** - Independent of external stub changes or breakage
4. **Controlled upgrades** - We update stubs when ready, not when upstream changes

#### Why Not Use Existing Stubs?

Many libraries provide incomplete or outdated stubs:

- PyTorch's official stubs use `Any` extensively
- NumPy stubs may not cover all array operations
- Third-party stub packages (e.g., `types-*`) can have version mismatches

SpectralMC requires **zero `Any` types**, so we maintain our own stubs.

### Organization

Type stubs are organized in the `stubs/` directory, mirroring the package structure:

```
stubs/
├── torch/
│   ├── __init__.pyi
│   ├── nn/
│   │   ├── __init__.pyi
│   │   ├── modules/
│   │   │   └── module.pyi
│   │   └── functional.pyi
│   ├── cuda/
│   │   └── __init__.pyi
│   └── optim/
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

The `mypy_path` in `pyproject.toml` points to `stubs/`:

```toml
[tool.mypy]
mypy_path = "stubs"
```

### Stub Requirements

All type stubs must adhere to these strict requirements:

#### 1. Type-Pure (Zero Tolerance)

Stubs must have **zero occurrences** of:

- ❌ No `Any` types
- ❌ No `cast()` calls
- ❌ No `# type: ignore` comments

```python
# ❌ INCORRECT - Contains Any
from typing import Any

def forward(input: Any) -> Any: ...

# ✅ CORRECT - Specific types
import torch

def forward(input: torch.Tensor) -> torch.Tensor: ...
```

#### 2. Minimal (Only Used APIs)

Include **only the APIs actually used** by SpectralMC. Don't stub the entire library:

```python
# ✅ CORRECT - Only used classes/functions
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...

# ❌ INCORRECT - Stubbing unused methods
class Linear(Module):
    # ... 50 methods we don't use ...
```

**Rationale**: Smaller stubs are easier to maintain and review. Add APIs incrementally as needed.

#### 3. Documented

All stubs must include docstrings explaining their purpose and scope:

```python
"""
Strict, project-specific stub for the **top-level** :pymod:`torch` namespace.

Only the public surface exercised by SpectralMC is declared. The stub must
remain *type-pure*: **no** ``Any``, **no** ``cast``, **no** ``type: ignore``.

Maintained as of: PyTorch 2.1.2 (CUDA 11.8)
Last updated: 2025-01-09
"""
```

### Example Stub Structure

#### Complete Stub: `stubs/torch/__init__.pyi`

```python
"""
Strict, project‑specific stub for the **top‑level** :pymod:`torch` namespace.

Only the public surface exercised by SpectralMC is declared. The stub must
remain *type‑pure*: **no** ``Any``, **no** ``cast``, **no** ``type: ignore``.
"""

from __future__ import annotations

from typing import Protocol, overload, Sequence

# ============================================================================
# Dtype System
# ============================================================================

class dtype:
    """Represents a PyTorch data type (e.g., float32, complex128)."""
    ...

float32: dtype
float64: dtype
complex64: dtype
complex128: dtype
int32: dtype
int64: dtype
bool: dtype

def get_default_dtype() -> dtype: ...
def set_default_dtype(d: dtype) -> None: ...

# ============================================================================
# Tensor Class
# ============================================================================

class Tensor:
    """PyTorch tensor - only methods used by SpectralMC."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def device(self) -> device: ...

    @property
    def dtype(self) -> dtype: ...

    def detach(self) -> Tensor: ...
    def clone(self) -> Tensor: ...
    def requires_grad_(self, mode: bool = True) -> Tensor: ...

    # Arithmetic
    def __add__(self, other: Tensor | float) -> Tensor: ...
    def __sub__(self, other: Tensor | float) -> Tensor: ...
    def __mul__(self, other: Tensor | float) -> Tensor: ...
    def __truediv__(self, other: Tensor | float) -> Tensor: ...

# ============================================================================
# Device Management
# ============================================================================

class device:
    """Represents a compute device (CPU or CUDA)."""

    def __init__(self, device_str: str) -> None: ...

    @property
    def type(self) -> str: ...

# ============================================================================
# Tensor Creation
# ============================================================================

def tensor(
    data: Sequence[float] | Sequence[Sequence[float]],
    dtype: dtype | None = None,
    device: device | None = None,
) -> Tensor: ...

def zeros(*size: int, dtype: dtype | None = None, device: device | None = None) -> Tensor: ...
def ones(*size: int, dtype: dtype | None = None, device: device | None = None) -> Tensor: ...
def randn(*size: int, dtype: dtype | None = None, device: device | None = None) -> Tensor: ...

def manual_seed(seed: int) -> None: ...
```

**Key features**:
- Docstrings explain scope
- Only used APIs included
- Specific types (no `Any`)
- Overloads where needed

### Adding New Dependencies

When adding a new third-party dependency to SpectralMC:

#### Step 1: Create Stub Directory

```bash
mkdir -p stubs/new_library
touch stubs/new_library/__init__.pyi
```

#### Step 2: Stub Only Used APIs

Identify which functions/classes you're using:

```bash
# Find all imports of the library
grep -r "from new_library import" src/spectralmc/
grep -r "import new_library" src/spectralmc/
```

Stub **only those APIs**, not the entire library.

#### Step 3: Write Type-Pure Stubs

```python
"""
Strict stub for new_library.

Only APIs used by SpectralMC. Type-pure: no Any, no cast, no type: ignore.

Maintained as of: new_library 1.2.3
Last updated: 2025-01-XX
"""

from __future__ import annotations

# Only stub the functions you're actually using
def function_we_use(arg: int) -> str: ...

class ClassWeUse:
    def method_we_use(self, x: float) -> float: ...
```

#### Step 4: Verify with mypy

```bash
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```

If mypy reports missing types, add them to the stub. **Never use `# type: ignore`.**

#### Step 5: Document in Stub

Add docstring explaining:
- Which version of the library this stub covers
- What APIs are included (and excluded)
- When it was last updated

### Maintaining Stubs

#### When to Update

Update stubs when:

1. **Upgrading library version** - Check if API changed
2. **Using new APIs** - Add missing functions/classes
3. **mypy errors** - Fill in gaps in type information

#### How to Update

1. Read library changelog for breaking changes
2. Update stub to match new API
3. Run mypy to verify
4. Update "Last updated" date in docstring

#### Testing Stubs

Stubs are checked by mypy automatically when type-checking `src/` and `tests/`:

```bash
# Stub errors will appear as mypy errors
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```

### Common Patterns

#### Protocol for Duck Typing

Use `Protocol` for structural subtyping:

```python
from typing import Protocol

class SupportsTrain(Protocol):
    """Any object with a .train() method."""
    def train(self, mode: bool = True) -> None: ...
```

#### Overloads for Function Variants

Use `@overload` for functions with multiple signatures:

```python
from typing import overload

@overload
def zeros(size: int, *, dtype: dtype | None = None) -> Tensor: ...

@overload
def zeros(size: tuple[int, ...], *, dtype: dtype | None = None) -> Tensor: ...

def zeros(*args: int | tuple[int, ...], dtype: dtype | None = None) -> Tensor: ...
```

#### TypeVar for Generics

```python
from typing import TypeVar

T = TypeVar("T")

def to_device(data: T, device: device) -> T: ...
```

### Stub Validation Checklist

Before committing new/updated stubs:

- [ ] Docstring explains scope and version
- [ ] Only used APIs included
- [ ] Zero `Any` types
- [ ] Zero `cast()` calls
- [ ] Zero `# type: ignore` comments
- [ ] `mypy` passes with no errors
- [ ] No `grep -r "Any" stubs/new_library/` results

---

## Summary

SpectralMC's coding standards ensure:

- **Consistent formatting** with Black (zero configuration)
- **Strict type safety** with mypy (zero tolerance for `Any`, `cast`, `type: ignore`)
- **Complete type coverage** via custom stubs for all dependencies
- **Reproducible type checking** independent of external changes
- **Executable documentation** through comprehensive type annotations

Before committing code:

```bash
# Inside Docker container
docker compose -f docker/docker-compose.yml exec spectralmc black .
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```

Both must pass with zero errors.

See also:
- [Functional Programming](./functional_programming.md) - ADT patterns, Result types, pattern matching
- [Immutability Doctrine](./immutability_doctrine.md) - Frozen dataclasses and functional updates
- [Pydantic Patterns](./pydantic_patterns.md) - Model validation and configuration
- [Documentation Standards](./documentation_standards.md) - Docstring requirements
