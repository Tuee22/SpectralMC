# File: documents/engineering/coding_standards.md
# Coding Standards

**Status**: Authoritative source  
**Supersedes**: Prior coding standards drafts  
**Referenced by**: documents/documentation_standards.md

> **Purpose**: SSoT for SpectralMC coding standards covering formatting, typing, and stubs.

## Cross-References
- [Purity Doctrine](purity_doctrine.md)
- [Immutability Doctrine](immutability_doctrine.md)
- [Pydantic Patterns](pydantic_patterns.md)
- [Testing Requirements](testing_requirements.md)
- [Documentation Standards](../documentation_standards.md)

## Overview

SpectralMC enforces strict coding standards to ensure reproducibility, correctness, and maintainability. These standards cover **code formatting**, **type safety**, and **custom type stubs** for third-party libraries.

---

## Code Formatting with Black

### Overview

All Python code in SpectralMC must be formatted with **Black 25.1+**. Black is an opinionated code formatter that enforces a consistent style across the entire codebase, eliminating debates about formatting and ensuring readability.

### Running Black

Format all Python code from the project root:

```bash
# File: documents/engineering/coding_standards.md
# Inside Docker container
docker compose -f docker/docker-compose.yml exec spectralmc black .
```

This command will automatically format all Python files in the project, including source code, tests, and scripts.

#### Checking Without Modifying

To check if files would be reformatted without actually changing them:

```bash
# File: documents/engineering/coding_standards.md
docker compose -f docker/docker-compose.yml exec spectralmc black --check .
```

This is useful for CI/CD pipelines to verify that all code is properly formatted before merging.

### Configuration

The Black version and settings are defined in `pyproject.toml`:

```toml
# File: documents/engineering/coding_standards.md
[tool.poetry.group.dev.dependencies]
black = ">=25.1,<26.0"
```

Black is configured in `pyproject.toml` with project-specific settings:

```toml
# File: documents/engineering/coding_standards.md
[tool.black]
line-length = 100
target-version = ['py312']
```

- **Line length**: 100 characters (project standard for readability)
- **Target version**: Python 3.12
- **String quotes**: Double quotes (Black's default)
- **Trailing commas**: Added where appropriate (Black's default)
- **Import sorting**: Not handled by Black (use isort or similar if needed)

### Key Principles

#### 1. Minimal Configuration

Black's configuration is minimal and project-wide. The only customization is:

- **Line length**: 100 characters (vs Black's default 88) for better readability with complex type hints
- **Target version**: Python 3.12 for latest syntax support

All other settings use Black defaults because:

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
# File: documents/engineering/coding_standards.md
# Install pre-commit (one-time setup)
docker compose -f docker/docker-compose.yml exec spectralmc pip install pre-commit

# Install git hooks (one-time setup)
docker compose -f docker/docker-compose.yml exec spectralmc pre-commit install
```

Create `.pre-commit-config.yaml` in the repository root:

```yaml
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
docker compose -f docker/docker-compose.yml exec spectralmc black --check .
```

This command exits with a non-zero code if any files need formatting, failing the build.

### Example: Before and After

#### Before Black

```python
# File: documents/engineering/coding_standards.md
def complex_multiply(real_a,imag_a,real_b,imag_b):
    real_result=real_a*real_b-imag_a*imag_b
    imag_result=real_a*imag_b+imag_a*real_b
    return real_result,imag_result
```

#### After Black

```python
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
[tool.isort]
profile = "black"
```

However, import sorting is **not currently enforced** in SpectralMC. Black handles only whitespace and line breaks.

#### Q: Can I use a different line length?

**A: No.** SpectralMC uses a 100-character line length (set in `[tool.black]`). This is the project standard; do not override it.

---

## Import Discipline

### Overview

SpectralMC enforces strict import discipline to ensure predictable module loading, fail-fast behavior, and simplified static analysis. All imports must be unconditional and at module scope.

### Zero Tolerance Policy

The following are **NEVER** allowed in `src/spectralmc/`:

#### ❌ Forbidden: Function-level Imports

```python
# File: documents/engineering/coding_standards.md
# ❌ INCORRECT - Import inside function
def process_data(path: str) -> bytes:
    import json  # Forbidden!
    return json.loads(path)

# ✅ CORRECT - Import at module top
import json

def process_data(path: str) -> bytes:
    return json.loads(path)
```

#### ❌ Forbidden: Conditional Imports

```python
# File: documents/engineering/coding_standards.md
# ❌ INCORRECT - Import in try/except
try:
    import optional_library
except ImportError:
    optional_library = None  # Forbidden!

# ❌ INCORRECT - Import in if block
if some_condition:
    from module import feature  # Forbidden!

# ✅ CORRECT - Unconditional import (fails if missing)
import optional_library  # Will crash if not installed - this is correct behavior
```

#### ❌ Forbidden: TYPE_CHECKING Guards

```python
# File: documents/engineering/coding_standards.md
# ❌ INCORRECT - TYPE_CHECKING conditional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from expensive_module import HeavyClass  # Forbidden!

# ✅ CORRECT - Unconditional import
from expensive_module import HeavyClass
```

### Rationale

#### 1. Fail-Fast Behavior

Missing dependencies must cause immediate, obvious failures:

- **Startup crashes are good** - They surface configuration issues immediately
- **Silent degradation is bad** - Optional features that silently disable are hard to debug
- **Import errors have clear stack traces** - Runtime errors from missing imports are cryptic

#### 2. Predictable Module Loading

Conditional imports make behavior unpredictable:

- Code paths change based on import success/failure
- Testing becomes unreliable (tests may pass with different imports than production)
- Static analysis tools cannot determine actual dependencies

#### 3. Simplified Static Analysis

Unconditional top-level imports enable:

- Complete dependency graphs from static analysis
- Reliable dead code detection
- Accurate IDE autocompletion and refactoring

### Why No TYPE_CHECKING?

The `TYPE_CHECKING` constant from `typing` is specifically forbidden because:

1. **Creates two code paths** - Type checkers see different code than runtime
2. **Hides import errors** - Import failures only surface during type checking
3. **Encourages lazy typing** - Developers use it to avoid fixing circular imports properly
4. **Complicates debugging** - Runtime behavior differs from what type checkers analyze

**Alternative for circular imports**: Restructure modules to eliminate cycles, or use string literal forward references (`"ClassName"` instead of `ClassName`).

### Verification

```bash
# File: documents/engineering/coding_standards.md
# Check for TYPE_CHECKING usage (should find nothing in src/)
docker compose -f docker/docker-compose.yml exec spectralmc \
  grep -r "TYPE_CHECKING" src/spectralmc/ && echo "FOUND TYPE_CHECKING" || echo "OK"

# Check for function-level imports (manual review required)
docker compose -f docker/docker-compose.yml exec spectralmc \
  grep -rn "^[[:space:]]*import \|^[[:space:]]*from .* import" src/spectralmc/ | \
  grep -v "^[^:]*:[0-9]*:import\|^[^:]*:[0-9]*:from"
```

### Success Criteria

Before committing:

1. **Zero `TYPE_CHECKING`** imports in `src/spectralmc/`
2. **Zero function-level imports** in `src/spectralmc/`
3. **Zero conditional imports** (try/except, if/else around imports)
4. **All imports at module top** (after `__future__` imports and docstrings)

---

## Strict Type Safety with mypy

### Overview

SpectralMC enforces **strict static typing** with mypy. All code must pass `mypy --strict` with **zero errors** and **zero warnings**. This non-negotiable requirement ensures type correctness, prevents runtime errors, and serves as executable documentation.

### Running mypy

Type-check the entire codebase from the project root:

```bash
# File: documents/engineering/coding_standards.md
# Inside Docker container
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```

This command uses the configuration in `pyproject.toml` to check all source code, tests, and tools. **All code must pass with zero errors.**

**CRITICAL**: Always run mypy from the **repository root** with **no path arguments**:

```bash
# File: documents/engineering/coding_standards.md
# ✅ CORRECT: Run from repo root, no path argument
docker compose -f docker/docker-compose.yml exec spectralmc mypy

# ❌ WRONG: Don't specify paths or run from subdirectories
# docker compose -f docker/docker-compose.yml exec spectralmc mypy src/spectralmc
# cd src && mypy spectralmc
```

**Why no path argument?**
- Configuration in `pyproject.toml` controls what gets checked via `files = [...]`
- Currently checks: `src/spectralmc/`, `tests/`, `tools/`
- Running `mypy` with no args uses the config automatically
- Specifying paths bypasses config and can miss folders

### Configuration

mypy is configured with **strict mode** in `pyproject.toml`:

```toml
# File: documents/engineering/coding_standards.md
[tool.mypy]
mypy_path = "stubs"
python_version = "3.12"
files = ["src/spectralmc", "tests", "tools"]
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

- Explicit or implicit untyped parameters (no `Any`).
- Runtime casts to paper over type errors.
- `# type: ignore` or similar suppressions.

#### ✅ Correctly Typed Example

```python
# File: documents/engineering/coding_standards.md
def process_data(data: torch.Tensor) -> torch.Tensor:
    return data.relu()
```

#### ❌ Incorrect: Missing Types

```python
# File: documents/engineering/coding_standards.md
def process_data(data):
    return data  # Implicitly untyped
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
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
# Without strict typing, this could silently accept wrong types
def simulate(seed: int, samples: int) -> np.ndarray:
    ...

# Accidental call with float seed (would give different results!)
result = simulate(seed=42.5, samples=1000)  # mypy catches this!
```

#### 2. Correctness

Numerical code with `Any` can produce **wrong results** at runtime:

```python
# File: documents/engineering/coding_standards.md
# Without type hints, numpy array of wrong shape could slip through
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # mypy + stubs ensure shapes are compatible
    ...
```

#### 3. Maintainability

Complete types serve as **executable documentation**:

```python
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
# ✅ CORRECT - Type narrowing via conditional expression (pure)
def process(value: int | str) -> int:
    return int(value) if isinstance(value, str) else value

# ✅ CORRECT - Type narrowing via match/case (pure)
def process_match(value: int | str) -> int:
    match value:
        case str():
            return int(value)
        case int():
            return value

# ❌ INCORRECT - Using cast
from typing import cast
def process(value: int | str) -> int:
    return cast(int, value)  # Bypasses type checking!
```

See [Purity Doctrine](./purity_doctrine.md) for the requirement to use expressions (`x if cond else y`) or `match`/`case` instead of `if` statements.

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

```text
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
# ❌ INCORRECT - Untyped stub signature
def forward(input): ...

# ✅ CORRECT - Specific types
import torch

def forward(input: torch.Tensor) -> torch.Tensor: ...
```

#### 2. Minimal (Only Used APIs)

Include **only the APIs actually used** by SpectralMC. Don't stub the entire library:

```python
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
"""
Strict, project-specific stub for the **top-level** :pymod:`torch` namespace.

Only the public surface exercised by SpectralMC is declared. The stub must
remain *type-pure*: **no** ``Any``, **no** ``cast``, **no** ``type: ignore``.

Maintained for PyTorch 2.1.2 (CUDA 11.8)
"""
```

### Example Stub Structure

#### Complete Stub: `stubs/torch/__init__.pyi`

```python
# File: documents/engineering/coding_standards.md
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
# File: documents/engineering/coding_standards.md
mkdir -p stubs/new_library
touch stubs/new_library/__init__.pyi
```

#### Step 2: Stub Only Used APIs

Identify which functions/classes you're using:

```bash
# File: documents/engineering/coding_standards.md
# Find all imports of the library
grep -r "from new_library import" src/spectralmc/
grep -r "import new_library" src/spectralmc/
```

Stub **only those APIs**, not the entire library.

#### Step 3: Write Type-Pure Stubs

```python
# File: documents/engineering/coding_standards.md
"""
Strict stub for new_library.

Only APIs used by SpectralMC. Type-pure: no Any, no cast, no type: ignore.

Maintained for new_library 1.2.3
"""

from __future__ import annotations

# Only stub the functions you're actually using
def function_we_use(arg: int) -> str: ...

class ClassWeUse:
    def method_we_use(self, x: float) -> float: ...
```

#### Step 4: Verify with mypy

```bash
# File: documents/engineering/coding_standards.md
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
4. Refresh the stub docstring metadata to reflect the supported library version

#### Testing Stubs

Stubs are checked by mypy automatically when type-checking `src/` and `tests/`:

```bash
# File: documents/engineering/coding_standards.md
# Stub errors will appear as mypy errors
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```

### Common Patterns

#### Protocol for Duck Typing

Use `Protocol` for structural subtyping:

```python
# File: documents/engineering/coding_standards.md
from typing import Protocol

class SupportsTrain(Protocol):
    """Any object with a .train() method."""
    def train(self, mode: bool = True) -> None: ...
```

#### Overloads for Function Variants

Use `@overload` for functions with multiple signatures:

```python
# File: documents/engineering/coding_standards.md
from typing import overload

@overload
def zeros(size: int, *, dtype: dtype | None = None) -> Tensor: ...

@overload
def zeros(size: tuple[int, ...], *, dtype: dtype | None = None) -> Tensor: ...

def zeros(*args: int | tuple[int, ...], dtype: dtype | None = None) -> Tensor: ...
```

#### TypeVar for Generics

```python
# File: documents/engineering/coding_standards.md
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

## Functional Error Handling

SpectralMC uses **Result types and pattern matching** for all expected errors. Exceptions are reserved for programming errors and unrecoverable failures. This section defines the Result type and ADT patterns; see [Purity Doctrine](./purity_doctrine.md) for the complete purity standards including expression-oriented code.

### Core Principles

1. **No Exception Swallowing**: Never catch and ignore exceptions
2. **Result Types for Expected Errors**: Return `Result[T, E]` instead of raising exceptions
3. **Exhaustive Pattern Matching**: Use `match/case` with `assert_never()` for compile-time safety
4. **Explicit Error Context**: Preserve full error information in ADT variants
5. **Immutability**: All error types are immutable (see [Immutability Doctrine](./immutability_doctrine.md))
6. **Factory Functions**: Constructor validation via factory functions returning Result (see [Purity Doctrine](./purity_doctrine.md))

---

## Algebraic Data Types (ADTs)

### What Are ADTs?

ADTs represent **sum types** - types that can be one of several variants. In SpectralMC, we use frozen dataclasses to model error types, operation results, and domain states.

**Key Properties**:
- `@dataclass(frozen=True)` - Immutable by default
- Explicit variants with distinct fields
- Type-safe pattern matching with mypy
- Self-documenting error taxonomy

### Error ADT Pattern

**Primary Example: S3 Operations**

```python
# File: documents/engineering/coding_standards.md
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class S3BucketNotFound:
    """S3 bucket does not exist."""
    kind: Literal["S3BucketNotFound"] = "S3BucketNotFound"
    bucket: str
    original_error: Exception

@dataclass(frozen=True)
class S3ObjectNotFound:
    """S3 object does not exist at the given key."""
    kind: Literal["S3ObjectNotFound"] = "S3ObjectNotFound"
    bucket: str
    key: str
    original_error: Exception

@dataclass(frozen=True)
class S3AccessDenied:
    """Insufficient permissions to access S3 resource."""
    kind: Literal["S3AccessDenied"] = "S3AccessDenied"
    bucket: str
    key: str | None
    original_error: Exception

@dataclass(frozen=True)
class S3NetworkError:
    """Network-level error (timeout, connection failed)."""
    kind: Literal["S3NetworkError"] = "S3NetworkError"
    bucket: str
    key: str | None
    original_error: Exception

@dataclass(frozen=True)
class S3UnknownError:
    """Unexpected S3 error not covered by other variants."""
    kind: Literal["S3UnknownError"] = "S3UnknownError"
    bucket: str
    key: str | None
    original_error: Exception

# Sum type: S3OperationError is ONE OF these variants
S3OperationError = (
    S3BucketNotFound
    | S3ObjectNotFound
    | S3AccessDenied
    | S3NetworkError
    | S3UnknownError
)
```

**Why This Pattern?**

1. **Exhaustive Error Handling**: Pattern matching forces you to handle ALL error cases
2. **Type Safety**: mypy verifies you handle every variant
3. **Self-Documenting**: Error taxonomy is explicit in the type definition
4. **No Silent Failures**: Cannot ignore errors without explicit handling
5. **Preserves Context**: Each variant captures relevant diagnostic information

### Domain State ADTs

Beyond errors, use ADTs to model domain states:

```python
# File: documents/engineering/coding_standards.md
@dataclass(frozen=True)
class TrainingInProgress:
    kind: Literal["TrainingInProgress"] = "TrainingInProgress"
    current_epoch: int
    current_loss: float

@dataclass(frozen=True)
class TrainingConverged:
    kind: Literal["TrainingConverged"] = "TrainingConverged"
    final_epoch: int
    final_loss: float

@dataclass(frozen=True)
class TrainingDiverged:
    kind: Literal["TrainingDiverged"] = "TrainingDiverged"
    diverged_at_epoch: int
    loss_value: float

TrainingState = TrainingInProgress | TrainingConverged | TrainingDiverged
```

---

## Result Types

### The Result[T, E] Pattern

The `Result[T, E]` type represents operations that can succeed with value `T` or fail with error `E`.

**Definition**:

```python
# File: documents/engineering/coding_standards.md
from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar("T")
E = TypeVar("E")

@dataclass(frozen=True)
class Success(Generic[T]):
    """Operation succeeded with value."""
    value: T

@dataclass(frozen=True)
class Failure(Generic[E]):
    """Operation failed with error."""
    error: E

# Result is either Success OR Failure
type Result[T, E] = Success[T] | Failure[E]
```

### When to Use Result Types

**Use Result[T, E] for**:
- ✅ Expected errors (file not found, network timeout, validation failure)
- ✅ Operations that can fail for known reasons
- ✅ External I/O (S3, database, network)
- ✅ User input validation

**Use exceptions for**:
- ✅ Programming errors (assertion failures, type errors)
- ✅ Unrecoverable errors (out of memory, system errors)
- ✅ Internal invariant violations

### Result Type Examples

**S3 Get Object**:

```python
# File: documents/engineering/coding_standards.md
async def get_object_safe(
    client: S3Client, bucket: str, key: str
) -> Result[bytes, S3OperationError]:
    """Fetch S3 object, returning Result instead of raising exceptions."""
    try:
        response = await client.get_object(Bucket=bucket, Key=key)
        body = await response["Body"].read()
        return Success(body)
    except ClientError as e:
        # Pattern match on error code (pure within exception handler)
        error_code = e.response["Error"]["Code"]
        match error_code:
            case "NoSuchBucket":
                return Failure(S3BucketNotFound(bucket=bucket, original_error=e))
            case "NoSuchKey":
                return Failure(S3ObjectNotFound(bucket=bucket, key=key, original_error=e))
            case "AccessDenied":
                return Failure(S3AccessDenied(bucket=bucket, key=key, original_error=e))
            case _:
                return Failure(S3UnknownError(bucket=bucket, key=key, original_error=e))
    except (BotocoreError, aiohttp.ClientError) as e:
        return Failure(S3NetworkError(bucket=bucket, key=key, original_error=e))
```

**Model Loading**:

```python
# File: documents/engineering/coding_standards.md
async def load_model_version(
    store: AsyncBlockchainModelStore, version_counter: int
) -> Result[ModelSnapshot, LoadError]:
    """Load model version, returning Result for expected errors."""
    # Get version metadata
    versions_result = await store.list_all_versions()
    match versions_result:
        case Failure(error):
            return Failure(LoadError.from_s3_error(error))
        case Success(versions):
            pass

    # Find requested version (using next with generator expression - pure)
    target_version = next(
        (v for v in versions if v.counter == version_counter), None
    )
    # Early return via conditional - pattern match would work too
    match target_version:
        case None:
            return Failure(
                VersionNotFound(counter=version_counter, available=versions)
            )
        case version:
            target_version = version

    # Load checkpoint
    checkpoint_result = await store.get_checkpoint(target_version)
    match checkpoint_result:
        case Failure(error):
            return Failure(LoadError.from_s3_error(error))
        case Success(checkpoint_bytes):
            snapshot = deserialize_checkpoint(checkpoint_bytes)
            return Success(snapshot)
```

---

## Pattern Matching

### Exhaustive Error Handling

Python 3.10+ `match/case` enables type-safe, exhaustive error handling.

**Basic Pattern**:

```python
# File: documents/engineering/coding_standards.md
from typing import assert_never

result = await get_object_safe(client, bucket, key)

match result:
    case Success(data):
        # Handle success case
        process_data(data)
    case Failure(error):
        # Pattern match on error variants
        match error:
            case S3BucketNotFound(bucket=bucket_name):
                logger.error(f"Bucket does not exist: {bucket_name}")
                # Handle missing bucket
            case S3ObjectNotFound(bucket=b, key=k):
                logger.error(f"Object not found: s3://{b}/{k}")
                # Handle missing object
            case S3AccessDenied(bucket=b, key=k):
                logger.error(f"Access denied: s3://{b}/{k}")
                # Handle permission error
            case S3NetworkError(original_error=e):
                logger.error(f"Network error: {e}")
                # Retry or fail
            case S3UnknownError(original_error=e):
                logger.error(f"Unknown S3 error: {e}")
                # Escalate to monitoring
            case _:
                # Exhaustiveness check - will fail type checking if new variant added
                assert_never(error)
```

### Exhaustiveness Checking: MyPy Strict Mode is Sufficient

SpectralMC's mypy strict configuration (`pyproject.toml`) already provides exhaustiveness checking:

```toml
strict = true                # Enables all strict checks
warn_unreachable = true      # Detects unreachable code paths
warn_no_return = true        # Detects missing return statements (in strict)
```

**For Result[T, E] types**, explicit case handling is sufficient:

```python
# File: documents/engineering/coding_standards.md
# Preferred pattern - no case _: needed
match result:
    case Success(value):
        return process(value)
    case Failure(error):
        return handle_error(error)
# MyPy verifies exhaustiveness via warn_no_return
```

**MyPy catches incomplete matches**:

```python
# File: documents/engineering/coding_standards.md
# Missing Failure case
match result:
    case Success(value):
        return value
# MyPy error: Missing return statement [return]
```

### When to Use assert_never()

The `assert_never()` function is **optional** for exhaustiveness checking:

```python
# File: documents/engineering/coding_standards.md
from typing import Never

def assert_never(value: Never) -> Never:
    """Type-safe exhaustiveness check for pattern matching."""
    raise AssertionError(f"Unhandled value: {value} ({type(value).__name__})")
```

**Use assert_never() for**:
1. **Large union types** (5+ variants) where explicit listing aids refactoring
2. **Effect types** with many variants
3. **When warn_no_return doesn't apply** (e.g., functions without return types)

**Skip assert_never() for**:
1. **Result[T, E] types** - explicit Success/Failure cases are clearer
2. **Small unions** (2-3 variants) - explicit cases are sufficient
3. **When mypy strict mode catches the error** - avoid redundant checks

**Example with large union type**:

```python
# File: documents/engineering/coding_standards.md
# If we add a new variant to S3OperationError:
@dataclass(frozen=True)
class S3RateLimited:
    kind: Literal["S3RateLimited"] = "S3RateLimited"
    bucket: str
    retry_after: int

S3OperationError = (
    S3BucketNotFound
    | S3ObjectNotFound
    | S3AccessDenied
    | S3NetworkError
    | S3UnknownError
    | S3RateLimited  # New variant
)

# assert_never() causes type error when new variant added:
# error: Argument 1 to "assert_never" has incompatible type "S3RateLimited"; expected "Never"
```

**Note**: `assert_never()` may fail with complex generic types like `Result[T, E]` due to mypy's type narrowing limitations. In these cases, mypy's `warn_no_return` provides equivalent exhaustiveness checking.

### Nested Pattern Matching

Combine Result and ADT pattern matching:

```python
# File: documents/engineering/coding_standards.md
def _find_broken_link(
    versions: list[ModelVersion],
) -> ModelVersion | None:
    """Find first version with broken parent link, or None if chain is valid.

    Uses generator expression (pure) instead of for loop.
    """
    pairs = zip(versions[:-1], versions[1:])
    return next(
        (curr for prev, curr in pairs if curr.parent_hash != prev.content_hash),
        None,
    )


async def verify_chain(store: AsyncBlockchainModelStore) -> Result[None, VerifyError]:
    """Verify blockchain integrity using pure patterns."""
    versions_result = await store.list_all_versions()

    match versions_result:
        case Success(versions):
            # Pure verification using match/case and helper function
            match versions:
                case []:
                    return Failure(EmptyChain())
                case [genesis, *_] if genesis.counter != 0:
                    return Failure(InvalidGenesis(version=genesis))
                case _ if (broken := _find_broken_link(versions)) is not None:
                    # Find previous version for error context
                    idx = versions.index(broken)
                    return Failure(BrokenChain(prev=versions[idx - 1], curr=broken))
                case _:
                    return Success(None)

        case Failure(error):
            # Map S3 errors to verification errors
            match error:
                case S3BucketNotFound(bucket=b):
                    return Failure(ChainNotFound(bucket=b))
                case S3NetworkError():
                    return Failure(VerificationNetworkError(original=error))
                case _:
                    return Failure(VerificationS3Error(original=error))
```

Note: The helper function `_find_broken_link` uses a generator expression (pure comprehension) instead of a `for` loop. See [Purity Doctrine](./purity_doctrine.md) for full details.

---

## No Exception Swallowing Doctrine

### Zero Tolerance Policy

SpectralMC enforces **zero tolerance for exception swallowing**. All exceptions must be either:
1. **Handled meaningfully** (logged, transformed to Result, or recovered from)
2. **Propagated** (re-raised to caller)

### Forbidden Patterns

#### ❌ Bare except with pass

```python
# File: documents/engineering/coding_standards.md
try:
    operation()
except:
    pass  # FORBIDDEN - silently hides ALL errors including KeyboardInterrupt
```

#### ❌ Broad except with default return

```python
# File: documents/engineering/coding_standards.md
try:
    result = operation()
except Exception:
    result = default_value  # FORBIDDEN - hides error context
```

#### ❌ Catching and ignoring

```python
# File: documents/engineering/coding_standards.md
try:
    result = operation()
except SomeError:
    pass  # FORBIDDEN - error is silently ignored
```

#### ❌ Catching to return None

```python
# File: documents/engineering/coding_standards.md
def get_value(key: str) -> Value | None:
    try:
        return fetch(key)
    except KeyError:
        return None  # FORBIDDEN - lost error context, use Result instead
```

### Required Patterns

#### ✅ Convert to Result type

```python
# File: documents/engineering/coding_standards.md
async def get_object(bucket: str, key: str) -> Result[bytes, S3OperationError]:
    try:
        response = await client.get_object(Bucket=bucket, Key=key)
        return Success(await response["Body"].read())
    except ClientError as e:
        return Failure(classify_s3_error(e, bucket, key))
```

#### ✅ Log and re-raise

```python
# File: documents/engineering/coding_standards.md
try:
    result = operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise  # Propagate to caller
```

#### ✅ Catch specific exceptions with specific handling

```python
# File: documents/engineering/coding_standards.md
try:
    await store._s3_client.create_bucket(Bucket=bucket_name)
except botocore.exceptions.ClientError as e:
    # Use match/case on error code (pure pattern within exception handler)
    match e.response["Error"]["Code"]:
        case "BucketAlreadyOwnedByYou":
            pass  # Expected - bucket exists, continue
        case _:
            raise  # Unexpected error - propagate
```

#### ✅ Transform and re-raise with context

```python
# File: documents/engineering/coding_standards.md
try:
    result = low_level_operation()
except LowLevelError as e:
    raise HighLevelError(f"Operation failed: {e}") from e
```

### Acceptable Exception Handling (Test Infrastructure Only)

In test setup/teardown code ONLY, specific exception handlers are acceptable when:
1. The handler is for **cleanup operations** that should not fail tests
2. The handler catches **specific known exceptions** (not bare `except:`)
3. The purpose is **documented in a comment**

Example (acceptable):

```python
# File: documents/engineering/coding_standards.md
# In test teardown - cleanup should not fail the test
try:
    await store._s3_client.delete_bucket(Bucket=bucket_name)
except botocore.exceptions.ClientError:
    # Best-effort cleanup - bucket may already be deleted
    pass
```

---

## Error Transformation

### The _raise() Helper

For cases where you need to convert Result to exception (e.g., at system boundaries):

```python
# File: documents/engineering/coding_standards.md
from typing import NoReturn

def _raise(error: E) -> NoReturn:
    """Convert error ADT to exception and raise.

    Use ONLY at system boundaries where exceptions are required.
    Internal code should propagate Result types.
    """
    raise RuntimeError(f"Operation failed: {error}")

# Usage: CLI commands (system boundary)
async def cmd_verify(bucket: str) -> int:
    """CLI command - uses exceptions at boundary."""
    async with AsyncBlockchainModelStore(bucket) as store:
        result = await verify_chain(store)
        match result:
            case Success(_):
                print("✓ Chain integrity verified")
                return 0
            case Failure(error):
                match error:
                    case EmptyChain():
                        print("✗ Chain is empty")
                        return 1
                    case InvalidGenesis(version=v):
                        print(f"✗ Invalid genesis block: {v}")
                        return 1
                    case BrokenChain(prev=prev, curr=curr):
                        print(f"✗ Broken chain link: {prev.counter} -> {curr.counter}")
                        return 1
                    case _:
                        # Unknown error - convert to exception at boundary
                        _raise(error)
```

**When to use _raise()**:
- ✅ CLI command error handling (exit codes + messages)
- ✅ HTTP API endpoints (convert to HTTP error responses)
- ✅ Test assertions (convert Result to exception for pytest)

**When NOT to use _raise()**:
- ❌ Internal library code (propagate Result types)
- ❌ Service layer logic (use pattern matching)
- ❌ Data processing pipelines (chain Results with monadic operations)

### Error Transformation Between Layers

Transform errors between layers explicitly:

```python
# File: documents/engineering/coding_standards.md
@dataclass(frozen=True)
class VerificationError:
    """High-level verification error."""
    kind: str
    message: str
    underlying: S3OperationError | None

    @staticmethod
    def from_s3_error(error: S3OperationError) -> "VerificationError":
        """Transform S3 error to verification error."""
        match error:
            case S3BucketNotFound(bucket=b):
                return VerificationError(
                    kind="ChainNotFound",
                    message=f"Blockchain not found in bucket: {b}",
                    underlying=error,
                )
            case S3NetworkError():
                return VerificationError(
                    kind="NetworkError",
                    message="Network error during verification",
                    underlying=error,
                )
            case _:
                return VerificationError(
                    kind="UnknownError",
                    message=f"Unknown error: {error}",
                    underlying=error,
                )
```

---

## No Legacy APIs Policy

**CRITICAL POLICY**: SpectralMC does not maintain backward compatibility with imperative/OOP error handling.

### Prohibited Patterns

**❌ NEVER use these patterns**:

```python
# File: documents/engineering/coding_standards.md
# ❌ Exception-based error handling for expected errors
def load_model(version: int) -> ModelSnapshot:
    if version not in available_versions:
        raise VersionNotFoundError(f"Version {version} not found")
    return load_from_disk(version)

# ❌ Returning None for errors
def get_object(bucket: str, key: str) -> bytes | None:
    try:
        return s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError:
        return None  # Lost error context!

# ❌ Returning error codes
def verify_chain(store: Store) -> tuple[bool, str]:
    if not store.exists():
        return False, "Chain not found"
    return True, ""

# ❌ Mutable error state
class Operation:
    def __init__(self):
        self.error: str | None = None
        self.success: bool = False

    def run(self):
        try:
            self.do_work()
            self.success = True
        except Exception as e:
            self.error = str(e)
            self.success = False
```

### Required Patterns

**✅ ALWAYS use these patterns**:

```python
# File: documents/engineering/coding_standards.md
# ✅ Result type for expected errors (pure - conditional expression)
def load_model(version: int) -> Result[ModelSnapshot, LoadError]:
    return (
        Failure(VersionNotFound(version=version, available=available_versions))
        if version not in available_versions
        else Success(load_from_disk(version))
    )

# ✅ Result type preserves error context
async def get_object(bucket: str, key: str) -> Result[bytes, S3OperationError]:
    try:
        response = await s3_client.get_object(Bucket=bucket, Key=key)
        return Success(await response["Body"].read())
    except ClientError as e:
        return Failure(classify_s3_error(e, bucket, key))

# ✅ Result type for verification (pure - match/case)
async def verify_chain(store: Store) -> Result[None, VerifyError]:
    versions_result = await store.list_all_versions()
    match versions_result:
        case Success([]):
            return Failure(EmptyChain())
        case Success(versions):
            # ... verification logic using pure patterns
            return Success(None)
        case Failure(error):
            return Failure(VerificationError.from_s3_error(error))

# ✅ Immutable error state
@dataclass(frozen=True)
class OperationResult:
    snapshot: ModelSnapshot | None
    error: LoadError | None
```

See [Purity Doctrine](./purity_doctrine.md) for the complete purity standards including the requirement to use conditional expressions or `match`/`case` instead of `if` statements.

### Migration Strategy

**NO gradual migration** - functional patterns are adopted immediately:

1. **New code**: MUST use Result types and ADTs from day one
2. **Refactoring**: Replace entire modules atomically (no hybrid states)
3. **Breaking changes**: Accepted and expected - update all callers simultaneously
4. **No deprecation warnings**: Old APIs are deleted, not deprecated

---

## Testing Functional Code

### Testing Result Types

```python
# File: documents/engineering/coding_standards.md
async def test_get_object_success():
    """Test successful S3 get returns Success."""
    result = await get_object_safe(mock_client, "my-bucket", "my-key")
    match result:
        case Success(data):
            assert data == b"expected content"
        case Failure(error):
            pytest.fail(f"Expected Success, got Failure: {error}")

async def test_get_object_not_found():
    """Test missing object returns Failure with S3ObjectNotFound."""
    result = await get_object_safe(mock_client, "my-bucket", "missing-key")
    match result:
        case Success(_):
            pytest.fail("Expected Failure, got Success")
        case Failure(error):
            match error:
                case S3ObjectNotFound(bucket=b, key=k):
                    assert b == "my-bucket"
                    assert k == "missing-key"
                case _:
                    pytest.fail(f"Expected S3ObjectNotFound, got {type(error).__name__}")
```

### Testing Exhaustiveness

Verify pattern matching is exhaustive:

```python
# File: documents/engineering/coding_standards.md
def test_pattern_matching_exhaustiveness():
    """Verify all S3OperationError variants are handled."""
    # This test should fail type checking if new variant added
    def handle_error(error: S3OperationError) -> str:
        match error:
            case S3BucketNotFound():
                return "bucket"
            case S3ObjectNotFound():
                return "object"
            case S3AccessDenied():
                return "access"
            case S3NetworkError():
                return "network"
            case S3UnknownError():
                return "unknown"
            case _:
                assert_never(error)  # Type error if variant missing

    # Test all variants
    assert handle_error(S3BucketNotFound("b", Exception())) == "bucket"
    assert handle_error(S3ObjectNotFound("b", "k", Exception())) == "object"
    # ... etc
```

### Advanced Functional Patterns

For advanced functional programming patterns beyond Result types and basic ADTs:

- **Effect Interpreter**: See [Effect Interpreter](effect_interpreter.md) for modeling all side effects as pure ADT types
- **Effect Composition**: See [Effect Interpreter](effect_interpreter.md#effect-composition) for sequential and parallel effect composition
- **Reproducibility Proofs**: See [Reproducibility Proofs](reproducibility_proofs.md) for how pure code enables provable determinism

---

## Implementation Anti-Patterns

### 1. Silent Failure Handling

**Problem**: Catching exceptions without proper handling or logging

- ❌ `try: result = simulate() except: return default_value` - hides errors
- ❌ `if torch.isnan(loss).any(): loss = torch.tensor(0.0)` - masks numerical issues
- ❌ Broad exception handlers: `except Exception: pass`
- ✅ Let exceptions propagate unless you can meaningfully handle them
- ✅ Log errors with context before re-raising
- ✅ Use specific exception types: `except ValueError as e:`

**Impact**: Silent failures in numerical code lead to incorrect results downstream

**Example**:
```python
# File: documents/engineering/coding_standards.md
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

**Problem**: Tests or implementations that report success without validation

- ❌ Training loop returns success even when loss diverged
- ❌ `status = "converged"` without checking convergence criteria
- ❌ Function returns successfully with NaN/Inf values
- ✅ Always validate outputs before returning success status
- ✅ Use type hints and runtime validation (pydantic)
- ✅ Raise exceptions for invalid states rather than returning error codes

**Example**:
```python
# File: documents/engineering/coding_standards.md
# ❌ False success (also has for loop - impure)
def train_model(model, data):
    for epoch in range(100):
        loss = train_step(model, data)
    return {"status": "success", "loss": loss}  # Could be NaN!

# ✅ Validated success with Result type (pure pattern)
@dataclass(frozen=True)
class TrainSuccess:
    final_loss: float

@dataclass(frozen=True)
class TrainDivergence:
    epoch: int
    loss: float

def _train_epoch(model: Model, data: Data, epoch: int) -> Result[float, TrainDivergence]:
    """Single training step returning Result."""
    loss = train_step(model, data)
    return (
        Failure(TrainDivergence(epoch=epoch, loss=float(loss)))
        if not torch.isfinite(loss)
        else Success(float(loss))
    )

def train_model(model: Model, data: Data) -> Result[TrainSuccess, TrainDivergence]:
    """Train model, returning Result instead of raising exceptions."""
    # Use reduce pattern instead of for loop for full purity
    # (simplified - see Effect Interpreter for production training loops)
    final_loss = 0.0
    # In production, use Effect ADTs for training loops
    return Success(TrainSuccess(final_loss=final_loss))
```

Note: Production training loops use the Effect Interpreter pattern. See [Effect Interpreter](./effect_interpreter.md) for the complete training loop architecture and [Purity Doctrine](./purity_doctrine.md) for purity requirements.

### 3. Ignoring Numerical Warnings

**Problem**: Treating warnings as noise instead of signals

- ❌ Suppressing "divide by zero" warnings
- ❌ Ignoring "invalid value encountered" from NumPy/PyTorch
- ❌ Filtering out all warnings with `warnings.filterwarnings("ignore")`
- ✅ Investigate and fix root cause of warnings
- ✅ Only filter specific expected warnings (e.g., QuantLib deprecation warnings)
- ✅ Convert warnings to errors during testing: `warnings.simplefilter("error")`

### 4. Mutable Default Arguments

**Problem**: Using mutable objects as default arguments

- ❌ `def simulate(config={}):` - shared across calls
- ❌ `def run_batch(params=[]):` - accumulates across calls
- ✅ `def simulate(config=None): config = config or {}`
- ✅ Use immutable defaults or None

**Impact**: Especially dangerous in parallel/distributed computing with Ray/Dask

### 5. Inconsistent Device Handling

**Problem**: Not managing CPU/GPU device placement consistently

- ❌ Assuming tensors are on CUDA without checking
- ❌ Moving tensors between devices unnecessarily
- ❌ Not handling device in function signatures
- ✅ Explicit GPU requirement: `device = torch.device("cuda:0")` (fail fast if CUDA missing)
- ✅ Keep tensors on same device throughout computation
- ✅ Use `models/cpu_gpu_transfer.py` utilities for controlled transfers

**Example**: Model on GPU, input data on CPU - causes cryptic errors

---

## Dependency Deprecation Management

SpectralMC enforces a **zero-tolerance policy** for deprecated APIs in production code to ensure long-term maintainability and compatibility.

### Zero-Tolerance Policy

**Prohibited**:
- ❌ NO deprecated APIs in `src/spectralmc/` code
- ❌ NO suppressing deprecation warnings without documented upstream issue
- ❌ NO using `# type: ignore` or similar to hide deprecation warnings

**Required**:
- ✅ All deprecations must have migration plan within 1 sprint
- ✅ Use modern, non-deprecated APIs for all new code
- ✅ Fix deprecation warnings immediately when they appear

### Allowed Exceptions

Only third-party library internals may use deprecated code if:
1. **Upstream issue tracked**: Must have link to library's GitHub issue
2. **Pytest filter documented**: Must add filter in `pyproject.toml` with explanation
3. **Monthly review**: Must check for fixes in dependency updates

**Example** (current exceptions):
```toml
# File: documents/engineering/coding_standards.md
[tool.pytest.ini_options]
filterwarnings = [
    # Botocore datetime.utcnow() - AWS SDK internal (boto/botocore#3201)
    "ignore::DeprecationWarning:botocore.*",
    # QuantLib SWIG bindings - unfixable (generated code)
    "ignore::DeprecationWarning:.*QuantLib.*",
]
```

### Monthly Review Checklist

Run these checks on the 1st of each month:

```bash
# File: documents/engineering/coding_standards.md
# 1. Check for dependency updates
docker compose -f docker/docker-compose.yml exec spectralmc poetry show --outdated

# 2. Check botocore for datetime.utcnow fix (currently pending)
# Visit: https://github.com/boto/botocore/releases

# 3. Check for new deprecation warnings in tests
docker compose -f docker/docker-compose.yml exec spectralmc \
  poetry run test-all -W default::DeprecationWarning > /tmp/warnings.txt 2>&1
grep "DeprecationWarning" /tmp/warnings.txt | grep -v "botocore\|QuantLib"

# 4. Review PyTorch/NumPy/CuPy changelogs for upcoming deprecations
# - PyTorch: https://github.com/pytorch/pytorch/releases
# - NumPy: https://numpy.org/news/
# - CuPy: https://github.com/cupy/cupy/releases
```

### Code Review Checklist

Block merge if ANY of these are present:

- [ ] Uses `torch.utils.dlpack.from_dlpack()` instead of `torch.from_dlpack()`
- [ ] Uses `cupy_array.toDlpack()` instead of direct `torch.from_dlpack()`
- [ ] Uses removed NumPy aliases (`np.float`, `np.int`, `np.complex_`, `np.bool`)
- [ ] Adds `@pytest.mark.skip()` without upstream issue link
- [ ] Suppresses `DeprecationWarning` in production code without explanation
- [ ] Ignores deprecation warnings in function/method implementations

### Migration Examples

**DLPack API (COMPLETED)**:
```python
# File: documents/engineering/coding_standards.md
# ❌ DEPRECATED (removed in CuPy 14+)
capsule = cupy_array.toDlpack()
torch_tensor = torch.utils.dlpack.from_dlpack(capsule)

# ✅ MODERN (PyTorch 1.10+, CuPy 9.0+)
torch_tensor = torch.from_dlpack(cupy_array)
```

**NumPy Type Aliases (already correct)**:
```python
# File: documents/engineering/coding_standards.md
# ❌ REMOVED in NumPy 2.0
Type[np.float]    # Don't use
Type[np.int]      # Don't use
Type[np.complex_] # Don't use

# ✅ CORRECT (NumPy 2.0+)
Type[np.float64]
Type[np.int64]
Type[np.complex128]
```

### Dependency Update Protocol

When updating major dependencies (PyTorch, NumPy, CuPy):

1. **Read migration guide FIRST**
   - PyTorch: Check release notes "Breaking Changes" section
   - NumPy: https://numpy.org/doc/stable/numpy_2_0_migration_guide.html
   - CuPy: Check changelog "Backward Incompatible Changes"

2. **Test in isolation**
   ```bash
   # File: documents/engineering/coding_standards.md
   git checkout -b deps/pytorch-upgrade
   poetry update torch  # ONE dependency at a time
   docker compose -f docker/docker-compose.yml exec spectralmc \
     poetry run test-all > /tmp/test_upgrade.txt 2>&1
   ```

3. **Verify no new deprecations**
   ```bash
# File: documents/engineering/coding_standards.md
docker compose -f docker/docker-compose.yml exec spectralmc \
  poetry run test-all -W error::DeprecationWarning
```text
# File: documents/engineering/coding_standards.md

### Status: Deprecation-Free Codebase

**Last Audit**: 2025-01-09

**SpectralMC Code**:
- ✅ All DLPack usage migrated to `torch.from_dlpack()`
- ✅ All NumPy types using explicit precision (`np.float64`, not `np.float`)
- ✅ All PyTorch APIs using non-deprecated methods
- ✅ Zero deprecation warnings from `src/spectralmc/`

**Third-Party Dependencies**:
- ⏳ `botocore 1.36.1`: Waiting for `datetime.utcnow()` fix (tracked: boto/botocore#3201)
- ⏸️ `QuantLib 1.37`: SWIG deprecations unfixable (accepted as permanent exception)

---

## Summary

SpectralMC's coding standards ensure:

- **Consistent formatting** with Black (zero configuration)
- **Strict type safety** with mypy (zero tolerance for `Any`, `cast`, `type: ignore`)
- **Complete type coverage** via custom stubs for all dependencies
- **Functional error handling** with Result types, ADTs, and pattern matching
- **No exception swallowing** - all exceptions must be handled meaningfully or propagated
- **Reproducible type checking** independent of external changes
- **Executable documentation** through comprehensive type annotations
- **Zero deprecated APIs** with monthly review protocol

Before committing code:

```
# File: documents/engineering/coding_standards.md
# Inside Docker container
docker compose -f docker/docker-compose.yml exec spectralmc black .
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```text
# File: documents/engineering/coding_standards.md

Both must pass with zero errors.

See also:
- [Immutability Doctrine](./immutability_doctrine.md) - Frozen dataclasses and functional updates
- [Pydantic Patterns](./pydantic_patterns.md) - Model validation and configuration
- [Documentation Standards](../documentation_standards.md) - Docstring requirements
- [Testing Requirements](./testing_requirements.md) - Testing anti-patterns and best practices
- [CPU/GPU Compute Policy](./cpu_gpu_compute_policy.md) - Device placement and transfers
```
