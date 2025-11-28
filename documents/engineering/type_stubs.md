# Custom Type Stubs (.pyi files)

## Overview

SpectralMC maintains **custom type stubs** for all third-party libraries in the `stubs/` directory. These `.pyi` files provide complete, project-specific type information that may be stricter than upstream library stubs.

**Related Standards**: [Type Safety](type_safety.md), [PyTorch Facade](pytorch_facade.md)

---

## Rationale

Custom type stubs ensure:

1. **Complete type coverage** - All external dependencies have full type information
2. **Project-specific typing** - Stricter than upstream (e.g., no `Any` where we know the actual type)
3. **Reproducible type checking** - Independent of external stub changes or breakage
4. **Controlled upgrades** - We update stubs when ready, not when upstream changes

### Why Not Use Existing Stubs?

Many libraries provide incomplete or outdated stubs:

- PyTorch's official stubs use `Any` extensively
- NumPy stubs may not cover all array operations
- Third-party stub packages (e.g., `types-*`) can have version mismatches

SpectralMC requires **zero `Any` types** (see [Type Safety](type_safety.md)), so we maintain our own stubs.

---

## Organization

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

---

## Stub Requirements

All type stubs must adhere to these strict requirements:

### 1. Type-Pure (Zero Tolerance)

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

### 2. Minimal (Only Used APIs)

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

### 3. Documented

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

---

## Example Stub Structure

### Complete Stub: `stubs/torch/__init__.pyi`

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

---

## Adding New Dependencies

When adding a new third-party dependency to SpectralMC:

### Step 1: Create Stub Directory

```bash
mkdir -p stubs/new_library
touch stubs/new_library/__init__.pyi
```

### Step 2: Stub Only Used APIs

Identify which functions/classes you're using:

```bash
# Find all imports of the library
grep -r "from new_library import" src/spectralmc/
grep -r "import new_library" src/spectralmc/
```

Stub **only those APIs**, not the entire library.

### Step 3: Write Type-Pure Stubs

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

### Step 4: Verify with mypy

```bash
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```

If mypy reports missing types, add them to the stub. **Never use `# type: ignore`.**

### Step 5: Document in Stub

Add docstring explaining:
- Which version of the library this stub covers
- What APIs are included (and excluded)
- When it was last updated

---

## Maintaining Stubs

### When to Update

Update stubs when:

1. **Upgrading library version** - Check if API changed
2. **Using new APIs** - Add missing functions/classes
3. **mypy errors** - Fill in gaps in type information

### How to Update

1. Read library changelog for breaking changes
2. Update stub to match new API
3. Run mypy to verify
4. Update "Last updated" date in docstring

### Testing Stubs

Stubs are checked by mypy automatically when type-checking `src/` and `tests/`:

```bash
# Stub errors will appear as mypy errors
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```

---

## Common Patterns

### Protocol for Duck Typing

Use `Protocol` for structural subtyping:

```python
from typing import Protocol

class SupportsTrain(Protocol):
    """Any object with a .train() method."""
    def train(self, mode: bool = True) -> None: ...
```

### Overloads for Function Variants

Use `@overload` for functions with multiple signatures:

```python
from typing import overload

@overload
def zeros(size: int, *, dtype: dtype | None = None) -> Tensor: ...

@overload
def zeros(size: tuple[int, ...], *, dtype: dtype | None = None) -> Tensor: ...

def zeros(*args: int | tuple[int, ...], dtype: dtype | None = None) -> Tensor: ...
```

### TypeVar for Generics

```python
from typing import TypeVar

T = TypeVar("T")

def to_device(data: T, device: device) -> T: ...
```

---

## Stub Validation Checklist

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

- **Custom stubs** in `stubs/` for all third-party libraries
- **Type-pure** - zero `Any`, `cast`, or `type: ignore`
- **Minimal** - only stub APIs actually used
- **Documented** - explain scope and version
- **Update incrementally** as new APIs are used
- **Verify with mypy** before committing

See also: [Type Safety](type_safety.md), [PyTorch Facade](pytorch_facade.md)
