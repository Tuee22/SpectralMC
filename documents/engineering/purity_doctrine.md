# Purity Doctrine

## Core Principle

**Pure functions depend only on their inputs and produce only their outputs.**

Purity is the foundation of SpectralMC's architecture. Pure code is testable, reproducible, and composable. Side effects are isolated to the Effect Interpreter, keeping business logic deterministic and verifiable.

---

## Overview

SpectralMC enforces strict purity standards for all non-test code. A **pure function**:

1. **Has no mutation** - all data is immutable
2. **Uses expressions, not control-flow statements** - no `for`, `if`, `while`
3. **Has no side effects** - no I/O, logging, or external state
4. **Does not raise exceptions** - errors are returned via Result types
5. **Depends only on inputs** - same inputs always produce same outputs

**Related Standards**:
- [Immutability Doctrine](./immutability_doctrine.md) - Frozen dataclasses
- [Coding Standards](./coding_standards.md) - Result types and ADTs (SSoT)
- [Effect Interpreter](./effect_interpreter.md) - Side effects as ADTs

---

## Forbidden Patterns

### 1. `for` Loops

**Anti-pattern**:
```python
def process_items(items: list[int]) -> list[int]:
    result = []
    for item in items:
        result.append(item * 2)
    return result
```

**Why forbidden**:
- Requires mutable accumulator (`result = []`)
- Iterative mutation (`result.append()`)
- Not expression-oriented

**Correct pattern** (comprehension):
```python
def process_items(items: list[int]) -> list[int]:
    return [item * 2 for item in items]
```

### 2. `if` Statements

**Anti-pattern**:
```python
def process(value: int | str) -> int:
    if isinstance(value, str):
        return int(value)
    return value
```

**Why forbidden**:
- Control flow via statements
- Multiple return points
- Not expression-oriented

**Correct pattern** (conditional expression):
```python
def process(value: int | str) -> int:
    return int(value) if isinstance(value, str) else value
```

**Correct pattern** (match/case for complex branching):
```python
def handle_result(result: Result[Model, LoadError]) -> str:
    match result:
        case Success(model):
            return model.name
        case Failure(error):
            return f"Error: {error}"
```

### 3. `while` Loops

**Anti-pattern**:
```python
def find_first(items: list[int], predicate: Callable[[int], bool]) -> int | None:
    i = 0
    while i < len(items):
        if predicate(items[i]):
            return items[i]
        i += 1
    return None
```

**Why forbidden**:
- Mutable loop counter
- Control flow statements
- Imperative style

**Correct pattern** (generator expression with next):
```python
def find_first(items: list[int], predicate: Callable[[int], bool]) -> int | None:
    return next((item for item in items if predicate(item)), None)
```

### 4. `raise` for Expected Errors

**Anti-pattern**:
```python
@dataclass(frozen=True)
class TensorTransfer:
    source_device: Device
    target_device: Device

    def __post_init__(self) -> None:
        if self.source_device == self.target_device:
            raise ValueError("Invalid transfer: same device")
```

**Why forbidden**:
- Exceptions break control flow
- Caller cannot statically know failure modes
- Error handling via try/except is impure

**Correct pattern** (factory returning Result):
```python
@dataclass(frozen=True)
class TensorTransfer:
    """Validated tensor transfer - only constructible via factory."""
    source_device: Device
    target_device: Device


@dataclass(frozen=True)
class InvalidTransferError:
    """Error when source and target device are identical."""
    device: Device


def tensor_transfer(
    source: Device, target: Device
) -> Result[TensorTransfer, InvalidTransferError]:
    """Create TensorTransfer, returning Failure if devices are identical."""
    return (
        Failure(InvalidTransferError(device=source))
        if source == target
        else Success(TensorTransfer(source_device=source, target_device=target))
    )
```

### 5. Side Effects in Pure Functions

**Anti-pattern**:
```python
def compute_price(model: Model, params: Params) -> float:
    logger.info(f"Computing price with {params}")  # Side effect!
    print(f"Model: {model.name}")  # Side effect!
    return model.forward(params)
```

**Why forbidden**:
- I/O operations are not reproducible
- Logging changes global state
- Function behavior depends on external state

**Correct pattern** (pure computation):
```python
def compute_price(model: Model, params: Params) -> float:
    """Pure price computation - no logging, no I/O."""
    return model.forward(params)
```

Side effects belong in the Effect Interpreter. See [Effect Interpreter](./effect_interpreter.md).

---

## Allowed Constructs

### 1. `match`/`case` on Pure Types

Pattern matching on ADTs (Result, custom sum types) is **pure**:

```python
def describe_training_state(state: TrainingState) -> str:
    """Pure function using match/case on ADT."""
    match state:
        case TrainingInProgress(epoch=e, loss=loss):
            return f"Training epoch {e}, loss={loss:.4f}"
        case TrainingConverged(epoch=e, loss=loss):
            return f"Converged at epoch {e}, final loss={loss:.4f}"
        case TrainingDiverged(epoch=e, loss=loss):
            return f"Diverged at epoch {e}, loss={loss:.4f}"
```

**Why allowed**:
- Exhaustive handling enforced by `assert_never()`
- No mutation, no side effects
- Expression-oriented (each case returns a value)
- Closest to functional pattern matching in Python

### 2. Comprehensions (Without Side Effects)

List, dict, set, and generator comprehensions are **pure** when they:
- Do not call functions with side effects
- Do not mutate external state
- Do not perform I/O

```python
# Pure comprehensions
squares: list[int] = [x * x for x in range(10)]
evens: list[int] = [x for x in items if x % 2 == 0]
lookup: dict[str, int] = {name: idx for idx, name in enumerate(names)}
unique: set[int] = {abs(x) for x in values}

# Impure comprehension (FORBIDDEN)
logged: list[int] = [log_and_return(x) for x in items]  # Side effect!
```

### 3. Conditional Expressions

The ternary operator is **pure**:

```python
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Pure clamping using conditional expressions."""
    return min_val if value < min_val else (max_val if value > max_val else value)
```

### 4. Definitions

`def`, `class`, `return`, `import` are **allowed** - they define structure, not control flow:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    learning_rate: float
    batch_size: int

def create_config(lr: float, bs: int) -> Config:
    return Config(learning_rate=lr, batch_size=bs)
```

### 5. `assert_never()` for Exhaustiveness

The `assert_never()` function is allowed to `raise` because:
- It's a compile-time exhaustiveness check
- It should **never** execute at runtime
- If it executes, it indicates a programming error (missing case)

```python
from typing import assert_never

def handle_effect(effect: Effect) -> Result[None, EffectError]:
    match effect:
        case TensorTransfer():
            return handle_transfer(effect)
        case StreamSync():
            return handle_sync(effect)
        case _:
            assert_never(effect)  # Allowed - compile-time check
```

---

## Constructor Validation Pattern

When dataclass construction requires validation, use **factory functions** instead of `__post_init__` that raises:

### Before (Impure - raises)

```python
@dataclass(frozen=True)
class BoundSpec:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError(f"lower ({self.lower}) must be < upper ({self.upper})")
```

### After (Pure - factory returns Result)

```python
@dataclass(frozen=True)
class BoundSpec:
    """Validated bound specification - only constructible via factory."""
    lower: float
    upper: float


@dataclass(frozen=True)
class InvalidBoundsError:
    """Error when lower >= upper."""
    lower: float
    upper: float


def bound_spec(lower: float, upper: float) -> Result[BoundSpec, InvalidBoundsError]:
    """Create BoundSpec, returning Failure if bounds are invalid."""
    return (
        Failure(InvalidBoundsError(lower=lower, upper=upper))
        if lower >= upper
        else Success(BoundSpec(lower=lower, upper=upper))
    )
```

### Usage Pattern

```python
def use_bounds(lower: float, upper: float) -> Result[float, InvalidBoundsError]:
    """Example of using factory function."""
    match bound_spec(lower, upper):
        case Success(bounds):
            return Success((bounds.upper - bounds.lower) / 2)
        case Failure(error):
            return Failure(error)
```

---

## Purity Summary Table

| Construct | Pure? | Use Instead |
|-----------|-------|-------------|
| `for` loop | No | Comprehension: `[f(x) for x in items]` |
| `if` statement | No | Conditional: `x if cond else y` or `match`/`case` |
| `while` loop | No | Comprehension, recursion, or `next()` with generator |
| `raise` | No | Return `Result[T, E]` type |
| `try`/`except` | No | Pattern match on `Result` |
| `print()` | No | Effect ADT |
| `logger.*()` | No | Effect ADT |
| `match`/`case` | **Yes** | (on pure types) |
| Comprehension | **Yes** | (if no side effects) |
| Conditional expr | **Yes** | |
| `def`/`class` | **Yes** | |
| `return` | **Yes** | |
| `assert_never()` | **Yes** | (compile-time exhaustiveness) |

---

## Enforcement

### Static Analysis

mypy cannot detect all purity violations. Use code review and grep audits:

```bash
# Check for for-loops in pure code (manual review needed)
grep -rn "^\s*for " src/spectralmc/ | grep -v "# IMPURE OK:"

# Check for if-statements in pure code (manual review needed)
grep -rn "^\s*if " src/spectralmc/ | grep -v "# IMPURE OK:" | grep -v "__post_init__"

# Check for raise statements (should only be assert_never)
grep -rn "^\s*raise " src/spectralmc/ | grep -v "assert_never" | grep -v "# IMPURE OK:"

# Check for print statements
grep -rn "print(" src/spectralmc/ && echo "FOUND print()" || echo "OK"

# Check for logging in pure modules
grep -rn "logger\." src/spectralmc/ | grep -v "# EFFECT:"
```

### Code Review Checklist

Before approving any PR:

- [ ] No `for` loops (use comprehensions)
- [ ] No `if` statements (use conditional expressions or `match`/`case`)
- [ ] No `while` loops (use comprehensions or recursion)
- [ ] No `raise` except in `assert_never()` or system boundaries
- [ ] No `print()` or `logger.*()` in pure functions
- [ ] Factory functions return `Result` for validation
- [ ] Comprehensions have no side effects

---

## Exceptions

### Test Code

Test code may use impure constructs:

- `assert` statements (pytest requires them)
- `pytest.raises()` context manager
- `for` loops in test setup/teardown

Tests verify behavior; they are not business logic.

### System Boundaries

At system boundaries (CLI commands, HTTP handlers), impure code is acceptable:

- Logging at entry/exit points
- `raise` after exhaustive Result handling
- I/O operations

Mark system boundary code with `# IMPURE OK: system boundary` comment.

### Effect Interpreter

The Effect Interpreter itself is impure by design - it executes effects:

- GPU operations
- Storage I/O
- RNG state manipulation

All impure operations are concentrated in the interpreter. Business logic remains pure.

---

## Rationale

### Why Purity Matters

1. **Testability**: Pure functions are trivially testable - mock inputs, assert outputs
2. **Reproducibility**: Same inputs always produce same outputs
3. **Composability**: Pure functions compose without hidden interactions
4. **Parallelism**: Pure functions are safe to parallelize
5. **Reasoning**: Pure functions are easier to understand and debug

### SpectralMC-Specific Benefits

- **Bit-exact reproducibility**: Required for quantitative finance
- **Checkpoint/resume equivalence**: Pure training steps guarantee identical results
- **Effect isolation**: All side effects captured in Effect ADTs
- **Type-safe error handling**: Result types prevent silent failures

---

## References

- [Immutability Doctrine](./immutability_doctrine.md) - Frozen dataclass requirements
- [Coding Standards](./coding_standards.md) - Result types and ADT patterns (SSoT)
- [Effect Interpreter](./effect_interpreter.md) - Side effect isolation
- [Reproducibility Proofs](./reproducibility_proofs.md) - How purity enables provable determinism

---

**Last updated**: 2025-11-30
**Status**: Active doctrine, zero tolerance for violations in pure code
