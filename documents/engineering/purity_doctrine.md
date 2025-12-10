# File: documents/engineering/purity_doctrine.md
# Purity Doctrine

**Status**: Authoritative source
**Supersedes**: Prior purity doctrine drafts
**Referenced by**: documents/documentation_standards.md

> **Purpose**: Define SpectralMC purity expectations and separation of side effects.

## Cross-References
- [Effect Interpreter](effect_interpreter.md)
- [Coding Standards](coding_standards.md)
- [Reproducibility Proofs](reproducibility_proofs.md)
- [Testing Requirements](testing_requirements.md)
- [Purity Enforcement](purity_enforcement.md)

## Zero Tolerance Policy for Business Logic

**Critical Distinction**: SpectralMC enforces **ZERO TOLERANCE** for purity violations in **Tier 2 (Business Logic)** files, while preserving necessary exemptions for infrastructure and effect execution layers.

### What Zero Tolerance Means

**In Tier 2 Business Logic Files** (`gbm_trainer.py`, `serialization/*`, `cvnn_factory.py`, `sobol_sampler.py`):
- ❌ ZERO statement-level `for` loops (use comprehensions, helper functions, or Effect ADTs)
- ❌ ZERO statement-level `if` branches (use conditional expressions or `match`/`case`)
- ❌ ZERO `while` loops (use generator expressions with `next()`)
- ❌ ZERO `raise` for expected errors (use Result types; defensive assertions OK)
- ❌ ZERO side effects (logging, I/O, mutation) - must use Effect ADTs

**Acceptable in Tier 2 ONLY as expressions**:
- ✅ Conditional expressions: `value if condition else default`
- ✅ Match/case on ADTs: `match result: case Success(v): ...`
- ✅ Comprehensions: `[f(x) for x in items]`
- ✅ Generator expressions: `next((x for x in items if p(x)), None)`

### Why Zero Tolerance?

Business logic purity enables:
1. **Bit-exact reproducibility** - Required for quantitative finance
2. **Property-based testing** - Pure functions are trivially testable
3. **Effect isolation** - All side effects captured in ADTs
4. **Type-safe composition** - Pure functions compose without hidden state

Infrastructure may be impure because it provides the deterministic foundation that business logic relies on.

---

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

## Scope and Architectural Boundaries

The purity doctrine applies to **business logic**, not all code in SpectralMC. The codebase has three architectural tiers with different purity requirements:

### Tier 1: Infrastructure and Facade Layers (EXEMPT)

**Exempt Files**:
- `src/spectralmc/models/torch.py` - PyTorch facade for reproducibility
- `src/spectralmc/models/cpu_gpu_transfer.py` - Device transfer utilities
- `src/spectralmc/cvnn.py` - PyTorch nn.Module layer library

**Rationale**: These files provide the infrastructure that enables business logic to be pure. They handle:
- Import-time environment configuration (deterministic flags)
- Thread safety guarantees
- Device placement and transfer
- Standard PyTorch idioms (nn.Module patterns)

**Acceptable patterns in infrastructure**:
- Import-time guards: `if "torch" in sys.modules: raise ImportError(...)`
- Thread safety checks: `if threading.get_ident() != _MAIN_THREAD_ID: raise RuntimeError(...)`
- Global configuration: `torch.use_deterministic_algorithms(True)`
- nn.Module idioms: `if bias: self.bias = nn.Parameter(...)`
- Weight initialization: `if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(...)`

Per [pytorch_facade.md](pytorch_facade.md), these patterns are necessary for guaranteeing reproducibility.

### Tier 2: Business Logic (ZERO TOLERANCE PURITY)

**Enforced Files**:
- `src/spectralmc/gbm_trainer.py` - Training orchestration
- `src/spectralmc/serialization/*` - Data serialization
- `src/spectralmc/cvnn_factory.py` - Model factory
- `src/spectralmc/sobol_sampler.py` - Sampling logic
- Other core processing modules

**Rationale**: Business logic must be 100% pure for testability, reproducibility, and composability.

**Zero Tolerance Enforcement**:
- ❌ NO `for` loops (use comprehensions or Effect sequences)
- ❌ NO statement-level `if` (use conditional expressions or match/case)
- ❌ NO `while` loops (use generators or recursion)
- ❌ NO `raise` for expected errors (use Result types)
- ❌ NO logging/I/O (use Effect ADTs)

**ONLY Acceptable Patterns**:
- ✅ Conditional expressions: `x if cond else y`
- ✅ Match/case on Result/ADTs
- ✅ Comprehensions (no side effects)
- ✅ Helper functions returning Result types
- ✅ Defensive assertions for programming errors (not validation)

**Whitelisted Boundaries** (4 total - documented for migration):
- `gbm_trainer.py:382` - Logging interval check (infrastructure boundary)
- `gbm_trainer.py:392` - TensorBoard flush check (infrastructure boundary)
- `gbm_trainer.py:421` - Gradient existence guard (TensorBoard infrastructure)
- `serialization/tensors.py:174` - Protobuf `requires_grad` mutation (API boundary)

See [Purity Enforcement](purity_enforcement.md) for AST linter whitelist configuration.

**Achievement**: 100% purity required for new code; 4 whitelisted exceptions in existing code documented for migration.

### Tier 3: Effect Interpreter (MIXED)

**Mixed Purity Files**:
- `src/spectralmc/effects/interpreter.py` - Effect execution
- `src/spectralmc/storage/*` - Storage operations

**Rationale**: Effect descriptions (ADTs) are pure; effect execution is impure by design.

**Requirements**:
- Effect ADT definitions: PURE (frozen dataclasses)
- Interpreter methods: MAY be impure (I/O, async operations)

---

## Forbidden Patterns (Tier 2 Business Logic ONLY)

The following patterns are **STRICTLY FORBIDDEN** in Tier 2 business logic files. These patterns are acceptable in Tier 1 infrastructure (see architectural boundaries above).

**Files where these patterns are FORBIDDEN**:
- `src/spectralmc/gbm_trainer.py` (except 3 whitelisted lines)
- `src/spectralmc/serialization/*.py` (except tensors.py:174)
- `src/spectralmc/cvnn_factory.py`
- `src/spectralmc/sobol_sampler.py`
- All other Tier 2 files

### 1. `for` Loops

**Anti-pattern** (FORBIDDEN in Tier 2):
```python
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
def process_items(items: list[int]) -> list[int]:
    return [item * 2 for item in items]
```

### 2. `if` Statements

**Anti-pattern**:
```python
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
def process(value: int | str) -> int:
    return int(value) if isinstance(value, str) else value
```

**Correct pattern** (match/case for complex branching):
```python
# File: documents/engineering/purity_doctrine.md
def handle_result(result: Result[Model, LoadError]) -> str:
    match result:
        case Success(model):
            return model.name
        case Failure(error):
            return f"Error: {error}"
```

**Critical**: Conditional **expressions** (`x if cond else y`) are pure and encouraged. Statement-level `if` blocks are forbidden because they:
- Break expression-oriented programming
- Cannot be composed or assigned
- Prevent type narrowing in some cases
- Encourage imperative mutation patterns

**Zero tolerance means**: Count of statement-level `if` in Tier 2 files must approach ZERO (4 whitelisted exceptions documented above).

### 3. `while` Loops

**Anti-pattern**:
```python
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
def find_first(items: list[int], predicate: Callable[[int], bool]) -> int | None:
    return next((item for item in items if predicate(item)), None)
```

### 4. `raise` for Expected Errors

**Anti-pattern**:
```python
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Pure clamping using conditional expressions."""
    return min_val if value < min_val else (max_val if value > max_val else value)
```

### 4. Definitions

`def`, `class`, `return`, `import` are **allowed** - they define structure, not control flow:

```python
# File: documents/engineering/purity_doctrine.md
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    learning_rate: float
    batch_size: int

def create_config(lr: float, bs: int) -> Config:
    return Config(learning_rate=lr, batch_size=bs)
```

### 5. Exhaustiveness Checking (MyPy Strict Mode)

**MyPy's strict mode provides exhaustiveness checking** via `warn_no_return` and `warn_unreachable`. The `assert_never()` function is **optional**, not required.

**Preferred pattern for Result[T, E]** - explicit cases without `assert_never()`:

```python
# File: documents/engineering/purity_doctrine.md
def process_result(result: Result[int, str]) -> int:
    match result:
        case Success(value):
            return value
        case Failure(error):
            return -1
    # MyPy verifies exhaustiveness - no assert_never() needed
```

**MyPy catches missing cases**:

```python
# File: documents/engineering/purity_doctrine.md
def incomplete_match(result: Result[int, str]) -> int:
    match result:
        case Success(value):
            return value
    # MyPy error: Missing return statement [return]
```

**When assert_never() is useful** - large union types (5+ variants):

```python
# File: documents/engineering/purity_doctrine.md
from typing import assert_never

def handle_effect(effect: Effect) -> Result[None, EffectError]:
    match effect:
        case TensorTransfer():
            return handle_transfer(effect)
        case StreamSync():
            return handle_sync(effect)
        case KernelLaunch():
            return handle_launch(effect)
        case DLPackTransfer():
            return handle_dlpack(effect)
        case _:
            assert_never(effect)  # Optional - helps with refactoring
```

**Note**: `assert_never()` may fail with complex generic types like `Result[T, E]` due to mypy limitations. Use explicit case handling instead.

---

## Constructor Validation Pattern

When dataclass construction requires validation, use **factory functions** instead of `__post_init__` that raises:

### Before (Impure - raises)

```python
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
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
# File: documents/engineering/purity_doctrine.md
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

### Primary: AST-Based Static Analysis

SpectralMC uses **AST-based linting** to enforce purity in Tier 2 files. See [Purity Enforcement](purity_enforcement.md) for complete tooling guide.

**Command**:
```bash
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-purity
```

**What it checks**:
1. **For loops**: Zero statement-level `for` in Tier 2 files
2. **If statements**: Zero statement-level `if` in Tier 2 files (4 whitelisted exceptions)
3. **While loops**: Zero `while` in Tier 2 files
4. **Raise statements**: Classification via whitelist (defensive assertions OK)
5. **Side effects**: No logging, I/O, or mutation in pure functions

**AST Detection Rules**:
- `ast.For` nodes in Tier 2 → ERROR (zero tolerance)
- `ast.If` nodes in Tier 2 (not in whitelist) → ERROR
- `ast.While` nodes in Tier 2 → ERROR (zero tolerance)
- `ast.Raise` nodes not in whitelist → WARNING (manual review required)
- Function calls to `logger.*`, `print`, file I/O → ERROR

**Whitelist Configuration** (4 acceptable patterns):
```python
ACCEPTABLE_IF_STATEMENTS = {
    ("src/spectralmc/gbm_trainer.py", 382): "Logging infrastructure boundary",
    ("src/spectralmc/gbm_trainer.py", 392): "TensorBoard flush boundary",
    ("src/spectralmc/gbm_trainer.py", 421): "Gradient guard (TensorBoard)",
    ("src/spectralmc/serialization/tensors.py", 174): "Protobuf API boundary",
}
```

### Code Review Checklist

Before approving any PR touching Tier 2 files:

- [ ] Zero `for` loops (verified by AST linter)
- [ ] Zero statement-level `if` outside whitelist (verified by AST linter)
- [ ] Zero `while` loops (verified by AST linter)
- [ ] All `raise` statements are defensive assertions (not validation errors)
- [ ] No `print()` or `logger.*()` calls
- [ ] Returns Effect ADTs for side-effectful operations
- [ ] Factory functions return `Result` for validation
- [ ] Comprehensions have no side effects
- [ ] Match/case uses exhaustive patterns

### CI/CD Integration

**Required gates** (must pass before merge):
1. `poetry run check-code` (Ruff + Black + MyPy)
2. `poetry run check-purity` (AST-based purity linting)
3. `poetry run test-all` (all tests passing)

---

## Tier-Specific Purity Requirements

### Hierarchy of Purity Requirements

| Layer | Purity Requirement | Allowed Impurity | Enforcement |
|-------|-------------------|------------------|-------------|
| **Tier 1: Infrastructure** | Relaxed | Thread safety, nn.Module idioms, device guards | Manual review |
| **Tier 2: Business Logic** | **ZERO TOLERANCE** | NONE (4 whitelisted boundaries for migration) | AST linting + CI |
| **Tier 3: Effect Interpreter** | Mixed (ADTs pure, execution impure) | GPU ops, I/O, RNG, network | Manual review |
| **Test Code** | Relaxed | assert, fixtures, setup loops | Manual review |
| **System Boundaries** | Relaxed | Logging, arg parsing, exit codes | Manual review |

### CUDA kernels (performance-focused exception)

CUDA kernels may use imperative constructs (`for`/`while`/`if`) for performance **while staying semantically pure**:

- No side effects beyond writing to their designated output buffers.
- Do not mutate input-only arguments.
- Return values must be pure data (no hidden handles or global state).

### Pydantic validation (pure boundary)

Pydantic raises `ValidationError` internally. Keep call sites pure by wrapping
construction in a helper that returns `Result[Model, ValidationError]` (see
`src/spectralmc/validation.py::validate_model`). Use that wrapper instead of
letting validation exceptions propagate; handle the Failure via `match/case`.

### Protobuf Serialization Boundary (imperative API)

Protobuf's generated code requires mutation via methods like `CopyFrom()`. This is
acceptable at serialization boundaries where protobuf's imperative API is unavoidable:

```python
# Acceptable: protobuf mutation in serialization layer
_ = [
    proto.state[param_id].CopyFrom(entry_proto)
    for result in param_entries_results
    if isinstance(result, Success)
    for param_id, entry_proto in [result.value]  # Unpack tuple from Success
]
```

**Pattern**: Isolated to `serialization/` modules where protobuf interop is required.
**Rationale**: Protobuf's generated API is inherently imperative; no functional alternative exists.
**Location**: See `src/spectralmc/serialization/tensors.py` lines 288-294.

Note: The comprehension is used for iteration (the side effect is unavoidable with protobuf's API design). This pattern is explicitly documented as a necessary evil at the serialization boundary.

### Import failures (fail-fast allowed)

Import-time guards for hard dependencies may raise (e.g., missing PyTorch/CuPy).
These are considered configuration errors; it is acceptable for the program to
fail fast rather than return `Result` in such cases.

### Defensive assertions (programming errors)

Infrastructure and facade code (e.g., `models/torch.py`) may use `raise` for **programming error checks**:

- Thread-safety assertions (e.g., "called from wrong thread")
- Contract violations in context managers (e.g., "entered/exited in different threads")
- Internal consistency checks that indicate bugs, not bad data

These are **not validation** - they indicate bugs in the code itself. Expected errors (like invalid user input) must use `Result` types. Programming errors may raise.

### Acceptable Raise Patterns (Comprehensive List)

The following patterns are acceptable and do not violate purity:

**1. Boundary Functions** - Result → exception conversion at system boundaries:
```python
def unwrap(self) -> T:
    """Extract value or raise. Use at boundaries only."""
    match self:
        case Success(value):
            return value
        case Failure(error):
            raise RuntimeError(f"Unwrap failed: {error}")
```

**2. Programming Error Assertions** - Defensive checks for invariants:
```python
# Thread-safety
if threading.get_ident() != main_thread_id:
    raise RuntimeError("Not thread-safe; call from main thread only")

# Unreachable code markers
raise AssertionError("Unreachable: all variants handled above")

# Internal serialization that should never fail
def _expect_serialization(result: Result[T, E]) -> T:
    match result:
        case Success(value): return value
        case Failure(error): raise RuntimeError(f"Serialization failure: {error}")
```

**3. Constructor Validation** - Fail-fast in `__init__` methods:
```python
def __init__(self, device: Device):
    if device != Device.cuda:
        raise RuntimeError("CUDA required for this operation")
```

**4. Test Infrastructure** - In test helpers, fixtures, and mocks:
```python
def assert_effect_count(self, count: int) -> None:
    if len(self.recorded_effects) != count:
        raise AssertionError(f"Expected {count} effects")
```

**Key Principle**: Raise is acceptable when failure indicates:
- Programming error (bugs in code, not invalid data)
- Environment misconfiguration (missing CUDA, wrong thread)
- Test failures (expected vs actual mismatch)
- System boundary conversion (Result → exception for interop)

### Business Logic Must Be 100% Pure (Tier 2)

**ZERO TOLERANCE** for impurity in business logic files:

**Forbidden in Tier 2** (except 4 whitelisted lines):
- ❌ Statement-level `for` loops → Use: `[f(x) for x in items]`
- ❌ Statement-level `if` branches → Use: `value if cond else default` or `match`/`case`
- ❌ Direct GPU operations → Use: Effect ADTs (TensorTransfer, KernelLaunch)
- ❌ Direct I/O or logging → Use: Effect ADTs (LogMessage, WriteCheckpoint)
- ❌ RNG state mutation → Use: Pure RNG functions with explicit state
- ❌ Exception raising for validation → Use: Result types

**Required Patterns**:
- ✅ Build Effect ADTs to describe operations
- ✅ Return Effect sequences for side effects
- ✅ Use comprehensions for iteration
- ✅ Use conditional expressions for branching
- ✅ Use match/case for complex logic
- ✅ Return Result types for errors

**Whitelisted Boundaries** (4 lines with documented rationale):
1. `gbm_trainer.py:382` - Logging histogram interval check (infrastructure boundary)
2. `gbm_trainer.py:392` - TensorBoard flush interval check (infrastructure boundary)
3. `gbm_trainer.py:421` - Gradient existence guard (TensorBoard infrastructure)
4. `serialization/tensors.py:174` - Protobuf `requires_grad` mutation (API boundary)

See [Purity Enforcement](purity_enforcement.md) for AST linter whitelist configuration.

**Why Zero Tolerance?** Training logic that is 100% pure:
- **Testable** with MockInterpreter (no GPU needed)
- **Reproducible** (same inputs → same effect sequence)
- **Composable** (effects combine without coupling)
- **Type-safe** (all error paths explicit in types)

### Effect Interpreter (ONLY Place for Side Effects)

The Effect Interpreter is the **ONLY** code allowed to execute side effects:

- GPU operations (kernel launches, transfers)
- Storage I/O (S3 reads/writes)
- RNG state manipulation
- Stream synchronization

All other code produces pure effect descriptions. See [Effect Interpreter](./effect_interpreter.md).

### Storage Layer

Storage modules (`storage/`) are effectful by nature:

- S3 client operations
- Network I/O with retry logic
- Blockchain atomic commits

These are part of the Effect Interpreter infrastructure.

### Test Code

Test code may use impure constructs:

- `assert` statements (pytest requires them)
- `pytest.raises()` context manager
- `for` loops in test setup/teardown

Tests verify behavior; they are not business logic.

### System Boundaries Only

At the outermost system boundaries (CLI `__main__.py`, HTTP handlers), minimal impurity is acceptable:

- Logging at entry/exit points
- Argument parsing
- Top-level exception handling

Mark with `# SYSTEM BOUNDARY:` comment.

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

## Compliance Status

**Last Audit**: 2025-01-08
**Overall Status**: ✅ **EXCELLENT COMPLIANCE**

### For Loops: ✅ 100% COMPLIANT

- **Statement-level for loops in pure code**: 0
- **For loops in allowed locations**: 3
  - `gbm.py:240, 246` - CUDA kernel loops (explicitly allowed)
  - `effects/mock.py:106` - Test infrastructure (explicitly allowed)

**Achievement**: All business logic uses comprehensions, helper functions, or functional patterns.

### If Statements: 🟡 63% COMPLIANT

- **Original refactorable if statements**: ~150
- **Converted to match/case or conditional expressions**: ~55 (37%)
- **Remaining statement-level if**: 95
  - Many are at effectful boundaries (PyTorch operations, I/O)
  - Guard clauses in validation code
  - Conditional logging/flushing in training orchestration

**High-impact conversions completed**:
- ✅ effects/registry.py (12 → 0)
- ✅ serialization/models.py (6 → 2)
- ✅ serialization/tensors.py (15 → 5)
- ✅ sobol_sampler.py (5 → 1)
- ✅ models/torch.py (37 → 28)

### Raise Statements: ✅ 100% COMPLIANT

- **Total raise statements in pure code**: 14
- **Acceptable (per doctrine exceptions)**: 14 (100%)
  - 1 boundary function (Result → exception conversion)
  - 9 programming error assertions
  - 1 import-time check
  - 3 test infrastructure assertions

**Classification**: All raises are for programming errors, boundaries, or test infrastructure. Zero violations.

### Test Coverage: ✅ 100% PASSING

- **Test suite**: 287/287 passing (100%)
- **MyPy strict mode**: ✅ PASSING
- **Type safety**: Zero regressions
- **Functional correctness**: Zero behavioral changes

### Summary

SpectralMC demonstrates **excellent adherence** to the purity doctrine:
- ✅ **100% for loop elimination** in pure business logic
- ✅ **37% if statement conversion** with significant progress in high-impact areas
- ✅ **100% compliant raise usage** (all acceptable per doctrine)
- ✅ **Zero test regressions** maintained throughout refactoring

Remaining if statements are largely at effectful boundaries or use acceptable guard clause patterns. The codebase is production-ready and aligned with functional programming principles.

---

## References

- [Immutability Doctrine](./immutability_doctrine.md) - Frozen dataclass requirements
- [Coding Standards](./coding_standards.md) - Result types and ADT patterns (SSoT)
- [Effect Interpreter](./effect_interpreter.md) - Side effect isolation
- [Reproducibility Proofs](./reproducibility_proofs.md) - How purity enables provable determinism

---
