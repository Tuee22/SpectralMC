# File: documents/engineering/purity_enforcement.md
# Purity Enforcement

**Status**: Authoritative source
**Supersedes**: None
**Referenced by**: purity_doctrine.md, coding_standards.md, code_quality.md

> **Purpose**: Comprehensive guide for automated purity enforcement in SpectralMC business logic via AST-based linting. Defines detection rules (PUR001-PUR005) and the check_purity.py CLI tool.

## Cross-References
- [Purity Doctrine](purity_doctrine.md)
- [Coding Standards](coding_standards.md)
- [Code Quality](code_quality.md)
- [Testing Requirements](testing_requirements.md)
- [Total Pure Modelling](total_pure_modelling.md)

---

## Overview

SpectralMC enforces **ZERO TOLERANCE** for purity violations in Tier 2 business logic
through automated AST-based linting. This document specifies the complete enforcement
system and keeps pure ADT flows aligned with [total_pure_modelling.md](total_pure_modelling.md).

### Purpose

- **Prevent regressions**: Detect purity violations via check_purity.py
- **Fast feedback**: Developers see violations in < 5 seconds locally
- **Clear guidance**: Error messages explain violations with before/after examples
- **Automated fixes**: Simple patterns (if → ternary) auto-fixable with `--fix` flag

### Scope

**Enforced**: Tier 2 business logic files only
**Exempt**: Tier 1 infrastructure, Tier 3 effect interpreters, test files
**Rules**: 5 AST-based detection rules (PUR001-PUR005)

---

## File Classification

### Tier 1: Infrastructure (EXEMPT from purity rules)

**Files**:
- `src/spectralmc/models/torch.py` - Torch runtime configuration for reproducibility
- `src/spectralmc/models/cpu_gpu_transfer.py` - Device transfer utilities
- `src/spectralmc/cvnn.py` - PyTorch nn.Module layer library

**Rationale**: Infrastructure provides deterministic foundation; may use imperative patterns for thread safety, nn.Module idioms, device guards.

### Tier 2: Business Logic (ZERO TOLERANCE PURITY)

**Files**:
- `src/spectralmc/cvnn_factory.py` - Model factory
- `src/spectralmc/sobol_sampler.py` - Sampling logic
- `src/spectralmc/serialization/*.py` - Data serialization
- `src/spectralmc/errors/*.py` - Error definitions
- **All other files in** `src/spectralmc/` (default)

**Enforcement**: AST linter with 5 rules (PUR001-PUR005)

### Tier 3: Effect Interpreter (EXEMPT from purity rules)

**Files**:
- `src/spectralmc/effects/interpreter.py` - Effect execution
- `src/spectralmc/effects/mock.py` - Test interpreter
- `src/spectralmc/storage/*.py` - Storage operations
- `src/spectralmc/gbm_trainer.py` - Mixed orchestrator (training loop + interpreter wiring)
- `src/spectralmc/__main__.py` - CLI entry point
- `src/spectralmc/storage/__main__.py` - Storage CLI

**Rationale**: Effect execution is impure by design (GPU ops, I/O, RNG mutation).

### Exempt Files

**Patterns**:
- `**/proto/*_pb2.py` - Generated protobuf code
- `**/test_*.py` - Test files (separate rules)
- `tests/**/*.py` - Test directory
- `tools/**/*.py` - Tooling scripts
- `stubs/**/*.py` - Type stubs

---

## AST Detection Rules

### Rule PUR001: No For Loops in Business Logic

**Detects**: `ast.For` nodes in Tier 2 files
**Error Message**: "for loop forbidden in business logic (PUR001). Use comprehension, map, or Effect sequence."
**Automatic Exemptions**: CUDA kernels (@cuda.jit decorated functions)

**Anti-pattern**:
```python
# File: documents/engineering/purity_enforcement.md
result = []
for item in items:
    result.append(item * 2)
return result
```

**Fix** (comprehension):
```python
# File: documents/engineering/purity_enforcement.md
return [item * 2 for item in items]
```

**Rationale**: for loops require mutable accumulators and iterative mutation, making code harder to reason about and test.

---

### Rule PUR002: No While Loops in Business Logic

**Detects**: `ast.While` nodes in Tier 2 files
**Error Message**: "while loop forbidden in business logic (PUR002). Use recursion or generator with next()."
**Automatic Exemptions**: CUDA kernels (@cuda.jit decorated functions)

**Anti-pattern**:
```python
# File: documents/engineering/purity_enforcement.md
i = 0
while i < len(items):
    if predicate(items[i]):
        return items[i]
    i += 1
return None
```

**Fix** (generator + next):
```python
# File: documents/engineering/purity_enforcement.md
return next((item for item in items if predicate(item)), None)
```

**Rationale**: while loops require mutable state and imperative control flow.

---

### Rule PUR003: No If Statements in Business Logic

**Detects**: `ast.If` nodes in Tier 2 files (excludes `ast.IfExp` - ternary operator)
**Error Message**: "if statement forbidden in business logic (PUR003). Use match/case or conditional expression (x if cond else y)."
**Automatic Exemptions**: Guard clauses returning Failure(...), CUDA kernels (@cuda.jit decorated functions)

**Whitelisted Exceptions** (2 total):
```python
# File: documents/engineering/purity_enforcement.md
ACCEPTABLE_IF_STATEMENTS = {
    ("src/spectralmc/async_normals.py", 311): "RNG advance guard (checkpoint resume)",
    ("src/spectralmc/serialization/tensors.py", 174): "Protobuf API boundary",
}
```

**Anti-pattern** (simple if/else):
```python
# File: documents/engineering/purity_enforcement.md
if isinstance(value, str):
    return int(value)
return value
```

**Fix** (conditional expression):
```python
# File: documents/engineering/purity_enforcement.md
return int(value) if isinstance(value, str) else value
```

**Anti-pattern** (complex branching):
```python
# File: documents/engineering/purity_enforcement.md
if isinstance(error, NetworkError):
    return "Network failure"
elif isinstance(error, AuthError):
    return "Auth failure"
else:
    return "Unknown failure"
```

**Fix** (match/case):
```python
# File: documents/engineering/purity_enforcement.md
match error:
    case NetworkError():
        return "Network failure"
    case AuthError():
        return "Auth failure"
    case _:
        return "Unknown failure"
```

**Rationale**: Statement-level if requires multiple return points and is not expression-oriented. Conditional expressions (`x if cond else y`) are pure and encouraged.

**Exception Pattern** (guard clauses):
Guard clauses that return `Failure(...)` are acceptable:
```python
# File: documents/engineering/purity_enforcement.md
if error_condition:
    return Failure(SomeError(...))
# Continue processing...
```

---

### Rule PUR004: No Raise for Expected Errors

**Detects**: `ast.Raise` nodes in Tier 2 files (except in `assert_never()` calls)
**Error Message**: "raise forbidden for expected errors (PUR004). Return Result[T, E] type."
**Automatic Exemptions**: Boundary unwrap functions (method named 'unwrap' with NoReturn annotation)

**Whitelisted Patterns**:
- `raise AssertionError(...)` with keywords: "unreachable", "invariant", "programming error"
- `raise RuntimeError(...)` with keywords: "thread", "invariant"
- `raise TypeError(...)` with keywords: "invariant"
- Inside `assert_never()` calls

**Anti-pattern**:
```python
# File: documents/engineering/purity_enforcement.md
def __post_init__(self) -> None:
    if self.lower >= self.upper:
        raise ValueError("Invalid bounds")
```

**Fix** (factory function):
```python
# File: documents/engineering/purity_enforcement.md
def bound_spec(lower: float, upper: float) -> Result[BoundSpec, InvalidBoundsError]:
    return (
        Failure(InvalidBoundsError(lower=lower, upper=upper))
        if lower >= upper
        else Success(BoundSpec(lower=lower, upper=upper))
    )
```

**Rationale**: Exceptions break control flow and hide failure modes from type system. Result types enable static type checking of error paths.

**Acceptable Raise** (programming errors):
```python
# File: documents/engineering/purity_enforcement.md
# Thread safety violation
if threading.get_ident() != main_thread_id:
    raise RuntimeError("Wrong thread - programming error")

# Unreachable code
raise AssertionError("Unreachable: all variants handled")
```

---

### Rule PUR005: No Side Effects in Business Logic

**Detects**: Calls to `print()`, `logger.*()` in Tier 2 files
**Error Message**: "Side effect forbidden in business logic (PUR005). Use Effect ADT."

**Forbidden Functions**:
- `print(...)`
- `logger.debug(...)`, `logger.info(...)`, `logger.warning(...)`, `logger.error(...)`, `logger.critical(...)`

**Anti-pattern**:
```python
# File: documents/engineering/purity_enforcement.md
def compute_price(model: Model, params: Params) -> float:
    logger.info(f"Computing price with {params}")
    print(f"Model: {model.name}")
    return model.forward(params)
```

**Fix** (pure computation):
```python
# File: documents/engineering/purity_enforcement.md
def compute_price(model: Model, params: Params) -> float:
    """Pure computation - no logging, no I/O."""
    return model.forward(params)
```

**For debugging/logging**, use Effect ADT (see [Effect Interpreter](effect_interpreter.md)):
```python
# File: documents/engineering/purity_enforcement.md
def compute_with_logging(model: Model, params: Params) -> tuple[float, list[LoggingEffect]]:
    """Pure computation returning effects."""
    price = model.forward(params)
    effects = [LogMessage(level="info", message=f"Price computed: {price}")]
    return price, effects
```

**Rationale**: Side effects make code non-deterministic and hard to test. Effect ADTs enable pure business logic with testable effect descriptions.

---

## Automatic Pattern Recognition

The purity checker automatically recognizes documented exception patterns, eliminating false positives from acceptable code constructs.

### Overview

Instead of manually whitelisting every guard clause or CUDA kernel, the AST checker detects these patterns automatically through structural analysis. This provides:

- **Zero false positives** for documented exception patterns
- **No line number maintenance** (patterns recognized structurally, not by location)
- **Clear separation** between violations and acceptable patterns
- **Developer-friendly output** (violations require action, exemptions are silent)

### Recognized Patterns

| Pattern | Detection Method | Example | Rule(s) Affected |
|---------|------------------|---------|------------------|
| Guard clause | Single-statement `if` returning `Failure(...)` | `if x < 0: return Failure(...)` | PUR003 |
| CUDA kernel | `@cuda.jit` decorator on parent function | `@cuda.jit def kernel(...): ...` | PUR001, PUR002, PUR003 |
| Boundary unwrap | Method named `unwrap` with `NoReturn` annotation | `def unwrap(self) -> NoReturn: raise ...` | PUR004 |

### Pattern Details

#### Guard Clause Detection (PUR003)

**Pattern**: Single-statement `if` body that returns `Failure(...)`.

**Rationale**: Guard clauses are the idiomatic way to perform early validation before proceeding with the main logic. Per purity doctrine lines 180-186, these are acceptable.

**Example (automatically exempted)**:
```python
# File: src/spectralmc/async_normals.py
def create(cls, rows: int, cols: int, *, dtype: cp.dtype) -> Result[...]:
    # ✅ CORRECT - Guard clause (automatically exempted)
    if min(rows, cols) <= 0:
        return Failure(InvalidShape(rows=rows, cols=cols))

    # Continue with main logic
    return Success(cls(rows, cols, dtype=dtype))
```

**Detection algorithm**:
1. Check if node is `ast.If`
2. Verify body has exactly one statement
3. Verify statement is `ast.Return`
4. Verify return value is `Failure(...)` call

**Counter-example (still flagged)**:
```python
# File: src/spectralmc/example.py (hypothetical)
# ❌ WRONG - Not a guard clause (mutates state)
if x < 0:
    x = 0  # Mutation, not early return
return x * 2
```

#### CUDA Kernel Detection (PUR001, PUR002, PUR003)

**Pattern**: Any `for`/`while`/`if` inside function decorated with `@cuda.jit`.

**Rationale**: CUDA kernels are Tier 3 GPU boundaries. Imperative patterns (`if`, `for`, `while`) are necessary for GPU efficiency and cannot be refactored to pure functional equivalents without severe performance penalties.

**Example (automatically exempted)**:
```python
# File: src/spectralmc/gbm.py
from numba import cuda

# ✅ CORRECT - CUDA kernel (all imperative patterns automatically exempted)
@cuda.jit
def SimulateBlackScholes(io, r, d, v, dt, sqrt_dt, timesteps, simulate_log_return):
    idx = cuda.grid(1)
    if idx < io.shape[1]:  # Thread boundary check (exempt)
        X = io[0, idx]
        if simulate_log_return:  # Algorithm branch (exempt)
            drift = r - d - 0.5 * v * v
            for i in range(timesteps):  # Sequential loop (exempt)
                dW = io[i, idx] * sqrt_dt
                X *= exp(drift * dt + v * dW)
                io[i, idx] = X
```

**Detection algorithm**:
1. Walk up AST tree from violation node
2. Find first parent `ast.FunctionDef`
3. Check function's `decorator_list` for `@cuda.jit`
4. Recognizes both forms: `@cuda.jit` and `@cuda.jit(...)`

**Scope**: Exempts ALL purity rules inside CUDA kernels (PUR001, PUR002, PUR003).

#### Boundary Unwrap Detection (PUR004)

**Pattern**: `raise` statement inside method named `unwrap` with `NoReturn` return annotation.

**Rationale**: Boundary unwrap functions convert `Result → Exception` at system boundaries where exceptions are expected (e.g., scripting interfaces, integration with exception-based APIs).

**Example (automatically exempted)**:
```python
# File: src/spectralmc/result.py
from typing import NoReturn

@dataclass(frozen=True)
class Failure:
    error: E

    # ✅ CORRECT - Boundary unwrap (automatically exempted)
    def unwrap(self) -> NoReturn:
        """Convert Result to exception at system boundary."""
        raise RuntimeError(f"Called unwrap() on Failure: {self.error}")
```

**Detection algorithm**:
1. Walk up AST tree from `raise` node
2. Find first parent `ast.FunctionDef`
3. Check if function name is `"unwrap"`
4. Check if return annotation is `NoReturn`

**Counter-example (still flagged)**:
```python
# File: src/spectralmc/example.py (hypothetical)
# ❌ WRONG - Not a boundary unwrap (raises for expected error)
def process(x: int) -> int:
    if x < 0:
        raise ValueError("negative")  # Expected error, should use Result type
    return x
```

### Implementation Notes

**Performance**: Pattern recognition adds negligible overhead (< 5% to AST traversal time).

**Maintainability**: All detection logic is in `tools/purity/rules.py`:
- `_is_guard_clause()` - Guard clause detection
- `_is_cuda_jit_decorator()` - CUDA decorator recognition
- `_is_cuda_kernel()` - CUDA kernel boundary detection
- `_is_boundary_unwrap()` - Boundary unwrap function detection

**Testing**: Pattern detection is tested via integration tests (run full codebase, verify 0 violations).

**False Negatives**: Pattern detection is intentionally narrow to avoid false negatives. If a pattern doesn't match exactly (e.g., guard clause with multiple statements), it will be flagged. This is safe: the developer sees the violation and can either refactor or add a whitelist entry.

---

## Tool Implementation

### check_purity.py CLI

**Location**: `tools/check_purity.py`

**Usage**:
```bash
# File: documents/engineering/purity_enforcement.md
# Check all Tier 2 files
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-purity

# Check specific files
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-purity src/spectralmc/cvnn_factory.py

# Auto-fix simple violations (if → ternary)
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-purity --fix

# Explain a rule
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-purity --explain PUR003

# Verbose output (show file classification)
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-purity --verbose
```

**Exit Codes**:
- `0`: No violations found
- `1`: Violations found or error occurred

### Integration with check-code

The purity checker runs as **Step 3** in the code quality pipeline:

```bash
# File: documents/engineering/purity_enforcement.md
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code
```

**Pipeline**:
1. **Ruff** - Linting with auto-fix
2. **Black** - Code formatting
3. **Purity Checker** ← NEW (after Black, before MyPy)
4. **MyPy** - Type checking

---

## Code Review Checklist

### Verification Checks (run locally by developer)

- [ ] `poetry run check-purity` passes (zero violations)
- [ ] `poetry run check-code` passes (Ruff + Black + Purity + MyPy)
- [ ] `poetry run test-all` passes (all 227 tests)

### Manual Verification (reviewer)

#### Purity Verification
- [ ] No for loops in Tier 2 (verified by linter)
- [ ] No statement-level if outside whitelist (verified by linter)
- [ ] No while loops (verified by linter)
- [ ] All raise statements are defensive assertions, not validation errors
- [ ] No print() or logger.*() calls in business logic

#### Functional Correctness
- [ ] Side effects modeled as Effect ADTs
- [ ] Factory functions return Result for validation
- [ ] Comprehensions have no side effects
- [ ] Match/case uses exhaustive patterns with assert_never()
- [ ] Result types used for all expected errors

#### Type Safety
- [ ] MyPy strict mode passes
- [ ] No Any, cast(), or type: ignore
- [ ] All functions fully typed (parameters and return)

---

## Migration Path

### Current Whitelisted Violations (2 total)

These patterns exist in current code and are documented for migration:

1. **async_normals.py:311** - RNG advance guard (checkpoint resume)
   ```python
   # File: src/spectralmc/async_normals.py
   if rerun_step <= checkpoint_step:  # WHITELISTED
       return Failure(RNGAdvanceError(...))
   ```
   **Migration**: Consider expression-only guard or restructure state handling

2. **serialization/tensors.py:174** - Protobuf requires_grad mutation
   ```python
   # File: src/spectralmc/serialization/tensors.py
   if proto.requires_grad:  # WHITELISTED
       tensor.requires_grad_(True)
   ```
   **Migration**: Conditional expression:
   ```python
   # File: src/spectralmc/serialization/tensors.py
   tensor = (
       tensor.requires_grad_(True)
       if proto.requires_grad
       else tensor
   )
   ```

### Whitelist Configuration

**Location**: `tools/purity/rules.py`

```python
# File: documents/engineering/purity_enforcement.md
ACCEPTABLE_IF_STATEMENTS: dict[tuple[str, int], str] = {
    ("src/spectralmc/async_normals.py", 311): "RNG advance guard (checkpoint resume)",
    ("src/spectralmc/serialization/tensors.py", 174): "Protobuf API boundary",
}
```

**Update Policy**: Whitelist entries should be removed as violations are fixed. New entries require documentation and approval.

---

## Automated Remediation

### Auto-Fixable Patterns

#### Pattern 1: Simple If/Else → Ternary

**Before**:
```python
# File: documents/engineering/purity_enforcement.md
if condition:
    return value_a
else:
    return value_b
```

**After** (`--fix` applies automatically):
```python
# File: documents/engineering/purity_enforcement.md
return value_a if condition else value_b
```

**Requirements**:
- Single statement in if branch (return or assign)
- Single statement in else branch (return or assign)
- No complex logic in condition

#### Pattern 2: Simple If/Else Assignment → Ternary

**Before**:
```python
# File: documents/engineering/purity_enforcement.md
if condition:
    x = value_a
else:
    x = value_b
```

**After** (`--fix` applies automatically):
```python
# File: documents/engineering/purity_enforcement.md
x = value_a if condition else value_b
```

### Manual Refactoring Required

#### Pattern 3: For Loop → Comprehension

**Before**:
```python
# File: documents/engineering/purity_enforcement.md
result = []
for item in items:
    result.append(transform(item))
return result
```

**After** (manual):
```python
# File: documents/engineering/purity_enforcement.md
return [transform(item) for item in items]
```

**Reason**: Requires understanding of accumulator pattern; too risky for auto-fix.

#### Pattern 4: Complex If → Match/Case

**Before**:
```python
# File: documents/engineering/purity_enforcement.md
if isinstance(x, TypeA):
    return handle_a(x)
elif isinstance(x, TypeB):
    return handle_b(x)
else:
    return handle_default(x)
```

**After** (manual):
```python
# File: documents/engineering/purity_enforcement.md
match x:
    case TypeA():
        return handle_a(x)
    case TypeB():
        return handle_b(x)
    case _:
        return handle_default(x)
```

**Reason**: Requires understanding of type hierarchy and pattern matching.

---

## Success Metrics

### Compliance Tracking

**Command**:
```bash
# File: documents/engineering/purity_enforcement.md
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-purity --verbose
```

**Metrics**:
1. **Violation Count by Rule**:
   - PUR001 (for loops): Target 0
   - PUR002 (while loops): Target 0
   - PUR003 (if statements): Target 2 (whitelisted only)
   - PUR004 (raise): Target 0 (except defensive assertions)
   - PUR005 (side effects): Target 0

2. **Purity Compliance Rate**:
   - Formula: `(clean_files / total_tier2_files) * 100%`
   - Current: 100% (2 whitelisted exceptions documented)
   - Target: 100% with 0 whitelist entries

3. **Auto-fix Success Rate**:
   - Track violations fixed automatically vs manually
   - Monitor `--fix` flag adoption

### Weekly Compliance Report

**Generate report**:
```bash
# File: documents/engineering/purity_enforcement.md
docker compose -f docker/docker-compose.yml exec spectralmc python tools/generate_purity_report.py
```

**Report includes**:
- Total violations by rule
- Violations by file
- Whitelist status (entries added/removed)
- Trend analysis (violations over time)

---

## Troubleshooting

### False Positives

**Issue**: Linter flags code that should be exempt

**Solution**: Verify file is correctly classified
```bash
# File: documents/engineering/purity_enforcement.md
poetry run check-purity --verbose src/spectralmc/your_file.py
```

**Check tier classification** in `pyproject.toml`:
```toml
# File: documents/engineering/purity_enforcement.md
[tool.purity]
tier1_infrastructure = [...]
tier3_effects = [...]
```

### Auto-Fix Not Working

**Issue**: `--fix` flag doesn't modify file

**Reason**: Pattern is too complex for auto-fix

**Solution**: Manually refactor using suggested fix from error message:
```bash
# File: documents/engineering/purity_enforcement.md
poetry run check-purity --explain PUR003
```

---

## Automation Policy

- No git hooks, `.pre-commit-config.yaml`, or CI/CD workflows are permitted for purity enforcement.
- Run `poetry run check-purity` manually inside Docker; share outputs alongside code reviews.
- See [User Automation Reference](user_automation_reference.md) for the explicit prohibition and manual command list.

---

## References

- [Purity Doctrine](purity_doctrine.md) - Complete purity standards
- [Coding Standards](coding_standards.md) - Result types and ADT patterns
- [Effect Interpreter](effect_interpreter.md) - Effect ADT system
- [Testing Requirements](testing_requirements.md) - Test purity exceptions

---
