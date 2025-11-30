# Immutability Doctrine

## Core Principle

**NEVER bypass immutability guarantees provided by the language or type system.**

Immutability is a contract, not a suggestion. Bypassing immutability protections creates runtime bugs that cannot be caught by static analysis.

---

## Forbidden Patterns

### ❌ 1. Using `object.__setattr__()` on Frozen Dataclasses

**Anti-pattern**:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelVersion:
    counter: int
    hash: str

version = ModelVersion(counter=1, hash="abc")
object.__setattr__(version, "counter", 999)  # ❌ FORBIDDEN
```

**Why forbidden**:
- Bypasses `frozen=True` protection
- Violates immutability contract
- Creates mutable state where code assumes immutability
- Cannot be detected by mypy or type checkers
- Undermines correctness guarantees (e.g., blockchain integrity)

**Correct approach**:
```python
# Create new instance instead
new_version = ModelVersion(counter=999, hash="abc")
```

### ❌ 2. Using `__dict__` Manipulation on Frozen Objects

**Anti-pattern**:
```python
version.__dict__["counter"] = 999  # ❌ FORBIDDEN
```

**Why forbidden**: Same reasons as `object.__setattr__()`

### ❌ 3. Using `vars()` to Mutate Frozen Objects

**Anti-pattern**:
```python
vars(version)["counter"] = 999  # ❌ FORBIDDEN
```

**Why forbidden**: Same reasons as `object.__setattr__()`

### ❌ 4. Bypassing Tuple Immutability

**Anti-pattern**:
```python
# Attempting to modify tuple internals (not possible in Python, but illustrates intent)
t = (1, 2, 3)
# Any attempt to bypass tuple immutability is forbidden
```

### ❌ 5. Mutating "Immutable" Collections via Internal References

**Anti-pattern**:
```python
from typing import FrozenSet

internal_list = [1, 2, 3]
frozen = frozenset(internal_list)
internal_list.append(4)  # ❌ FORBIDDEN - violates immutability assumption
```

---

## Allowed Patterns

### ✅ 1. Creating New Instances

```python
@dataclass(frozen=True)
class Config:
    timeout: int
    retries: int

old_config = Config(timeout=30, retries=3)
new_config = Config(timeout=60, retries=3)  # ✅ Correct
```

### ✅ 2. Using `dataclasses.replace()`

```python
from dataclasses import replace

old_config = Config(timeout=30, retries=3)
new_config = replace(old_config, timeout=60)  # ✅ Correct
```

### ✅ 3. Functional Updates

```python
def update_timeout(config: Config, new_timeout: int) -> Config:
    """Return new Config with updated timeout."""
    return Config(timeout=new_timeout, retries=config.retries)
```

---

## Enforcement

### Static Analysis

**mypy cannot catch `object.__setattr__()` bypasses**. Use code review and grep audits.

**Audit commands**:
```bash
# Search for forbidden patterns
grep -rn "object.__setattr__" src/ tests/ examples/
grep -rn "__dict__\[" src/ tests/ examples/ | grep -v "# OK:"
grep -rn "vars(" src/ tests/ examples/ | grep "="
```

### Testing Immutability

**Tests MUST NOT use bypass patterns** even for testing immutability:

**Anti-pattern** (FORBIDDEN):
```python
def test_immutability():
    with pytest.raises(Exception):
        object.__setattr__(frozen_obj, "field", new_value)  # ❌ WRONG
```

**Correct pattern**:
```python
from dataclasses import FrozenInstanceError

def test_immutability():
    with pytest.raises(FrozenInstanceError):
        frozen_obj.field = new_value  # ✅ Correct
```

---

## Rationale

### Why Immutability Matters

1. **Correctness**: Immutable objects cannot be accidentally modified
2. **Thread safety**: Immutable objects are safe to share across threads without locks
3. **Hashing**: Immutable objects can be safely used as dict keys and set members
4. **Reasoning**: Code is easier to understand when state cannot change unexpectedly
5. **Debugging**: Immutable objects eliminate an entire class of bugs

### Critical Use Cases in SpectralMC

#### 1. Blockchain Integrity (`ModelVersion`)

```python
@dataclass(frozen=True)
class ModelVersion:
    counter: int
    content_hash: str
    parent_hash: str
    # ...
```

- Versions form a Merkle chain
- Mutating a version breaks chain integrity
- Cannot detect corruption if versions can change
- **Impact**: Silent data corruption, undetectable tampering

#### 2. Configuration Objects

```python
@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    # ...
```

- Frozen configs prevent accidental modification
- Ensures consistency across application lifecycle
- **Impact**: Unpredictable behavior, hard-to-reproduce bugs

#### 3. Caching

```python
@dataclass(frozen=True)
class CacheKey:
    model_id: str
    version: int
```

- Immutable keys required for reliable caching
- Mutation invalidates cache assumptions
- **Impact**: Cache hits return wrong data, silent corruption

---

## Violations

### Reporting Violations

Report violations as bugs with **HIGH** severity.

**Bug Report Template**:
```
Title: Immutability bypass using object.__setattr__()
Severity: HIGH
File: src/spectralmc/storage/chain.py:42
Pattern: object.__setattr__(version, "counter", new_value)
Fix: Use dataclasses.replace() or create new instance
Reference: documents/engineering/immutability_doctrine.md
```

### Fixing Violations

#### In Production Code

Replace bypasses with functional update patterns:

**Before**:
```python
object.__setattr__(frozen_obj, "field", new_value)
```

**After**:
```python
from dataclasses import replace
frozen_obj = replace(frozen_obj, field=new_value)
```

#### In Test Code

Use normal assignment to trigger proper errors:

**Before**:
```python
with pytest.raises(Exception):
    object.__setattr__(frozen_obj, "field", new_value)
```

**After**:
```python
from dataclasses import FrozenInstanceError

with pytest.raises(FrozenInstanceError):
    frozen_obj.field = new_value
```

---

## Prevention

### Code Review Checklist

Before approving any PR:

- [ ] No `object.__setattr__()` on frozen dataclasses
- [ ] No `__dict__` manipulation on immutable objects
- [ ] No `vars()` mutation on frozen objects
- [ ] All immutability violations reported as HIGH severity bugs

### Automated Checks

**Grep audit** (run periodically):
```bash
# Returns exit code 1 if any violations found
if grep -r "object.__setattr__" src/; then
  echo "ERROR: Immutability bypass detected in src/"
  exit 1
fi
```

---

## References

- **PEP 557**: Data Classes (frozen parameter)
- **Python Data Model**: `__setattr__` special method
- **Functional Programming**: Immutability principles
- **SpectralMC Engineering Standards**: `documents/engineering/index.md`

---

**Last updated**: 2025-11-28
**Status**: Active doctrine, zero tolerance for violations
