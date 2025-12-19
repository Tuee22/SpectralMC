# File: documents/engineering/warnings_policy.md
# Warnings Policy

**Status**: Authoritative source
**Supersedes**: coding_standards.md (deprecation warnings sections)
**Referenced by**: CLAUDE.md, coding_standards.md, testing_requirements.md, code_quality.md

> **Purpose**: Zero-tolerance warnings policy for SpectralMC across Python, PyTorch, CuPy, NumPy, Pydantic, Docker, and development tools.

## TL;DR

- **Zero warnings** = universal success criterion for all commits
- **Never suppress** warnings with filters, flags, or log level changes
- **Always upgrade** to latest/best supported APIs when deprecation warnings appear
- **Fix root causes** instead of masking symptoms
- **Rare exception**: External library bugs with verified upstream issues (strict criteria apply)

## Philosophy

Warnings are not noise—they indicate real problems that require fixes:

1. **Warnings signal technical debt**: Deprecated APIs will break in future versions
2. **Warnings predict failures**: Today's warning is tomorrow's runtime error
3. **Warnings enable upgrades**: Zero warnings means we can safely upgrade dependencies
4. **Warnings improve quality**: Code without warnings is more maintainable and debuggable

**Core Principle**: If the code emits a warning, the code is wrong. Fix the code.

## Coverage Areas

This policy applies to all warning sources:

### Python
- `DeprecationWarning` - API scheduled for removal
- `FutureWarning` - Behavior changes coming
- `PendingDeprecationWarning` - First-stage deprecation
- `RuntimeWarning` - Suspicious runtime behavior
- `ImportWarning` - Import system issues
- `ResourceWarning` - Unclosed files/connections
- Any custom warnings from libraries

### PyTorch

**CUDA Warnings**:
- Driver compatibility warnings → upgrade CUDA drivers
- Device placement warnings → verify tensor devices match
- Memory warnings → reduce batch size or implement gradient checkpointing
- CUDA context errors → restart application, check for context leaks

**Autograd Warnings**:
- In-place operations on leaf variables → use `.clone()` before modification
- Backward hook warnings → verify hook signatures match current PyTorch API
- Gradient checkpointing warnings → update to supported checkpoint API

**dtype Warnings**:
- Implicit type conversions → make conversions explicit
- Mixed precision warnings → verify autocast usage is correct
- Type mismatch in operations → ensure consistent dtypes

**Memory Warnings**:
- OOM (Out of Memory) → reduce batch size, enable gradient accumulation
- Memory fragmentation → clear cache with `torch.cuda.empty_cache()`
- Pinned memory warnings → verify data loader `pin_memory` usage

**Policy**: GPU warnings indicate real hardware issues. Never suppress. Let CUDA errors fail loudly.

### CuPy

**CUDA Compatibility**:
- CUDA version mismatch → install correct CuPy version for your CUDA
- Driver warnings → upgrade NVIDIA drivers
- Compute capability warnings → verify GPU is supported

**DLPack Deprecation** (RESOLVED):
- Old API: `cupy_array.toDlpack()` + `torch.utils.dlpack.from_dlpack()`
- Modern API: `torch.from_dlpack(cupy_array)` (PyTorch 1.10+, CuPy 9.0+)
- Status: All SpectralMC code migrated to modern API

**Memory Pool**:
- Memory pool warnings → check allocator configuration
- Stream warnings → verify CUDA stream management

### NumPy

**Numerical Warnings**:
- `RuntimeWarning: divide by zero` → use `torch.where()` or explicit zero handling
- `RuntimeWarning: invalid value encountered` → check for NaN/Inf inputs
- `RuntimeWarning: overflow encountered` → use appropriate dtypes or clipping

**Type Deprecation** (NumPy 2.0):
- Removed aliases: `np.float`, `np.int`, `np.complex_`, `np.bool`
- Use explicit types: `np.float64`, `np.int64`, `np.complex128`, `np.bool_`
- Status: All SpectralMC code uses explicit types

**Comparison Warnings**:
- Element-wise comparison warnings → use appropriate comparison methods
- Dtype comparison warnings → ensure compatible types

### Pydantic

**Serialization Warnings**:
- `PydanticSerializationUnexpectedValue` → add `field_serializer` for custom types
- Type coercion warnings → verify model field types match input types
- Validation warnings → check field constraints and validators

**Common Issue - MappingProxyType**:
```python
# File: documents/engineering/warnings_policy.md
# ❌ WRONG - MappingProxyType causes serialization warning
from types import MappingProxyType
from pydantic import BaseModel
from typing import Mapping

class OptimizerState(BaseModel):
    param_states: Mapping[int, ParamState]  # Frozen as MappingProxyType

    @model_validator(mode="after")
    def freeze_collections(self):
        object.__setattr__(self, "param_states",
            MappingProxyType(dict(self.param_states)))
        return self

# Serialization: opt.model_dump() → Warning about mappingproxy

# ✅ CORRECT - Add field_serializer
from pydantic import BaseModel, field_serializer, model_validator
from typing import Mapping
from types import MappingProxyType

class OptimizerState(BaseModel):
    param_states: Mapping[int, ParamState]

    @model_validator(mode="after")
    def freeze_collections(self):
        object.__setattr__(self, "param_states",
            MappingProxyType(dict(self.param_states)))
        return self

    @field_serializer("param_states", when_used="always")
    def serialize_param_states(
        self, value: Mapping[int, ParamState]
    ) -> dict[int, ParamState]:
        """Convert MappingProxyType to dict for serialization."""
        return dict(value)
```

**Validation Warnings**:
- Field validation failures → check input data matches model schema
- Custom validator warnings → verify validator return types
- Strict mode warnings → enable strict mode for better type safety

### Docker
- `DEPRECATED` instruction warnings
- Base image deprecation notices
- Layer cache warnings
- Build argument warnings

### Development Tools
- MyPy warnings
- Ruff warnings (should be errors via configuration)
- Black formatting warnings
- pytest collection warnings

### Runtime
- GPU driver warnings
- CUDA context warnings
- Numerical runtime warnings (NaN, Inf, overflow)
- Resource leaks (unclosed files, connections)

## Policy by Category

### 1. Deprecation Warnings

**Policy**: Always upgrade to the latest/best supported API.

**Process**:
1. Read the deprecation message carefully—it usually explains the replacement
2. Check the library's documentation or changelog for migration guidance
3. Update your code to use the new API
4. Test thoroughly to ensure behavior is unchanged
5. Remove any workarounds that existed for the old API

**Example - DLPack Migration** (COMPLETED):
```python
# File: documents/engineering/warnings_policy.md
# ❌ DEPRECATED (removed in CuPy 14+)
import cupy as cp
import torch

cupy_array = cp.array([1.0, 2.0, 3.0])
capsule = cupy_array.toDlpack()
torch_tensor = torch.utils.dlpack.from_dlpack(capsule)

# ✅ MODERN (PyTorch 1.10+, CuPy 9.0+)
import cupy as cp
import torch

cupy_array = cp.array([1.0, 2.0, 3.0])
torch_tensor = torch.from_dlpack(cupy_array)
```

**Example - NumPy Type Aliases** (COMPLETED):
```python
# File: documents/engineering/warnings_policy.md
# ❌ REMOVED in NumPy 2.0
import numpy as np
from typing import Type

float_type: Type[np.float]        # Don't use
int_type: Type[np.int]            # Don't use
complex_type: Type[np.complex_]   # Don't use

# ✅ CORRECT (NumPy 2.0+)
import numpy as np
from typing import Type

float_type: Type[np.float64]
int_type: Type[np.int64]
complex_type: Type[np.complex128]
```

**Example - Python asyncio**:
```python
# File: documents/engineering/warnings_policy.md
# Warning: "asyncio.coroutine is deprecated, use async def"

# ❌ DEPRECATED
import asyncio

@asyncio.coroutine
def fetch_data():
    result = yield from make_request()
    return result

# ✅ CORRECT
async def fetch_data():
    result = await make_request()
    return result
```

### 2. Docker Build Warnings

**Policy**: Fix warnings instead of muting them.

**Common Issues**:
- `DEPRECATED` Dockerfile instructions → use current syntax
- Legacy base image versions → update to latest stable
- Missing `--platform` flag → add explicit platform

**Example**:
```dockerfile
# File: documents/engineering/warnings_policy.md
# ❌ WRONG - Using deprecated MAINTAINER
MAINTAINER dev@spectralmc.com

# ✅ CORRECT - Use LABEL
LABEL maintainer="dev@spectralmc.com"
```

### 3. Type Checker Warnings

**Policy**: Add proper type annotations or fix type mismatches.

**MyPy Philosophy**: If MyPy emits a warning, the types are incomplete or incorrect.

**Example**:
```python
# File: documents/engineering/warnings_policy.md
# ❌ WRONG - Silencing MyPy warning
from typing import Any

result: Any = dangerous_operation()  # type: ignore

# ✅ CORRECT - Fix the type annotation
from typing import Result, Success, Failure

result: Result[str, Error] = dangerous_operation()
match result:
    case Success(value):
        print(value)
    case Failure(error):
        handle_error(error)
```

### 4. GPU Warnings

**Policy**: GPU warnings indicate real hardware issues. Never suppress. Let CUDA errors fail loudly.

**Common Issues**:
- CUDA driver compatibility → upgrade drivers
- Device memory allocation → reduce batch size
- CUDA context errors → restart application
- Silent CPU fallback → FORBIDDEN in SpectralMC

**SpectralMC GPU Requirement**:
```python
# File: documents/engineering/warnings_policy.md
# ❌ WRONG - Silent CPU fallback masks GPU unavailability
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.randn(1000, 1000, device=device)

# ✅ CORRECT - Explicit GPU requirement (fail fast if CUDA missing)
import torch

assert torch.cuda.is_available(), "CUDA required for SpectralMC"
device = torch.device("cuda:0")
tensor = torch.randn(1000, 1000, device=device)
```

**CUDA Memory**:
```python
# File: documents/engineering/warnings_policy.md
# ❌ WRONG - Catching CUDA OOM silently
import torch

try:
    large_tensor = torch.randn(100000, 100000, device="cuda:0")
except RuntimeError:
    # Silent fallback hides memory issues
    large_tensor = torch.randn(100000, 100000, device="cpu")

# ✅ CORRECT - Let CUDA errors fail, or reduce batch size explicitly
import torch

# Option 1: Fail loudly
large_tensor = torch.randn(100000, 100000, device="cuda:0")

# Option 2: Explicit batch size reduction with logging
try:
    large_tensor = torch.randn(100000, 100000, device="cuda:0")
except RuntimeError as e:
    if "out of memory" in str(e):
        logger.error("CUDA OOM: reduce batch size")
        raise  # Re-raise, don't suppress
    raise
```

### 5. Pydantic Warnings

**Policy**: Pydantic warnings indicate incomplete model definitions or serialization issues.

**Serialization - MappingProxyType Example**:

See full example in Coverage Areas > Pydantic section above.

**Root Cause**: SpectralMC uses `MappingProxyType` to enforce immutability on Pydantic models. When serializing with `model_dump(mode="python")`, Pydantic expects `dict` but encounters `MappingProxyType`, triggering a warning.

**Solution**: Add `@field_serializer` to convert `MappingProxyType` back to `dict` during serialization.

**Validation Warnings**:
```python
# File: documents/engineering/warnings_policy.md
# ❌ WRONG - Ignoring validation warnings
from pydantic import BaseModel, field_validator
import warnings

warnings.filterwarnings("ignore", module="pydantic")

class TrainingConfig(BaseModel):
    learning_rate: float

    @field_validator("learning_rate")
    def validate_lr(cls, v):
        # Returns wrong type, causes warning
        return str(v)

# ✅ CORRECT - Fix validator return type
from pydantic import BaseModel, field_validator

class TrainingConfig(BaseModel):
    learning_rate: float

    @field_validator("learning_rate")
    @classmethod
    def validate_lr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v
```

### 6. Test Warnings

**Policy**: Fix the code being tested, not the test configuration.

**Common Issues**:
- Unclosed resources → add proper cleanup
- Deprecated test fixtures → update fixture definitions
- Import path warnings → reorganize imports

**Example**:
```python
# File: documents/engineering/warnings_policy.md
# ❌ WRONG - Suppressing test warnings
import pytest

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_feature():
    result = old_deprecated_api()
    assert result is not None

# ✅ CORRECT - Fix the code causing the warning
def test_feature():
    result = new_supported_api()  # Use modern API
    assert result is not None
```

### 7. Runtime Warnings

**Policy**: Fix the runtime behavior causing the warning.

**Common Issues**:
- Unclosed connections → use context managers
- Resource leaks → add proper cleanup
- Numerical warnings → fix math operations

**Example - Resource Leaks**:
```python
# File: documents/engineering/warnings_policy.md
# ❌ WRONG - Ignoring ResourceWarning
import json

def load_data():
    f = open("data.json")
    return json.load(f)  # File never closed

# ✅ CORRECT - Use context manager
import json
from pathlib import Path

def load_data():
    with open("data.json") as f:
        return json.load(f)
```

**Example - Numerical Warnings**:
```python
# File: documents/engineering/warnings_policy.md
# ❌ WRONG - Suppressing divide by zero warning
import warnings
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)
result = a / b  # Can produce Inf/NaN

# ✅ CORRECT - Fix the math
import torch

result = torch.where(b != 0, a / b, torch.zeros_like(a))
```

**Example - NumPy Invalid Value**:
```python
# File: documents/engineering/warnings_policy.md
# ❌ WRONG - Ignoring invalid value warning
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered")
result = np.sqrt(negative_array)  # Produces NaN

# ✅ CORRECT - Handle negative values explicitly
import numpy as np

result = np.where(array >= 0, np.sqrt(array), 0.0)
# Or use complex dtype if negative sqrt is intended:
result = np.sqrt(array.astype(np.complex128))
```

## How to Fix Warnings

### Step-by-Step Process

1. **Reproduce the warning locally**
   - Ensure you can see the warning in your development environment
   - Note the exact warning message and context

2. **Identify the source**
   - Is this from our code or an external library?
   - Check the stack trace to find the origin

3. **Understand the cause**
   - Read the warning message carefully
   - Check documentation for the deprecated API
   - Search the library's GitHub issues if unclear

4. **Implement the fix**
   - For deprecations: upgrade to the new API
   - For type warnings: add/fix type annotations
   - For runtime warnings: fix the behavior
   - For Docker warnings: update Dockerfile syntax

5. **Verify the fix**
   - Re-run the command that produced the warning
   - Confirm the warning no longer appears
   - Ensure functionality is preserved

6. **Commit with clear message**
   - Explain what was deprecated and what replaced it
   - Example: "fix: upgrade from DLPack capsule API to torch.from_dlpack"

## Exceptions (Rare)

**Default stance**: Fix or work around all warnings. Suppression is a last resort.

### When Suppression is Permitted

**ONLY** when all of the following conditions are met:

1. **Verified external**: The warning originates from a third-party library, not our code
2. **Beyond our control**: We cannot fix it by changing our usage of the library
3. **Upstream issue exists**: There is an open issue on the library's GitHub tracker
4. **Documented**: The suppression includes a comment with the issue URL
5. **Monthly review**: Add a reminder to check if the upstream issue is resolved

### Current Exceptions

SpectralMC has **4 documented exceptions** in `pyproject.binary.toml`:

| Warning | Library | Status | Review | Upstream Issue |
|---------|---------|--------|--------|----------------|
| datetime.utcnow() | botocore/aiobotocore | Pending | Monthly | boto/botocore#3201 |
| SWIG builtin type | QuantLib | Permanent | N/A | Unfixable (generated code) |
| IFFT imaginary component | SpectralMC | Expected | N/A | Untrained models only |
| Numba grid size | Numba | Expected | N/A | Small test datasets only |

### How to Document Exceptions

**pytest filterwarnings** (preferred for test warnings):
```toml
# File: documents/engineering/warnings_policy.md
# pyproject.binary.toml
[tool.pytest.ini_options]
filterwarnings = [
    # botocore datetime.utcnow() - AWS SDK internal code
    # Upstream: https://github.com/boto/botocore/issues/3201
    # Review: Monthly (1st of month)
    "ignore::DeprecationWarning:botocore.*",
    "ignore::DeprecationWarning:aiobotocore.*",

    # QuantLib SWIG bindings - unfixable generated code
    # Status: Permanent exception
    # Pattern matches SWIG type definition warnings from importlib
    "ignore:builtin type .* has no __module__ attribute:DeprecationWarning:.*importlib.*",

    # IFFT imaginary component - expected for untrained models
    # Status: Test-only (production models have trained IFFT)
    "ignore:IFFT imaginary component.*exceeds tolerance:RuntimeWarning",

    # Numba grid size - expected for small test datasets
    # Status: Test-only (production uses large grids without occupancy issues)
    "ignore:Grid size.*will likely result in GPU under-utilization::numba.cuda.dispatcher",
]
```

**Python warnings module** (use sparingly):
```python
# File: documents/engineering/warnings_policy.md
# At module level, with full documentation
import warnings

# External library bug: https://github.com/org/lib/issues/456
# Review: Monthly (1st of month) - Check if upstream fix is available
warnings.filterwarnings(
    "ignore",
    message="specific warning text",
    category=DeprecationWarning,
    module="external_library"
)
```

### Monthly Review Process

Run these checks on the **1st of each month**:

```bash
# File: documents/engineering/warnings_policy.md
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

## Anti-Patterns

These practices violate the warnings policy:

### 1. Blanket Suppression
❌ **WRONG**: Silencing all warnings of a type
```python
# File: documents/engineering/warnings_policy.md
import warnings
warnings.filterwarnings("ignore")  # Hides ALL warnings
```

✅ **CORRECT**: Fix the code or add specific, documented exceptions

### 2. Downgrading Log Levels
❌ **WRONG**: Hiding warnings by reducing verbosity
```bash
# File: documents/engineering/warnings_policy.md
docker compose -f docker/docker-compose.yml up --build --quiet  # Masks warnings
```

✅ **CORRECT**: Run with normal verbosity and fix warnings

### 3. Deferring Fixes
❌ **WRONG**: Planning to fix warnings "later"
```python
# File: documents/engineering/warnings_policy.md
# TODO: Fix this deprecation warning later
result = old_api(data)  # Emits DeprecationWarning
```

✅ **CORRECT**: Fix before committing

### 4. Using Deprecated APIs
❌ **WRONG**: "It still works, so we'll keep using it"
```python
# File: documents/engineering/warnings_policy.md
# DeprecationWarning: toDlpack() is deprecated
capsule = cupy_array.toDlpack()
```

✅ **CORRECT**: Upgrade immediately when deprecation notices appear

### 5. Suppressing in Application Code
❌ **WRONG**: Adding filters in production code
```python
# File: documents/engineering/warnings_policy.md
# In src/spectralmc/gbm.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

✅ **CORRECT**: Fix the deprecated usage; only add filters in test config for external library bugs

### 6. Disabling Tool Warnings
❌ **WRONG**: Configuring tools to ignore warnings
```toml
# File: documents/engineering/warnings_policy.md
# pyproject.toml
[tool.mypy]
warn_return_any = false  # Disables important warnings
```

✅ **CORRECT**: Keep strict settings and fix type issues

## Success Criteria

All of the following must show zero warnings:

### Local Development
```bash
# File: documents/engineering/warnings_policy.md
# Code quality check
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code

# Docker build
docker compose -f docker/docker-compose.yml up --build -d
```

### Testing
```bash
# File: documents/engineering/warnings_policy.md
# Test suite - ALWAYS redirect to file for complete output
docker compose -f docker/docker-compose.yml exec spectralmc \
  poetry run test-all > /tmp/test-all.txt 2>&1

# Read complete output to verify zero warnings (except documented exceptions)
cat /tmp/test-all.txt
```

### Runtime
- Zero CUDA warnings in logs
- Zero numerical warnings (NaN, Inf, overflow)
- Zero Pydantic serialization warnings
- Zero GPU memory warnings
- Zero resource warnings (unclosed files/connections)

## Examples by Tool

### Python Deprecation Warning
```python
# File: documents/engineering/warnings_policy.md
# Warning: "asyncio.coroutine is deprecated, use async def"

# ❌ WRONG
import asyncio

@asyncio.coroutine
def fetch_data():
    result = yield from make_request()
    return result

# ✅ CORRECT
async def fetch_data():
    result = await make_request()
    return result
```

### Docker DEPRECATED Warning
```dockerfile
# File: documents/engineering/warnings_policy.md
# DEPRECATED: The legacy builder is deprecated

# ❌ WRONG - Continue using legacy builder
# (Old Dockerfile syntax or buildkit disabled)

# ✅ CORRECT - Use BuildKit syntax
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Ensure DOCKER_BUILDKIT=1 in environment
```

### PyTorch CUDA Warning
```python
# File: documents/engineering/warnings_policy.md
# Warning: "CUDA out of memory"

# ❌ WRONG - Silent CPU fallback
import torch

try:
    tensor = torch.randn(100000, 100000, device="cuda:0")
except RuntimeError:
    tensor = torch.randn(100000, 100000, device="cpu")

# ✅ CORRECT - Fail loudly or reduce batch size explicitly
import torch

# Option 1: Let it fail (correct for SpectralMC - GPU required)
tensor = torch.randn(100000, 100000, device="cuda:0")

# Option 2: Explicit reduction with logging (if applicable)
import logging
logger = logging.getLogger(__name__)

try:
    tensor = torch.randn(100000, 100000, device="cuda:0")
except RuntimeError as e:
    if "out of memory" in str(e):
        logger.error("CUDA OOM: reduce batch size and retry")
        raise  # Re-raise, don't suppress
    raise
```

### CuPy DLPack Warning
```python
# File: documents/engineering/warnings_policy.md
# Warning: "toDlpack() is deprecated"

# ❌ DEPRECATED (removed in CuPy 14+)
import cupy as cp
import torch

cupy_array = cp.array([1.0, 2.0, 3.0])
capsule = cupy_array.toDlpack()
torch_tensor = torch.utils.dlpack.from_dlpack(capsule)

# ✅ MODERN (PyTorch 1.10+, CuPy 9.0+)
import cupy as cp
import torch

cupy_array = cp.array([1.0, 2.0, 3.0])
torch_tensor = torch.from_dlpack(cupy_array)
```

### NumPy Type Warning
```python
# File: documents/engineering/warnings_policy.md
# Warning: "np.float is deprecated"

# ❌ REMOVED in NumPy 2.0
import numpy as np

array = np.array([1.0, 2.0], dtype=np.float)
int_array = np.array([1, 2], dtype=np.int)

# ✅ CORRECT (NumPy 2.0+)
import numpy as np

array = np.array([1.0, 2.0], dtype=np.float64)
int_array = np.array([1, 2], dtype=np.int64)
```

### Pydantic Serialization Warning
```python
# File: documents/engineering/warnings_policy.md
# Warning: "PydanticSerializationUnexpectedValue: Expected dict, got mappingproxy"

# ❌ WRONG - MappingProxyType causes warning
from types import MappingProxyType
from pydantic import BaseModel, model_validator
from typing import Mapping

class OptimizerState(BaseModel):
    param_states: Mapping[int, ParamState]

    @model_validator(mode="after")
    def freeze_collections(self):
        object.__setattr__(self, "param_states",
            MappingProxyType(dict(self.param_states)))
        return self

# Calling opt.model_dump(mode="python") emits warning

# ✅ CORRECT - Add field_serializer
from types import MappingProxyType
from pydantic import BaseModel, field_serializer, model_validator
from typing import Mapping

class OptimizerState(BaseModel):
    param_states: Mapping[int, ParamState]

    @model_validator(mode="after")
    def freeze_collections(self):
        object.__setattr__(self, "param_states",
            MappingProxyType(dict(self.param_states)))
        return self

    @field_serializer("param_states", when_used="always")
    def serialize_param_states(
        self, value: Mapping[int, ParamState]
    ) -> dict[int, ParamState]:
        """Convert MappingProxyType to dict for serialization."""
        return dict(value)
```

### Numerical Warning (NumPy/PyTorch)
```python
# File: documents/engineering/warnings_policy.md
# Warning: "RuntimeWarning: divide by zero encountered"

# ❌ WRONG - Suppressing warning
import warnings
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)
result = a / b  # Can produce Inf/NaN

# ✅ CORRECT - Fix the math
import torch

result = torch.where(b != 0, a / b, torch.zeros_like(a))
```

## Cross-References

This document is the authoritative source for warnings policy. Related documentation:

- 📖 [CLAUDE.md](../../CLAUDE.md) - Quick reference for zero warnings requirement
- 📖 [Coding Standards](coding_standards.md) - Type safety and code quality (links here for deprecations)
- 📖 [Testing Requirements](testing_requirements.md) - Test warnings and pytest configuration
- 📖 [Code Quality](code_quality.md) - Ruff/Black/MyPy configuration
- 📖 [Docker Build Philosophy](docker_build_philosophy.md) - Docker build warnings
- 📖 [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md) - GPU runtime warnings
