# SpectralMC Engineering Standards

## Overview

This directory contains the engineering standards and best practices for the SpectralMC project. These standards are **requirements** for maintaining the scientific reproducibility and performance guarantees that make SpectralMC reliable for GPU-accelerated Monte Carlo learning.

### Why These Standards Matter

SpectralMC operates in an environment where reproducibility is paramount:

- **Financial Quantitative Models**: Results must be bit-for-bit identical across different environments
- **GPU-Accelerated Computation**: Performance bottlenecks can occur silently if data movement isn't carefully controlled
- **Scientific Computing**: Reproducible results are essential for validating mathematical models
- **Complex Systems**: Strict typing prevents runtime errors in computationally expensive operations

Every standard serves one of these critical goals:

1. **Reproducibility**: Deterministic behavior across platforms, environments, and time
2. **Performance**: Explicit control over expensive operations like GPU/CPU transfers
3. **Reliability**: Catch errors at compile-time rather than during expensive computations
4. **Maintainability**: Clear, self-documenting code that can be understood and modified safely

### The Cost of Non-Compliance

Violating these guidelines can result in:
- Non-reproducible training runs costing hours of GPU time
- Silent performance degradation due to unexpected PCIe transfers
- Runtime type errors in production quantitative models
- Inability to validate scientific results across different environments

**All code must adhere to these guidelines without exception.**

---

## Standards Documentation

### Code Quality

1. **[Coding Standards](coding_standards.md)** ⭐ Single Source of Truth
   - Black formatting requirements and configuration
   - mypy strict mode enforcement (zero tolerance for `Any`, `cast`, `type: ignore`)
   - Custom `.pyi` type stubs for third-party libraries
   - **5 implementation anti-patterns** to avoid
   - **Dependency deprecation management** with zero-tolerance policy

2. **[Functional Programming](functional_programming.md)** ⭐ New
   - Algebraic Data Types (ADTs) for errors and domain states
   - Result[T, E] pattern for expected errors
   - Pattern matching with exhaustive error handling
   - **No legacy APIs policy** - pure functional patterns only
   - Error transformation and `_raise()` helper for system boundaries

### Development Patterns

3. **[Pydantic Patterns](pydantic_patterns.md)**
   - Model validation and serialization
   - Configuration classes with nested validation
   - `ConfigDict(extra="forbid")` requirement
   - Type-safe usage with Pydantic models

4. **[PyTorch Facade](pytorch_facade.md)**
   - Facade pattern for guaranteed reproducibility
   - Deterministic settings and thread safety
   - Device and dtype management
   - Complex-valued neural networks (`(real, imag)` tensor pair pattern)
   - ComplexValuedModel protocol for type-safe duck typing

5. **[Immutability Doctrine](immutability_doctrine.md)**
   - Never bypass immutability guarantees
   - Forbidden patterns (`object.__setattr__()`, `__dict__` manipulation)
   - Functional update patterns (`dataclasses.replace()`)
   - Blockchain integrity and correctness guarantees

### Documentation

6. **[Documentation Standards](documentation_standards.md)**
   - Module documentation requirements
   - Google-style docstrings for functions and classes
   - Technical whitepaper organization

### Testing & Compute Policy

7. **[Testing Requirements](testing_requirements.md)** ⭐ Comprehensive Guide
   - Test structure and type safety
   - **GPU required** - all tests require GPU, no silent CPU fallbacks
   - Deterministic testing requirements
   - **13 testing anti-patterns** to avoid
   - **Test output handling** best practices
   - **Blockchain storage test coverage**

8. **[CPU/GPU Compute Policy](cpu_gpu_compute_policy.md)**
   - Two-phase architecture (CPU init → GPU compute)
   - Acceptable CPU usage patterns
   - GPU compute requirements and enforcement
   - TensorTree transfers for explicit device movement
   - Test guidelines for device placement

### Infrastructure

9. **[Docker Build Philosophy](docker_build_philosophy.md)**
   - Dual build strategy (binary vs source)
   - Poetry-first dependency management
   - Layer optimization for build cache
   - BUILD_FROM_SOURCE flag usage

10. **[GPU Build Troubleshooting](gpu_build_troubleshooting.md)**
    - Legacy GPU support (GTX 970, compute capability < 6.0)
    - Source build configuration
    - Common build errors and solutions
    - Validation and testing

### Model Versioning

11. **[Blockchain Storage](blockchain_storage.md)** ⭐ Production Model Versioning
    - S3-based blockchain model versioning system
    - Atomic commits with 10-step CAS protocol
    - InferenceClient (pinned/tracking modes)
    - Chain verification and garbage collection
    - CLI tools and TensorBoard integration
    - Training integration with auto-commit

---

## Cross-Reference Guide

### Type Safety
- **Primary**: [Coding Standards](coding_standards.md) - Complete type safety requirements
- **Related**: [Functional Programming](functional_programming.md) - ADT type safety with pattern matching
- **Related**: [Pydantic Patterns](pydantic_patterns.md) - Runtime type validation
- **Related**: [PyTorch Facade](pytorch_facade.md) - Type-safe device and dtype helpers

### Immutability
- **Primary**: [Immutability Doctrine](immutability_doctrine.md) - Immutable data structures
- **Related**: [Functional Programming](functional_programming.md) - Frozen dataclasses for ADTs
- **Related**: [Pydantic Patterns](pydantic_patterns.md) - `frozen=True` for Pydantic models

### Error Handling
- **Primary**: [Functional Programming](functional_programming.md) - Result types, ADTs, pattern matching
- **Related**: [Testing Requirements](testing_requirements.md) - Testing error cases
- **Related**: [Coding Standards](coding_standards.md) - Type safety for error handling

### PyTorch Patterns
- **Primary**: [PyTorch Facade](pytorch_facade.md) - Deterministic execution, complex-valued networks
- **Related**: [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md) - Device management and TensorTree
- **Related**: [Coding Standards](coding_standards.md) - PyTorch type stubs

### Performance
- **Primary**: [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md) - Device placement and transfers
- **Related**: [PyTorch Facade](pytorch_facade.md) - Deterministic algorithms (5-15% slower)
- **Related**: [Docker Build Philosophy](docker_build_philosophy.md) - Build-time performance optimizations

---

## Related Documentation

- [Product Documentation](../product/index.md) - Deployment guides and operations
- [Domain Knowledge](../domain/index.md) - Scientific theory and research
- [Main Project README](../../CLAUDE.md) - Quick reference guide

---

## Quick Reference

### Before Committing

All code must pass these checks:

```bash
# Inside Docker container
docker compose -f docker/docker-compose.yml exec spectralmc black .
docker compose -f docker/docker-compose.yml exec spectralmc mypy
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all
```

### Forbidden Patterns

**Type Safety** (see [Coding Standards](coding_standards.md)):
- ❌ `Any` type annotations
- ❌ `cast()` function calls
- ❌ `# type: ignore` comments

**Immutability** (see [Immutability Doctrine](immutability_doctrine.md)):
- ❌ `object.__setattr__()` on frozen dataclasses
- ❌ `__dict__` manipulation on immutable objects

**Error Handling** (see [Functional Programming](functional_programming.md)):
- ❌ Exceptions for expected errors (use Result types)
- ❌ Returning `None` to indicate errors
- ❌ Mutable error state

**GPU/CPU** (see [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md)):
- ❌ Implicit GPU/CPU transfers (`.cuda()`, `.cpu()`, `.to(device)`)

**Testing** (see [Testing Requirements](testing_requirements.md)):
- ❌ `pytest.skip()` without fixing root cause
- ❌ Tests that pass when features are broken
- ❌ Analyzing truncated test output

**Implementation** (see [Coding Standards](coding_standards.md)):
- ❌ Silent exception handling
- ❌ Mutable default arguments
- ❌ Ignoring numerical warnings

### Required Patterns

**Type Safety** (see [Coding Standards](coding_standards.md)):
- ✅ Explicit type annotations on all functions
- ✅ mypy strict mode clean

**Functional Programming** (see [Functional Programming](functional_programming.md)):
- ✅ Result[T, E] for expected errors
- ✅ ADTs with `@dataclass(frozen=True)`
- ✅ Pattern matching with exhaustive error handling
- ✅ `assert_never()` for exhaustiveness checks

**Documentation** (see [Documentation Standards](documentation_standards.md)):
- ✅ Google-style docstrings on public APIs

**Code Quality** (see [Coding Standards](coding_standards.md)):
- ✅ Black-formatted code

**PyTorch** (see [PyTorch Facade](pytorch_facade.md)):
- ✅ PyTorch facade imported first
- ✅ Complex-valued layers use `(real, imag)` tensor pairs

**GPU/CPU** (see [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md)):
- ✅ Explicit GPU/CPU transfers via `move_tensor_tree()`

**Pydantic** (see [Pydantic Patterns](pydantic_patterns.md)):
- ✅ Pydantic models with `ConfigDict(extra="forbid")`

---

## Summary

Following these standards ensures that SpectralMC maintains:

- **Strict type safety** with zero tolerance for `Any`, `cast`, or `type: ignore`
- **Functional error handling** with Result types and ADTs (no legacy exception-based APIs)
- **Reproducible execution** through the PyTorch facade pattern and CPU initialization
- **GPU-centric compute** with deterministic CPU initialization for reproducibility
- **Comprehensive documentation** for all modules and functions
- **Consistent formatting** with Black
- **Robust testing** with full type coverage and GPU-first design

All code must pass `mypy --strict` and `black --check` before being committed. The custom type stubs in `stubs/` ensure complete type coverage for all dependencies, while the CPU/GPU compute policy guarantees reproducible, deterministic execution with GPU performance.

### Recent Changes (2025-11-28)

- **Consolidated**: Merged `code_formatting.md`, `type_safety.md`, and `type_stubs.md` into single [Coding Standards](coding_standards.md) SSoT document
- **Added**: New [Functional Programming](functional_programming.md) document covering ADTs, Result types, pattern matching, and no legacy APIs policy
- **Enhanced**: [PyTorch Facade](pytorch_facade.md) now includes complex-valued network patterns and ComplexValuedModel protocol
- **Removed**: `project_patterns.md` - content distributed to [PyTorch Facade](pytorch_facade.md) and [Functional Programming](functional_programming.md)
