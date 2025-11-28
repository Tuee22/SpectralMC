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

### Code Quality & Type Safety

1. **[Code Formatting](code_formatting.md)**
   - Black configuration and automated formatting
   - Zero configuration policy
   - Integration with pre-commit hooks

2. **[Type Safety](type_safety.md)**
   - mypy strict mode enforcement
   - Zero-tolerance policy (no `Any`, `cast`, or `type: ignore`)
   - Required annotations for all functions and class attributes

3. **[Type Stubs](type_stubs.md)**
   - Custom `.pyi` files for third-party libraries
   - Project-specific typing stricter than upstream
   - Reproducible type checking independent of external changes

4. **[Pydantic Patterns](pydantic_patterns.md)**
   - Model definition and validation
   - Configuration classes with nested validation
   - Type-safe usage with Pydantic models

### Documentation

5. **[Documentation Standards](documentation_standards.md)**
   - Module documentation requirements
   - Google-style docstrings for functions and classes
   - Technical whitepaper organization

### PyTorch & Project Patterns

6. **[PyTorch Facade](pytorch_facade.md)**
   - Facade pattern for guaranteed reproducibility
   - Deterministic settings and thread safety
   - Device and dtype management

7. **[Project Patterns](project_patterns.md)**
   - Complex-valued neural network conventions
   - Protocol usage for type safety
   - GPU/CPU transfer patterns (TensorTree)
   - Error handling patterns

### Testing & Compute Policy

8. **[Testing Requirements](testing_requirements.md)**
   - Test structure and type safety
   - GPU testing with `@pytest.mark.gpu`
   - Deterministic testing requirements

9. **[CPU/GPU Compute Policy](cpu_gpu_compute_policy.md)**
   - Two-phase architecture (CPU init → GPU compute)
   - Acceptable CPU usage patterns
   - GPU compute requirements and enforcement
   - Test guidelines for device placement

### Infrastructure

10. **[Docker Build Philosophy](docker_build_philosophy.md)**
    - Dual build strategy (binary vs source)
    - Poetry-first dependency management
    - Layer optimization for build cache
    - BUILD_FROM_SOURCE flag usage

11. **[GPU Build Troubleshooting](gpu_build_troubleshooting.md)**
    - Legacy GPU support (GTX 970, compute capability < 6.0)
    - Source build configuration
    - Common build errors and solutions
    - Validation and testing

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

- ❌ `Any` type annotations
- ❌ `cast()` function calls
- ❌ `# type: ignore` comments
- ❌ Implicit GPU/CPU transfers (`.cuda()`, `.cpu()`, `.to(device)`)
- ❌ Direct pytest usage (must use `poetry run test-all`)
- ❌ Mutable default arguments
- ❌ Silent exception handling

### Required Patterns

- ✅ Explicit type annotations on all functions
- ✅ Google-style docstrings on public APIs
- ✅ Black-formatted code
- ✅ mypy strict mode clean
- ✅ Explicit GPU/CPU transfers via `move_tensor_tree()`
- ✅ PyTorch facade imported first
- ✅ Pydantic models with `ConfigDict(extra="forbid")`

---

## Summary

Following these standards ensures that SpectralMC maintains:

- **Strict type safety** with zero tolerance for `Any`, `cast`, or `type: ignore`
- **Reproducible execution** through the PyTorch facade pattern and CPU initialization
- **GPU-centric compute** with deterministic CPU initialization for reproducibility
- **Comprehensive documentation** for all modules and functions
- **Consistent formatting** with Black
- **Robust testing** with full type coverage and GPU-first design

All code must pass `mypy --strict` and `black --check` before being committed. The custom type stubs in `stubs/` ensure complete type coverage for all dependencies, while the CPU/GPU compute policy guarantees reproducible, deterministic execution with GPU performance.
