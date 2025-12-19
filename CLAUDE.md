# File: CLAUDE.md
# SpectralMC Development Guide

**Status**: Reference only  
**Supersedes**: None  
**Referenced by**: README.md; AGENTS.md  

> **Purpose**: Quick-start and operational guide for SpectralMC agents; mirrors AGENTS.md constraints.
> **📖 Authoritative Reference**: [AGENTS.md](AGENTS.md)

## Project Overview

SpectralMC is a GPU-accelerated library for online machine learning using Monte Carlo simulation. It trains complex-valued neural networks (CVNNs) using Monte Carlo data, drawing on techniques from Reinforcement Learning, particularly policy gradient methods.

**Primary Use Case**: Quantitative Finance - stochastic process modeling and derivative pricing with significantly reduced computational requirements compared to traditional Monte Carlo methods.

**Key Technology Stack**:
- Python 3.12
- PyTorch 2.7.0 (CUDA 12.8)
- CuPy (CUDA 12.x)
- NumPy, SciPy, scikit-learn
- Distributed: Dask, Ray
- Testing: pytest, hypothesis
- Type checking: mypy (strict mode)

## Architecture

### Core Components

**src/spectralmc/**
- `async_normals.py` - Asynchronous normal distribution generation on GPU
- `cvnn.py` - Complex-valued neural network implementation
- `cvnn_factory.py` - Factory for creating CVNN instances
- `gbm.py` - Geometric Brownian Motion simulation
- `gbm_trainer.py` - Training loop for GBM models
- `sobol_sampler.py` - Quasi-Monte Carlo sampling using Sobol sequences
- `quantlib.py` - QuantLib integration utilities
- `models/` - Model implementations
  - `torch.py` - PyTorch-based model definitions
  - `numerical.py` - Numerical model utilities
  - `cpu_gpu_transfer.py` - CPU/GPU memory transfer utilities
- `storage/` - Blockchain model versioning (S3-based, production-ready)
  - `chain.py` - Blockchain primitives (ModelVersion, hashing, semantic versioning)
  - `store.py` - AsyncBlockchainModelStore with atomic S3 commits
  - `checkpoint.py` - Checkpoint serialization/deserialization utilities
  - `inference.py` - InferenceClient with pinned/tracking modes
  - `verification.py` - Chain integrity verification
  - `gc.py` - Garbage collection for old versions
  - `tensorboard_writer.py` - TensorBoard logging integration
  - `errors.py` - Exception hierarchy for storage operations
  - `__main__.py` - CLI tool for storage operations
- `serialization/` - Protocol Buffer serialization
  - `common.py` - Enum converters (Precision, Device, dtype enums)
  - `simulation.py` - Simulation parameter converters
  - `models.py` - Model configuration converters
  - `training.py` - Training configuration converters
- `proto/` - Generated Protocol Buffer code
  - `common_pb2.py` - Common message types
  - `simulation_pb2.py` - Simulation messages
  - `models_pb2.py` - Model configuration messages
  - `training_pb2.py` - Training configuration messages

### Blockchain Model Versioning - Overview

SpectralMC uses blockchain-based versioning with S3 storage for production ML model control.

**Key Features**:
- Immutable version history (SHA256 content addressing)
- Semantic versioning (MAJOR.MINOR.PATCH)
- Atomic commits with ETag-based CAS
- InferenceClient (pinned/tracking modes)
- Chain verification and garbage collection

**For complete documentation**, see [Blockchain Storage](documents/engineering/blockchain_storage.md):
- S3 storage structure and CLI commands
- Storage architecture and 10-step atomic commit protocol
- InferenceClient modes (pinned vs tracking)
- Chain verification algorithm
- Training integration examples
- Complete API reference

### Workflow

1. **Monte Carlo Simulation**: Generate finite samples from parametric distributions directly on GPU
2. **Fourier Transform**: Use FFT to estimate the sample's characteristic function
3. **CVNN Training**: Update complex-valued neural network parameters to approximate the characteristic function
4. **CVNN Inference**: Produce estimated distributions for computing means, moments, quantiles, and other metrics

## 🐳 Docker Development

### Critical Development Rules
- ❌ **NEVER** run commands directly on host (poetry, pytest, mypy)
- ✅ **ALWAYS** use: `docker compose -f docker/docker-compose.yml exec spectralmc <command>`
- ✅ **ALL** commands run inside container

### Docker Commands
```bash
# File: CLAUDE.md
# Start services (SpectralMC, MinIO, TensorBoard)
cd docker && docker compose up -d

# Execute commands inside container
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code
```

**For complete Docker workflows**, see [Docker Build Philosophy](documents/engineering/docker_build_philosophy.md).

## 🎮 GPU Support

SpectralMC supports both modern and legacy GPUs through a dual-build strategy.

**Status**: ✅ PRODUCTION READY - 227/227 tests passing (100%, validated Nov 30, 2025)
- 11/11 GPU tests passing
- 216/216 CPU tests passing
- LAPACK/OpenBLAS support working (validated Nov 30, 2025)

**GPU Compatibility**:
- **Modern GPUs** (GTX 1060+, RTX series): Binary wheels, 10-15 min build
- **Legacy GPUs** (GTX 970/980, Maxwell sm_52): Source build, 2-4 hours first build

**For complete GPU documentation**, see:
- [GPU Build Guide](documents/engineering/gpu_build.md) - GTX 970 compatibility and build instructions
- [Docker Build Philosophy](documents/engineering/docker_build_philosophy.md) - Dual-build strategy

## Type Safety - Quick Reference

SpectralMC enforces strict static typing with zero compromises.

**Core Rules**:
- ❌ NO `Any` types, `cast()`, or `# type: ignore`
- ✅ Complete type hints on all functions
- ✅ Run: `docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code`

**Code Quality Pipeline** (`poetry run check-code`):
1. **Ruff** - Linting with auto-fix
2. **Black** - Code formatting
3. **MyPy** - Static type checking

**For complete documentation**, see [Coding Standards](documents/engineering/coding_standards.md) - Type Safety section.

## Testing

### Running Tests

SpectralMC uses pytest. **All tests require GPU** - silent CPU fallbacks are strictly forbidden. **All commands must run inside Docker container**:

```bash
# File: CLAUDE.md
# Run all tests - redirect to file for complete output
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all > /tmp/test-all.txt 2>&1

# Run specific test file
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_gbm.py > /tmp/test-gbm.txt 2>&1
```

**IMPORTANT**: Always redirect output to /tmp/ files and read complete output with Read tool.

**For complete testing documentation**, see:
- [Testing Requirements](documents/engineering/testing_requirements.md) - GPU enforcement, determinism, 13 anti-patterns
- [Testing Architecture](documents/engineering/testing_architecture.md) - DRY doctrine, helper consolidation, fixture patterns, pytest best practices

### Test Configuration

- **Test directory**: `tests/`
- **GPU required**: All tests assume GPU is available - missing GPU causes test failure
- **No fallbacks**: Silent CPU fallbacks are forbidden - `pytest.skip("CUDA not available")` is not allowed
- **Fixtures**: Global GPU memory cleanup in `tests/conftest.py`

### Mixed Precision Testing

SpectralMC REQUIRES mixed precision testing for all numerical tests.

**Quick Start**:
```python
# File: tests/test_example.py
# PyTorch model tests - use full_dtype fixture
def test_model(full_dtype: torch.dtype) -> None:
    model = MyModel()  # Auto-uses full_dtype
    x = torch.randn(10)  # Auto-uses full_dtype
    assert x.dtype == full_dtype

# Numerical simulation tests - use precision fixture
def test_simulation(precision: Precision) -> None:
    params = make_simulation_params(dtype=precision)
    result = run_simulation(params)
```

**Shared Fixtures** (from `tests.helpers`):
- `full_dtype` - PyTorch dtype with context manager
- `full_dtype_enum` - FullPrecisionDType enum
- `precision` - Numerical Precision enum

**Shared Constants** (from `tests.helpers`):
- `FULL_PRECISION_DTYPES` - (float32, float64) tuple
- `PRECISIONS` - (Precision.float32, Precision.float64) tuple
- `RTOL_FLOAT32`, `RTOL_FLOAT64` - Precision-specific tolerances

**For complete policy**, see [Testing Requirements: Mixed Precision Testing Policy](documents/engineering/testing_requirements.md#mixed-precision-testing-policy).

## Anti-Patterns - Quick Reference

SpectralMC enforces strict standards to prevent common errors in testing and implementation.

**For complete anti-patterns documentation**, see:
- [Testing Requirements](documents/engineering/testing_requirements.md) - 13 testing anti-patterns
- [Coding Standards](documents/engineering/coding_standards.md) - 5 implementation anti-patterns

### Key Testing Anti-Patterns to Avoid

- ❌ Tests that pass when features are broken
- ❌ Using `pytest.skip("CUDA not available")` - missing GPU must fail, not skip
- ❌ Silent CPU fallback: `"cuda" if torch.cuda.is_available() else "cpu"`
- ❌ Testing actions without validating results
- ❌ Analyzing truncated test output

### Key Implementation Anti-Patterns to Avoid

- ❌ Silent failure handling
- ❌ Ignoring numerical warnings
- ❌ Mutable default arguments
- ❌ Inconsistent device handling
- ❌ For loops in business logic (Tier 2) - use comprehensions
- ❌ If statements in business logic (Tier 2) - use conditional expressions or match/case
- ❌ While loops in business logic (Tier 2) - use generators
- ❌ Raise for expected errors (Tier 2) - use Result types
- ❌ Side effects in business logic (print, logger) - use Effect ADTs

## Recovery Checklist

When you encounter test failures or bugs:

1. **Read the complete error output** (redirect to file if truncated)
2. **Reproduce the failure** reliably with minimal example
3. **Check for numerical issues** (NaN, Inf, loss divergence)
4. **Validate inputs and outputs** at each stage
5. **Use debugger** to inspect tensor values, gradients, device placement
6. **Check git diff** - what changed since tests last passed?
7. **Run single test in isolation** - rule out test interaction
8. **Verify random seeds** - ensure reproducibility
9. **Check GPU memory** - OOM can cause silent failures
10. **Profile if slow** - don't guess at bottlenecks
11. **Run purity checker** - `docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-purity`

## Warnings Policy

SpectralMC enforces zero warnings across all tools and environments.

**For complete documentation**, see [Warnings Policy](documents/engineering/warnings_policy.md).

### Key Points

- ❌ NO warnings in test output
- ❌ NO deprecation warnings in production code
- ❌ NO suppressing warnings without documented exception
- ✅ Fix root causes immediately
- ✅ Monthly review on 1st of month

### Success Criteria

Zero warnings in all of the following:

```bash
# File: CLAUDE.md
# Code quality check
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code

# Test suite (redirect to file for complete output)
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all > /tmp/test-all.txt 2>&1

# Docker build
docker compose -f docker/docker-compose.yml up --build -d
```

### Current Exceptions

Four documented exceptions in pyproject.binary.toml:
1. **botocore** datetime.utcnow() - AWS SDK internal (pending upstream fix)
2. **QuantLib** SWIG deprecations - unfixable generated code (permanent)
3. **IFFT** imaginary component - expected for untrained models (test-only)
4. **Numba** grid size - expected for small test datasets (test-only)

See [Warnings Policy](documents/engineering/warnings_policy.md#exceptions-rare) for details.

## When to Rebuild Docker Images

> **📖 Authoritative Reference**: [Docker Build Philosophy](documents/engineering/docker_build_philosophy.md#entry-point-script-management)

### Entry Point Scripts and Dual-Pyproject Architecture

**Architecture**: SpectralMC uses dual pyproject files (pyproject.binary.toml and pyproject.source.toml) that are copied to pyproject.toml at build time:
- Binary build: `RUN cp pyproject.binary.toml pyproject.toml` (docker/Dockerfile line 88)
- Source build: `RUN cp pyproject.source.toml pyproject.toml` (docker/Dockerfile.source line 137)

The `[tool.poetry.scripts]` section is shared and must be synchronized between both files.

**Problem**: Volume mount syncs source pyproject files but not the generated pyproject.toml or Poetry-generated entry point scripts in `/usr/local/bin/`.

**When to rebuild**:
```bash
# Rebuild required when [tool.poetry.scripts] changes in either pyproject file
docker compose -f docker/docker-compose.yml up --build -d
```

**Why rebuild?**
- Each Dockerfile copies pyproject.binary.toml (or pyproject.source.toml) to pyproject.toml at build time
- Entry point scripts are generated from the copied pyproject.toml by `poetry install`
- Volume mount overlays source pyproject files but NOT the generated pyproject.toml or `/usr/local/bin/` scripts
- Changes to pyproject.binary.toml or pyproject.source.toml are visible immediately (via volume mount)
- But generated pyproject.toml and scripts require image rebuild to regenerate

**Important**: Keep `[tool.poetry.scripts]` synchronized in both pyproject.binary.toml and pyproject.source.toml.

**Symptoms of stale scripts**:
```
Warning: 'script-name' is an entry point defined in pyproject.toml, but it's not installed as a script.
```

**Quick fix** (temporary, until next rebuild):
```bash
docker compose -f docker/docker-compose.yml exec spectralmc poetry install
```

**Permanent fix**:
```bash
docker compose -f docker/docker-compose.yml up --build -d
```

### Build Triggers

Rebuild the Docker image when:
- ✅ `[tool.poetry.scripts]` changes in pyproject.binary.toml or pyproject.source.toml (new/removed/renamed scripts)
- ✅ `[tool.poetry.dependencies]` changes in pyproject.binary.toml or pyproject.source.toml (new/updated packages)
- ✅ Dockerfile or Dockerfile.source changes
- ❌ Source code changes (`src/`, `tests/`, `tools/`) - volume mount handles these
- ❌ Configuration changes (poetry.toml, shared pyproject sections like [tool.mypy]) - volume mount handles these

### Policy: No Custom Entrypoint Scripts

**CRITICAL**: Do NOT add custom entrypoint scripts or startup hooks to work around script installation.

**Forbidden approaches**:
- ❌ Custom `ENTRYPOINT` scripts that run `poetry install`
- ❌ Startup scripts that regenerate entry points
- ❌ Runtime workarounds for build-time artifacts

**Required approach**:
- ✅ Rebuild image when `[tool.poetry.scripts]` changes in either pyproject file
- ✅ Keep `[tool.poetry.scripts]` synchronized in both pyproject.binary.toml and pyproject.source.toml
- ✅ Use Poetry entry points exclusively
- ✅ Maintain Docker image immutability

**Why**: Docker images are immutable build artifacts. Scripts are generated at build time, not runtime.

## Prevention: Pre-commit Checklist

Before committing:

- [ ] All tests pass (CPU and GPU)
- [ ] mypy type checking passes (strict mode)
- [ ] Purity check passes (`poetry run check-purity`)
- [ ] No NaN/Inf in test outputs
- [ ] Convergence validated, not just "runs without error"
- [ ] No pytest.skip() added
- [ ] No overly broad exception handlers added
- [ ] No hardcoded magic numbers without comments
- [ ] Type hints on all functions
- [ ] Docstrings on public functions
- [ ] No TODO comments without GitHub issues

## PyProject Architecture Enforcement

SpectralMC uses dual pyproject files that must stay synchronized in shared sections.

### Automated Validation

**Check synchronization (part of check-code)**:
```bash
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code
```

### Automation Policy

- No `.pre-commit-config.yaml`, git hooks, or CI/CD workflows are allowed in this repository.
- Run synchronization and quality checks manually inside Docker before handoff.

### Synchronization Rules

**When editing shared sections** (scripts, mypy, black, pytest, pydantic-mypy, dev dependencies):
1. Edit in BOTH pyproject.binary.toml AND pyproject.source.toml
2. Run `poetry run check-code` to verify synchronization
3. Commit both files together
4. Share manual check outputs (no automated hooks exist)

**When editing different sections** (dependencies, sources):
- Edit only the relevant file (binary or source)
- Different sections are expected to diverge
- Validator allows these differences

## 🔒 Git Workflow Policy

**Critical Rule**: Claude Code and ALL LLMs are NOT authorized to commit, push, or create branches.

### Absolutely Forbidden Git Operations
- ❌ **NEVER** run `git commit` (including `--amend`, `--no-verify`, etc.)
- ❌ **NEVER** run `git push` (including `--force`, `--force-with-lease`, etc.)
- ❌ **NEVER** run `git checkout -b` or `git branch` to create new branches
- ❌ **NEVER** run `git switch -c` to create and switch branches
- ❌ **NEVER** run `git add` followed by commit operations
- ❌ **NEVER** create commits or branches under any circumstances
- ❌ **NEVER** modify git history with rebase, reset, or amend

### Required Workflow - ONLY Permitted Git Operations
- ✅ `git status` - Check working directory status
- ✅ `git diff` - Review changes
- ✅ `git log` - View commit history
- ✅ `git branch` (without -b flag) - List existing branches
- ✅ Make all code changes as requested
- ✅ Run tests and validation (via Docker: `docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all`)
- ✅ Leave ALL changes as **uncommitted** working directory changes on current branch
- ✅ User reviews changes using `git status` and `git diff`
- ✅ User manually creates branches, commits, and pushes when satisfied

**Rationale**: All changes must be human-reviewed before entering version control. This ensures code quality, prevents automated commit mistakes, maintains clear authorship, and prevents unauthorized branch proliferation. The user alone decides when and how to commit changes to version control.

## Dependency Management - Quick Reference

SpectralMC enforces a zero-tolerance policy for deprecated APIs in production code.

**For complete documentation**, see [Coding Standards](documents/engineering/coding_standards.md) - Dependency Deprecation Management section.

### Key Points

- ❌ NO deprecated APIs in `src/spectralmc/` code
- ✅ Monthly review checklist (1st of each month)
- ✅ Immediate migration when deprecations appear
- ✅ DLPack API migration completed
- ✅ NumPy 2.0 compatibility achieved

## Engineering Standards

See `documents/engineering/README.md` for comprehensive engineering standards including:

- **Code Quality**: Black formatting, mypy strict mode, custom type stubs
- **Immutability Doctrine**: Never bypass immutability guarantees (frozen dataclasses, etc.)
- **Development Patterns**: Pydantic models, PyTorch facade, project-specific patterns
- **Testing**: Requirements, GPU testing, CPU/GPU compute policy
- **Infrastructure**: Docker build philosophy (binary vs source builds, Poetry management)

All code must pass `mypy --strict`, `black --check`, and `poetry run test-all` before committing.

## Documentation

- **[Engineering Standards](documents/engineering/README.md)** - Development practices and code quality
- **[Product Documentation](documents/product/index.md)** - Deployment and operations guides
- **[Domain Knowledge](documents/domain/index.md)** - Scientific theory and research papers

## Contact

Author: matt@resolvefintech.com
