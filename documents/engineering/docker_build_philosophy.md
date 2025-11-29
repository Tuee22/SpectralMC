# Docker Build Philosophy

## Overview

SpectralMC uses a **dual-mode Docker build strategy** to support both modern GPUs (binary packages) and legacy Maxwell GPUs like the GTX 970 (source builds). This document explains the build philosophy, Poetry-first dependency management, and layer optimization strategies.

**Related Standards**: [Code Formatting](code_formatting.md), [Type Safety](type_safety.md)

---

## Build Modes

SpectralMC supports two build modes controlled by the `BUILD_FROM_SOURCE` environment variable:

### Binary Build (Default)

**When to use**: Modern GPUs with compute capability ≥ 6.0 (Pascal, Volta, Turing, Ampere, Ada, Hopper, Blackwell)

```bash
# Standard build (fast, pre-compiled packages)
cd docker && docker compose up --build -d
```

**Characteristics**:
- **Build time**: 10-15 minutes
- **Dockerfile**: `docker/Dockerfile`
- **Base image**: `nvidia/cuda:12.8.1-devel-ubuntu22.04`
- **PyTorch**: Pre-compiled binary wheel (PyTorch 2.7.1+cu128)
- **CuPy**: Pre-compiled binary wheel (CuPy 13.4.0+, cuda12x)
- **GPU support**: Compute capability 6.0+ (GTX 1060+, Tesla P100+, RTX series including RTX 5090)
- **Use case**: Development, modern hardware, fast iteration

### Source Build (Legacy GPUs)

**When to use**: Legacy Maxwell GPUs with compute capability 5.2 (GTX 970, GTX 980)

```bash
# Source build for GTX 970 (sm_52 support)
BUILD_FROM_SOURCE=true docker compose up --build -d
```

**Characteristics**:
- **Build time**: 2-4 hours (first build only, cached thereafter)
- **Dockerfile**: `docker/Dockerfile.source`
- **Base image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **PyTorch**: Built from source with sm_52 support (PyTorch 2.4.1)
- **CuPy**: Latest version from PyPI (CuPy 13.x, cuda11x)
- **GPU support**: Compute capability 5.2+ (GTX 970+)
- **Use case**: Legacy hardware, validated on GTX 970

**Why 2-4 hours?**
- PyTorch compilation from source with CUDA support
- Compiling thousands of CUDA kernels for sm_52
- Worth it for legacy GPU compatibility

---

## Poetry-First Dependency Management

SpectralMC uses **Poetry** as the single source of truth for all Python dependencies. This ensures consistency across development, testing, and production environments.

### Installation Pattern

**Critical Rule**: Install pip **once** at the beginning, then use **only Poetry** for all subsequent dependencies.

```dockerfile
# ✅ CORRECT Pattern
# Step 1: Install pip once
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Step 2: Upgrade pip and install Poetry via pip
RUN python -m pip install --upgrade pip setuptools wheel poetry

# Step 3: Configure Poetry
RUN poetry config virtualenvs.create false

# Step 4: All subsequent dependencies via Poetry
RUN poetry install --no-interaction --no-root
```

**Forbidden patterns**:
- ❌ Installing Poetry via curl script: `curl -sSL https://install.python-poetry.org | python3.12 -`
- ❌ Using pip after Poetry is installed
- ❌ Mixing pip and Poetry for dependencies

### Why Poetry via pip?

1. **Consistency**: Same installation method for all Python packages
2. **Simplicity**: No external script download, less network dependency
3. **Reproducibility**: pip install is more deterministic than curl script
4. **Docker Best Practice**: Minimize external dependencies in builds

---

## poetry.lock Handling

SpectralMC **does NOT copy `poetry.lock`** into Docker images. The lockfile is regenerated inside the container from `pyproject.toml`.

### Configuration

```dockerfile
# ✅ CORRECT - Only copy pyproject.toml
COPY pyproject.toml ./

# ❌ INCORRECT - Don't copy poetry.lock
# COPY pyproject.toml poetry.lock* ./
```

Both `.dockerignore` and `.gitignore` include `poetry.lock`:

**`.dockerignore`**:
```
# Poetry - exclude lock to reduce context size, will be regenerated
poetry.lock
```

**`.gitignore`**:
```python
# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
poetry.lock
```

### Rationale

1. **Environment-specific**: Lockfiles can vary by platform (Linux/macOS/Windows)
2. **Docker isolation**: Each build should resolve dependencies fresh
3. **Avoid conflicts**: poetry.lock from host may conflict with container environment
4. **pyproject.toml is source of truth**: Version constraints in pyproject.toml control what gets installed

**Trade-off**: Builds are less deterministic (different minor versions possible), but more flexible across environments.

---

## BUILD_FROM_SOURCE Flag

The `BUILD_FROM_SOURCE` environment variable controls which build path is taken.

### Setting the Flag

**In docker-compose.yml**:
```yaml
services:
  spectralmc:
    build:
      args:
        BUILD_FROM_SOURCE: "${BUILD_FROM_SOURCE:-false}"
```

**On command line**:
```bash
# Binary build (default)
docker compose -f docker/docker-compose.yml up --build -d

# Source build (GTX 970)
BUILD_FROM_SOURCE=true docker compose -f docker/docker-compose.yml up --build -d
```

### How It Works

SpectralMC uses **separate Dockerfiles** for binary and source builds:

- **Binary builds** → `docker/Dockerfile` (CUDA 12.8.1, PyTorch 2.7.1+cu128)
- **Source builds** → `docker/Dockerfile.source` (CUDA 11.8.0, PyTorch 2.4.1 source)

The `docker-compose.yml` file uses shell parameter expansion to select the correct Dockerfile:

```yaml
spectralmc:
  build:
    dockerfile: docker/${BUILD_FROM_SOURCE:+Dockerfile.source}${BUILD_FROM_SOURCE:-Dockerfile}
```

**Selection logic**:
- `BUILD_FROM_SOURCE=true` → expands to `docker/Dockerfile.source`
- `BUILD_FROM_SOURCE=false` (or unset) → expands to `docker/Dockerfile`

**Why separate Dockerfiles?**
- **Different base images**: CUDA 12.8.1 (RTX 5090) vs CUDA 11.8.0 (GTX 970)
- **Clean separation**: Each Dockerfile optimized for its use case
- **Simpler maintenance**: No complex conditional logic within Dockerfiles
- **Docker limitation**: Cannot conditionally select base image with ARG in single Dockerfile

---

## Layer Optimization Strategy

SpectralMC's Dockerfile is organized to **maximize Docker build cache hits** and **minimize rebuild time**.

### Layer Structure

```
┌─────────────────────────────────────┐
│ Layer 1-2: System Dependencies     │  Changes: NEVER
│ - apt packages, build tools         │  Cache: Always hit
│ - Python 3.12 installation          │
├─────────────────────────────────────┤
│ Layer 3: Poetry Installation       │  Changes: RARELY
│ - pip, setuptools, wheel, poetry    │  Cache: Usually hit
├─────────────────────────────────────┤
│ Layer 4: pyproject.toml COPY       │  Changes: OCCASIONALLY
│ - Only dependency manifest          │  Cache: Often hit
├─────────────────────────────────────┤
│ Layer 5: Poetry Install             │  Changes: WHEN DEPS CHANGE
│ - poetry install (all dependencies) │  Cache: Sometimes hit
├─────────────────────────────────────┤
│ Layer 6-10: PyTorch/CuPy Build     │  Changes: BUILD_FROM_SOURCE
│ - Binary OR source build            │  Cache: Build-mode dependent
│ - Most expensive layer if source    │
├─────────────────────────────────────┤
│ Layer 11+: Application Code         │  Changes: FREQUENTLY
│ - COPY entire codebase              │  Cache: Rarely hit
│ - poetry install (install package)  │
└─────────────────────────────────────┘
```

### Rationale

1. **Early layers rarely change** → Almost always cached
2. **Middle layers change with dependencies** → Cached if pyproject.toml unchanged
3. **Late layers change frequently** → Always rebuild, but fast (just copying code)

**Result**: Most rebuilds only recompile the application layer (~30 seconds), not PyTorch (~2 hours).

### Example Build Times

| Scenario | Binary Build | Source Build |
|----------|--------------|--------------|
| **First build** | 5-10 min | 2-4 hours |
| **Code change only** | 30 sec | 30 sec |
| **Dependency change** | 3-5 min | 5-10 min |
| **PyTorch upgrade** | 5 min | 2-4 hours |

**Key takeaway**: Source build penalty is paid **once**, then cached.

---

## Binary vs Source Build Comparison

| Aspect | Binary Build | Source Build |
|--------|--------------|--------------|
| **Dockerfile** | `docker/Dockerfile` | `docker/Dockerfile.source` |
| **Base image** | CUDA 12.8.1 | CUDA 11.8.0 + cuDNN8 |
| **Build time (first)** | 10-15 minutes | 2-4 hours |
| **Build time (cached)** | 2-5 minutes | 2-5 minutes |
| **PyTorch version** | 2.7.1+cu128 | 2.4.1 (source) |
| **CuPy version** | 13.4+ (cuda12x) | 13.x (cuda11x) |
| **NumPy support** | NumPy 2.x | NumPy 2.x |
| **GPU support** | Compute ≥ 6.0 | Compute ≥ 5.2 (sm_52) |
| **Supported GPUs** | GTX 1060+, RTX series (incl. RTX 5090), Tesla P100+ | GTX 970+, GTX 980 |
| **Use case** | Development, modern hardware | Legacy hardware (GTX 970) |
| **Recommended for** | Most users | GTX 970/980 owners only |

### When to Use Each

**Use Binary Build if**:
- You have a modern GPU (GTX 1060 or newer)
- You want fast iteration during development
- You're CI/CD testing on cloud GPUs

**Use Source Build if**:
- You have a GTX 970 or GTX 980
- You need compute capability 5.2 support
- You're validating on legacy hardware

---

## Complete Dockerfile Example

**Note**: SpectralMC now uses **two separate Dockerfiles**:
- `docker/Dockerfile` - Binary builds (CUDA 12.8.1)
- `docker/Dockerfile.source` - Source builds (CUDA 11.8.0)

Here's the key sections showing the Poetry-first pattern (from `docker/Dockerfile`):

```dockerfile
# Layer 3: Poetry Installation via pip
RUN python -m pip install --upgrade pip setuptools wheel poetry && \
    poetry config virtualenvs.create false

# Layer 4: Copy dependency manifest (NOT poetry.lock)
COPY pyproject.toml ./

# Layer 5: Install dependencies via Poetry
RUN poetry install --no-interaction --no-root

# Layer 6-10: PyTorch installation (conditional on BUILD_FROM_SOURCE)
ARG BUILD_FROM_SOURCE=false

RUN if [ "$BUILD_FROM_SOURCE" = "true" ]; then \
        echo "=== Building PyTorch from source (sm_52) ===" && \
        git clone --branch v2.4.1 --recursive https://github.com/pytorch/pytorch.git && \
        cd pytorch && python setup.py bdist_wheel && \
        pip install --force-reinstall dist/*.whl; \
    else \
        echo "=== Installing PyTorch binary ===" && \
        pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118; \
    fi

# Layer 11+: Application code
COPY ./ /spectralmc/
RUN poetry install --no-interaction
```

---

## Troubleshooting

### Build fails with "poetry.lock not found"

**Solution**: This is expected. Do NOT copy poetry.lock. Poetry generates it automatically.

### Different package versions than expected

**Cause**: Poetry resolved dependencies differently than your local machine.

**Solution**: This is expected behavior. pyproject.toml version constraints control what's installed.

### Source build takes too long

**First build**: 2-4 hours is normal for PyTorch compilation.

**Subsequent builds**: Should be ~30 seconds (cached). If not, check Docker cache (`docker builder prune`).

### Want to force rebuild from scratch

```bash
# Clear Docker build cache
docker builder prune -a

# Rebuild
BUILD_FROM_SOURCE=true docker compose up --build -d
```

---

## Summary

- **Dual build strategy**: Binary (fast) vs Source (GTX 970 support)
- **Separate Dockerfiles**:
  - `docker/Dockerfile` for binary builds (CUDA 12.8.1, PyTorch 2.7.1+cu128)
  - `docker/Dockerfile.source` for source builds (CUDA 11.8.0, PyTorch 2.4.1)
- **Dockerfile selection**: docker-compose.yml uses shell parameter expansion based on BUILD_FROM_SOURCE
- **Poetry-first**: Install via pip (binary) or curl script (source), use only Poetry for dependencies
- **No poetry.lock in Docker**: Regenerated from pyproject.toml
- **Layer optimization**: Each Dockerfile optimized for its use case
- **BUILD_FROM_SOURCE flag**: Selects which Dockerfile to use
- **Binary build**: 10-15 minutes (supports RTX 5090 sm_120)
- **Source build**: 2-4 hours first time (supports GTX 970 sm_52), cached thereafter
- **Code changes**: ~2-5 minutes rebuild time (both modes)

See also: [Type Safety](type_safety.md), [Testing Requirements](testing_requirements.md)
