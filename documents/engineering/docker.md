# Docker & Environment Variables

**Status**: Authoritative source
**Supersedes**: None
**Referenced by**: [docker_workflow.md](docker_workflow.md), [docker_build_philosophy.md](docker_build_philosophy.md), [testing.md](testing.md), [command_reference.md](command_reference.md)

> **Purpose**: Prescribe Docker container standards, environment variable policy, and build artifact management for SpectralMC GPU development.

## TL;DR

- All development happens inside Docker containers (no local Python/Poetry installation)
- Environment variables are the Single Source of Truth - defined in `docker/Dockerfile` and `docker/Dockerfile.source`
- Standardized cache directories under `/opt/spectralmc/` namespace for all development tools
- Build artifacts live under `/opt/` namespace (protobuf, PyTorch wheels for source builds)
- Direct pytest execution blocked - use `poetry run test-all` commands only
- All containers run as root for simplicity and GPU access
- See [docker_workflow.md](docker_workflow.md) for daily development patterns and command reference

## Environment Variables

All environment variables are defined in `docker/Dockerfile` (binary builds) and `docker/Dockerfile.source` (source builds). This section documents each variable's purpose and rationale.

### Python Path Configuration

#### PATH="/root/.local/bin:/usr/local/bin:$PATH"

Prepends local binary directories to system PATH.

- **Purpose**: Ensures Poetry and other user-installed tools take precedence over system binaries
- **Rationale**: Standard Unix convention for user-installed binaries; `/root/.local/bin` for pip user installs
- **Default without it**: System binaries in `/usr/bin` would take precedence, potentially using wrong versions
- **Defined in**: Both Dockerfile (line 22) and Dockerfile.source (line 20)

### CuPy JIT Compilation

#### CUPY_COMPILE_WITH_PTX=1

Enables PTX (Parallel Thread Execution) compilation for CuPy.

- **Purpose**: Allows CuPy to compile kernels at runtime for the local GPU architecture
- **Rationale**: Critical for supporting newer GPUs (RTX 5090) not available when CuPy was built
- **How it works**: CuPy generates PTX intermediate code that gets JIT-compiled for the actual GPU
- **Trade-off**: Slightly slower first run (JIT compilation overhead) but supports all GPU architectures
- **Default without it**: CuPy limited to pre-compiled architectures, fails on newer GPUs
- **Defined in**: Both Dockerfile (line 26) and Dockerfile.source (line 26)

#### CUPY_NVCC_GENERATE_CODE=current

Tells CuPy to generate code for the current GPU architecture.

- **Purpose**: Optimizes CuPy kernels for the specific GPU in the container
- **Rationale**: Paired with `CUPY_COMPILE_WITH_PTX`, ensures optimal performance on local hardware
- **How it works**: CuPy queries the GPU at runtime and compiles for that specific compute capability
- **Benefit**: Binary builds support GTX 1060+ (sm_60) through RTX 5090 (sm_120) automatically
- **Default without it**: CuPy would compile for a generic architecture, losing performance
- **Defined in**: Both Dockerfile (line 27) and Dockerfile.source (line 27)

### Locale Configuration (Source Builds Only)

#### LANG=C.UTF-8

Sets the system language and character encoding.

- **Purpose**: Ensures proper Unicode handling in Python and CLI output
- **Rationale**: UTF-8 is required for scientific symbols (∇, ∈, ∀) in PyTorch build logs
- **Default without it**: ASCII locale can cause encoding errors during PyTorch compilation
- **Why source builds only**: Binary builds have simpler output; source builds show verbose compiler logs
- **Defined in**: Dockerfile.source only (line 22)

#### LC_ALL=C.UTF-8

Sets locale for all categories (time, numeric, monetary, etc.).

- **Purpose**: Comprehensive locale setting that overrides all other LC_* variables
- **Rationale**: Prevents locale-related errors during long PyTorch compilation (2-4 hours)
- **Default without it**: Mixed locale settings can cause build failures or garbled output
- **Defined in**: Dockerfile.source only (line 23)

### Build Cache Configuration (Source Builds Only)

#### CCACHE_DIR=/root/.ccache

Sets the directory for ccache (compiler cache).

- **Purpose**: Speeds up PyTorch recompilation by caching previously compiled object files
- **Rationale**: PyTorch has thousands of C++ files; ccache can reduce rebuild time from 2-4 hours to 30 minutes
- **Location**: `/root/.ccache` (persistent across rebuilds if volume-mounted)
- **Cache size**: Configured to 20GB in Dockerfile.source (line 113)
- **Default without it**: Every PyTorch rebuild compiles all files from scratch
- **Defined in**: Dockerfile.source only (line 94)

#### CMAKE_CXX_COMPILER_LAUNCHER=ccache

Tells CMake to use ccache for C++ compilation.

- **Purpose**: Integrates ccache with PyTorch's CMake build system
- **Rationale**: PyTorch uses CMake; this ensures ccache intercepts all C++ compilation
- **How it works**: CMake invokes `ccache g++` instead of `g++` directly
- **Benefit**: Cached compilations are ~10x faster
- **Default without it**: CMake bypasses ccache, no caching occurs
- **Defined in**: Dockerfile.source only (line 95)

#### CMAKE_CUDA_COMPILER_LAUNCHER=ccache

Tells CMake to use ccache for CUDA compilation.

- **Purpose**: Integrates ccache with CUDA kernel compilation
- **Rationale**: PyTorch has hundreds of CUDA kernels; caching reduces rebuild time significantly
- **How it works**: CMake invokes `ccache nvcc` instead of `nvcc` directly
- **Trade-off**: CUDA compilation is slower than C++, but ccache still provides 5-10x speedup
- **Default without it**: CUDA recompilation takes 1-2 hours every rebuild
- **Defined in**: Dockerfile.source only (line 96)

### pip Configuration (Source Builds Only)

#### PIP_FIND_LINKS=/opt/pytorch-wheel

Tells pip to search for packages in a local directory.

- **Purpose**: Allows pip to find the locally-built PyTorch wheel instead of downloading from PyPI
- **Rationale**: Source builds create a custom PyTorch 2.4.1 wheel with sm_52 support
- **How it works**: pip checks `/opt/pytorch-wheel/` before searching PyPI
- **Location**: PyTorch wheel is built to `/opt/pytorch-wheel/` in Dockerfile.source (line 126)
- **Default without it**: pip would download incompatible PyTorch from PyPI, overwriting custom build
- **Defined in**: Dockerfile.source only (line 97)

#### PIP_CONSTRAINT=/tmp/constraints-source.txt

Specifies version constraints for pip installations.

- **Purpose**: Pins dependency versions to ensure compatibility with PyTorch 2.4.1 source build
- **Rationale**: Binary builds use Poetry's lockfile; source builds need additional constraints
- **Contents**: Specifies compatible versions of numpy, scipy, etc. for CUDA 11.8
- **Location**: `docker/constraints-source.txt` copied to `/tmp/` (line 101)
- **Default without it**: pip might install incompatible dependency versions, breaking the build
- **Defined in**: Dockerfile.source only (line 98)

### AWS / S3 Configuration (docker-compose.yml only)

These variables are defined in `docker/docker-compose.yml` for the `spectralmc` and `tensorboard` services.

#### AWS_ACCESS_KEY_ID=minioadmin

MinIO access key for S3-compatible storage.

- **Purpose**: Authenticates boto3 S3 client with local MinIO instance
- **Rationale**: Required for blockchain model versioning and TensorBoard log storage
- **Value**: `minioadmin` (MinIO default credentials for development)
- **Security**: Development-only; production deployments must use secure credentials
- **Default without it**: S3 operations fail with authentication errors
- **Defined in**: docker-compose.yml (lines 108, 143)

#### AWS_SECRET_ACCESS_KEY=minioadmin

MinIO secret key for S3-compatible storage.

- **Purpose**: Completes S3 authentication (access key + secret key pair)
- **Rationale**: Required by boto3 S3 client for signing requests
- **Value**: `minioadmin` (matches MinIO default)
- **Security**: Development-only; must be rotated for production
- **Default without it**: S3 operations fail with authentication errors
- **Defined in**: docker-compose.yml (lines 109, 144)

#### AWS_REGION=us-east-1

AWS region for S3 client.

- **Purpose**: Required by boto3 even for local MinIO endpoints
- **Rationale**: boto3 requires a region; value doesn't matter for local MinIO
- **Value**: `us-east-1` (arbitrary choice, MinIO ignores it)
- **Default without it**: boto3 raises "NoRegionError"
- **Defined in**: docker-compose.yml (lines 110, 145)

#### AWS_ENDPOINT_URL=http://minio:9000

S3 endpoint URL for boto3 client.

- **Purpose**: Redirects boto3 from AWS to local MinIO instance
- **Rationale**: boto3 defaults to AWS; this overrides to use MinIO
- **Value**: `http://minio:9000` (MinIO container on Docker network)
- **Network**: Uses Docker Compose internal DNS (`minio` resolves to container IP)
- **Default without it**: boto3 tries to connect to AWS, fails without credentials
- **Defined in**: docker-compose.yml (lines 111, 146)

#### S3_ENDPOINT=http://minio:9000

Alternative S3 endpoint variable.

- **Purpose**: Some S3 libraries check `S3_ENDPOINT` instead of `AWS_ENDPOINT_URL`
- **Rationale**: Ensures compatibility with multiple S3 client libraries
- **Value**: Same as `AWS_ENDPOINT_URL`
- **Redundancy**: Both variables set for maximum compatibility
- **Default without it**: Some libraries might try AWS instead of MinIO
- **Defined in**: docker-compose.yml (lines 112, 147)

#### S3_USE_HTTPS=0

Disables HTTPS for S3 connections.

- **Purpose**: MinIO development instance uses HTTP, not HTTPS
- **Rationale**: HTTPS requires certificates; unnecessary for local development
- **Value**: `0` (disabled)
- **Security**: Development-only; production must use HTTPS
- **Default without it**: S3 client attempts HTTPS, connection fails
- **Defined in**: docker-compose.yml (lines 113, 148)

#### S3_VERIFY_SSL=0

Disables SSL certificate verification.

- **Purpose**: Paired with `S3_USE_HTTPS=0` to disable SSL entirely
- **Rationale**: No certificates in development environment
- **Value**: `0` (disabled)
- **Security**: Development-only; production must verify SSL certificates
- **Default without it**: S3 client attempts certificate verification, fails
- **Defined in**: docker-compose.yml (lines 114, 149)

### Python Runtime Configuration

#### PYTHONUNBUFFERED=1

Forces Python to run in unbuffered mode for immediate output visibility.

- **Purpose**: Ensures logs appear in real-time via `docker logs`
- **Rationale**: Critical for Docker containers - buffered output causes delays in log visibility
- **Default without it**: Python buffers output, making debugging difficult
- **Defined in**: Both Dockerfile and Dockerfile.source

#### PYTHONPATH="/spectralmc"

Tells Python where to find importable modules.

- **Purpose**: Allows `from spectralmc import ...` style imports from `/spectralmc` directory
- **Rationale**: Enables module resolution in development environment
- **Default without it**: Python can't find spectralmc modules without explicit package installation
- **Defined in**: Both Dockerfile and Dockerfile.source

#### PYTHONPYCACHEPREFIX="/opt/pycache"

Centralizes Python bytecode (`.pyc`) files in a single directory outside the source tree.

- **Purpose**: Keeps source directories clean, enables centralized cache management
- **Rationale**: Prevents `.pyc` file pollution in source tree, improves build reproducibility
- **Location**: `/opt/pycache` contains all compiled Python modules
- **Default without it**: `.pyc` files scattered in `__pycache__/` throughout source tree
- **Defined in**: Both Dockerfile and Dockerfile.source

### Poetry Configuration

#### POETRY_NO_INTERACTION=1

Runs Poetry in non-interactive mode, never prompting for user input.

- **Purpose**: Ensures automated builds never hang waiting for input
- **Rationale**: Essential for Docker builds and CI/CD pipelines
- **Default without it**: Poetry may prompt for confirmation, hanging builds
- **Defined in**: Both Dockerfile and Dockerfile.source

#### POETRY_HOME=/usr/local

Sets where Poetry itself is installed.

- **Purpose**: Makes Poetry accessible system-wide following Unix conventions
- **Rationale**: Consistent installation location across all containers
- **Default without it**: Poetry installs to user home directory (`~/.local`)
- **Defined in**: Both Dockerfile and Dockerfile.source

### pip Configuration

#### PIP_NO_CACHE_DIR=1

Disables pip's download cache to reduce Docker image size.

- **Purpose**: Minimizes Docker layer sizes by preventing cache accumulation
- **Rationale**: Package caches aren't needed across builds in containers
- **Default without it**: Pip caches downloads in `~/.cache/pip`, increasing image size
- **Defined in**: Both Dockerfile and Dockerfile.source

#### PIP_DISABLE_PIP_VERSION_CHECK=1

Stops pip from checking for newer versions on every command.

- **Purpose**: Speeds up pip operations by skipping version checking
- **Rationale**: Version checking unnecessary in containers with locked dependencies
- **Default without it**: Pip checks PyPI for updates on every invocation, adding latency
- **Defined in**: Both Dockerfile and Dockerfile.source

#### PIP_BREAK_SYSTEM_PACKAGES=1

Allows pip to install packages system-wide outside virtual environments.

- **Purpose**: Enables system-wide package installation in containers
- **Rationale**: Containers have isolated environments; no need for virtualenvs
- **Note**: Complements `poetry.toml` (virtualenvs.create = false) configuration
- **Default without it**: Pip may refuse to install without virtualenv activation
- **Defined in**: Both Dockerfile and Dockerfile.source

### Tool Cache Configuration

#### XDG_CACHE_HOME="/opt/spectralmc/cache"

Follows XDG Base Directory specification for tool cache storage.

- **Purpose**: Centralizes cache files for XDG-compliant development tools
- **Rationale**: Predictable cache location, easier cleanup and monitoring
- **Tools**: mypy, pytest, ruff, and other XDG-compliant applications
- **Default without it**: Tools default to `~/.cache/`, scattered across filesystem
- **Defined in**: Both Dockerfile and Dockerfile.source

#### MYPY_CACHE_DIR=/opt/spectralmc/mypy_cache

Dedicated cache directory for MyPy type checker.

- **Purpose**: Speeds up incremental type checking by caching previous runs
- **Rationale**: MyPy cache significantly reduces check time on subsequent runs
- **Location**: `/opt/spectralmc/mypy_cache`
- **Default without it**: MyPy uses `.mypy_cache/` in current directory
- **Defined in**: Both Dockerfile and Dockerfile.source

#### RUFF_CACHE_DIR=/opt/spectralmc/ruff_cache

Dedicated cache directory for Ruff linter.

- **Purpose**: Speeds up incremental linting by caching previous runs
- **Rationale**: Ruff cache enables fast incremental linting
- **Location**: `/opt/spectralmc/ruff_cache`
- **Default without it**: Ruff uses default cache location
- **Defined in**: Both Dockerfile and Dockerfile.source

#### PYTEST_CACHE_DIR=/opt/spectralmc/pytest_cache

Dedicated cache directory for pytest.

- **Purpose**: Stores pytest cache data (test failures, profiling info, last-failed tracking)
- **Rationale**: Enables pytest's `--lf` (last-failed) and `--ff` (failed-first) features
- **Location**: `/opt/spectralmc/pytest_cache`
- **Default without it**: pytest uses `.pytest_cache/` in current directory
- **Defined in**: Both Dockerfile and Dockerfile.source

### Container Detection

#### DOCKER_CONTAINER=1

Custom flag indicating code is running in Docker.

- **Purpose**: Allows application code and tools to detect Docker environment
- **Rationale**: Enables conditional behavior for containerized environments
- **Use case**: Skip host-specific operations, adjust resource limits, enable container-specific logging
- **Default without it**: No standard way to detect Docker environment
- **Defined in**: Both Dockerfile and Dockerfile.source

#### CI=false

Disables CI-specific behavior in development tools.

- **Purpose**: Explicitly marks environment as non-CI for tool behavior
- **Rationale**: Many tools (pytest, mypy, ruff, poetry) change behavior in CI mode
- **CI mode changes**: More verbose output, stricter checks, different caching
- **Default without it**: Tools auto-detect CI via environment variables like GITHUB_ACTIONS, GITLAB_CI
- **Why explicit**: Makes intent clear that this is a development environment, not CI
- **Defined in**: Both Dockerfile and Dockerfile.source

## Environment Variable Policy

**Critical Rule**: All environment variables should be set exclusively in Dockerfiles.

### Prohibited Practices

❌ **NEVER** set or override environment variables in:

- Python scripts (`tools/*.py`)
- Shell scripts
- Poetry scripts in `pyproject.binary.toml` or `pyproject.source.toml`
- Test configuration files
- `docker-compose.yml` (except for secrets/service-specific values like AWS credentials)

### Rationale

1. **Single Source of Truth**: Dockerfiles are the canonical definition of the container environment
2. **Predictability**: Same environment in development, testing, and production
3. **Simplicity**: No conditional logic based on environment type
4. **Debuggability**: Environment is defined in one place, easy to audit
5. **Immutability**: Container behavior doesn't change based on how it's invoked

### Allowed Exceptions

The ONLY exceptions are runtime secrets and service-specific configuration in `docker-compose.yml`:

- AWS/S3 credentials (MinIO access keys)
- Service endpoints (MinIO, TensorBoard)
- Network configuration (CUDA device visibility)

These are appropriate for `docker-compose.yml` because:
- They vary between deployment environments
- They connect services in the Docker Compose stack
- They're not part of the build-time container definition

### What This Means

- Build tools must NOT set environment variables
- Test runners must NOT modify `PYTHONPATH`, cache directories, or Python configuration
- All builds use identical configuration regardless of context (dev, CI, production)
- Build tools must NOT hardcode paths like `/opt/spectralmc/*` - use environment variables instead
- All build artifact paths must be read via `os.environ[]` (no fallbacks)

### Enforcement

Code review must reject any PRs that set environment variables outside Dockerfiles (except allowed exceptions above).

Use fail-fast pattern in Python code:

```python
# ✅ CORRECT - Use os.environ[] without fallbacks
cache_dir = Path(os.environ["MYPY_CACHE_DIR"])

# ❌ WRONG - Fallback defeats enforcement
cache_dir = Path(os.getenv("MYPY_CACHE_DIR", "/default"))
```

If different behavior is needed for different contexts, use:

1. **Feature flags**: Application-level configuration
2. **Build arguments**: Docker ARG (compile-time only, not runtime)
3. **Configuration files**: Explicit config files read by application

Do NOT use environment variables for context-dependent behavior.

## ENV Command Format Requirements

**CRITICAL**: All environment variables MUST be defined in a single multi-line ENV command. Multiple separate ENV commands are PROHIBITED.

### Required Format

```dockerfile
# Environment configuration (single source of truth)
ENV VAR1=value1 \
    VAR2=value2 \
    VAR3=value3 \
    VAR4=value4
```

### Prohibited Format

```dockerfile
# ❌ WRONG - Multiple ENV commands scattered
ENV VAR1=value1
ENV VAR2=value2

# ...later in file...
ENV VAR3=value3
ENV VAR4=value4
```

### Rationale

1. **Single Source of Truth**: All environment variables visible in one location
2. **Docker Layer Optimization**: Single ENV command = single layer, faster builds
3. **Maintainability**: Easy to audit complete environment configuration
4. **Prevents Duplication**: Cannot accidentally define same variable twice
5. **Enforces Discipline**: No ad-hoc ENV additions scattered throughout Dockerfile

### Exceptions

**NONE**. There are no exceptions to this rule. ALL environment variables must be in the single consolidated ENV block at the top of the Dockerfile.

### Location

Place the consolidated ENV block immediately after:
- ARG declarations
- WORKDIR (only if WORKDIR comes before any other commands)

In SpectralMC Dockerfiles:
- **Dockerfile (binary)**: After ARG BUILD_FROM_SOURCE (line ~22)
- **Dockerfile.source**: After ARG DEBIAN_FRONTEND (line ~20)

### Validation

To verify compliance:

```bash
# Count ENV commands in Dockerfile - should be exactly 1
grep -c "^ENV " docker/Dockerfile

# Count ENV commands in Dockerfile.source - should be exactly 1
grep -c "^ENV " docker/Dockerfile.source
```

Both counts MUST equal 1 (or 0 if no ENV commands exist).

## Build Directory Structure

All build artifacts and caches live under `/opt/` hierarchy with project namespacing:

### SpectralMC Cache Directories

- `/opt/spectralmc/cache` - General cache (XDG_CACHE_HOME)
- `/opt/spectralmc/mypy_cache` - MyPy type checker cache
- `/opt/spectralmc/pytest_cache` - Pytest cache (test failures, profiling)
- `/opt/spectralmc/ruff_cache` - Ruff linter cache

### SpectralMC Build Artifacts

- `/opt/spectralmc_proto` - Generated protobuf modules
- `/opt/pytorch` - Source-built PyTorch (source builds only)
- `/opt/cupy` - Source-built CuPy (source builds only)
- `/opt/pytorch-wheel` - PyTorch wheel staging (source builds only)

### Shared Python Cache

- `/opt/pycache` - Python bytecode cache (PYTHONPYCACHEPREFIX)
  - Shared across all Python containers for performance
  - Lives outside project namespace to enable cross-project sharing

**Rationale**: Namespacing prevents conflicts when multiple containers run simultaneously. SpectralMC uses `/opt/spectralmc/` while other projects use their own namespaces. Centralized caches enable faster builds and better resource management.

## Testing Policy

### timeout Wrapper

Direct timeout usage with test commands is blocked (Dockerfile line 111-113, Dockerfile.source line 181-183):

```bash
# ❌ BLOCKED
timeout 60 pytest tests/

# ✅ REQUIRED
poetry run test-all  # Tests complete when they complete
```

**Rationale**: Tests should complete naturally; timeout defeats determinism and hides slow tests.

## Infrastructure Services

SpectralMC development requires MinIO for S3-compatible storage (blockchain model versioning):

| Service       | Image                      | Internal Port | Purpose                        |
| ------------- | -------------------------- | ------------- | ------------------------------ |
| spectralmc    | spectralmc:local (custom)  | N/A           | GPU development and training   |
| minio         | quay.io/minio/minio:latest | 9000 (API)    | S3-compatible storage          |
| minio         | quay.io/minio/minio:latest | 9001 (Console)| MinIO web console              |
| tensorboard   | spectralmc:local (custom)  | 6006          | Training metrics visualization |
| createbuckets | minio/mc:latest            | N/A           | One-shot bucket creation       |

**Named Volumes** (persistent data):

- MinIO: `minio-data:/data`
- Remove with: `docker compose -f docker/docker-compose.yml down -v`

**Network**: All services on `spectralmc-net` bridge network (internal DNS)

**Note**: No service ports are published to host by default. Services communicate via Docker internal networking. Use `docker compose exec` to access services from host.

## Success Criteria

- [ ] All environment variables in single ENV block per Dockerfile
- [ ] Exactly 1 ENV command per Dockerfile (verify with `grep -c "^ENV " docker/Dockerfile`)
- [ ] Cache directories exist under `/opt/spectralmc/` namespace
- [ ] Python bytecode cache in `/opt/pycache`
- [ ] `docker compose ps` shows all services running
- [ ] `poetry run test-all` passes
- [ ] `poetry run check-code` passes
- [ ] MinIO accessible via internal DNS (`minio:9000`)
- [ ] TensorBoard accessible on port 6006
- [ ] timeout wrapper blocks test timeouts
- [ ] GPU accessible via `nvidia-smi` inside container

## GPU Support

SpectralMC requires NVIDIA GPU with Docker support:

```yaml
# docker-compose.yml configuration
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: ["gpu"]
```

**Requirements**:
- NVIDIA GPU (GTX 970+ for source builds, GTX 1060+ for binary builds)
- NVIDIA drivers installed on host
- nvidia-container-toolkit installed
- Docker configured for GPU access

**Validation**:

```bash
# Inside container
nvidia-smi  # Should show GPU
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

See [GPU Build Guide](gpu_build.md) for complete GPU setup documentation.

## Security Hardening

- **Non-root users**: Containers run as root for GPU access and development simplicity (no production deployment)
- **Secrets management**: Never commit credentials to Dockerfiles - use docker-compose.yml environment section
- **Named volumes**: Prefer named volumes over bind mounts for data persistence
- **Base images**: NVIDIA CUDA images provide security updates and GPU compatibility

## Cross-References

- 📖 [Docker Workflow](docker_workflow.md) — Daily development patterns and container management
- 📖 [Docker Build Philosophy](docker_build_philosophy.md) — Dual-build strategy, entry point scripts, layer optimization
- 📖 [Build Artifact Management](build_artifact_management.md) — Build output, cache policies, and .gitignore patterns
- 📖 [Testing](testing.md) — Test execution and output management
- 📖 [Command Reference](command_reference.md) — Complete command table for Docker-based development
- 📖 [GPU Build Guide](gpu_build.md) — Legacy GPU source build instructions
