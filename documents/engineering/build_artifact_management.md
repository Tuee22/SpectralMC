# File: documents/engineering/build_artifact_management.md
# Build Artifact Management

**Status**: Authoritative source  
**Supersedes**: prior informal lockfile notes  
**Referenced by**: engineering/README.md; command_reference.md; docker_build_philosophy.md

> **Purpose**: Define what counts as a build artifact in SpectralMC, where artifacts live, and how to keep them out of source control. Aligns with effectful naming; honors SpectralMC’s dual Docker build paths.

## Executive Summary

- **Lockfiles (`poetry.lock`, `package-lock.json`) are treated as build artifacts**: not tracked, regenerated inside containers.
- **Artifacts stay inside containers** (e.g., `/opt/**`, build caches); do not copy them back to the host or commit them.
- **Sources only** in git: `pyproject.toml`, code, stubs, docs.

## Doctrine

### Core Principle
Only source-of-truth inputs are versioned. Anything produced by a build step is an artifact and must be ignored.

### Locations
- Container-only outputs: `/opt/**` (wheels, compiled kernels, **protobuf-generated Python code**), `.venv` if created transiently in builds.
- Lockfiles: regenerated during `poetry install` in Docker builds; excluded from git and Docker context.
- Caches/binaries: `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `dist/`, `build/`, `*.egg-info/`.
- **Protobuf-generated code**: `*_pb2.py`, `*_pb2.pyi`, `*_pb2_grpc.py` files generated from `.proto` sources; stored in `/opt/spectralmc_proto/` inside containers, referenced but never committed.

### Rationale
1. Reproducibility across binary vs source builds without stale locks.
2. Smaller Docker contexts and faster rebuilds.
3. Avoid leaking compiled outputs or cache churn into reviews.

## What Is Versioned (Sources)
- `pyproject.toml`, `poetry.toml`, source code (including `.proto` files), stubs, scripts, docs.
- Minimal configs required to rebuild deterministically inside Docker.
- **Note**: `.proto` schema files are versioned as source; the generated `*_pb2.py` files are build artifacts.

## What Is Not Versioned (Artifacts)
- `poetry.lock`, `package-lock.json`.
- Build outputs under `/opt/**`, `dist/`, `build/`, `*.egg-info/`.
- Python caches: `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`.
- **Protobuf-generated code**: `*_pb2.py`, `*_pb2.pyi`, `*_pb2_grpc.py` files (generated from `.proto` sources during container build).

## Ignore Policy

- `.gitignore`: exclude lockfiles and build/caches with comments describing why/how to regenerate.
- `.dockerignore`: exclude the same artifacts to keep build context minimal.
- See [Documentation Standards](../documentation_standards.md#gitignore-and-dockerignore-rules) for comment format.

## Regeneration Workflow (Inside Docker)

```bash
# File: documents/engineering/build_artifact_management.md
# Binary or source build resolves deps without host lockfiles
docker compose -f docker/docker-compose.yml exec spectralmc poetry install

# Full test run (captures outputs)
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all > /tmp/test-output.txt 2>&1
```

Artifacts produced during these steps remain inside the container.

## Verification

```bash
# File: documents/engineering/build_artifact_management.md
# Ensure artifacts are ignored
git status --short | grep -E "poetry.lock|package-lock.json|dist/|build/|egg-info|_pb2\.py" && echo "Artifacts present (fix ignores)"

# Confirm Docker context excludes lockfiles and generated code
cd docker && docker build --progress=plain . | grep -E "poetry.lock|package-lock.json|_pb2\.py"

# Verify protobuf-generated code is in /opt (not in repo)
docker compose -f docker/docker-compose.yml exec spectralmc ls -la /opt/spectralmc_proto/*_pb2.py
```

## Cross-References

- [Docker Build Philosophy](docker_build_philosophy.md) — build modes and lockfile handling in Dockerfiles.
- [Docker Workflow](docker_workflow.md) — container-only command contract.
- [Command Reference](command_reference.md) — canonical commands for installs/tests.
- [Documentation Standards](../documentation_standards.md) — linking/metadata/ignore file guidance.

## Protobuf Migration Status

**Current State (Pending Migration)**:
- Protobuf-generated files (`src/spectralmc/proto/*_pb2.py`) are currently checked into version control
- `.proto` source files are in the repository (correct)

**Target State**:
- Generated `*_pb2.py` files should be excluded from git (added to `.gitignore`)
- Generated code should be built into `/opt/spectralmc_proto/` during Docker build
- Source code should import from `/opt/spectralmc_proto/` (not from checked-in files)

**Migration Steps**:
1. Update `.gitignore` to exclude `src/spectralmc/proto/*_pb2.py` and `src/spectralmc/proto/*_pb2.pyi`
2. Update Dockerfile to generate protobuf code during build into `/opt/spectralmc_proto/`
3. Update Python import paths to reference `/opt/spectralmc_proto/`
4. Remove checked-in `*_pb2.py` files from git: `git rm src/spectralmc/proto/*_pb2.py*`

## Maintenance

- Update when new artifact types appear or ignore rules change.
- Keep policy consistent with Dockerfiles (binary vs source builds) and compose args.
