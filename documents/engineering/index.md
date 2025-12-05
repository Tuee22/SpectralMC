# File: documents/engineering/index.md
# Engineering Standards Index

**Status**: Authoritative source  
**Supersedes**: None  
**Referenced by**: documents/documentation_standards.md; documents/README.md; documents/product/index.md; documents/domain/index.md

> **Purpose**: Index and single-source map for SpectralMC engineering standards.

## Cross-References
- [Documentation Standards](../documentation_standards.md)
- [CLAUDE.md](../../CLAUDE.md)

## Canonical Standards (SSoT)
- [Coding Standards](coding_standards.md) — Type safety, ADTs, formatting, stubs.
- [Testing Requirements](testing_requirements.md) — GPU-only testing, determinism, timeouts.
- [Purity Doctrine](purity_doctrine.md) — Pure functions and effect isolation.
- [Immutability Doctrine](immutability_doctrine.md) — Frozen data models and safe updates.
- [Effect Interpreter Doctrine](effect_interpreter.md) — Effect ADTs and interpreter boundaries.
- [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md) — Device placement and execution rules.
- [PyTorch Facade Pattern](pytorch_facade.md) — Deterministic import-order guardrails.
- [Reproducibility Proofs](reproducibility_proofs.md) — Formal guarantees and proofs.
- [Docker Build Philosophy](docker_build_philosophy.md) — Build topology and dependency policy.
- [Blockchain Model Versioning](blockchain_storage.md) — Storage architecture and atomic commits.
- [GPU Build Guide](gpu_build.md) — Legacy GPU build path (reference to Docker policy).
- [Pydantic Best Practices](pydantic_patterns.md) — Configuration models and validators.

## Navigation Notes
- Standards are authoritative; overlays must link here first and only list deltas.
- All documents use snake_case filenames and the shared metadata header defined in `documentation_standards.md`.
- Run link validation after renames/moves and update this index when new standards are added.
