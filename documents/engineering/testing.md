# File: documents/engineering/testing.md
# Testing

**Status**: Reference only
**Supersedes**: None (reference document)
**Referenced by**: engineering/README.md; command_reference.md; documentation_standards.md

> **Purpose**: GPU-only testing standard for SpectralMC, aligned with effectful naming while keeping SpectralMC-specific policies (determinism, CUDA-only).
> **📖 Authoritative Reference**: [Testing Requirements](testing_requirements.md)

## Quick Summary

**Key Testing Expectations:**
- Tests are **GPU-only**: module-level `assert torch.cuda.is_available()` and explicit `torch.device("cuda:0")`; no CPU fallbacks or `pytest.skip`.
- Runner: `docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all` (arguments allowed); never run `pytest` directly.
- Determinism: seed PyTorch/NumPy/CuPy where randomness appears; respect per-test 60s timeout; preserve full logs (no shell timeouts).
- Type safety: tests are fully typed, mypy-strict-clean, and obey purity rules for code under test.

**Related Documentation:**
- [Testing Requirements](testing_requirements.md) — SSoT for GPU enforcement, determinism, timeouts, fixtures, and anti-patterns
- [Testing Architecture](testing_architecture.md) — DRY patterns, helper consolidation, fixture organization
- [Code Quality](code_quality.md) — Purity/type rules that tests must satisfy
- [Docker Workflow](docker_workflow.md) — Docker-only contract for running tests
- [Command Reference](command_reference.md) — Canonical `poetry run test-all` invocations and log handling
- [Documentation Standards](../documentation_standards.md) — Metadata and linking requirements for test docs
