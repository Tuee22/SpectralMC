# File: documents/engineering/testing.md
# Testing

**Status**: Reference only  
**Supersedes**: None (reference document)  
**Referenced by**: engineering/README.md; command_reference.md; documentation_standards.md

> **Purpose**: GPU-only testing standard for SpectralMC, aligned with effectful naming while keeping SpectralMC-specific policies (determinism, CUDA-only).  
> **📖 Authoritative Reference**: [Testing Requirements](testing_requirements.md)

## Cross-References

- [Testing Architecture](testing_architecture.md) — DRY helpers/fixtures and suite layout
- [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md) — Device placement boundaries
- [Total Pure Modelling](total_pure_modelling.md) — Model shapes/states mirrored in fixtures
- [Reproducibility Proofs](reproducibility_proofs.md) — Determinism guarantees
- [Blockchain Storage](blockchain_storage.md) — Commit semantics for snapshot persistence
- [Documentation Standards](../documentation_standards.md) — Metadata/linking rules
- [Docker Workflow](docker_workflow.md) — Container-only execution of `poetry run test-all`

## Quick Summary

- **GPU-only**: global guard asserts CUDA availability; no per-test `torch.cuda.is_available()` checks, no CPU fallbacks/skips. Use `spectralmc.models.torch.Device` rather than raw `torch.device("cuda:0")`.
- **Runner**: `docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all` (arguments allowed); never run `pytest` directly.
- **Determinism**: seed via shared helper (`tests.helpers.seed_all_rngs`) where randomness appears; respect the 60s per-test timeout; preserve full logs (no shell timeouts).
- **DRY helpers**: reuse `tests/helpers` for simulation/config builders (`make_simulation_params`, `make_black_scholes_config`, `make_gbm_cvnn_config`), bounds, param diff, assertions. Avoid inlined RNG capture/config duplication in tests/examples.
- **Type safety/purity**: tests are fully typed, mypy-strict-clean, and obey purity rules for code under test.

**Related Documentation:**
- [Testing Requirements](testing_requirements.md) — SSoT for GPU enforcement, determinism, timeouts, fixtures, and anti-patterns
- [Testing Architecture](testing_architecture.md) — DRY patterns, helper consolidation, fixture organization
- [Code Quality](code_quality.md) — Purity/type rules that tests must satisfy
- [Total Pure Modelling](total_pure_modelling.md) — Models to mirror in fixtures and generators
- [Docker Workflow](docker_workflow.md) — Docker-only contract for running tests
- [Command Reference](command_reference.md) — Canonical `poetry run test-all` invocations and log handling
- [Documentation Standards](../documentation_standards.md) — Metadata and linking requirements for test docs

## Full-Stack CVNN Pricer Test

`tests/test_e2e/test_full_stack_cvnn_pricer.py` exercises the GPU-first training → snapshot → storage → reload → inference path end to end, following the constraints in [testing_requirements.md](testing_requirements.md) and suite layout in [testing_architecture.md](testing_architecture.md).

- **Scope (per [total_pure_modelling.md](total_pure_modelling.md))**: Builds a minimal CVNN via `cvnn_factory` with deterministic seeds, constructs GBM simulation params, and instantiates `GbmCVNNPricer` on `torch.device("cuda:0")` with full-precision dtypes. Randomness is seeded locally to satisfy [reproducibility_proofs.md](reproducibility_proofs.md).
- **Training phase**: Runs a short `TrainingConfig` (small batches/steps) to respect the 60s per-test guard from [testing_requirements.md](testing_requirements.md#timeout-policy) while still mutating weights/optimizer state.
- **Snapshot + storage**: Captures `GbmCVNNPricer.snapshot()` and commits it to the async blockchain store fixture, validating commit semantics from [blockchain_storage.md](blockchain_storage.md). Verifies optimizer state, RNG states, `sobol_skip`, and CVNN parameters are preserved bit-for-bit.
- **Reload + parity checks**: Rehydrates a fresh pricer from the committed snapshot and asserts parameter equality against the pre-commit model using helpers defined in [testing_architecture.md](testing_architecture.md#helper-function-consolidation) and device rules in [cpu_gpu_compute_policy.md](cpu_gpu_compute_policy.md).
- **Inference determinism**: Runs `predict_price` on a small set of `BlackScholes.Inputs` both before and after reload, asserting finite outputs and exact equality to demonstrate deterministic inference through the full pipeline.
- **Execution contract**: Requires GPU availability at module import (global guard), must be invoked via `docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_e2e/test_full_stack_cvnn_pricer.py` per [docker_workflow.md](docker_workflow.md) and [command_reference.md](command_reference.md).
