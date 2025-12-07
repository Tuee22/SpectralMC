# File: documents/functional_refactoring_plan.md
# Functional Refactoring Plan

**Status**: Draft (action plan)  
**Scope**: Full-program purity & Result-based error handling across SpectralMC  
**Audience**: SpectralMC contributors implementing the purity refactor  
**Use**: Running checklist — mark `[ ]` → `[x]` as items complete

> **Objective**: Eliminate raises and imperative control-flow in business logic by threading typed ADTs/Results end-to-end. Preserve fail-fast import guards; keep CUDA kernels imperative but pure (output-only mutation).

---

## Guiding Principles (SSoT: [coding_standards](documents/engineering/coding_standards.md), [purity_doctrine](documents/engineering/purity_doctrine.md), [pydantic_patterns](documents/engineering/pydantic_patterns.md), [documentation_standards](documents/documentation_standards.md))
- **Result everywhere**: Domain operations return `Result[T, E]` with explicit ADT errors; callers pattern-match exhaustively.
- **No `raise` in business logic**: Use ADTs for expected errors; preserve import-time fail-fast for missing hard deps (torch, cupy, etc.).
- **Expression-oriented**: Replace statement loops/ifs with comprehensions, `map/reduce`, `match/case`.
- **Pydantic purity**: Wrap Pydantic construction with `validate_model` → `Result` (see `src/spectralmc/validation.py`). Validators may raise internally; boundary converts to `Result`.
- **CUDA kernel exception**: Imperative loops allowed inside kernels if they only mutate their designated outputs.
- **Strict typing**: No `Any`, `cast`, or `type: ignore`.
- **Consistency rule**: For each domain, keep API/impl/tests in sync before moving on. Do not mix Result returns with silent `None` fallbacks or ad-hoc raises—either propagate `Failure` explicitly or mark the boundary as fail-fast (imports only).
- **Call-site discipline**: Every Result-returning call must be handled (`match`/`case`); no `.unwrap()` or implicit truthiness. Avoid assigning `None` on failure—propagate a typed error instead.
- **Docs/stubs parity**: Update documentation snippets and stubs with the new Result signatures in the same phase; stale examples/tests create drift.
- **Phase acceptance criteria**: For each domain phase, “done” means API updated, ADTs documented, all call sites patched, tests passing, docs/stubs/examples updated, and no lingering raises/imperative loops in that domain (except import fail-fast, CUDA kernels).
- **Scoped rollout**: Keep changes per domain; avoid mixing old/new behaviors. If a shim is needed, isolate it with a TODO for removal in the next phase.
- **Test gating per phase**: Run targeted suites (e.g., sampler tests when touching sampler) plus `poetry run check-code` to prevent drift.
- **Traceability**: In each phase, note the SSoT sections satisfied (purity, typing, import discipline) to keep intent explicit.

---

## Work Breakdown Structure (Checklist)

### 0) Foundations (already added)
- [x] Add `validate_model` helper returning `Result[TModel, ValidationError]`.
- [x] Update doctrine to allow fail-fast imports and Result-wrapped Pydantic usage.

### 1) Define Domain Error ADTs
- [~] Create granular, typed error variants per domain to avoid reusing generic strings (sampler/async normals done; remaining domains pending):
  - [x] **Sampler errors**: `DimensionMismatch`, `InvalidBounds`, `NegativeSamples`, `ValidationFailed`.
  - [x] **Async normals errors**: `InvalidDType`, `InvalidShape`, `SeedOutOfRange`, `QueueEmpty`, `QueueBusy`.
  - [x] **Serialization errors**: `UnknownDType`, `InvalidTensorState`, `UnsupportedPrecision`, `TensorProtoMismatch`.
  - [x] **Torch facade errors**: `UnsupportedTorchDType`, `InvalidDevice`, `InvalidAdamState`, `TensorStateOnGPU`.
  - [x] **Trainer errors**: `CommitFailed`, `CommitSkipped`, `TrainingStepFailed`, `CheckpointWriteFailed`.
  - [x] **CVNN factory errors**: `UnhandledConfigNode`, `ModelOnWrongDevice`, `SerializationDeviceMismatch`.
  - [x] **Sobol/Black-Scholes config errors**: `InvalidConfig`, `InvalidSimulationParams`.

- Deliverables:
  - [x] `spectralmc/errors/<domain>.py` modules with frozen dataclasses or Literal-tagged sum types. (Sampler)
  - [x] Aggregated `Result` aliases in each domain for ergonomic signatures. (`GBMConfigResult` introduced for the new config ADTs)

### 2) Sampler Layer (`sobol_sampler.py`)
- [x] Replace constructor raises with a pure factory: `build_sobol_sampler(...) -> Result[SobolSampler[T], SobolSamplerError]`.
- [x] Replace `sample` raises with `Result[list[T], SobolSamplerError]`.
- [x] Thread Pydantic validation through `validate_model`; convert failures to `Failure`.
- [~] Update call sites: `gbm_trainer`, tests (`tests/test_sobol_sampler.py`, `tests/test_gbm.py`), docs/examples to `match` on `Result`. (tests updated; trainer now returns `Failure(SamplerInitFailed)` on sampler errors—docs/examples still pending)

### 3) Async Normals (`async_normals.py`)
- [x] Define ADTs for dtype/shape/seed/enqueue errors.
- [x] Convert constructors and methods (`enqueue`, `get_matrix`, `snapshot`, etc.) to return `Result`. (public config creation now via `ConcurrentNormGeneratorConfig.create` → `validate_model`)
- [x] Replace runtime raises with `Failure`; keep CUDA kernel imperative body.
- [x] Update tests to `match` on `Result` and assert error variants. (validation tests now use `validate_model`)

### 4) Serialization (`serialization/common.py`, `serialization/tensors.py`, `serialization/simulation.py`, `serialization/models.py`)
- [x] Introduce serialization ADTs and convert converters to `Result` (common/simulation/models/tensors now signal typed failures).
- [x] Convert remaining helpers to `Result` return types; remove any residual raises (Adam/Model checkpoint converters now emit `SerializationResult`).
- [x] Update consumers (trainer, storage, tests) to handle `Failure` (storage/test coverage updated; other surfaces still to follow).

### 5) Torch Facade & Numerics (`models/torch.py`, `models/numerical.py`, `models/cpu_gpu_transfer.py`)
- [x] Add error ADTs for unsupported dtype/device and invalid Adam state.
- [x] Wrap constructors/helpers in `Result`; keep import-time ImportError as fail-fast (per doctrine).
- [x] Update tensor-state serializers/deserializers to emit `Result` (`TensorState`/`AdamOptimizerState` now return `TorchFacadeResult` and get converted downstream).
- [x] Refactor call sites (trainer, serialization) to pattern-match.

### 6) CVNN Factory (`cvnn_factory.py`)
- [x] Replace `RuntimeError` paths with error ADTs and `Result`-returning factory (builder + load/get now return `CVNNFactoryResult`, plus new `cvnn_factory` errors).
- [x] Update consumers/tests to `match` on `Result` (all CVNN/test helpers now unwrap via `_expect_success`).

### 7) Trainer (`gbm_trainer.py`)
- [x] Introduce trainer error ADTs (commit failure, effect failure, invalid config). (`SamplerInitFailed`, `InvalidTrainerConfig` added; commit/storage shims pending)
- [x] Make public entry points (`train`, `train_via_effects`, `snapshot`, `predict_price`) return `Result`. (`train`/`train_via_effects` return `Result`; `predict_price` now returns `Result` as well).
- [x] Thread downstream Result handling from sampler, async_normals, serialization, torch facade. (MC price path propagates `NormalsUnavailable/NormalsGenerationFailed`; serialization/torch now matching.)
- [x] Convert effect-building to propagate `Result` errors instead of raising.
- [x] Update tests/examples to pattern-match and assert error variants.

### 8) GBM / Simulation (`gbm.py`)
- [x] Wrap configuration/build paths in `Result`; keep CUDA kernel loops as-is but ensure inputs validated via Result before launch. (`build_simulation_params` and `build_black_scholes_config` now return `Result`; serialization + tests call these builders before instantiating `BlackScholes`)
- [x] Update callers to pattern-match. (trainer/test call sites updated; docs/examples now show Result handling)

### 9) Tests & Examples
- [x] Update all tests to assert on `Success`/`Failure` variants instead of expecting exceptions. (See `tests/test_gbm`, checker/trainer tests, blockchain/storage suites now matching on builder results)
- [x] Replace `pytest.raises` where applicable (except for import fail-fast or system-boundary allowances). (Nothing new added for config – all new builders match rather than raise)
- [x] Update examples to show Result usage patterns. (`training_with_blockchain_storage`, inference clients now pattern-match builder results)

### 10) Documentation & Stubs
- [~] Refresh engineering docs to reflect Result-based APIs and error ADTs per domain. (Training integration doc, plan, and examples highlight builder flow; other docs/stubs may still need review)
- [ ] Ensure stubs mirror new signatures (no `Any`, no `cast`, no `type: ignore`).
- [~] Update README/CLAUDE/AGENTS references if API changes are user-visible. (No user-facing docs yet updated – pending review)

---

## Sequencing & Incremental Delivery
1. Sampler layer (safe, contained surface).  
2. Async normals (GPU-focused but isolated).  
3. Serialization (shared surface; unlocks trainer refactor).  
4. Torch facade/numerics (shared utilities).  
5. CVNN factory (narrow surface).  
6. Trainer (largest blast radius).  
7. GBM configuration paths.  
8. Tests/examples/doc refresh.

Each phase should:
- Add ADTs and pure factories.
- Migrate one module’s public surface to `Result`.
- Update all call sites + tests.
- Run `poetry run check-code` + targeted tests.
- Document changes in changelog/README as needed.

---

## Constraints & Exceptions
- **Allowed raises**: Import-time failures for hard deps (torch, cupy) remain raises (fail-fast).
- **CUDA kernels**: Imperative bodies allowed; must not mutate inputs except designated outputs.
- **Storage/effects layers**: Already allowed impurity per doctrine; keep boundary clean when calling into business logic.

---

## Success Criteria
- Zero `raise` in business logic modules (except import guards).
- Public APIs return `Result[...]` with explicit error ADTs.
- No statement-level loops/ifs in business logic; expression-oriented or match/case.
- Tests and examples use `Result` handling; no `pytest.raises` for business logic errors.
- Docs/stubs reflect new signatures; `check-code` and suites pass. 
