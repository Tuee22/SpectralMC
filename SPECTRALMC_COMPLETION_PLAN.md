# SpectralMC Completion Plan: Blockchain Model Store

## Executive Summary

This plan implements a **blockchain-like model versioning system** for SpectralMC that provides mathematical guarantees of determinism and atomicity for continuous online learning.

### Core Guarantees

1. **Deterministic Training**: Two trainers starting from identical checkpoint produce bit-identical results
2. **Sequential Commits**: Training steps form an immutable chain (Git-like fast-forward only)
3. **Atomic Linearization**: Concurrent commits resolve to single linear chain via optimistic concurrency
4. **Reproducibility Forever**: Retraining from any checkpoint in history reproduces the exact remainder of chain
5. **Tamper Detection**: Content hashes form Merkle chain where corruption is cryptographically detectable

### Architectural Principles

- **No Illegal States**: All conditional parameters replaced with algebraic data types
- **Type Safety**: 100% mypy strict compliance, zero escape hatches
- **Async-First**: All I/O operations use asyncio for non-blocking concurrency
- **Protobuf Wire Format**: Efficient binary serialization with schema evolution
- **Pydantic Validation**: Runtime validation of business logic at deserialization boundaries
- **MinIO Storage**: S3-compatible object store with atomic conditional writes (If-Match headers)
- **TensorBoard Metrics**: Version-specific training metrics with full history preservation

---

## Phase 1: Eliminate Illegal States (Bottom-Up, Incremental Testing)

### Objective

Refactor all function signatures and class constructors to make illegal states unrepresentable at compile time using algebraic data types, Literal types, and Pydantic validators.

**Development Strategy**: Build bottom-up with testing after each sub-phase. Each sub-phase is independently testable and builds on previous work without breaking existing functionality.

---

### Sub-Phase 1A: Core Type System (COMPLETED ✅)

**Modules**: Foundation types with no external dependencies

**Status**: COMPLETED - All tests passing (20/20)

### 1.1 Device Transfer Destination (cpu_gpu_transfer.py)

**Problem**: Boolean `pin_memory` parameter is meaningless when target device is CUDA.

**Files Modified**: `src/spectralmc/models/cpu_gpu_transfer.py`

**Changes Required**:
- Create `TransferDestination` enum with variants: CPU, CPU_PINNED, CUDA
- Replace all `move_tensor_tree()` signatures to use single enum parameter instead of `(device, pin_memory)` tuple
- Update `_copy_tensor()` internal function similarly
- Update all call sites across codebase (search for `move_tensor_tree` and `pin_memory`)
- Add methods to enum for converting to torch.device and extracting pinning boolean

**Success Criteria**:
- Cannot construct call with `pin_memory=True` when destination is CUDA
- Mypy enforces all call sites use enum
- All existing tests pass with identical behavior

### 1.2 DType Precision Split (torch.py)

**Problem**: `DType.to_precision()` raises KeyError for float16/bfloat16 which have no Precision equivalent.

**Files Modified**: `src/spectralmc/models/torch.py`, `src/spectralmc/gbm.py`

**Changes Required**:
- Split `DType` enum into two separate enums: `FullPrecisionDType` (float32/64, complex64/128) and `ReducedPrecisionDType` (float16, bfloat16)
- Create `AnyDType` union type for contexts accepting either variant
- Add `.to_precision()` method only to `FullPrecisionDType` (type system prevents calling on reduced precision)
- Update `TensorState` to accept `AnyDType` (serialization supports all types)
- Update `SimulationParams` to require `FullPrecisionDType` (MC simulation requires full precision)
- Update all type hints and imports across codebase

**Success Criteria**:
- Impossible to call `.to_precision()` on float16/bfloat16 (compile-time error)
- Impossible to create MC simulation with reduced precision dtypes
- Can still serialize tensors in any dtype

### 1.3 Explicit Width Specification (cvnn_factory.py)

**Problem**: `width: Optional[int] = None` has implicit "preserve input width" semantics.

**Files Modified**: `src/spectralmc/cvnn_factory.py`

**Changes Required**:
- Create `WidthSpec` base class with frozen Pydantic model
- Create two concrete subclasses: `PreserveWidth` (tag-only) and `ExplicitWidth(value: int)`
- Update `LinearCfg.width` field to use `WidthSpec` type with `PreserveWidth()` as default
- Update factory function `_make_linear()` to pattern match on width spec variants
- Update all config construction sites to use explicit types

**Success Criteria**:
- Intent is explicit at construction (no ambiguous None values)
- Factory logic uses pattern matching instead of None checks
- Behavior identical to current implementation

### 1.4 Validated Training Config (gbm_trainer.py)

**Problem**: Training hyperparameters passed as loose function arguments without cross-field validation.

**Files Modified**: `src/spectralmc/gbm_trainer.py`, test files using `train()` method

**Changes Required**:
- Create `TrainingConfig` Pydantic model with frozen=True
- Add field validators: `num_batches: PositiveInt`, `batch_size: PositiveInt`, `learning_rate: Annotated[float, Field(gt=0.0, lt=1.0)]`
- Add model validator checking `num_batches * batch_size < GPU_MEMORY_LIMIT`
- Replace `train()` method signature to accept single `TrainingConfig` parameter
- Update all call sites (tests and examples)

**Success Criteria**:
- Cannot train with negative batches or unrealistic learning rates
- OOM risk detected at config construction time
- Pydantic validation errors are clear and actionable

### 1.5 CUDA Block Size Constraints (gbm.py)

**Problem**: `threads_per_block: int` allows invalid values (must be power of 2 in range [32, 1024]).

**Files Modified**: `src/spectralmc/gbm.py`

**Changes Required**:
- Define `ThreadsPerBlock = Literal[32, 64, 128, 256, 512, 1024]` type alias
- Update `SimulationParams.threads_per_block` to use literal type
- Add model validator ensuring `network_size * batches_per_mc_run < GPU_MEMORY_LIMIT`

**Success Criteria**:
- Type checker rejects invalid block sizes at compile time
- GPU memory constraints validated at model construction

### 1.6 Buffer Size Validation (async_normals.py)

**Problem**: Buffer size can exceed matrix dimensions (wastes memory).

**Files Modified**: `src/spectralmc/async_normals.py`

**Changes Required**:
- Create `BufferConfig` Pydantic model with `size: PositiveInt`
- Add class method `BufferConfig.create(size, matrix_rows, matrix_cols)` that validates `size <= rows * cols`
- Update `ConcurrentNormGenerator.__init__()` to accept `BufferConfig` instead of raw int

**Success Criteria**:
- Cannot construct oversized buffers
- Validation happens at config creation with clear error messages

### 1.7 Sobol Configuration (sobol_sampler.py)

**Problem**: Redundant runtime validation for non-negative integers.

**Files Modified**: `src/spectralmc/sobol_sampler.py`

**Changes Required**:
- Create `SobolConfig` Pydantic model with `seed: NonNegativeInt` and `skip: NonNegativeInt`
- Update `SobolSampler.__init__()` to accept config object
- Remove manual validation checks (Pydantic handles it)

**Success Criteria**:
- No redundant runtime checks
- Validation is declarative via Pydantic types

### Phase 1 Execution Plan (Bottom-Up, Incremental Testing)

**Sub-Phase 1A: Core Type System** ✅ COMPLETED
- **Sections**: 1.1 (Device Transfer), 1.2 (DType Precision Split)
- **Dependencies**: None (foundation layer)
- **Tests**: `test_models_cpu_gpu_transfer.py`, `test_models_torch.py`
- **Verification**: `docker compose -f docker/docker-compose.yml exec spectralmc pytest tests/test_models_cpu_gpu_transfer.py tests/test_models_torch.py`
- **Result**: ✅ 20/20 tests passing

**Sub-Phase 1B: Simple Configurations** ✅ COMPLETED
- **Sections**: 1.7 (Sobol Configuration), 1.6 (Buffer Size Validation)
- **Dependencies**: Only Pydantic (no domain logic)
- **Tests**: `test_sobol_sampler.py`, `test_async_normals.py`
- **Verification**: `docker compose -f docker/docker-compose.yml exec spectralmc pytest tests/test_sobol_sampler.py tests/test_async_normals.py`
- **Result**: ✅ 18/18 tests passing (11 sobol, 7 async_normals)

**Sub-Phase 1C: Simulation Parameters**
- **Sections**: 1.5 (CUDA Block Size Constraints in gbm.py)
- **Dependencies**: FullPrecisionDType from 1A, Pydantic
- **Tests**: `test_gbm.py`
- **Verification**: `docker compose -f docker/docker-compose.yml exec spectralmc pytest tests/test_gbm.py`
- **Rationale**: Uses FullPrecisionDType, must come after 1A

**Sub-Phase 1D: Model Configuration**
- **Sections**: 1.3 (Explicit Width Specification in cvnn_factory.py)
- **Dependencies**: Pydantic, core types
- **Tests**: `test_cvnn_factory.py`
- **Verification**: `docker compose -f docker/docker-compose.yml exec spectralmc pytest tests/test_cvnn_factory.py`
- **Rationale**: Factory patterns, model building logic

**Sub-Phase 1E: Training Configuration**
- **Sections**: 1.4 (Validated Training Config in gbm_trainer.py)
- **Dependencies**: All above sections (uses GBM, CVNN, etc.)
- **Tests**: `test_gbm_trainer.py`
- **Verification**: `docker compose -f docker/docker-compose.yml exec spectralmc pytest tests/test_gbm_trainer.py`
- **Rationale**: Highest-level abstraction, depends on all previous work

**Sub-Phase 1F: Full Integration**
- **Verification**: Run complete test suite
- **Command**: `docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all > /tmp/phase1-complete.txt 2>&1`
- **Success Criteria**: All existing tests pass, mypy --strict passes

### Phase 1 Summary

**Files Modified**: 7 files
**New Types Created**: 8 ADTs/enums/configs
**Illegal States Eliminated**: 7 categories
**Estimated Time**: 1-2 days (with incremental testing: ~2-3 hours per sub-phase)

**Type Safety Requirements**:
- ✅ `mypy --strict` must pass with **zero errors**
- ❌ **NO** `Any` types (explicit or implicit)
- ❌ **NO** `cast()` expressions
- ❌ **NO** `# type: ignore` comments
- ✅ All functions have complete type hints
- ✅ All Pydantic models use `ConfigDict(extra="forbid")`

**Docker Verification Commands**:
```bash
# Type checking (strict, no Any/cast/type:ignore allowed)
docker compose -f docker/docker-compose.yml exec spectralmc mypy src/spectralmc --strict --disallow-any-explicit > /tmp/mypy-phase1.txt 2>&1

# Check for forbidden constructs
docker compose -f docker/docker-compose.yml exec spectralmc grep -r "# type: ignore" src/spectralmc/ && echo "FOUND type:ignore" || echo "OK: No type:ignore"
docker compose -f docker/docker-compose.yml exec spectralmc grep -r "cast(" src/spectralmc/ && echo "FOUND cast()" || echo "OK: No cast()"

# Full test suite
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all > /tmp/test-phase1.txt 2>&1
```

**Type Safety Status (as of 2025-11-06)**:

*Critical Type Errors Fixed* (3 errors):
- ✅ `gbm.py:204` - ConcurrentNormGenerator now uses BufferConfig.create()
- ✅ `gbm_trainer.py:217` - Added type narrowing for AnyDType → FullPrecisionDType
- ✅ `gbm_trainer.py:241` - SobolSampler now uses SobolConfig

*Remaining Work* (25 explicit Any errors):
These require replacing `Any` annotations with concrete types. Files affected:
- `sobol_sampler.py` - lines 50, 71 (2 errors)
- `async_normals.py` - lines 86, 135 (2 errors)
- `gbm.py` - lines 51, 56, 90, 153, 163, 170, 178 (7 errors)
- `models/torch.py` - lines 243, 314, 345, 405, 431 (5 errors)
- `cvnn_factory.py` - lines 53, 57, 64, 73, 87, 93, 100 (7 errors)
- `gbm_trainer.py` - lines 121, 133 (2 errors)

**Note**: These Any types are primarily in:
1. Pydantic `arbitrary_types_allowed` model configurations (can use `object` instead)
2. Generic type parameters that need proper TypeVars or Protocol definitions
3. Optimizer state dictionaries that need structured types

All `cast()` and `# type: ignore` already eliminated ✅

---

## Phase 2: Protocol Buffer Schema Layer

### Objective

Define complete Protobuf schemas for all Pydantic models, enabling efficient binary serialization with forward/backward compatibility.

### 2.1 Dependencies and Setup

**Files Modified**: `pyproject.toml`

**Changes Required**:
- Add `aioboto3 = ">=13.0,<14.0"` for async S3 client
- Add `types-aiobotocore = {version = ">=2.13,<3.0", extras = ["s3"]}` for type stubs
- Add `tensorboard = ">=2.16,<3.0"` for training metrics visualization
- Verify `protobuf = ">=6.30,<7.0"` already present

**Directory Created**: `src/spectralmc/proto/`

**Files Created in proto/**:
- `__init__.py` (re-exports all generated _pb2 modules)
- `common.proto`
- `tensors.proto`
- `simulation.proto`
- `training.proto`
- `models.proto`

### 2.2 common.proto

**Contents**:
- Enum definitions: `Precision`, `Device`, `TransferDestination`, `FullPrecisionDType`, `ReducedPrecisionDType`
- `ModelVersion` message: semantic version, counter, content hash, parent hash (blockchain link), creation metadata, global training step, **TensorBoard logdir path**
- `TorchEnv` message: PyTorch version, CUDA version, GPU name, hostname, snapshot timestamp
- `ArchitectureFingerprint` message: structure hash, parameter count, human-readable description

**Design Notes**:
- Use `google.protobuf.Timestamp` for datetime fields
- Store hashes as `bytes` (32 bytes for SHA256)
- Version counter is `uint64` (monotonically increasing, never wraps)
- TensorBoard logdir is S3 path (e.g., `s3://bucket/tensorboard/v0000000042/`)

### 2.3 tensors.proto

**Contents**:
- `TensorState` message: SafeTensors blob, shape array, dtype (oneof union for full/reduced), SHA256 checksum
- `AdamParamState` message: step counter, exp_avg tensor, exp_avg_sq tensor, optional max_exp_avg_sq (AMSGrad)
- `AdamParamGroup` message: parameter indices, all Adam hyperparameters (lr, betas, eps, etc.)
- `AdamOptimizerState` message: map of parameter states, repeated param groups
- `RNGState` message: NumPy/PyTorch/CUDA RNG serialized states, skip offsets map for Sobol/normal generators
- `ModelCheckpoint` message: version metadata, environment, architecture fingerprint, model state dict, optimizer state, RNG state, global step, training loss, timestamp

**Design Notes**:
- Use `map<int32, AdamParamState>` for efficient parameter state lookup
- RNG state includes skip offsets to enable deterministic fast-forward
- Checkpoint is complete snapshot of all mutable state

### 2.4 simulation.proto

**Contents**:
- `SimulationParams` message: timesteps, network size, batches per run, threads per block, seed, buffer size, skip, dtype
- `BlackScholesConfig` message: simulation params, log return flag, normalize forwards flag
- `OptionContract` message: X0, K, T, r, d, v (all double)
- `PricingResults` message: put/call prices, intrinsic values, convexity
- `BoundSpec` message: lower/upper bounds for Sobol sampler
- `SobolConfig` message: seed, skip, dimensions map

### 2.5 training.proto

**Contents**:
- `TrainingConfig` message: num batches, batch size, learning rate, optional LR schedule
- `StepMetrics` message: step number, batch time, loss, gradient norm, learning rate, timestamp
- `GbmCVNNPricerConfig` message: BlackScholes config, Sobol config, CVNN config reference (string key), optimizer state, global step

**Design Notes**:
- CVNN architecture stored separately (large, referenced by key)
- Enables lightweight config updates without reserializing full model

### 2.6 models.proto

**Contents**:
- Enums: `ActivationKind`, `LayerKind`
- `WidthSpec` message: oneof union of `PreserveWidth` (empty) and `ExplicitWidth(value)`
- Layer config messages: `LinearCfg`, `NaiveBNCfg`, `CovBNCfg`, `SequentialCfg`, `ResidualCfg`
- `LayerCfg` message: oneof union of all layer types
- `CVNNConfig` message: dtype, repeated layers, seed, optional final activation

**Design Notes**:
- Use oneof for ADT-style tagged unions
- Recursive messages (SequentialCfg contains repeated LayerCfg)

### 2.7 Build Process

**Commands to Run**:
- `cd src/spectralmc/proto && protoc --python_out=. --pyi_out=. *.proto`
- Generates `*_pb2.py` (runtime) and `*_pb2.pyi` (type stubs) for each .proto file

**Generated Files**: 10 files (5 runtime + 5 stubs)

### Phase 2 Summary

**Files Created**: 6 Protobuf schemas + 1 __init__.py
**Generated Files**: 10 Python files
**Estimated Time**: 1 day

**Verification Commands**:
- `ls src/spectralmc/proto/*_pb2.py` (verify generation)
- `python -c "from spectralmc.proto import common_pb2; print(common_pb2.Precision.FLOAT32)"` (verify imports)
- `mypy src/spectralmc/proto/` (verify type stubs)

---

## Phase 3: Pydantic ↔ Protobuf Conversion Layer

### Objective

Implement type-safe bidirectional converters between Pydantic models (runtime validation) and Protobuf messages (wire format).

### 3.1 Module Structure

**Directory Created**: `src/spectralmc/serialization/`

**Files Created**:
- `__init__.py`: Protocol definitions, utility functions (compute_sha256, verify_checksum)
- `common.py`: Converters for Precision, Device, DType enums
- `tensors.py`: Converters for TensorState, AdamOptimizerState, RNGState
- `simulation.py`: Converters for SimulationParams, BlackScholesConfig, SobolConfig
- `training.py`: Converters for TrainingConfig, GbmCVNNPricerConfig
- `models.py`: Converters for CVNNConfig, LayerCfg (all variants)

### 3.2 Conversion Protocol

**Define in __init__.py**:
- `ProtoSerializable` Protocol with generic types for Pydantic model and Protobuf message
- Two static methods: `to_proto(pydantic_model) -> proto_message` and `from_proto(proto_message) -> pydantic_model`
- Utility functions for SHA256 checksum computation and verification
- All converters implement this protocol for type safety

### 3.3 Converter Implementation Pattern

**For Each Domain**:
- Create converter class with static methods (no instance state)
- `to_proto()`: Pattern match on Pydantic fields, construct Protobuf message, compute checksums
- `from_proto()`: Extract Protobuf fields, validate checksums, construct Pydantic model (triggers validation)
- Handle oneof unions carefully (tagged unions map to Pydantic discriminated unions)
- Handle optional fields using `HasField()` checks

**Key Decisions**:
- Validation happens during `from_proto()` via Pydantic constructors
- Protobuf defines schema only, no business logic
- Checksums computed during serialization, verified during deserialization
- Conversion failures raise clear exceptions with context

### 3.4 Special Cases

**Recursive Types** (LayerCfg contains SequentialCfg which contains LayerCfg):
- Use recursive function calls for nested conversion
- Maintain type safety through entire traversal

**Union Types** (oneof in Protobuf):
- Use structural pattern matching on `WhichOneof()` result
- Map to Pydantic discriminated unions

**Maps** (optimizer state dict):
- Convert Protobuf map to Python dict preserving types
- Handle integer keys correctly (Protobuf map keys are strings)

### Phase 3 Summary

**Files Created**: 6 converter modules (~1200 LOC)
**Converters Implemented**: ~15 converter classes
**Estimated Time**: 2-3 days

**Verification Strategy**:
- Write round-trip tests for each converter (Pydantic → Proto → Pydantic)
- Verify checksums catch corruption
- Verify validation errors propagate correctly

---

## Phase 4: Blockchain Model Store with Async I/O

### Objective

Implement MinIO-backed model versioning with blockchain guarantees using async I/O for concurrency.

### 4.1 Storage Module Structure

**Directory Created**: `src/spectralmc/storage/`

**Files Created**:
- `__init__.py`: Public API exports
- `errors.py`: Exception hierarchy (StorageError, CommitError, NotFastForwardError, ConflictError, ChecksumError, VersionNotFoundError, ChainCorruptionError)
- `chain.py`: ModelVersion dataclass, blockchain primitives (hash computation, semantic version bumping)
- `store.py`: BlockchainModelStore main class (async S3 operations)
- `inference.py`: InferenceClient (version pinning, polling, hot-swapping)
- `verification.py`: Chain integrity verification utilities

### 4.2 Error Hierarchy

**Define Exception Classes**:
- `StorageError`: Base class
- `CommitError(StorageError)`: Base for commit failures
- `NotFastForwardError(CommitError)`: Parent is not current head (includes parent counter and head counter in message)
- `ConflictError(CommitError)`: ETag mismatch during CAS (includes expected and actual ETags)
- `ChecksumError(StorageError)`: Data corruption detected (includes expected and actual hashes)
- `VersionNotFoundError(StorageError)`: Requested version doesn't exist
- `ChainCorruptionError(StorageError)`: Integrity violation (includes version number and reason)

### 4.3 Blockchain Primitives (chain.py)

**ModelVersion Dataclass**:
- Fields: semantic version string, counter (monotonic int), content hash, parent hash, creation datetime, creator identifier, global step, steps since parent
- Methods: `to_path()` for S3 key generation, `to_dict()` for JSON serialization, `is_genesis()` boolean check, `parse()` class method from S3 metadata
- All fields frozen (immutable)

**Utility Functions**:
- `compute_checkpoint_hash(bytes) -> bytes`: SHA256 hash of checkpoint
- `auto_bump_version(parent_version, architecture_changed, params_changed) -> str`: Determine new semantic version based on change type (major/minor/patch)

### 4.4 Model Store (store.py)

**BlockchainModelStore Class**:

**Initialization**:
- Constructor accepts: bucket name, MinIO endpoint URL, optional prefix
- Creates aioboto3 session (reused for all operations)
- Provides properties for S3 key paths (chain head, version directories)

**Commit Algorithm** (`commit_version()` async method):
1. Validate checkpoint global_step consistency with parent
2. Serialize checkpoint to Protobuf bytes
3. Compute content hash (SHA256)
4. Build ModelVersion metadata with auto-bumped semantic version
5. Upload artifacts in parallel using `asyncio.gather()`: checkpoint.pb, metadata.json, content_hash.txt
6. Fetch current chain head + ETag from S3
7. Verify parent hash matches current head (fast-forward check)
8. Attempt atomic CAS write using `If-Match` header with ETag
9. On **conflict** (another trainer won):
   - Immediately rollback uploaded artifacts (delete version directory)
   - Raise ConflictError with parent/head version info
   - Caller should retry from latest version
10. On **success**:
    - Append to audit log (best-effort, non-critical)
    - Return new ModelVersion
    - **TensorBoard events uploaded separately AFTER commit succeeds**

**Important**: TensorBoard events are NOT uploaded during commit. They are uploaded post-commit to avoid orphaned logs from failed commits.

**Read Operations**:
- `fetch_latest_version()`: Read chain.json, return ModelVersion
- `fetch_checkpoint(version)`: Download checkpoint.pb, verify content hash, deserialize Protobuf
- `list_all_versions()`: Paginate through S3 versions directory, return sorted list

**Concurrency Primitives**:
- `atomic_commit_with_retry()` async context manager: Wraps commit with exponential backoff retry logic
- Automatically refetches latest version after conflicts
- Configurable max retries and base delay

### 4.5 Inference Client (inference.py)

**InferenceClient Class**:

**Modes**:
- Pinned mode (`version != None`): Load specific version, never update
- Tracking mode (`version == None`): Poll for updates, hot-swap model atomically

**Lifecycle**:
- `start()` async method: Load initial model, spawn background polling task (if tracking)
- `stop()` async method: Cancel polling task, cleanup resources
- `predict()` async method: Run model in thread pool (blocking operation), return results

**Polling Mechanism**:
- Background asyncio task sleeps for `poll_interval` seconds
- Fetches latest version from store
- Compares counter to current version
- If newer: load new checkpoint, build model, atomically swap reference
- Model reference swap is atomic due to Python GIL (single assignment)

**Thread Safety**:
- Predictions and polling run concurrently
- Model reference read is atomic (no locks needed)

### 4.6 Chain Verification (verification.py)

**verify_chain() Async Function**:
- Algorithm:
  1. Fetch all versions from store
  2. Verify genesis block (counter=0, empty parent_hash)
  3. Iterate through chain verifying:
     - Parent hash matches previous content hash (Merkle chain property)
     - Counter increments by exactly 1
     - Global step increases monotonically
     - Content hash matches actual checkpoint hash
  4. Return True if intact, raise ChainCorruptionError on first violation

**find_corruption() Async Function**:
- Wrapper that catches ChainCorruptionError and returns corrupted version
- Returns None if chain is intact

### 4.7 S3 Directory Structure

**Bucket Layout**:
```
s3://bucket/prefix/
  ├── chain.json                    # Atomic pointer to head (versioned via ETag)
  ├── audit_log.jsonl               # Append-only history (includes rejected commits)
  ├── tensorboard/                  # TensorBoard event logs (version-organized)
  │   ├── v0000000000/
  │   │   └── events.out.tfevents...
  │   ├── v0000000001/
  │   │   └── events.out.tfevents...
  │   └── v0000000042/
  │       └── events.out.tfevents...
  └── versions/
      ├── v0000000000_1.0.0_hash/
      │   ├── checkpoint.pb
      │   ├── metadata.json
      │   └── content_hash.txt
      ├── v0000000001_1.0.1_hash/
      └── v0000000042_1.2.3_hash/
```

**Design Rationale**:
- Immutable versions (never modified after creation)
- Single mutable file (chain.json) with atomic updates
- Audit log captures all commit attempts for debugging
- Content-addressable via hash suffix (enables deduplication)
- TensorBoard logs organized by version counter for easy navigation

### 4.8 TensorBoard Integration

**Objective**: Enable viewing training metrics for any version in the model chain through TensorBoard UI.

**Architecture Overview**:
- Training loop writes TensorBoard events to version-specific S3 directories
- Each version has isolated TensorBoard logdir: `s3://bucket/tensorboard/v{counter:010d}/`
- TensorBoard natively reads from S3 (via `tensorboard --logdir=s3://...`)
- User can view single version, multiple versions (comparison), or entire chain history

**Training-Side Integration**:

**Metrics Writer Class** (create in `src/spectralmc/storage/tensorboard.py`):
- `TensorBoardWriter` class wraps `torch.utils.tensorboard.SummaryWriter`
- Constructor accepts: store reference, version counter (pending), S3 bucket/prefix
- Writes to **local temp directory** during training (fast, non-blocking)
- Metrics logged: loss, gradient norm, learning rate, batch time
- On close: does NOT automatically upload to S3
- Provides `upload_to_s3()` async method to upload events AFTER successful commit

**Two-Phase Commit Pattern**:
```
Phase 1: Train with local TensorBoard
  ├─ writer = TensorBoardWriter(temp_dir="/tmp/tb_v11")
  ├─ Train model, log metrics locally
  └─ writer.close()

Phase 2: Commit checkpoint
  ├─ version = await store.commit_version(checkpoint, parent)
  └─ On success:
      └─ await writer.upload_to_s3(version.tensorboard_logdir)
```

**Commit Integration**:
- TensorBoard logdir path computed from version counter AFTER commit succeeds
- Path stored in `ModelVersion.tensorboard_logdir` field (Protobuf schema)
- Upload happens asynchronously after commit, but before returning to caller
- On commit failure: local temp directory cleaned up, no S3 pollution

**Benefits of Post-Commit Upload**:
- No orphaned TensorBoard logs from failed commits
- Clean S3 storage (only committed versions have metrics)
- Failed trainers don't waste bandwidth uploading events that will be discarded

**Viewing-Side Integration**:

**CLI Tool** (create `src/spectralmc/storage/tensorboard_cli.py`):
- `launch_tensorboard()` function: Given version counter(s), construct logdir_spec and spawn TensorBoard
- Single version mode: `launch_tensorboard(store, version=42)` → `tensorboard --logdir=s3://bucket/tensorboard/v0000000042`
- Comparison mode: `launch_tensorboard(store, versions=[40, 41, 42])` → TensorBoard with multiple runs
- Full history mode: `launch_tensorboard(store, versions=None)` → All versions as runs
- Returns TensorBoard process handle and URL

**TensorBoard UI Usage**:
- TensorBoard displays each version as a separate run named `v0000000042`
- User can filter runs by regex (e.g., `v000000042|v000000043` to view specific versions)
- User can compare metrics across versions using TensorBoard's built-in comparison features
- Scalars, histograms, and other logged data are version-specific

**Version Selection UX**:

**Option 1 - Command-Line Launch**:
```bash
# View single version
python -m spectralmc.storage.tensorboard_cli --version 42

# Compare multiple versions
python -m spectralmc.storage.tensorboard_cli --versions 40,41,42

# View full history
python -m spectralmc.storage.tensorboard_cli --all
```

**Option 2 - Python API**:
```python
from spectralmc.storage import BlockchainModelStore
from spectralmc.storage.tensorboard_cli import launch_tensorboard

store = BlockchainModelStore(...)

# Launch TensorBoard for specific version
tb_proc, tb_url = launch_tensorboard(store, version=42)
print(f"TensorBoard running at {tb_url}")

# Compare three versions
launch_tensorboard(store, versions=[40, 41, 42])
```

**Option 3 - Manifest File for TensorBoard**:
- Generate `tensorboard_manifest.txt` mapping friendly names to S3 paths
- Content: `v42:s3://bucket/tensorboard/v0000000042,v43:s3://bucket/tensorboard/v0000000043`
- Launch: `tensorboard --logdir_spec=$(cat tensorboard_manifest.txt)`

**Implementation Details**:

**Async Metrics Upload**:
- TensorBoard writes to local temp directory first (fast, non-blocking)
- Background thread uploads tfevents files to S3 incrementally
- On flush/close, ensure all events uploaded before returning
- Use aioboto3 for concurrent uploads of multiple event files

**Metrics to Log** (align with `StepMetrics` Protobuf message):
- Scalars: loss, gradient norm, learning rate, batch time
- Histograms: parameter distributions, gradient distributions
- Custom metrics: convergence indicators, option pricing errors
- Global step aligned with checkpoint global_step for consistency

**TensorBoard S3 Configuration**:
- TensorBoard supports S3 via `tensorflow` or standalone S3 filesystem
- Requires AWS credentials (or MinIO credentials configured as AWS-compatible)
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL` (for MinIO)
- Alternative: Use `s3fs` Python package as backend

**Files to Create**:
- `src/spectralmc/storage/tensorboard.py`: TensorBoardWriter class
- `src/spectralmc/storage/tensorboard_cli.py`: CLI tool for launching TensorBoard
- `tests/test_storage/test_tensorboard.py`: Tests for metrics writing and retrieval

**Integration with Training Loop**:
- Modify `GbmCVNNPricer.train()` to accept optional `TensorBoardWriter`
- Writer logs to local temp directory during training
- Write metrics after each batch: `tb_writer.add_scalar('loss', loss, global_step)`
- After successful commit, upload events: `await tb_writer.upload_to_s3(version.tensorboard_logdir)`
- On commit failure, clean up temp directory

**Training Workflow Example**:
```python
# Create local TensorBoard writer
tb_writer = TensorBoardWriter(temp_dir="/tmp/training")

# Train with local logging
trainer.train(config, tensorboard_writer=tb_writer)
tb_writer.close()

# Attempt commit
try:
    version = await store.commit_version(checkpoint, parent, "trainer-1")
    # Upload TensorBoard on success
    await tb_writer.upload_to_s3(version.tensorboard_logdir)
except ConflictError:
    # Clean up temp directory
    tb_writer.cleanup()
    # Retry from latest version
```

**Benefits**:
- Full training history preserved and visualizable
- Compare metrics across versions to track model evolution
- Debug training issues by examining metrics at any point in chain
- Version-specific metrics isolated (no cross-contamination)
- Leverage TensorBoard's mature UI (no custom dashboard needed)
- No orphaned metrics from failed commits

### 4.9 Garbage Collection and Orphaned Artifact Cleanup

**Objective**: Clean up incomplete commits and orphaned artifacts from failed/crashed trainers.

**Orphaned Artifacts Scenarios**:

1. **Incomplete Rollback**: Trainer crashes after uploading checkpoint but before rollback completes
   - Leaves orphaned `versions/v{counter}/` directory in S3
   - Directory exists but is not referenced in `chain.json`

2. **Network Failures**: Partial uploads where some files written but commit never attempted
   - Incomplete version directories (missing metadata.json or content_hash.txt)

3. **Race Condition Residue**: Multiple trainers attempting same counter, losing trainer's artifacts not fully cleaned

**Detection Algorithm**:

**Orphaned Version Detection** (create in `src/spectralmc/storage/garbage_collection.py`):
```
1. Fetch all version counters from chain.json history
2. List all directories in s3://bucket/versions/
3. For each directory:
   - Extract counter from path (v{counter}_...)
   - Check if counter exists in chain
   - If NOT in chain: mark as orphaned
   - If in chain but incomplete (missing files): mark as corrupted
4. Return list of orphaned/corrupted version paths
```

**Cleanup Strategy**:

**Conservative Cleanup** (default):
- Only delete versions older than `grace_period` (default: 24 hours)
- Rationale: Gives trainers time to retry after transient failures
- Never delete recent artifacts (might be mid-commit)

**Aggressive Cleanup** (manual):
- Delete all orphaned artifacts immediately
- Use with caution (might delete in-progress uploads)

**Implementation**:

**GarbageCollector Class**:
```python
class GarbageCollector:
    def __init__(self, store: BlockchainModelStore, grace_period_hours: int = 24)

    async def find_orphaned_versions() -> List[OrphanedArtifact]:
        """Scan S3 and identify orphaned artifacts."""
        # Returns list of orphaned version paths with metadata

    async def cleanup_orphaned_versions(dry_run: bool = True) -> CleanupReport:
        """Delete orphaned artifacts (with dry-run option)."""
        # Returns report of deleted artifacts and bytes freed

    async def verify_version_completeness(version: int) -> bool:
        """Check if version directory has all required files."""
        # Returns True if checkpoint.pb, metadata.json, content_hash.txt exist
```

**Cleanup Schedule**:

**Option 1 - Manual Cleanup**:
```bash
# Dry run (show what would be deleted)
python -m spectralmc.storage.garbage_collection --dry-run

# Execute cleanup
python -m spectralmc.storage.garbage_collection --grace-period 24
```

**Option 2 - Automated Cleanup**:
- Run as cron job or scheduled task
- Execute daily: `0 2 * * * python -m spectralmc.storage.garbage_collection`
- Logs cleanup report to audit log

**Option 3 - Integrated Cleanup**:
- Run garbage collection automatically after N failed commits
- Configurable threshold: `auto_cleanup_after_failures=10`
- Cleans up orphaned artifacts from recent failed commits

**Safety Mechanisms**:

1. **Grace Period**: Never delete artifacts newer than grace_period
2. **Dry Run First**: Default mode is dry-run (explicit flag required for deletion)
3. **Audit Logging**: All deletions logged to `audit_log.jsonl` with timestamps and reasons
4. **Incremental Deletion**: Delete one version at a time (avoid mass deletion bugs)
5. **Verification**: Re-verify chain integrity after cleanup

**Metrics to Track**:
- Number of orphaned artifacts detected
- Total bytes reclaimed
- Cleanup execution time
- Errors encountered during cleanup

**Edge Cases**:

**Concurrent Cleanup**: Multiple cleanup processes running simultaneously
- Use S3 object locks or advisory locks (future enhancement)
- Current: Accept potential race conditions (cleanup is idempotent)

**Version Counter Gaps**: Missing counters in chain (e.g., v10, v11, v13 - missing v12)
- Not necessarily orphaned (counter might be skipped intentionally)
- Only delete if directory exists but not referenced in chain

**Partial Chains**: Chain verification fails partway through
- Don't cleanup until chain is verified (corruption might be deeper issue)
- Fix chain integrity first, then cleanup

**Files to Create**:
- `src/spectralmc/storage/garbage_collection.py`: GarbageCollector class
- `src/spectralmc/storage/__main__.py`: CLI entry point for manual cleanup
- `tests/test_storage/test_garbage_collection.py`: Tests for cleanup logic

**Integration with Phase 5 Testing**:
- Test orphaned artifact detection
- Test cleanup with grace period
- Test that cleanup doesn't delete recent artifacts
- Test cleanup after concurrent commit conflicts

### Phase 4 Summary

**Files Created**: 10 modules (~1200 LOC)
- Core storage: errors.py, chain.py, store.py, inference.py, verification.py
- TensorBoard: tensorboard.py, tensorboard_cli.py
- Garbage collection: garbage_collection.py, __main__.py
- Tests: test_tensorboard.py, test_garbage_collection.py

**Key Operations**: Async commit, fetch, verify, hot-swap, TensorBoard metrics logging, garbage collection
**Estimated Time**: 4-5 days (increased from 3-4 due to TensorBoard + GC)

**Verification Strategy**:
- Unit tests for each storage operation
- Integration tests with real MinIO instance
- Concurrency tests with multiple async clients (test orphaned artifact creation)
- TensorBoard integration tests (write metrics, view in UI)
- Garbage collection tests (orphaned detection, cleanup with grace period)

---

## Phase 5: Comprehensive Test Suite

### Objective

Write end-to-end tests demonstrating determinism, atomicity, and reproducibility with 100% coverage of storage layer.

### 5.1 Test Infrastructure

**Directory Created**: `tests/test_storage/`

**Shared Fixtures** (`conftest.py`):
- MinIO test container fixture (or mock)
- Empty store fixture with cleanup
- Genesis version fixture
- Dummy checkpoint builders
- Helper functions for checkpoint comparison
- Process spawning utilities for out-of-process tests
- Byte-level comparison utilities for determinism verification

**Test Categories**:

**In-Process Tests**: Run within single Python process
- Fast execution, easier debugging
- Test algorithmic correctness and logic
- Use asyncio for concurrency simulation
- **Limitation**: Don't test true serialization or process isolation

**Out-of-Process Tests**: Spawn separate Python processes
- True test of determinism across process boundaries
- Validates serialization/deserialization fidelity
- Tests actual concurrency (not just async simulation)
- **Critical for**: Reproducibility guarantees, cross-machine validation

**Both test types required** to prove blockchain guarantees

### 5.2 Test Modules and Objectives

**test_deterministic_training.py**:

**In-Process Tests**:
- Test: Create two trainers in same process with identical configs, train for N steps, compare checkpoints
- Verification: Checkpoint bytes are identical (SHA256 hash matches)
- Limitation: Both trainers share same Python interpreter state

**Out-of-Process Tests**:
- Test: Spawn two separate Python processes, each loads identical config, trains for N steps
- Each process serializes checkpoint to separate temp file
- Parent process compares checkpoint files byte-by-byte
- Verification: Files are bit-identical (proves serialization determinism)
- **Critical**: This tests that different memory spaces produce identical results

**Cross-Environment Tests** (if GPU available):
- Test: Train on CPU in one process, train on GPU in another process (both serialize to CPU)
- Verification: CPU and GPU produce identical checkpoints (proves device-independence)
- Validates CPU-forced serialization eliminates GPU kernel non-determinism

**test_sequential_commits.py**:

**In-Process Tests**:
- Test: Create two trainers, Trainer A commits v1 from v0, Trainer B attempts commit from v0 (stale parent)
- Verification: Trainer B raises NotFastForwardError with correct parent/head info
- Fast: Runs in milliseconds using asyncio

**Out-of-Process Tests**:
- Test: Spawn Trainer A process, commit v1, wait for completion
- Spawn Trainer B process, attempt commit from v0 (stale parent)
- Verification: Trainer B exits with NotFastForwardError, no artifacts in S3
- **Critical**: Tests that S3 state correctly prevents stale commits across processes

**test_concurrent_commits.py**:

**In-Process Tests**:
- Test: Launch 10 async trainers using `asyncio.gather()`, all commit from same parent
- Verification: Exactly one succeeds, nine raise ConflictError
- Verification: Losing trainers' artifacts rolled back (scan S3 for orphaned versions)
- Fast: Tests optimistic concurrency logic

**Out-of-Process Tests** (critical for true concurrency):
- Test: Spawn 10 separate Python processes simultaneously using `multiprocessing.Pool`
- Each process loads v10, trains, attempts commit
- Use barrier synchronization to ensure all processes attempt commit at same instant
- Parent process collects exit codes and checks S3 state
- Verification: Exactly one process succeeds (exit code 0), nine fail (exit code 1)
- Verification: Only one v11 exists in S3, nine orphaned artifacts cleaned up
- Verification: Winning process has TensorBoard logs, losers have none
- **Critical**: Tests true race condition with OS-level process scheduling

**Stress Test**:
- Test: 100 concurrent processes attempting commit (tests scalability)
- Verification: Exactly one winner, 99 losers
- Verification: No S3 corruption, chain integrity maintained
- Measure: Conflict resolution latency, retry success rate

**test_chain_reproducibility.py**:

**In-Process Tests**:
- Test: Build chain v0→v15 in single process, load v10, retrain 5 steps, compare hashes
- Verification: Content hashes match v11-v15 exactly
- Tests: Reproducibility within same process

**Out-of-Process Tests** (critical):
- Test: Build chain v0→v15 in Process A, save to S3
- Process A exits (memory cleared)
- Spawn Process B, load v10 from S3, train 5 steps
- Compare Process B's checkpoints to original v11-v15 in S3
- Verification: Byte-identical checkpoints (proves cross-process reproducibility)
- **Critical**: Tests that serialization/deserialization preserves all state

**Cross-Restart Tests**:
- Test: Train v0→v5 in Process A, commit to S3, kill process
- Spawn Process B, load v5, train to v10, commit, kill process
- Spawn Process C, load v10, train to v15, commit
- Verification: Chain v0→v15 is valid, all parent hashes link correctly
- **Critical**: Tests reproducibility across multiple process restarts (simulates production)

**RNG State Preservation Test**:
- Test: Train to v5, snapshot RNG state (Sobol skip offsets, NumPy/PyTorch RNG)
- Continue training to v6 in same process, record outputs
- Restart process, load v5, continue to v6, record outputs
- Verification: Outputs from step 5→6 are bit-identical across restart
- **Critical**: Proves RNG state fully captured in checkpoint

**test_chain_integrity.py** (Tamper Detection):

**Checkpoint Corruption Tests**:
1. **Modified Checkpoint Data**:
   - Test: Build chain v0→v10, modify checkpoint.pb bytes at v5
   - Run `verify_chain()`, expect ChainCorruptionError at v5
   - Verification: Content hash mismatch detected

2. **Metadata Tampering**:
   - Test: Modify content_hash in metadata.json (change one byte)
   - Run `verify_chain()`, expect ChecksumError
   - Verification: Hash verification catches metadata corruption

3. **Broken Parent Link**:
   - Test: Modify parent_hash in v7 to point to v5 (skip v6)
   - Run `verify_chain()`, expect ChainCorruptionError
   - Verification: Merkle chain property violated, detected immediately

4. **Content Hash Collision Attempt**:
   - Test: Create two different checkpoints, attempt to give them same content_hash
   - Verification: SHA256 collision impossible, integrity maintained

**Counter Manipulation Tests**:
1. **Skipped Counter**:
   - Test: Create chain v0→v5, v7 (skip v6), update chain.json to point to v7
   - Run `verify_chain()`, expect ChainCorruptionError
   - Verification: Counter must increment by exactly 1

2. **Duplicate Counter**:
   - Test: Create two v5 checkpoints with different content, attempt to add both
   - Verification: Only one v5 can exist (enforced by atomic commit)

**Global Step Tampering**:
- Test: Modify global_step in v8 to be less than v7's global_step
- Run `verify_chain()`, expect ChainCorruptionError
- Verification: Global step must increase monotonically

**Partial File Corruption**:
- Test: Delete content_hash.txt from v4 directory
- Run `verify_version_completeness()`, expect failure
- Verification: Incomplete versions detected

**Chain Head Manipulation**:
- Test: Modify chain.json to point to non-existent version v99
- Attempt to load latest, expect VersionNotFoundError
- Verification: Invalid chain head rejected

**test_version_pinning.py**:

**In-Process Tests**:
- Test: Create inference client pinned to v20, commit v21, verify client uses v20
- Verification: Client.current_version.counter == 20 after new version published

**Out-of-Process Tests**:
- Test: Spawn inference client process pinned to v20
- Spawn trainer process that commits v21
- Poll client process state (via shared file or IPC)
- Verification: Client remains on v20 despite new version in S3
- Test manual upgrade: Send signal to client, verify upgrade to v21

**Tracking Mode Tests**:
- Test: Spawn inference client in tracking mode (version=None)
- Commit new versions v21, v22, v23 in separate trainer process
- Verification: Client automatically upgrades to v21, then v22, then v23
- Measure: Polling interval, upgrade latency

**test_auto_semantic_versioning.py**:
- Test: Train with same architecture 5 times, verify patch version bumps (1.0.0 → 1.0.5)
- Test: Change architecture (add layer), verify major version bump (1.0.5 → 2.0.0)
- Test: Change parameter count, verify minor version bump (2.0.0 → 2.1.0)
- Verification: Semantic version follows architectural changes deterministically

**test_snapshot_completeness.py**:

**In-Process Snapshot/Restore**:
- Test: Train for 10 steps, snapshot, restore in same process, train 5 more
- Verification: Global step counter at 15, RNG produces same values, optimizer state preserved

**Out-of-Process Snapshot/Restore** (critical):
- Test: Process A trains for 10 steps, saves checkpoint to S3, exits
- Process B loads checkpoint, verifies global_step==10
- Process B trains 5 more steps
- Verification: Process B reaches global_step==15 with correct RNG/optimizer state
- **Critical**: Tests that all mutable state survives serialization

**test_failure_scenarios.py**:

**Network Failure Simulation**:
- Test: Mock S3 client to fail during checkpoint upload (after 50% uploaded)
- Verification: Commit fails, partial artifacts cleaned up (rollback)
- Test: Network failure during ETag fetch
- Verification: Commit fails gracefully, no orphaned artifacts

**S3 Unavailability**:
- Test: Stop MinIO container during commit
- Verification: Commit fails with connection error, retries with backoff
- Test: MinIO comes back online, retry succeeds
- Verification: Chain remains consistent despite transient failures

**Process Crash Simulation**:
- Test: Kill trainer process (SIGKILL) during checkpoint upload
- Verification: Orphaned artifacts left in S3 (incomplete rollback)
- Run garbage collector after grace period
- Verification: Orphaned artifacts cleaned up, chain integrity maintained

**Corrupted S3 Data**:
- Test: Manually corrupt checkpoint.pb bytes in S3
- Attempt to load version
- Verification: Checksum verification catches corruption, raises ChecksumError

**test_continuous_learning.py** (End-to-End):

**Single Trainer, Single Client**:
- Test: Spawn trainer process in loop (commits v1, v2, v3, ..., v10)
- Spawn inference client in tracking mode
- Client polls and upgrades as new versions appear
- Verification: Client tracks latest version, predictions use most recent model

**Multiple Trainers, Single Client**:
- Test: Spawn 3 trainer processes, all training concurrently from v0
- Only one commits v1 (others retry), winner commits v2, etc.
- Client tracks latest version throughout
- Verification: Chain remains linear despite concurrent trainers, client never sees inconsistent state

**Complete Production Simulation** (critical):
- Test: Spawn continuous trainer loop (commits new version every 10 seconds)
- Spawn 5 inference clients (3 tracking, 2 pinned)
- Run for 5 minutes
- Simulate failures: kill trainer process midway (orphaned artifacts), kill client process (graceful restart)
- Verification: Chain grows continuously, tracking clients upgrade, pinned clients stable, no corruption

**test_tensorboard_integration.py**:

**Successful Commit TensorBoard Upload**:
- Test: Train with local TensorBoard writer, commit succeeds, upload events to S3
- Verification: Events exist at `s3://bucket/tensorboard/v{counter}/`
- Launch TensorBoard viewer, verify metrics visible in UI

**Failed Commit No Upload**:
- Test: Train with TensorBoard writer, commit fails (ConflictError)
- Verification: No events in S3 at `s3://bucket/tensorboard/v{counter}/`
- Verification: Local temp directory cleaned up

**Out-of-Process TensorBoard Viewing**:
- Test: Train and commit v5, v6, v7 in separate processes
- Spawn TensorBoard viewer process for versions [5, 6, 7]
- Verification: TensorBoard UI shows all three runs, metrics correct
- Test comparison: Verify TensorBoard can compare loss curves across versions

**Orphaned TensorBoard Cleanup**:
- Test: Simulate trainer crash after commit but before TensorBoard upload
- Manually upload orphaned TensorBoard events (not linked in ModelVersion)
- Run garbage collector
- Verification: Orphaned TensorBoard logs cleaned up

**test_garbage_collection.py**:

**Orphaned Detection**:
- Test: Manually create orphaned version directory (not in chain)
- Run `find_orphaned_versions()`
- Verification: Orphaned version detected with correct metadata

**Grace Period Enforcement**:
- Test: Create orphaned artifacts with various timestamps (1 hour old, 25 hours old)
- Run GC with 24-hour grace period
- Verification: Only artifacts >24 hours deleted, recent artifacts preserved

**Dry Run**:
- Test: Create orphaned artifacts, run GC with dry_run=True
- Verification: Report shows what would be deleted, no actual deletions
- Run GC with dry_run=False
- Verification: Artifacts deleted as reported

**Concurrent Cleanup Safety**:
- Test: Spawn two GC processes simultaneously
- Verification: No errors, artifacts cleaned up exactly once (idempotent)

**Process Crash Simulation**:
- Test: Start trainer upload, kill with SIGKILL mid-upload
- Wait 25 hours, run GC
- Verification: Partial upload artifacts cleaned up, chain intact

### 5.3 End-to-End Integration Tests

**test_e2e_production_workflow.py**:

**Complete Lifecycle Test**:
1. **Genesis**: Create initial model, commit v0
2. **Training Loop**: Train for 100 steps, commit v1, repeat 10 times (v0→v10)
3. **Concurrent Training**: Spawn 5 trainers from v10, only one commits v11
4. **Inference Deployment**: Deploy 10 inference clients (5 tracking v11, 5 pinned to v10)
5. **Continued Training**: Continue training to v20
6. **Client Upgrade**: Tracking clients upgrade automatically, pinned clients stable
7. **Verification**: Run chain verification on entire chain
8. **TensorBoard**: View metrics for all versions, compare v10 vs v20 performance
9. **Cleanup**: Run GC, verify no orphaned artifacts

**Cross-Process Determinism E2E**:
1. Train v0→v5 in Process A
2. Process A exits, memory cleared
3. Train v5→v10 in Process B
4. Process B exits
5. Train v10→v15 in Process C
6. Verification: Chain v0→v15 is valid, content hashes correct
7. **Critical Test**: Restart from v0 in Process D, train to v15
8. Verification: Process D produces bit-identical chain (proves perfect reproducibility)

**Multi-Day Continuous Learning Simulation**:
- Test: Simulate 7 days of training (accelerated)
- Day 1-3: Single trainer commits every hour (72 versions)
- Day 4: Add second trainer (concurrent commits, conflicts)
- Day 5: Kill both trainers, restart from latest (tests recovery)
- Day 6: Add third trainer (more conflicts)
- Day 7: Run GC to clean up failed commits from days 4-6
- Verification: Chain grows to ~150 versions, no corruption, all deterministic

**Disaster Recovery Test**:
- Test: Build chain v0→v50
- Simulate S3 failure: Stop MinIO, attempt commit
- Verification: Commit fails, trainer caches checkpoint locally
- Restore MinIO
- Verification: Trainer retries, commit succeeds, chain at v51
- **Critical**: No data loss despite storage outage

### 5.4 Test Coverage Targets

**Storage Module**: 100% line coverage
**Serialization Module**: 100% coverage of converters
**Critical Paths**: All error conditions exercised
**In-Process vs Out-of-Process**: All critical tests have both variants

**Coverage Requirements by Test Type**:
- **In-Process**: Fast unit tests, 100% code coverage
- **Out-of-Process**: Critical integration tests, proves real-world guarantees
- **E2E Tests**: Production workflow simulation, validates entire system

### Phase 5 Summary

**Test Files Created**: 14 modules
- Core tests: deterministic_training, sequential_commits, concurrent_commits, chain_reproducibility
- Integrity tests: chain_integrity (comprehensive tamper detection)
- Feature tests: version_pinning, auto_semantic_versioning, snapshot_completeness
- Robustness tests: failure_scenarios, continuous_learning
- Integration tests: tensorboard_integration, garbage_collection
- **New**: test_e2e_production_workflow (comprehensive end-to-end)

**Total Tests**: 200+ test cases
- **In-Process**: ~120 tests (fast, unit-level)
- **Out-of-Process**: ~60 tests (integration, determinism proofs)
- **E2E**: ~20 tests (production workflows)

**Estimated Time**: 4-5 days (increased due to comprehensive out-of-process and E2E coverage)

**Test Execution Time**:
- In-process tests: ~5 minutes
- Out-of-process tests: ~20 minutes
- E2E tests: ~10 minutes
- Total: ~35 minutes (acceptable for comprehensive test suite)

**Verification Commands**:
- `poetry run pytest tests/test_storage/ -v`
- `poetry run pytest tests/test_storage/ --cov=spectralmc.storage --cov-report=html`
- `poetry run pytest tests/test_storage/ --cov=spectralmc.serialization --cov-report=term-missing`

---

## Phase 6: Documentation and Integration

### Objective

Document the system, provide usage examples, and integrate with existing infrastructure.

### 6.1 Update CLAUDE.md

**New Section**: "Model Versioning (Blockchain)"

**Content to Add**:
- Overview of blockchain guarantees
- Training workflow example (commit new versions with TensorBoard metrics)
- Inference workflow examples (pinned vs tracking modes)
- Chain verification commands
- TensorBoard integration (viewing metrics for any version)
- Best practices for semantic versioning

### 6.2 Usage Examples

**Create**: `examples/blockchain_training.py`
- Complete async training loop
- Demonstrates commit with retry
- Shows error handling for conflicts
- Includes TensorBoard metrics logging

**Create**: `examples/inference_client.py`
- Both pinned and tracking modes
- Demonstrates hot-swapping
- Shows graceful shutdown

**Create**: `examples/verify_chain.py`
- Load store
- Run chain verification
- Display version history

**Create**: `examples/tensorboard_viewer.py`
- Launch TensorBoard for specific versions
- Compare metrics across multiple versions
- View full training history

### 6.3 Architecture Documentation

**Create**: `docs/blockchain_architecture.md`
- Diagram of version chain structure
- Explanation of optimistic concurrency control
- MinIO configuration guide
- IAM policy examples for S3 permissions

**Create**: `docs/deployment.md`
- Docker Compose file for MinIO
- Environment variables configuration
- Bucket creation commands
- TensorBoard S3 configuration (credentials, endpoint)
- Monitoring and observability

### 6.4 Integration Tasks

**CI/CD Pipeline**:
- Add Protobuf compilation step to build process
- Add storage tests to test matrix (requires MinIO container)
- Add coverage reporting for new modules

**Type Checking**:
- Verify mypy passes with generated Protobuf stubs
- Add storage and serialization modules to mypy config

**Logging**:
- Add structured logging to storage operations
- Log commit attempts, conflicts, retries
- Add metrics for version count, chain length

**Migration**:
- Write utility to migrate existing checkpoints to blockchain format
- Document breaking changes
- Provide rollback strategy

### 6.5 README Updates

**Add Sections**:
- Blockchain model versioning overview
- MinIO setup instructions
- Quick start with storage layer
- Link to detailed architecture docs

### Phase 6 Summary

**Documentation Files**: 4 markdown files + README updates
**Example Scripts**: 4 complete examples (including TensorBoard viewer)
**Integration Changes**: CI/CD, logging, migration tools, TensorBoard setup
**Estimated Time**: 1 day

---

## Implementation Timeline (Bottom-Up with Incremental Testing)

| Phase | Sub-Phase | Description | Time | Test After | Status |
|-------|-----------|-------------|------|------------|--------|
| **Phase 1** | **1A** | Core Type System (Device, DType) | 2-3 hrs | ✅ 20/20 tests | ✅ DONE |
| | **1B** | Simple Configs (Sobol, Buffer) | 2-3 hrs | ✅ 18/18 tests | ✅ DONE |
| | **1C** | Simulation Params (CUDA blocks) | 2-3 hrs | 🧪 Test | 📋 TODO |
| | **1D** | Model Config (Width spec) | 2-3 hrs | 🧪 Test | 📋 TODO |
| | **1E** | Training Config | 2-3 hrs | 🧪 Test | 📋 TODO |
| | **1F** | Full Integration | 1 hr | 🧪 Full suite | 📋 TODO |
| | | **Phase 1 Total** | **1-2 days** | | |
| **Phase 2** | | Protobuf Schemas | 1 day | 🧪 Import tests | Phase 1 complete |
| **Phase 3** | | Conversion Layer | 2-3 days | 🧪 Round-trip tests | Phase 2 complete |
| **Phase 4** | | Blockchain Store + TB + GC | 4-5 days | 🧪 Integration tests | Phase 3 complete |
| **Phase 5** | | Comprehensive Test Suite | 4-5 days | 🧪 200+ tests | Phase 4 complete |
| **Phase 6** | | Documentation | 1-2 days | 🧪 Example runs | Phase 5 complete |
| | | **Total** | **13-18 days** | | |

**Key Changes from Original Plan**:
- ✅ **Incremental testing**: Test after each sub-phase instead of end of phase
- ✅ **Bottom-up dependencies**: Each sub-phase builds on tested foundation
- ✅ **Fast feedback**: Catch issues within 2-3 hours instead of 1-2 days
- ✅ **Docker-based**: All commands run inside container with output redirection
- ✅ **Cumulative validation**: Each test run includes previous sub-phases

**Testing Philosophy**:
- 🧪 Test frequently (after each sub-phase)
- 📊 Redirect to files for complete output analysis
- ✅ All tests must pass before proceeding
- 🔍 Type check (mypy --strict) before running tests

---

## Success Criteria

### Functional Requirements

- ✅ **Determinism (In-Process)**: Identical starting conditions produce identical results (bit-level)
- ✅ **Determinism (Out-of-Process)**: Cross-process reproducibility verified (proves serialization fidelity)
- ✅ **Determinism (Cross-Environment)**: CPU and GPU produce identical checkpoints (device-independent)
- ✅ **Atomicity**: Concurrent commits resolve to single linear chain (no divergence)
- ✅ **Reproducibility**: Retraining from any checkpoint reproduces exact remainder
- ✅ **Reproducibility (Cross-Restart)**: Multi-process restarts reproduce identical chains
- ✅ **Tamper Detection**: Chain verification catches all corruption scenarios (15+ attack vectors tested)
- ✅ **Version Pinning**: Inference clients can pin to specific versions
- ✅ **TensorBoard Integration**: Training metrics preserved for all committed versions
- ✅ **No Orphaned Artifacts**: Failed commits leave no artifacts in S3 (rollback + GC)
- ✅ **Concurrent Training**: Multiple trainers work simultaneously without corruption
- ✅ **RNG State Preservation**: Random number generators fully captured and restored across processes

### Non-Functional Requirements

- ✅ **Type Safety**: `mypy --strict` passes with zero errors
- ✅ **Test Coverage**: 100% coverage for storage module, 95%+ overall
- ✅ **In-Process Test Coverage**: 120+ tests, all code paths exercised
- ✅ **Out-of-Process Test Coverage**: 60+ tests, proves real-world determinism
- ✅ **E2E Test Coverage**: 20+ tests, validates production workflows
- ✅ **Performance**: Commit latency < 1 second for typical model sizes
- ✅ **Reliability**: Automatic retry with exponential backoff handles transient failures
- ✅ **Observability**: Structured logging for all storage operations
- ✅ **Test Execution Time**: Full test suite completes in ~35 minutes

---

## Rollout Strategy

### Development Phase
- Phases 1-3 deployed to development environment
- Unit tests run on every commit
- Type checking enforced in pre-commit hooks

### Staging Phase
- Phase 4 deployed to staging with single trainer
- Integration tests with real MinIO instance
- Performance benchmarking

### Validation Phase
- Phase 5: Full test suite execution
- Stress testing with concurrent trainers
- Chain verification on large version histories

### Production Phase
- Phase 6: Documentation complete
- Migration of existing checkpoints
- Gradual rollout with feature flags
- Monitoring and alerting configured

---

## Risk Mitigation

### Data Corruption
- **Risk**: Bit flips in storage or network
- **Mitigation**: SHA256 checksums on all artifacts, verified at read time

### Concurrent Write Conflicts
- **Risk**: High contention when many trainers commit simultaneously
- **Mitigation**: Exponential backoff with jitter, automatic retry up to N attempts

### Version Sprawl
- **Risk**: Unbounded version history growth
- **Mitigation**: Retention policies (future work), archive old versions to Glacier

### Schema Evolution
- **Risk**: Protobuf schema changes break old checkpoints
- **Mitigation**: Use Protobuf `optional` fields, never remove fields, maintain backward compatibility

### MinIO Availability
- **Risk**: Storage backend unavailable
- **Mitigation**: Trainer can cache checkpoints locally, retry commits when storage recovers

---

## Future Enhancements

### Compression
- Apply zstd compression to checkpoint.pb files
- Reduces storage costs by ~60% for typical models

### Lazy Loading
- Use SafeTensors memory-mapped loading
- Load only required tensors for inference

### Version Retention Policies
- Automatic archival to S3 Glacier after N days
- Keep latest M versions, all major versions
- Configurable retention rules

### Distributed Training
- Multiple trainers collaborate on single model
- Gradient aggregation before commit
- Requires consensus protocol (future research)

### Observability Dashboard
- Web UI showing version chain visualization
- Commit rate graphs and chain health metrics
- TensorBoard already provides training metrics visualization (no custom dashboard needed)
- Future: Integrate TensorBoard metrics into unified dashboard

---

## Appendix: Key Design Decisions

### Why Protobuf Over JSON?
- 5-10x smaller serialized size
- Faster serialization/deserialization
- Schema evolution with backward compatibility
- Type safety via code generation

### Why aioboto3 Over boto3?
- Non-blocking I/O for concurrent operations
- Better latency for multi-trainer scenarios
- Natural fit with async inference clients

### Why Optimistic Concurrency Over Locking?
- No distributed lock coordination needed
- Better scalability (no lock contention)
- Natural conflict resolution via retry

### Why Blockchain Structure?
- Tamper detection via Merkle chain
- Reproducibility guarantees
- Audit trail for compliance
- Natural versioning semantics

### Why Separate chain.json From Versions?
- Single atomic update point (chain head)
- Versions are immutable (never modified)
- Enables content-addressable storage
- Simplifies garbage collection

---

## Appendix B: Concurrent Training Scenario (Detailed Walkthrough)

### Scenario: Two Trainers Training Simultaneously from v10

**Setup**:
- Trainer A and Trainer B both load checkpoint v10
- Both train for 100 steps with identical or different configs
- Both attempt to commit new version simultaneously

### Execution Timeline

**T=0**: Both trainers load v10
```
Trainer A: Load v10 ✓
Trainer B: Load v10 ✓
Chain: v0 → v1 → ... → v10 (head)
```

**T=1-60**: Both trainers execute training
```
Trainer A: Training... (100 steps)
Trainer B: Training... (100 steps)
Both: Writing metrics to local TensorBoard dirs
```

**T=61**: Both trainers build checkpoints
```
Trainer A: checkpoint_a = build_checkpoint()
Trainer B: checkpoint_b = build_checkpoint()
```

**T=62**: Both trainers attempt commit (race condition)
```
Trainer A: await commit_version(checkpoint_a, parent=v10, "trainer-a")
Trainer B: await commit_version(checkpoint_b, parent=v10, "trainer-b")
```

### The Race: Step-by-Step

**Step 1**: Upload artifacts (parallel, no conflict)
```
Trainer A: Upload to s3://bucket/versions/v0000000011_hash_a/
Trainer B: Upload to s3://bucket/versions/v0000000011_hash_b/
Both: Success (different hash suffixes, no collision)
```

**Step 2**: Fetch chain head (parallel reads, same result)
```
Trainer A: GET chain.json → ETag="xyz123", head=v10
Trainer B: GET chain.json → ETag="xyz123", head=v10
```

**Step 3**: Verify fast-forward (both pass initially)
```
Trainer A: parent v10 == head v10 ✓
Trainer B: parent v10 == head v10 ✓
```

**Step 4**: Atomic CAS write (RACE CONDITION)
```
Trainer A: PUT chain.json If-Match="xyz123" → SUCCESS (writes first)
Trainer B: PUT chain.json If-Match="xyz123" → CONFLICT (ETag changed!)

Result:
- Trainer A wins (chain.json updated to point to v11 with ETag="abc789")
- Trainer B loses (If-Match fails because ETag is now "abc789")
```

**Step 5**: Rollback and recovery
```
Trainer A:
  ├─ Commit succeeded ✓
  ├─ Upload TensorBoard to s3://bucket/tensorboard/v0000000011/
  └─ Return ModelVersion(counter=11, ...)

Trainer B:
  ├─ Commit failed (ConflictError raised)
  ├─ Rollback: DELETE s3://bucket/versions/v0000000011_hash_b/
  ├─ Cleanup local TensorBoard dir
  └─ Raise ConflictError("Parent v10 != head v11")
```

**Final State**:
```
Chain: v0 → v1 → ... → v10 → v11 (Trainer A's checkpoint)
S3 Artifacts:
  ✓ s3://bucket/versions/v0000000011_hash_a/ (exists, referenced in chain)
  ✓ s3://bucket/tensorboard/v0000000011/ (Trainer A's metrics)
  ✗ No orphaned artifacts from Trainer B (rolled back)
```

### Recovery Strategy for Trainer B

**Option 1: Automatic Retry with Exponential Backoff**
```python
async with atomic_commit_with_retry(store, max_retries=5) as commit:
    # Retry loop:
    for attempt in range(5):
        try:
            checkpoint = trainer.build_checkpoint()
            parent = await store.fetch_latest_version()  # Now v11
            return await commit(checkpoint, parent, "trainer-b")
        except ConflictError:
            # Fetch latest, retry from v11
            await asyncio.sleep(backoff_delay)
```

**Retry behavior**:
1. Fetch latest (v11 from Trainer A)
2. **Important**: Trainer B must decide:
   - **Discard own work**: Load v11, train from there
   - **Merge strategies**: Not supported (no consensus protocol)

**Option 2: Accept Defeat and Catch Up**
```python
try:
    await store.commit_version(checkpoint, parent=v10, "trainer-b")
except NotFastForwardError:
    # Another trainer committed first
    latest = await store.fetch_latest_version()  # v11
    trainer.load_checkpoint(latest)
    # Continue training from winner's checkpoint
```

### Key Insights

1. **Only One Winner**: Blockchain linearization guarantee ensures exactly one v11 exists
2. **No Data Loss for Winner**: Trainer A's work is preserved in chain
3. **Work Lost for Loser**: Trainer B's 100 training steps are discarded (unless retrying from v11)
4. **No S3 Pollution**: Rollback ensures no orphaned artifacts
5. **Determinism Preserved**: If both had identical configs/steps, both produced identical checkpoints (but only one committed)

### Garbage Collection Safety Net

**Edge Case**: Trainer B crashes before rollback completes
```
Orphaned artifact: s3://bucket/versions/v0000000011_hash_b/
Created: 2025-11-06 10:00:00
Chain: Does not reference v0000000011_hash_b
```

**Garbage Collector** (runs daily):
```
1. Scan versions/ directory → find v0000000011_hash_b
2. Check chain.json → v11 points to hash_a (not hash_b)
3. Check timestamp → 25 hours old (past grace period)
4. Mark as orphaned
5. DELETE s3://bucket/versions/v0000000011_hash_b/ ✓
```

**Result**: Even if rollback fails, orphaned artifacts cleaned within 24 hours.

---

This plan provides complete specifications for implementing the blockchain model store while remaining high-level enough for Claude Code CLI to generate specific implementations during execution.
