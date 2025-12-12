# File: documents/engineering/blockchain_storage.md
# Blockchain Model Versioning

**Status**: Authoritative source  
**Supersedes**: Prior blockchain storage notes  
**Referenced by**: documents/product/deployment.md

> **Purpose**: Define blockchain-based model versioning architecture using S3 storage, atomic CAS commits, and semantic versioning for production ML models.

## Cross-References
- [effect_interpreter.md](effect_interpreter.md)
- [../product/deployment.md](../product/deployment.md)
- [../product/training_integration.md](../product/training_integration.md)

## Overview

SpectralMC uses a blockchain-based approach for model versioning with S3 storage, providing production-ready version control for ML models.

**Key Features**:
- **Immutable version history** with SHA256 content addressing
- **Semantic versioning** (MAJOR.MINOR.PATCH)
- **Merkle chain linking** for tamper detection
- **Atomic commits** with ETag-based CAS (Compare-And-Swap)
- **S3 storage** with aioboto3 async operations
- **Protocol Buffer serialization** for cross-platform compatibility
- **InferenceClient** with pinned/tracking modes
- **Chain verification** to detect corruption
- **Garbage collection** for old versions
- **TensorBoard integration** for metrics logging

---

## Storage Architecture

### AsyncBlockchainModelStore

**Location**: `src/spectralmc/storage/store.py`

Production S3-based storage with atomic CAS commits:
- 10-step atomic commit protocol using ETag/If-Match
- Conflict detection and fast-forward enforcement
- Automatic retry with exponential backoff

### S3 Storage Structure

```text
# File: documents/engineering/blockchain_storage.md
S3 Bucket Structure:
my-model-bucket/
├── chain.json                    # HEAD pointer with ETag
└── versions/
    ├── v0000000000_1.0.0_abcd1234/
    │   ├── checkpoint.pb         # Protocol Buffer serialized model
    │   ├── metadata.json         # Version info
    │   └── content_hash.txt      # SHA256 checksum
    └── v0000000001_1.0.1_ef567890/
        ├── checkpoint.pb
        ├── metadata.json
        └── content_hash.txt
```

**Key Features**:
- `chain.json`: Atomic HEAD pointer updated via CAS (ETag-based)
- `versions/`: Immutable version directories (never modified after creation)
- `checkpoint.pb`: Protocol Buffer serialized model state
- `metadata.json`: Version metadata (counter, semver, parent_hash, timestamp)
- `content_hash.txt`: SHA256 checksum for integrity verification

---

## Atomic Commit Protocol (10-step CAS)

```mermaid
flowchart TB
  Start[Read chain head ETag]
  BuildMeta[Build version metadata + content hash]
  UploadCheckpoint[Upload checkpoint file]
  UploadMeta[Upload metadata file]
  UploadHash[Upload content hash file]
  UpdateHead[CAS update chain json If Match ETag]
  CommitSuccess[Commit succeeds]
  Conflict[412 Precondition Failed]
  Rollback[Delete uploaded files]
  Retry[Retry with backoff]

  Start -->|read head| BuildMeta
  BuildMeta -->|prepare metadata| UploadCheckpoint
  UploadCheckpoint -->|upload checkpoint| UploadMeta
  UploadMeta -->|upload metadata| UploadHash
  UploadHash -->|upload hash| UpdateHead
  UpdateHead -->|success| CommitSuccess
  UpdateHead -->|precondition failed| Conflict
  Conflict -->|cleanup| Rollback
  Rollback -->|retry strategy| Retry
  Retry -->|restart sequence| Start
```

### CAS Guarantees

- **Atomicity**: Either all files committed or none (rollback on conflict)
- **Consistency**: chain.json always points to valid version
- **Isolation**: Concurrent commits detected via ETag mismatch
- **Durability**: S3 provides 99.999999999% durability

---

## Core Usage

### Committing Models

```python
# File: documents/engineering/blockchain_storage.md
from spectralmc.storage import AsyncBlockchainModelStore, commit_snapshot

# Initialize async store (S3)
async with AsyncBlockchainModelStore("my-model-bucket") as store:
    # Commit a trained model snapshot
    config = GbmCVNNPricerConfig(...)  # Your trained config
    version = await commit_snapshot(
        store,
        config,
        message="Trained for 1000 epochs"
    )

    print(f"Committed version {version.counter}: {version.content_hash[:8]}")
```

### Loading Models

```python
# File: documents/engineering/blockchain_storage.md
from spectralmc.storage import load_snapshot_from_checkpoint
from spectralmc.result import Success, Failure

async with AsyncBlockchainModelStore("my-model-bucket") as store:
    # Get HEAD version
    head_result = await store.get_head()

    match head_result:
        case Success(head):
            # Load checkpoint
            model_template = torch.nn.Linear(5, 5)
            config_template = make_config(model_template)

            snapshot = await load_snapshot_from_checkpoint(
                store,
                head,
                model_template,
                config_template
            )

            # Use snapshot.cvnn for inference
            model = snapshot.cvnn
        case Failure(error):
            print(f"Error getting HEAD: {error}")
```

---

## InferenceClient

Production model serving with version control integration.

### Mode Selection Decision Tree

```mermaid
flowchart TB
  Start{Deployment Environment}
  Production[Production Environment]
  Development[Development Environment]

  StabilityNeeded{Stability Critical}
  ABTest{AB Testing}

  Pinned[Pinned Mode]
  MultiPinned[Multiple Pinned Clients]
  Tracking[Tracking Mode]

  Start -->|production| Production
  Start -->|development or staging| Development

  Production -->|assess stability| StabilityNeeded
  StabilityNeeded -->|stability critical| ABTest
  StabilityNeeded -->|can update| Tracking

  ABTest -->|single version| Pinned
  ABTest -->|multiple versions| MultiPinned

  Development -->|auto update| Tracking

  Pinned -->|configure client| PinnedConfig[Version 42 Static]
  MultiPinned -->|configure clients| MultiConfig[ClientA v42 ClientB v43]
  Tracking -->|configure tracking| TrackingConfig[Track Head Poll30s]
```

### Pinned Mode (Production)

```python
# File: documents/engineering/blockchain_storage.md
from spectralmc.storage import InferenceClient

# Pin to specific version for production stability
async with InferenceClient(
    version_counter=42,  # Pin to v42
    poll_interval=60.0,
    store=store,
    model_template=model,
    config_template=config
) as client:
    # Always serves v42, never updates
    snapshot = client.get_model()
    predictions = run_inference(snapshot.cvnn, inputs)
```

### Tracking Mode (Development)

```python
# File: documents/engineering/blockchain_storage.md
# Auto-track latest version with hot-swapping
async with InferenceClient(
    version_counter=None,  # Track HEAD
    poll_interval=30.0,
    store=store,
    model_template=model,
    config_template=config
) as client:
    # Model auto-updates every 30 seconds
    snapshot = client.get_model()
    predictions = run_inference(snapshot.cvnn, inputs)
```

---

## Chain Verification

Detect tampering and corruption with automated verification.

### Chain Verification Algorithm

```mermaid
flowchart TB
  Start[Start Verification]
  LoadVersions[Load all versions from S3]
  CheckGenesis{Genesis Valid}
  GenesisOK[Genesis Valid]
  GenesisFail[Genesis Invalid]

  IterateVersions[Iterate through versions]
  CheckParentHash{Parent Hash Matches}
  CheckCounter{Counter Sequence}
  CheckSemver{Semver Sequence}

  ParentHashOK[Merkle chain intact]
  CounterOK[Counter monotonic]
  SemverOK[Semantic version valid]

  AllValid[All checks passed]
  CorruptionFound[CORRUPTION DETECTED]

  Start -->|load versions| LoadVersions
  LoadVersions -->|validate genesis| CheckGenesis
  CheckGenesis -->|valid| GenesisOK
  CheckGenesis -->|invalid| GenesisFail
  GenesisOK -->|iterate| IterateVersions
  IterateVersions -->|check parent| CheckParentHash
  CheckParentHash -->|match| ParentHashOK
  CheckParentHash -->|mismatch| CorruptionFound
  ParentHashOK -->|check counter| CheckCounter
  CheckCounter -->|sequential| CounterOK
  CheckCounter -->|gap or duplicate| CorruptionFound
  CounterOK -->|check semver| CheckSemver
  CheckSemver -->|valid| SemverOK
  CheckSemver -->|invalid| CorruptionFound
  SemverOK -->|report valid| AllValid
```

### Code Usage

```python
# File: documents/engineering/blockchain_storage.md
from spectralmc.storage import verify_chain, verify_chain_detailed
from spectralmc.result import Success, Failure

async with AsyncBlockchainModelStore("my-model-bucket") as store:
    # Functional verification with Result types
    result = await verify_chain(store)

    match result:
        case Success(report) if report.is_valid:
            print(f"Chain valid: {report.details}")
        case Success(report):
            # Corruption detected
            print(f"Corruption: {report.corruption_type}")
            print(f"At version: {report.corrupted_version.counter}")
        case Failure(error):
            print(f"S3 error during verification: {error}")

    # Alternative: Use verify_chain_detailed directly (returns CorruptionReport)
    report = await verify_chain_detailed(store)
    if not report.is_valid:
        print(f"Corruption: {report.corruption_type}")
        print(f"At version: {report.corrupted_version.counter}")
```

### Validation Checks

- Genesis block integrity (counter=0, empty parent_hash, semver="1.0.0")
- Merkle chain property: `parent_hash == previous.content_hash`
- Counter monotonicity (strictly increasing by 1)
- Semantic version progression

---

## Garbage Collection

Automated cleanup of old versions:

```python
# File: documents/engineering/blockchain_storage.md
from spectralmc.storage import run_gc, RetentionPolicy

async with AsyncBlockchainModelStore("my-model-bucket") as store:
    # Preview what would be deleted (dry run)
    report = await run_gc(
        store,
        keep_versions=10,          # Keep last 10 versions
        protect_tags=[5, 12, 20],  # Protect production releases
        dry_run=True
    )

    print(f"Would delete: {report.deleted_versions}")
    print(f"Would free: {report.bytes_freed / (1024**2):.2f} MB")

    # Actually delete
    report = await run_gc(store, keep_versions=10, dry_run=False)
```

### Safety Features

- Genesis (v0) always protected
- Configurable minimum versions (default: 3)
- Protected tags for production releases
- Dry-run preview before deletion

---

## TensorBoard Integration

Log model versions and training metrics:

```python
# File: documents/engineering/blockchain_storage.md
from spectralmc.storage import log_blockchain_to_tensorboard

async with AsyncBlockchainModelStore("my-model-bucket") as store:
    await log_blockchain_to_tensorboard(
        store,
        log_dir="runs/my_experiment",
        model_template=model,
        config_template=config
    )

# View with: tensorboard --logdir=runs/
```

**Logs**:
- Version metadata (counter, semver, content_hash, timestamp)
- Training metrics (global_step, param_count, sobol_skip)
- Summary statistics (total versions, versions per day)

---

## Training Integration

Automatic blockchain commits during training via `GbmCVNNPricer.train()`.

### Auto-commit after training completes

```python
# File: documents/engineering/blockchain_storage.md
from spectralmc.gbm_trainer import GbmCVNNPricer, TrainingConfig
from spectralmc.storage import AsyncBlockchainModelStore

async with AsyncBlockchainModelStore("my-model-bucket") as store:
    pricer = GbmCVNNPricer(config)

    training_config = TrainingConfig(
        num_batches=1000,
        batch_size=32,
        learning_rate=0.001
    )

    # Train with automatic commit when done
    pricer.train(
        training_config,
        blockchain_store=store,
        auto_commit=True,
        commit_message_template="Final checkpoint: step={step}, loss={loss:.4f}"
    )

    # Version automatically committed after training
    head = await store.get_head()
    print(f"Training committed as version {head.counter}")
```

### Periodic commits during training

```python
# File: documents/engineering/blockchain_storage.md
# Commit every 100 batches during training
pricer.train(
    training_config,
    blockchain_store=store,
    auto_commit=True,
    commit_interval=100,  # Commit every 100 batches
    commit_message_template="Checkpoint: step={step}, loss={loss:.4f}"
)
```

### Features

- **Backward compatible**: Training without `blockchain_store` still works
- **Graceful error handling**: Commit failures logged but don't crash training
- **Optimizer state preservation**: Full checkpoint including Adam state
- **Template interpolation**: Variables `{step}`, `{loss}`, `{batch}` in messages
- **Async-to-sync bridge**: Uses `asyncio.run()` to commit within sync training loop

### Validation

- `auto_commit=True` requires `blockchain_store` parameter
- `commit_interval` requires `blockchain_store` parameter

See `examples/training_with_blockchain_storage.py` for complete example.

---

## CLI Tools

### Usage

```bash
# File: documents/engineering/blockchain_storage.md
# Verify chain integrity (Docker-only)
docker compose -f docker/docker-compose.yml exec spectralmc \
  poetry run python -m spectralmc.storage verify my-model-bucket

# List all versions
docker compose -f docker/docker-compose.yml exec spectralmc \
  poetry run python -m spectralmc.storage list-versions my-model-bucket

# Inspect specific version
docker compose -f docker/docker-compose.yml exec spectralmc \
  poetry run python -m spectralmc.storage inspect my-model-bucket v0000000042

# Preview garbage collection
docker compose -f docker/docker-compose.yml exec spectralmc \
  poetry run python -m spectralmc.storage gc-preview my-model-bucket 10

# Run garbage collection (keep last 10, protect v5 and v12)
docker compose -f docker/docker-compose.yml exec spectralmc \
  poetry run python -m spectralmc.storage gc-run my-model-bucket 10 --protect-tags 5,12 --yes

# Log to TensorBoard
docker compose -f docker/docker-compose.yml exec spectralmc \
  poetry run python -m spectralmc.storage tensorboard-log my-model-bucket --log-dir runs/exp1
```

### Complete CLI Commands

- `verify` - Verify blockchain integrity
- `find-corruption` - Find first corrupted version
- `list-versions` - List all versions
- `inspect` - Inspect specific version in detail
- `gc-preview` - Preview garbage collection (dry run)
- `gc-run` - Run garbage collection
- `tensorboard-log` - Log blockchain to TensorBoard

---

## Test Coverage

All storage features have comprehensive test coverage:

- **CLI commands**: 22 tests (83% coverage of `__main__.py`)
  - verify, find-corruption, list-versions, inspect commands
  - gc-preview, gc-run with protected tags
  - tensorboard-log, error handling
- **InferenceClient**: 8 tests (pinned mode, tracking mode, lifecycle)
- **Chain verification**: 15 tests (genesis, merkle chain, corruption detection)
- **Garbage collection**: 15 tests (retention policies, safety checks)
- **TensorBoard**: 12 tests (logging, metadata, error handling)
- **Training integration**: 7 tests (auto_commit, periodic commits, optimizer state preservation)

**Total: 86 storage tests, 73% overall coverage**

### Run storage tests

```bash
# File: documents/engineering/blockchain_storage.md
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_storage/
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all tests/test_integrity/
```

---

## Related Documentation

- [Testing Requirements](testing_requirements.md) - Test output handling and anti-patterns
- [Coding Standards](coding_standards.md) - Type safety, Result types, ADTs, and error handling
- [Main Project Guide](../../CLAUDE.md) - Quick reference
