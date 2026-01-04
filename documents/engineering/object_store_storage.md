# File: documents/engineering/object_store_storage.md
# Object Store Model Versioning

**Status**: Authoritative source  
**Supersedes**: documents/engineering/blockchain_storage.md  
**Referenced by**: documents/documentation_standards.md; documents/engineering/README.md

> **Purpose**: Define SpectralMC's tamper-evident, append-only object store model for
> training artifacts, including explicit S3/MinIO assumptions and proof obligations.

## Cross-References
- [Effect Interpreter Doctrine](effect_interpreter.md)
- [Reproducibility Proofs](reproducibility_proofs.md)
- [TLA+ Reproducibility Proofs](tla.md)
- [Total Pure Modelling](total_pure_modelling.md)
- [Immutability Doctrine](immutability_doctrine.md)

## Executive Summary

SpectralMC uses an append-only, content-addressed storage model on S3-compatible
object stores. The model replaces blockchain framing with explicit storage
assumptions and a tamper-evident manifest chain. It is designed for trusted
networks and workers, with exactly-once semantics for new elements, reproducible
replay via manifest snapshots, and strong auditability without consensus.

## Storage Assumptions (Explicit)

These assumptions are required to prove tamper-evidence and exactly-once semantics
in TLA+. They are treated as axioms for S3 and MinIO backends.

### S3 Assumptions

- **Strong consistency**: GET, PUT, and LIST are strongly consistent. Source:
  https://aws.amazon.com/blogs/aws/amazon-s3-update-strong-read-after-write-consistency/
- **WORM immutability via Object Lock**: Object versions can be prevented from
  overwrite or deletion for a retention period or legal hold. Source:
  https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html
- **Versioning required for Object Lock**: Object Lock applies only to versioned
  buckets. Source: https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html
- **Conditional writes**: If-Match and If-None-Match are supported for CAS and
  exactly-once creation. Source: S3 API semantics and client behavior.

### MinIO Assumptions

- **WORM immutability**: Object Lock enforces write-once-read-many retention on
  versioned objects and blocks deletion of locked versions. Source:
  https://docs.min.io/enterprise/aistor-object-store/administration/object-locking-and-immutability/
- **S3 Object Lock compatibility**: MinIO Object Lock is API compatible with S3
  Object Lock semantics. Source:
  https://docs.min.io/enterprise/aistor-object-store/administration/object-locking-and-immutability/
- **Read/write quorum**: Erasure coding requires read and write quorum for
  operations to succeed; writes fail if quorum is not met. Source:
  https://docs.min.io/enterprise/aistor-object-store/operations/core-concepts/erasure-coding/
- **Conditional writes**: If-Match and If-None-Match are supported via S3-compatible
  APIs.

## Storage Model Overview

The storage model is a content-addressed, append-only log with a single mutable
pointer. It provides tamper-evident history without blockchain consensus.

**Core elements**:

- **Elements**: immutable content-addressed objects keyed by SHA256 of canonical
  serialization.
- **Manifests**: immutable commits that reference element IDs and include the hash
  of the previous manifest.
- **Head pointer**: a small CAS-updated object that points to the latest manifest.

## Object Layout (Conceptual)

```text
# File: documents/engineering/object_store_storage.md
s3://<bucket>/
  elements/
    <sha256>.bin
  manifests/
    <timestamp>-<seq>-<hash>.json
  refs/
    latest
  audit/
    <timestamp>.json
```

Notes:
- `elements/` are immutable and written with `If-None-Match: *` to guarantee
  exactly-once creation.
- `manifests/` are immutable and include `prev_manifest_hash` for tamper evidence.
- `refs/latest` is updated with `If-Match: <etag>` to enforce linear history.
- `audit/` is advisory and never used for correctness.

## Invariants and Proof Obligations

The TLA+ storage spec must prove the following invariants under the assumptions
above:

1. **Immutability**: Once written, an element or manifest cannot change.
2. **Exactly-once element creation**: Duplicate content results in a no-op, not a
   new element ID.
3. **Manifest chain integrity**: Each manifest references the correct previous
   manifest hash.
4. **Head linearity**: `refs/latest` advances only via CAS and always points to a
   valid manifest.
5. **Reproducible replay**: A manifest fully defines the ordered sequence of
   elements and configuration needed for deterministic reconstruction.

## Effect Interpreter Boundary

All storage operations are expressed as storage effects in the effect ADT and
interpreted by the effect interpreter. The TLA+ model includes the interpreter
behind the purity wall to ensure that the storage model aligns with effect
sequencing and deterministic execution.

## Legacy Naming

Some APIs still use legacy "blockchain" names (for example,
`AsyncBlockchainModelStore`). Documentation describes the object store model and
its invariants; API renames are tracked separately.
