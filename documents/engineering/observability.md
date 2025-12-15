# File: documents/engineering/observability.md
# Observability

**Status**: Authoritative source  
**Supersedes**: none  
**Referenced by**: engineering/README.md

> **Purpose**: Define SpectralMC’s observability expectations with logging and S3-native audit
> trails. No centralized monitoring stack is prescribed; S3 manifests are the source of truth
> for training and model state.

## Scope

- Logging modeled as effects; interpreters emit structured logs.
- S3 `head.json` + immutable version directories + audit log are canonical for state.
- No Prometheus/OpenTelemetry/metrics requirements; avoid adding metric dependencies.

## Principles

- **State lives in S3**: `head.json` guarded by ETag/object-lock and version directories keyed
  by counter+hash are definitive.
- **Structured logs only**: emit context-rich JSON logs from interpreters (device, version
  counter, content hash, audit ID) to aid debugging without inventing secondary state.
- **Deterministic breadcrumbs**: log RNG seeds, device placement, and S3 keys used so runs can
  be replayed from S3 alone.
- **Separation of concerns**: pure layers produce log effects; interpreters write logs and
  append audit records.

## Logging Expectations

- Include: version counter/hash, S3 object keys, device IDs, RNG seeds, commit message, and
  audit log record ID for each commit/load.
- Record failures that bypass commit (serialization failure, ETag mismatch, missing bucket)
  and note whether `head.json` changed.
- Avoid: ad-hoc metrics, Prometheus/OpenTelemetry exporters, or runtime metric definition.

## Logging

- Use `LogMessage` effects with level/message/context; avoid direct logger calls in pure code.
- Emit logs from interpreters; pure layers remain deterministic and free of I/O.
- Include device/model identifiers in log context to spot CPU fallbacks or mismatched models.
- Prefer structured JSON payloads to allow easy filtering without a metrics backend.

## Implementation Notes

- There is no prescribed metrics backend. Do not add Prometheus/OpenTelemetry dependencies.
- S3 bucket logging/object-lock may be enabled per deployment for additional auditability.
- Audit log entries (append-only in S3) and `head.json` ETag history are the primary sources
  for operational reconstruction; avoid duplicating this state elsewhere.

## Cross-References

- [Effect Interpreter Doctrine](effect_interpreter.md) — logging effects and interpreter duties.
- [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md) — device expectations for logging context.
- [Blockchain Model Versioning](blockchain_storage.md) — S3 SSoT and audit log design.
- [Monitoring and Alerting (Retired)](monitoring_and_alerting.md) — rationale for no Prometheus stack.
- [Documentation Standards](../documentation_standards.md) — metadata and linking.
