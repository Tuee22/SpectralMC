# Observability

**Status**: Authoritative source (SpectralMC scope)  
**Supersedes**: none  
**Referenced by**: engineering/README.md

> **Purpose**: Define SpectralMC’s observability expectations (metrics/logging) in a backend-agnostic way. Aligns naming with effectful while excluding non-SpectralMC domain detail.

## Scope

- Metrics for GPU execution, training workflows, and storage integrity.
- Logging modeled as effects; interpreters emit logs.
- Alerting severity/routing are documented separately once platform is selected.

## Principles

- **Type-safe metrics**: metrics are defined once with fixed label schemas; avoid runtime creation.
- **Deterministic, GPU-first signals**: capture device, dtype, and kernel identifiers to detect CPU fallbacks.
- **Low cardinality**: bound label value sets (model_id, kernel, device, version) to prevent explosion.
- **Separation of concerns**: program code emits metric/log effects; interpreters send to the telemetry backend.

## Core Metrics (minimum set)

- **GPU utilization**: per-device kernel occupancy and memory footprint.
- **Training health**: loss/gradient norms, step duration, checkpoint success/failure.
- **Storage integrity**: chain verification results, CAS commit/rollback counts.
- **RNG determinism**: RNG capture/restore success, divergence counters.

## Logging

- Use `LogMessage` effects with level/message/context; avoid direct logger calls in pure code.
- Include device/model identifiers in log context to spot CPU fallbacks or mismatched models.
- Ensure logs are emitted from interpreters; pure layers remain deterministic.

## Implementation Notes

- Backend is intentionally unspecified (Prometheus/OpenTelemetry/etc.); choose a consistent stack per deployment and keep schemas stable.
- Metric and log definitions live alongside interpreters to ensure GPU/storage context is available.
- Alerting rules should build on these metrics; see `monitoring_and_alerting.md` when adopted.

## Cross-References

- [Effect Interpreter Doctrine](effect_interpreter.md) — logging/metrics as effects.
- [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md) — device expectations for metric dimensions.
- [Blockchain Model Versioning](blockchain_storage.md) — integrity metrics.
- [Testing](testing.md) — ensures GPU-only paths and deterministic signals.
- [Documentation Standards](../documentation_standards.md) — metadata and linking.
