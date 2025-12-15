# File: documents/engineering/monitoring_and_alerting.md
# Monitoring and Alerting (Retired)

**Status**: Deprecated  
**Supersedes**: none  
**Referenced by**: engineering/README.md (for historical context)

> **Purpose**: Record that centralized monitoring/alerting (Prometheus, OpenTelemetry, etc.)
> is not part of the SpectralMC operational model. S3-backed manifests and audit logs are
> the source of truth for training and model state; use deployment-specific logging only.

## Cross-References

- [Observability](observability.md) — Logging and audit expectations without Prometheus
- [Blockchain Model Versioning](blockchain_storage.md) — S3 state and audit log SSoT
- [Documentation Standards](../documentation_standards.md) — Metadata and linking rules

## Current Policy

- No centralized monitoring stack is prescribed. Do not add Prometheus/OpenTelemetry hooks.
- Operational state comes from S3: `head.json`, immutable version directories, and the
  append-only audit log.
- Per-deployment logging (stdout/structured logs) may be used for debugging but must not
  attempt to mirror training or checkpoint state; treat S3 as the canonical record.
