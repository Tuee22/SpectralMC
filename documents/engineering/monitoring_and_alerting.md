# File: documents/engineering/monitoring_and_alerting.md
# Monitoring and Alerting

**Status**: Reference only  
**Supersedes**: none  
**Referenced by**: engineering/README.md (future)

> **Purpose**: Placeholder alignment with effectful naming. SpectralMC alerting rules depend on the chosen telemetry platform and must be defined deployment-by-deployment. Use this file to document adopted rules once the platform is selected.
> **📖 Authoritative Reference**: [Observability](observability.md)

## Interim Guidance

- Build alerts on the core metrics defined in [observability.md](observability.md) (GPU utilization, training health, storage integrity, RNG determinism).
- Define severity levels, routing, and runbooks in deployment-specific overlays (not in source) to avoid leaking environment details.
- Keep PromQL/alert expressions (or equivalent) co-located with deployment manifests; link them here when finalized.

## Action Items

- Choose telemetry stack (Prometheus/OpenTelemetry/etc.).
- Document severity/routing conventions and runbooks per deployment.
- Backfill links here once adopted.
