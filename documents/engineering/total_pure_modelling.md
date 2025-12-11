# File: documents/engineering/total_pure_modelling.md
# Total Pure Modelling

**Status**: Authoritative source  
**Supersedes**: N/A  
**Referenced by**: CLAUDE.md; documents/engineering/README.md

> **Purpose**: GPU-first guide for modelling SpectralMC state with total, pure ADTs and
> state machines that track real-world device placement, blockchain storage, and effect
> boundaries. Impossible states must be unrepresentable; timing toggles are forbidden.

## Executive Summary
- Model every GPU/CPU boundary, storage transition, and integration handshake as explicit
  variants. No boolean flags or ad hoc retries.
- Compute pure decisions first (device moves, effect routing, retries), then run effect
  interpreters to execute PyTorch ops, storage writes, or network calls.
- Keep tests and fixtures exhaustive across variants, including failure modes and poisoned
  data, so CUDA paths and blockchain commits stay aligned.

## How to Use This Guide
- Apply the core principles to any SpectralMC domain: PyTorch compute, effect
  interpretation, blockchain storage, or ingestion pipelines.
- Use the domain patterns as checklists; pull canonical shapes and constraints from the
  linked SSoTs before adding new variants.
- Keep pure logic separate from effects; tests cover only the pure decisions. Effect
  interpreters stay small and deterministic.

## Core Principles for SpectralMC
- Model only realizable GPU/CPU/device transitions; impossible placements cannot appear in
  the types. Device defaults (`cuda:0` vs `cpu`) are explicit.
- Every variant reaches a decision; no pending states without exit paths or capped retries.
- No timing hacks or env flags to hide drift; fix the model instead (see
  testing_requirements.md for determinism).
- Frontend/backend and orchestrators share state names and HTTP codes. Effect names mirror
  the same variants in logs and metrics.
- Compute pure decisions first; run CUDA kernels, transfers, blockchain commits, or socket
  connects only after a guard result.

## SpectralMC-Specific Patterns
### Device Placement and Transfers
- Represent placement explicitly (`OnGpu(device)`, `OnCpu`, `PinnedHost`, `RemoteShard`).
- Model transfer intent, not just action (`StayOnGpu`, `MoveToCpu(reason)`,
  `RejectTransfer(reason)`). Do not allow silent CPU fallbacks.
- Encode caps for host transfers and float downsampling in the type
  (`CpuTransferCapped(max_bytes)`), so callers cannot bypass size checks.

### Effect Interpretation
- Pair every pure decision ADT with an interpreter that owns effects (CUDA kernels,
  blockchain writes, S3 uploads). Link to effect_interpreter.md for interpreter structure.
- Keep interpreters total: every variant mapped; no default branches or implicit retries.
- Prefer expression-style control flow; avoid mutation of tensors or buffers inside
  interpreters unless explicitly modelled as effects.

### Pipelines, Training, and Retries
- Split ingest → validate → stage_on_gpu → train → persist checkpoints → publish metrics
  into variants with typed failure reasons.
- Make retries bounded in the type (`RetryScheduled(at, attempt)`, `RetryExhausted`) to
  prevent runaway GPU usage.
- Model poisoned data (`DeadLettered(reason)`), incompatible dtypes, and precision changes
  (`CastToBfloat16`) explicitly to prevent silent accuracy drift.

### Blockchain Storage and Integrity
- Capture local vs canonical truth (`LocalPending`, `Committed`, `RollbackRequired`,
  `ConflictDetected(remote_hash)`) so replay and integrity checks are deterministic.
- Keep state transitions aligned with immutability_doctrine.md: terminal states are
  terminal unless `RestartedFrom(previous_state)` is explicit.

### Product Surfaces (CLI/API)
- Model loading phases (`Bootstrapping`, `Hydrating`, `Ready(view)`), not booleans.
- Map UI/API outcomes deterministically: 401 ↔ `RedirectToLogin`, 403 ↔ `Denied`,
  409 ↔ `Conflict`. Avoid divergence between CLI exits and API responses.
- Gate WebSocket or gRPC connects behind guard results; no side effects before a pure
  decision is produced.

## Where Models Drift from Reality (GPU Edition)
- Treating “unknown device” as “CPU” hides performance bugs; represent `DeviceUnknown` or
  `PlacementLookupFailed` instead of defaulting.
- Missing host-transfer caps or lack of pinned memory states leads to OOM and thrashing.
- Implicit dtype casts (`float32`→`float16`) without a state cause silent accuracy loss.
- Background refresh or retries without explicit states create race conditions across
  trainers or workers.
- Conflating “empty batch” with “ingest failure” forces callers to poll and retries to flap.

## Patterns to Embrace
- Exhaustive matches on ADTs with no default branches.
- Deterministic bootstrap: empty storage jumps to a terminal choice; existing checkpoints
  trigger a single replay or refresh.
- Guard-driven orchestration: pure decisions first, interpreters second.
- Shared meaning across logs, metrics, and HTTP codes; device placement encoded in names
  (`effect=move_to_cpu`, `state=retry_scheduled`).
- Fixtures and generators that cover every variant, including error paths and transfer
  rejection reasons.

## Pitfalls to Avoid
- Truthy shortcuts (`if batch` or `if tensor.device.type == "cuda"`); use explicit variants.
- Silent CPU fallbacks or “best effort” transfers; represent and reject impossible paths.
- Infinite hydration/retry loops; always include exit states and attempt caps.
- Sleep/polling to “eventually” resolve; schedule retries explicitly with timestamps.
- Env flags that change behavior (`PYTEST_E2E`); rely on fixtures and dependency injection.
- Any side effect that bypasses the guard or interpreter (ungated kernel launch or
  blockchain write).

## Small Code Postcards
### GPU→CPU Transfer Decision (Python)
```python
# File: src/spectralmc/pipeline/device_transfer.py
from dataclasses import dataclass
from typing import Literal

import torch

MAX_CPU_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class TransferDecision:
    kind: Literal["stay_on_gpu", "move_to_cpu", "reject"]
    reason: str | None = None


def decide_transfer(tensor: torch.Tensor, target: torch.device) -> TransferDecision:
    if tensor.device == target:
        return TransferDecision(kind="stay_on_gpu")
    if target.type == "cpu" and tensor.element_size() * tensor.numel() > MAX_CPU_BYTES:
        return TransferDecision(kind="reject", reason="oversized_host_transfer")
    if target.type == "cpu" and not tensor.is_pinned():
        return TransferDecision(kind="reject", reason="unpinned_host_memory")
    return TransferDecision(kind="move_to_cpu")


def apply_transfer(tensor: torch.Tensor, decision: TransferDecision,
                   target: torch.device) -> torch.Tensor:
    match decision.kind:
        case "stay_on_gpu":
            return tensor
        case "move_to_cpu":
            return tensor.to(device=target, non_blocking=True)
        case "reject":
            raise RuntimeError(decision.reason or "transfer_rejected")
    raise ValueError("unreachable")
```

### Effect Interpreter for Ingest Pipeline (Python)
```python
# File: src/spectralmc/pipeline/ingest_effects.py
from dataclasses import dataclass
from typing import Literal

from spectralmc.effects import Effect, run_effects


@dataclass(frozen=True)
class IngestState:
    kind: Literal["validated", "retry", "dead_letter"]
    payload: bytes | None = None
    reason: str | None = None


def interpret_ingest(state: IngestState) -> list[Effect]:
    match state.kind:
        case "validated":
            return [Effect.stage_on_gpu(state.payload or b"")]
        case "retry":
            return [Effect.schedule_retry(state.reason or "unknown")]
        case "dead_letter":
            return [Effect.write_dead_letter(state.reason or "unknown")]
    raise ValueError("unreachable")


def handle_ingest(state: IngestState) -> None:
    effects = interpret_ingest(state)
    run_effects(effects)
```

## Cross-References
- [coding_standards.md](coding_standards.md#result-types)
- [purity_doctrine.md](purity_doctrine.md#pure-adt-boundaries)
- [purity_enforcement.md](purity_enforcement.md#runtime-guards)
- [cpu_gpu_compute_policy.md](cpu_gpu_compute_policy.md#device-placement-constraints)
- [effect_interpreter.md](effect_interpreter.md#interpreter-design)
- [testing_requirements.md](testing_requirements.md#determinism-and-gpu-assumptions)
- [testing_architecture.md](testing_architecture.md#fixtures-and-generators)
- [reproducibility_proofs.md](reproducibility_proofs.md#gpu-determinism)
