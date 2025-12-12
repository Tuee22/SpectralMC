# File: documents/engineering/pytorch_facade.md
# Torch Runtime (Facade Removed)

**Status**: Authoritative source  
**Supersedes**: Prior PyTorch facade notes  
**Referenced by**: documents/documentation_standards.md; documents/engineering/README.md

> **Purpose**: Document the move away from the import-time PyTorch facade toward a
> Total Pure Modelling approach: a pure runtime ADT plus an explicit effect that
> configures deterministic PyTorch execution exactly once.
> This follows the guard→decision→effect pipeline described in
> [total_pure_modelling.md](total_pure_modelling.md#core-principles-for-spectralmc).

## Cross-References
- [Coding Standards](coding_standards.md)
- [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md)
- [Reproducibility Proofs](reproducibility_proofs.md)
- [Effect Interpreter](effect_interpreter.md)
- [total_pure_modelling.md](total_pure_modelling.md)

## What Changed
- **Facade removed**: `spectralmc.models.torch` is no longer the enforced entry point.
- **Pure runtime ADT**: Model torch readiness as data (e.g., `TorchRuntime = Ready(cuda, cudnn) |
  Rejected(reason)`).
- **Explicit effect**: A single effect applies deterministic flags (`use_deterministic_algorithms`,
  cuDNN/TF32 disables, `CUBLAS_WORKSPACE_CONFIG`) and yields the `torch` handle to downstream code.
- **Dependency injection**: Interpreters and trainers receive the runtime handle instead of
  importing `torch` implicitly; tests can inject fakes or poisoned runtimes.
- **TPM alignment**: All torch usage follows the Total Pure Modelling rule: decide first (pure ADT),
  then run an interpreter to perform effects. See the device/transfer section in
  [total_pure_modelling.md](total_pure_modelling.md#device-placement-and-transfers).

## Required Workflow (guard → decision → effect)
1) **Guard/pure decision**: Probe CUDA, device count, cuDNN version → return `TorchRuntime`.
2) **Effect**: Interpret only the `ready` variant; set deterministic flags; return a validated
   `torch` handle (or fail closed).
3) **Injection**: Pass the handle into interpreters (GPU, training, MonteCarlo) and builders; never
   rely on import order or global defaults.
4) **Rejection path**: If `TorchRuntime` is `rejected`, log/emit metrics and stop the pipeline.

## Determinism Checklist (applied by the effect)
- `torch.use_deterministic_algorithms(True, warn_only=False)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.backends.cudnn.allow_tf32 = False`
- `torch.backends.cuda.matmul.allow_tf32 = False`
- `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")`

These settings remain mandatory; they just run under an effect instead of at import time.

## Threading and Purity
- No implicit main-thread guard via imports. If thread affinity is required, encode it in the
  runtime ADT (`RequiresMainThread | WorkerSafe`) and assert it in the interpreter.
- Call sites must stay pure: decide first, then run the effect to configure torch, then execute
  GPU/CPU work using the injected handle.

## Migration Notes
- Remove “import facade first” guidance from code and docs; replace with “obtain a validated
  TorchRuntime and inject the returned handle”.
- Legacy helpers like `default_device`/`default_dtype` should be re-expressed as explicit,
  validated effects or pure decisions around device/dtype creation. Do not rely on global defaults.
- Tests should build the runtime ADT in fixtures, apply the configuration effect once, and inject
  the handle into systems under test.

## Examples (concrete TPM-aligned patterns)

### Pure decision + effect application

```python
# File: src/spectralmc/runtime/torch_runtime.py (conceptual)
from dataclasses import dataclass
from typing import Literal
import os

@dataclass(frozen=True)
class TorchRuntime:
    kind: Literal["ready", "rejected"]
    cuda_version: str | None = None
    cudnn_version: int | None = None
    reason: str | None = None

def decide_torch_runtime() -> TorchRuntime:
    import torch  # local import to keep decision pure from globals
    if not torch.cuda.is_available():
        return TorchRuntime(kind="rejected", reason="cuda_unavailable")
    cudnn_ver = torch.backends.cudnn.version()
    if cudnn_ver is None:
        return TorchRuntime(kind="rejected", reason="cudnn_unavailable")
    return TorchRuntime(kind="ready", cuda_version=torch.version.cuda, cudnn_version=cudnn_ver)

def apply_torch_runtime(runtime: TorchRuntime):
    if runtime.kind != "ready":
        raise RuntimeError(runtime.reason or "torch_runtime_rejected")
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch
```

### Interpreter construction with injected torch (no import-order dependency)

```python
# File: src/spectralmc/effects/interpreter_factory.py (conceptual)
from spectralmc.runtime.torch_runtime import decide_torch_runtime, apply_torch_runtime
from spectralmc.effects.interpreter import SpectralMCInterpreter

runtime = decide_torch_runtime()
torch_handle = apply_torch_runtime(runtime)
torch_stream = torch_handle.cuda.Stream()
cupy_stream = cupy.cuda.Stream()
interpreter = SpectralMCInterpreter.create(torch_stream, cupy_stream, storage_bucket, torch_handle)
```

These examples follow the guard→decision→effect guidance in
[total_pure_modelling.md](total_pure_modelling.md#core-principles-for-spectralmc).

## FAQ
- **Why drop the facade?** Import-order guards hid failures and could not be modelled as data. The
  pure runtime ADT + effect approach lets us represent determinism readiness explicitly and keeps
  side effects inside interpreters, matching Total Pure Modelling.
- **Can we still enforce determinism?** Yes. The interpreter applying the runtime effect must fail
  closed if any flag cannot be set or if CUDA/cuDNN are missing.
- **How do we prevent accidental raw `import torch`?** Add lint checks and review gates; code paths
  should accept an injected torch handle rather than importing it directly.
