# File: documents/engineering/pytorch_facade.md
# Torch Runtime (Facade Removed)

**Status**: Authoritative source  
**Supersedes**: Prior PyTorch facade notes  
**Referenced by**: documents/documentation_standards.md; documents/engineering/README.md

> **Purpose**: Document the move away from the import-time PyTorch facade toward a
> Total Pure Modelling approach: a pure runtime ADT plus an explicit effect that
> configures deterministic PyTorch execution exactly once.

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

## Required Workflow
1) Produce the runtime decision *purely* (probe CUDA, check device count, validate cuDNN) and return
   a `TorchRuntime` value.
2) Interpret that decision with an explicit side-effect that sets deterministic flags and returns a
   validated `torch` module instance.
3) Pass the validated handle into interpreters (GPU, training, MonteCarlo) and builders; never rely
   on global import order.
4) If the runtime is `Rejected`, surface the reason in logs/metrics and short-circuit the pipeline.

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

## FAQ
- **Why drop the facade?** Import-order guards hid failures and could not be modelled as data. The
  pure runtime ADT + effect approach lets us represent determinism readiness explicitly and keeps
  side effects inside interpreters, matching Total Pure Modelling.
- **Can we still enforce determinism?** Yes. The interpreter applying the runtime effect must fail
  closed if any flag cannot be set or if CUDA/cuDNN are missing.
- **How do we prevent accidental raw `import torch`?** Add lint checks and review gates; code paths
  should accept an injected torch handle rather than importing it directly.
