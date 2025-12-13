"""SpectralMC model implementations and utilities.

This subpackage provides:

- **runtime/torch_runtime.py** - TorchRuntime decision + effect for deterministic torch
  configuration and handle injection.
- **torch.py** - Typed PyTorch helpers (Device/dtype enums, context managers, TensorState
  serialization helpers) consuming the injected torch handle from
  `spectralmc.runtime.torch_runtime.get_torch_handle`.
- **numerical.py** - Precision enum and numeric type utilities.
- **cpu_gpu_transfer.py** - Pure-functional TensorTree API for CPU-to-GPU transfers.

All modules are fully typed and mypy-strict-clean.
"""
