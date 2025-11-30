"""SpectralMC model implementations and utilities.

This subpackage provides:

- **torch.py** - PyTorch facade for deterministic execution with Device/DType enums,
  context managers, and TensorState serialization helpers.
- **numerical.py** - Precision enum and numeric type utilities.
- **cpu_gpu_transfer.py** - Pure-functional TensorTree API for CPU-to-GPU transfers.

All modules are fully typed and mypy-strict-clean.
"""
