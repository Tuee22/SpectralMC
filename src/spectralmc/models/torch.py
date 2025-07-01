"""
spectralmc.models.torch
=======================

Thin, *reproducibility‑aware* interop layer between the SpectralMC code‑base
and PyTorch.  Responsibilities:

*   Centralise every global switch that influences determinism.
*   Offer serialisable Pydantic wrappers (`dtype`, `TensorState`, …).
*   Provide a one‑shot `TorchEnv.snapshot()` to lock down the local runtime.
"""

from __future__ import annotations

import os
import platform
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Iterator, Tuple

import torch
from pydantic import BaseModel, ConfigDict

__all__: Tuple[str, ...] = (
    "dtype",
    "device",
    "TensorState",
    "TorchEnv",
    "default_dtype",
)

# ───────────────────────────── determinism at import ─────────────────────────

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
torch.use_deterministic_algorithms(True, warn_only=False)
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────── dtype helpers ───────────────────────────────

_DTYPE_STR_TO_TORCH: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
_TORCH_DTYPE_TO_STR: Dict[torch.dtype, str] = {
    v: k for k, v in _DTYPE_STR_TO_TORCH.items()
}


class dtype(str, Enum):  # pylint: disable=invalid-name
    """All Torch dtypes that SpectralMC supports – **serialisable**."""

    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    bfloat16 = "bfloat16"
    complex64 = "complex64"
    complex128 = "complex128"

    # ------------------------------------------------------------------ #
    # Torch interop
    # ------------------------------------------------------------------ #

    def to_torch(self) -> torch.dtype:
        return _DTYPE_STR_TO_TORCH[self.value]

    @classmethod
    def from_torch(cls, dt: torch.dtype) -> "dtype":
        try:
            return cls(_TORCH_DTYPE_TO_STR[dt])
        except KeyError as exc:  # pragma: no cover – guard rail
            raise ValueError(f"Unsupported torch.dtype {dt!r}") from exc


# ─────────────────────────────── device helpers ──────────────────────────────


class device(str, Enum):  # pylint: disable=invalid-name
    cpu = "cpu"
    cuda = "cuda"

    def to_torch(self) -> torch.device:  # noqa: D401
        return torch.device(self.value)

    @classmethod
    def from_torch(cls, dev: torch.device) -> "device":
        return cls(dev.type)


# ───────────────────────────── context managers ──────────────────────────────


@contextmanager
def default_dtype(dt: torch.dtype) -> Iterator[None]:
    """Temporarily switch the *default* floating point dtype."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# ───────────────────────────── safetensors wiring ────────────────────────────

from io import BytesIO
from safetensors import safe_open
from safetensors.torch import save as _sf_save  # type: ignore  (Mypy stub present)


class TensorState(BaseModel):
    """Loss‑less SafeTensors wrapper around a numeric `torch.Tensor`."""

    data: bytes
    shape: Tuple[int, ...]
    dtype: dtype

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ......................................................

    @staticmethod
    def from_tensor(t: torch.Tensor) -> "TensorState":
        t_cpu = t.detach().cpu()
        return TensorState(
            data=_sf_save({"tensor": t_cpu}),
            shape=tuple(t_cpu.shape),
            dtype=dtype.from_torch(t_cpu.dtype),
        )

    def to_tensor(self, *, device: torch.device) -> torch.Tensor:  # noqa: D401
        with safe_open(BytesIO(self.data), framework="pt", device=str(device)) as f:
            return f.get_tensor("tensor")


# ───────────────────────────── runtime fingerprint ───────────────────────────


class TorchEnv(BaseModel):
    """
    Minimal runtime fingerprint – store this next to a checkpoint so you can
    re‑create *exactly* the same environment later.
    """

    torch_version: str
    cuda_version: str | None
    cudnn_version: int | None
    gpu_name: str | None
    python_version: str
    seed: int

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ......................................................

    @classmethod
    def snapshot(cls, *, seed: int) -> "TorchEnv":
        return cls(
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
            cudnn_version=(
                torch.backends.cudnn.version() if torch.cuda.is_available() else None
            ),
            gpu_name=(
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            python_version=platform.python_version(),
            seed=seed,
        )
