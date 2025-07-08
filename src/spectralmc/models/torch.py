"""
spectralmc.models.torch
=======================

A *minimal‑surface* façade exposing only the PyTorch functionality required by
SpectralMC while enforcing **full determinism**.

---------------------------------------------------------------------------
Why a façade?
---------------------------------------------------------------------------
* Central place to pin all reproducibility flags (cuBLAS, cuDNN, TF32, etc.).
* Provide strongly‑typed helpers (`DType`, `Device`) that round‑trip cleanly
  with the real PyTorch singletons **without** leaking dynamic types into the
  public API.
* Supply thread‑safe context‑managers for temporarily overriding the global
  default *dtype* / *device*.  The implementation serialises callers with
  two distinct locks, so a single thread restoring one setting never tramples
  the other.
* Offer a *portable* yet *loss‑less* representation of tensors and Adam
  optimiser state that can be shipped over the wire as plain JSON / msg‑pack.

The original version aborted on CPU‑only machines; this edition **downgrades
gracefully** to a reduced‑feature mode when CUDA/cuDNN are absent. The public
API and mypy‑strict guarantees remain identical on both CPU and GPU.
"""

from __future__ import annotations

###############################################################################
# Environment tweaks (must run *before* the first ``import torch``) -----------
###############################################################################
import os
import platform
import threading
from contextlib import contextmanager
from enum import Enum
from io import BytesIO
from typing import Dict, Iterable, Iterator, List, Mapping, Protocol, Tuple

# Fix cuBLAS workspace deterministically (ignored on CPU installs)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch  # noqa: E402

###############################################################################
# Global determinism (valid on CPU and GPU) -----------------------------------
###############################################################################
torch.use_deterministic_algorithms(True, warn_only=False)
_HAS_CUDA = torch.cuda.is_available()

# ------------------------------------------------------------------ #
# cuDNN / TF32 toggles – guarded so CPU‑only builds don't choke
# ------------------------------------------------------------------ #
if _HAS_CUDA:
    if torch.backends.cudnn.version() is None:
        raise RuntimeError(
            "SpectralMC requires cuDNN for deterministic GPU execution, "
            "but it is missing in the current runtime."
        )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except AttributeError:  # pragma: no cover
        pass
else:
    # Best effort on CPU: set whatever attributes exist.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

###############################################################################
# Strongly‑typed helpers ------------------------------------------------------
###############################################################################
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


class DType(str, Enum):
    """Floating‑point and complex formats accepted by SpectralMC."""

    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    bfloat16 = "bfloat16"
    complex64 = "complex64"
    complex128 = "complex128"

    # ------------------------------------------------------------------ #
    def to_torch(self) -> torch.dtype:
        return _DTYPE_STR_TO_TORCH[self.value]

    @classmethod
    def from_torch(cls, dt: torch.dtype) -> "DType":
        try:
            return cls(_TORCH_DTYPE_TO_STR[dt])
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unsupported torch.dtype {dt!r}") from exc


class Device(str, Enum):
    """`cpu` or the *first* CUDA card (SpectralMC is single‑GPU)."""

    cpu = "cpu"
    cuda = "cuda:0"

    # ------------------------------------------------------------------ #
    def to_torch(self) -> torch.device:
        return torch.device(self.value)

    @classmethod
    def from_torch(cls, dev: torch.device) -> "Device":
        if dev.type == "cpu":
            return cls.cpu
        if dev.type == "cuda" and dev.index in {None, 0}:
            return cls.cuda
        raise ValueError("SpectralMC supports only CPU and the first CUDA device.")


###############################################################################
# Thread‑safe context‑managers ------------------------------------------------
###############################################################################
_dtype_lock = threading.RLock()
_device_lock = threading.RLock()


@contextmanager
def default_dtype(dt: torch.dtype) -> Iterator[None]:
    with _dtype_lock:
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dt)
        try:
            yield
        finally:
            torch.set_default_dtype(prev)


@contextmanager
def default_device(dev: torch.device) -> Iterator[None]:
    with _device_lock:
        prev = torch.tensor([]).device
        torch.set_default_device(dev)
        try:
            yield
        finally:
            torch.set_default_device(prev)


###############################################################################
# SafeTensor serialisation ----------------------------------------------------
###############################################################################
from pydantic import BaseModel, ConfigDict  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save as _sf_save  # noqa: E402


class TensorState(BaseModel):
    """
    A CPU‑only SafeTensor snapshot that travels as raw bytes yet remembers
    shape and dtype for integrity checks on deserialisation.
    """

    data: bytes
    shape: Tuple[int, ...]
    dtype: DType

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_torch(t: torch.Tensor) -> "TensorState":
        if t.device != Device.cpu.to_torch():
            raise RuntimeError("TensorState expects a CPU tensor.")
        blob = _sf_save({"tensor": t})
        return TensorState(
            data=blob, shape=tuple(t.shape), dtype=DType.from_torch(t.dtype)
        )

    def to_torch(self) -> torch.Tensor:
        with safe_open(
            BytesIO(self.data), framework="pt", device=Device.cpu.value
        ) as reader:
            tensor = reader.get_tensor("tensor")
        if (
            tuple(tensor.shape) != self.shape
            or DType.from_torch(tensor.dtype) != self.dtype
        ):
            raise RuntimeError("Tensor metadata mismatch after deserialisation.")
        return tensor

    # Convenience validator ---------------------------------------------------
    @staticmethod
    def from_bytes(raw: bytes) -> "TensorState":
        with safe_open(BytesIO(raw), framework="pt", device=Device.cpu.value) as reader:
            if reader.keys() != ["tensor"]:
                raise ValueError("SafeTensor must contain exactly one entry 'tensor'.")
            t = reader.get_tensor("tensor")
        return TensorState(
            data=raw, shape=tuple(t.shape), dtype=DType.from_torch(t.dtype)
        )


###############################################################################
# Runtime fingerprint ---------------------------------------------------------
###############################################################################
class TorchEnv(BaseModel):
    """Structured snapshot of the active PyTorch / CUDA environment."""

    torch_version: str
    cuda_version: str
    cudnn_version: int
    gpu_name: str
    python_version: str

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def snapshot(cls) -> "TorchEnv":
        if _HAS_CUDA:
            cuda_ver = torch.version.cuda or "<unknown>"
            cudnn_ver = torch.backends.cudnn.version() or -1
            gpu = torch.cuda.get_device_name(0)
        else:
            cuda_ver = "<not available>"
            cudnn_ver = -1
            gpu = "<cpu>"
        return cls(
            torch_version=torch.__version__,
            cuda_version=cuda_ver,
            cudnn_version=cudnn_ver,
            gpu_name=gpu,
            python_version=platform.python_version(),
        )


###############################################################################
# Adam optimiser helpers ------------------------------------------------------
###############################################################################
class _HasStateDict(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...


class AdamParamState(BaseModel):
    """Per‑parameter state required by Adam/AdamW."""

    step: int
    exp_avg: TensorState
    exp_avg_sq: TensorState
    max_exp_avg_sq: TensorState | None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, s: Mapping[str, object]) -> "AdamParamState":
        allowed = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        unexpected = set(s) - allowed
        if unexpected:  # pragma: no cover
            raise RuntimeError(f"Unexpected key(s) in Adam state: {unexpected}")

        def _req_tensor(obj: object, name: str) -> torch.Tensor:
            if not isinstance(obj, torch.Tensor):
                raise TypeError(f"Adam '{name}' must be torch.Tensor.")
            if obj.device != Device.cpu.to_torch():
                raise RuntimeError("All Adam state tensors must reside on CPU.")
            return obj

        step_raw = s["step"]
        if not isinstance(step_raw, int):
            raise TypeError("Adam 'step' must be int.")

        return cls(
            step=step_raw,
            exp_avg=TensorState.from_torch(_req_tensor(s["exp_avg"], "exp_avg")),
            exp_avg_sq=TensorState.from_torch(
                _req_tensor(s["exp_avg_sq"], "exp_avg_sq")
            ),
            max_exp_avg_sq=(
                TensorState.from_torch(
                    _req_tensor(s["max_exp_avg_sq"], "max_exp_avg_sq")
                )
                if s.get("max_exp_avg_sq") is not None
                else None
            ),
        )

    # ------------------------------------------------------------------ #
    def to_torch(self) -> Dict[str, object]:
        base: Dict[str, object] = {
            "step": self.step,
            "exp_avg": self.exp_avg.to_torch(),
            "exp_avg_sq": self.exp_avg_sq.to_torch(),
        }
        if self.max_exp_avg_sq is not None:
            base["max_exp_avg_sq"] = self.max_exp_avg_sq.to_torch()
        return base


class AdamParamGroup(BaseModel):
    params: List[int]
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool = False
    maximize: bool = False

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, g: Mapping[str, object]) -> "AdamParamGroup":
        return cls.model_validate(g)

    def to_torch(self) -> Dict[str, object]:
        return self.model_dump(mode="python")


class AdamOptimizerState(BaseModel):
    """Fully‑typed, serialisable capture of an Adam/AdamW optimiser."""

    param_states: Dict[int, AdamParamState]
    param_groups: List[AdamParamGroup]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, optim: _HasStateDict) -> "AdamOptimizerState":
        if optim.__class__.__name__ not in {"Adam", "AdamW"}:
            raise TypeError("Only torch.optim.Adam or AdamW are supported.")
        sd = optim.state_dict()
        if set(sd) != {"state", "param_groups"}:  # pragma: no cover
            raise RuntimeError(
                "state_dict() must contain exactly {'state', 'param_groups'}."
            )
        state_raw, groups_raw = sd["state"], sd["param_groups"]
        if not isinstance(state_raw, Mapping):
            raise TypeError("'state' must be a mapping.")
        if not isinstance(groups_raw, Iterable):
            raise TypeError("'param_groups' must be a sequence.")
        return cls(
            param_states={
                pid: AdamParamState.from_torch(st) for pid, st in state_raw.items()
            },
            param_groups=[AdamParamGroup.from_torch(pg) for pg in groups_raw],
        )

    # ------------------------------------------------------------------ #
    def to_torch(self) -> Dict[str, object]:
        return {
            "state": {pid: st.to_torch() for pid, st in self.param_states.items()},
            "param_groups": [g.to_torch() for g in self.param_groups],
        }


###############################################################################
# Public re‑exports -----------------------------------------------------------
###############################################################################
__all__: Tuple[str, ...] = (
    "DType",
    "Device",
    "TensorState",
    "TorchEnv",
    "AdamParamState",
    "AdamParamGroup",
    "AdamOptimizerState",
    "default_dtype",
    "default_device",
)
