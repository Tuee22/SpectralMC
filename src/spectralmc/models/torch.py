"""
spectralmc.models.torch
=======================

A minimal, fully‑typed façade exposing just enough of PyTorch for
SpectralMC while enforcing reproducible execution.
"""

from __future__ import annotations

###############################################################################
# Early environment fixes -----------------------------------------------------
###############################################################################
import os
import platform
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Iterable, Iterator, List, Mapping, Protocol, Tuple

# Ensure deterministic cuBLAS kernels (ignored if CUDA is absent)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch  # noqa: E402

###############################################################################
# Determinism knobs -----------------------------------------------------------
###############################################################################
torch.use_deterministic_algorithms(True, warn_only=False)
_HAS_CUDA = torch.cuda.is_available()

if _HAS_CUDA:
    if torch.backends.cudnn.version() is None:
        raise RuntimeError(
            "SpectralMC requires cuDNN for deterministic GPU execution, "
            "but it is missing from the current runtime."
        )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    # Even on pure‑CPU builds make sure these flags exist and are disabled
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False

###############################################################################
# Strongly typed helpers ------------------------------------------------------
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
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    bfloat16 = "bfloat16"
    complex64 = "complex64"
    complex128 = "complex128"

    # --- conversion helpers -------------------------------------------------
    def to_torch(self) -> torch.dtype:  # noqa: D401
        return _DTYPE_STR_TO_TORCH[self.value]

    @classmethod
    def from_torch(cls, dt: torch.dtype) -> "DType":
        if dt not in _TORCH_DTYPE_TO_STR:
            raise ValueError(f"Unsupported torch.dtype {dt!r}")
        return cls(_TORCH_DTYPE_TO_STR[dt])


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda:0"  # single‑GPU policy

    # --- conversion helpers -------------------------------------------------
    def to_torch(self) -> torch.device:  # noqa: D401
        return torch.device(self.value)

    @classmethod
    def from_torch(cls, dev: torch.device) -> "Device":
        if dev.type == "cpu":
            return cls.cpu
        if dev.type == "cuda" and dev.index in {None, 0}:
            return cls.cuda
        raise ValueError("Only CPU and the first CUDA card are supported.")


###############################################################################
# Lightweight global context managers ----------------------------------------
###############################################################################
@contextmanager
def default_dtype(dt: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextmanager
def default_device(dev: torch.device) -> Iterator[None]:
    previous = torch.tensor([]).device
    torch.set_default_device(dev)
    try:
        yield
    finally:
        torch.set_default_device(previous)


###############################################################################
# SafeTensor serialisation helpers -------------------------------------------
###############################################################################
from pydantic import BaseModel, ConfigDict  # noqa: E402
from safetensors.torch import load as _sf_load, save as _sf_save  # noqa: E402


class TensorState(BaseModel):
    """A CPU‑only SafeTensor snapshot with integrity metadata."""

    data: bytes
    shape: Tuple[int, ...]
    dtype: DType

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_torch(t: torch.Tensor) -> "TensorState":
        if t.device != Device.cpu.to_torch():
            raise RuntimeError("Tensor must reside on CPU to enter TensorState.")
        blob: bytes = _sf_save({"tensor": t})
        return TensorState(
            data=blob, shape=tuple(t.shape), dtype=DType.from_torch(t.dtype)
        )

    def to_torch(self) -> torch.Tensor:
        tensor_dict: dict[str, torch.Tensor] = _sf_load(
            self.data, device=Device.cpu.value
        )
        tensor = tensor_dict["tensor"]
        if (
            tuple(tensor.shape) != self.shape
            or DType.from_torch(tensor.dtype) != self.dtype
        ):
            raise RuntimeError("Tensor metadata mismatch on deserialisation.")
        return tensor

    @staticmethod
    def from_bytes(raw: bytes) -> "TensorState":
        tensor_dict: dict[str, torch.Tensor] = _sf_load(raw, device=Device.cpu.value)
        if set(tensor_dict) != {"tensor"}:
            raise ValueError("SafeTensor must contain exactly one entry named 'tensor'.")
        t = tensor_dict["tensor"]
        return TensorState(
            data=raw, shape=tuple(t.shape), dtype=DType.from_torch(t.dtype)
        )


###############################################################################
# Environment fingerprint -----------------------------------------------------
###############################################################################
class TorchEnv(BaseModel):
    torch_version: str
    cuda_version: str
    cudnn_version: int
    gpu_name: str
    python_version: str

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

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
    step: int
    exp_avg: TensorState
    exp_avg_sq: TensorState
    max_exp_avg_sq: TensorState | None

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, s: Mapping[str, object]) -> "AdamParamState":
        allowed = {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        extra = set(s) - allowed
        if extra:
            raise RuntimeError(f"Unexpected Adam state keys: {extra}")

        def _req_tensor(obj: object, name: str) -> torch.Tensor:
            if not isinstance(obj, torch.Tensor):
                raise TypeError(f"Adam '{name}' must be torch.Tensor.")
            if obj.device != Device.cpu.to_torch():
                raise RuntimeError("Adam state tensors must live on CPU.")
            return obj

        # Accept Python int or 0‑D CPU tensor
        step_raw = s["step"]
        if isinstance(step_raw, torch.Tensor):
            if step_raw.ndim != 0 or step_raw.device != Device.cpu.to_torch():
                raise TypeError("Adam 'step' must be a CPU scalar tensor.")
            step_val = int(step_raw.item())
        elif isinstance(step_raw, int):
            step_val = step_raw
        else:
            raise TypeError("Adam 'step' must be int or scalar tensor.")

        return cls(
            step=step_val,
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

    def to_torch(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "step": self.step,
            "exp_avg": self.exp_avg.to_torch(),
            "exp_avg_sq": self.exp_avg_sq.to_torch(),
        }
        if self.max_exp_avg_sq is not None:
            out["max_exp_avg_sq"] = self.max_exp_avg_sq.to_torch()
        return out


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
    param_states: Dict[int, AdamParamState]
    param_groups: List[AdamParamGroup]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_torch(cls, optim: _HasStateDict) -> "AdamOptimizerState":
        if optim.__class__.__name__ not in {"Adam", "AdamW"}:
            raise TypeError("Only torch.optim.Adam or AdamW are supported.")

        sd = optim.state_dict()
        if set(sd) != {"state", "param_groups"}:
            raise RuntimeError("state_dict() must contain exactly state & param_groups")

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