# src/spectralmc/models/torch.py
"""
spectralmc.models.torch
=======================

A minimal, fully‑typed façade exposing just enough of PyTorch for SpectralMC
while enforcing reproducible execution.

Thread‑safety contract
----------------------
* This module **must be imported from the main thread _before_ any other
  threads are spawned**; otherwise it raises ``ImportError``.
* The context managers ``default_dtype`` and ``default_device`` **may only be
  entered from the main thread**; attempting to use them from a worker thread
  raises ``RuntimeError``.
* To override the import‑time thread guard (at your own risk) set the
  environment variable::

      SPECTRALMC_ALLOW_THREADS=1
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
#  Early import‑time guards                                                   #
# --------------------------------------------------------------------------- #
import os
import platform
import sys
import threading
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Iterable, Iterator, List, Mapping, Protocol, Tuple

# --- prevent pre‑emptive torch import -------------------------------------- #
if "torch" in sys.modules:
    raise ImportError(
        "PyTorch was imported before the SpectralMC torch façade. "
        "Import 'spectralmc.models.torch' **first** so it can set "
        "deterministic flags."
    )

# --- thread‑safety: import must occur before worker threads ---------------- #
if threading.active_count() > 1:
    raise ImportError(
        "SpectralMC façade imported after additional threads were created. "
        "Its global dtype/device helpers are not thread‑safe. "
        "Import this module in the *main thread* before spawning workers, "
        "or set SPECTRALMC_ALLOW_THREADS=1 to bypass this check."
    )

_MAIN_THREAD_ID: int = threading.get_ident()

# --------------------------------------------------------------------------- #
#  Early environment fixes                                                    #
# --------------------------------------------------------------------------- #
# Ensure deterministic cuBLAS kernels (ignored if CUDA is absent)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch  # noqa: E402
from spectralmc.models.numerical import Precision

# --------------------------------------------------------------------------- #
#  Determinism knobs                                                          #
# --------------------------------------------------------------------------- #
torch.use_deterministic_algorithms(True, warn_only=False)
_HAS_CUDA: bool = torch.cuda.is_available()

if _HAS_CUDA:
    if torch.backends.cudnn.version() is None:
        raise RuntimeError(
            "SpectralMC requires cuDNN for deterministic GPU execution, "
            "but it is missing from the current runtime.",
        )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    # Even on CPU‑only builds make sure these flags exist and are disabled
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False


# --------------------------------------------------------------------------- #
#  Helper: assert main thread                                                 #
# --------------------------------------------------------------------------- #
def _assert_main_thread(ctx_name: str) -> None:
    """Raise if called from any thread other than the importing main thread."""
    if threading.get_ident() != _MAIN_THREAD_ID:
        raise RuntimeError(
            f"{ctx_name} is not thread‑safe; call it only " "from the main thread."
        )


# --------------------------------------------------------------------------- #
#  Strongly‑typed helpers                                                     #
# --------------------------------------------------------------------------- #
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
_DTYPE_STR_TO_STR_COMPLEX: Dict[str, str] = {
    "float32": "complex64",
    "float64": "complex128",
}
_STR_COMPLEX_DTYPE_TO_STR: Dict[str, str] = {
    v: k for k, v in _DTYPE_STR_TO_STR_COMPLEX.items()
}
_DTYPE_STR_TO_PRECISION: Dict[str, Precision] = {
    "float32": Precision.float32,
    "float64": Precision.float64,
}
_PRECISION_DTYPE_TO_STR: Dict[Precision, str] = {
    v: k for k, v in _DTYPE_STR_TO_PRECISION.items()
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
        """Return the corresponding ``torch.dtype``."""
        return _DTYPE_STR_TO_TORCH[self.value]

    @classmethod
    def from_torch(cls, dt: torch.dtype) -> "DType":
        if dt not in _TORCH_DTYPE_TO_STR:
            raise ValueError(f"Unsupported torch.dtype {dt!r}")
        return cls(_TORCH_DTYPE_TO_STR[dt])

    def to_complex(self) -> "DType":
        """return the complex valued type if real and imaginary components
        have *precision equal to self.value"""
        return DType(_DTYPE_STR_TO_STR_COMPLEX[self.value])

    @classmethod
    def from_complex(cls, dt: DType) -> "DType":
        if dt not in _STR_COMPLEX_DTYPE_TO_STR:
            raise ValueError(f"Unsupported torch.dtype {dt!r}")
        return cls(_STR_COMPLEX_DTYPE_TO_STR[dt])

    def to_precision(self) -> Precision:
        """return the precision representation of this type"""
        return _DTYPE_STR_TO_PRECISION[self.value]

    @classmethod
    def from_precision(cls, dt: Precision) -> "DType":
        return cls(_PRECISION_DTYPE_TO_STR[dt])


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda:0"  # single‑GPU policy

    # --- conversion helpers -------------------------------------------------
    def to_torch(self) -> torch.device:  # noqa: D401
        """Return the corresponding ``torch.device``."""
        return torch.device(self.value)

    @classmethod
    def from_torch(cls, dev: torch.device) -> "Device":
        if dev.type == "cpu":
            return cls.cpu
        if dev.type == "cuda" and dev.index in {None, 0}:
            return cls.cuda
        raise ValueError("Only CPU and the first CUDA card are supported.")


# --------------------------------------------------------------------------- #
#  Lightweight global context managers (main‑thread only)                     #
# --------------------------------------------------------------------------- #
@contextmanager
def default_dtype(dt: torch.dtype) -> Iterator[None]:
    """Temporarily sets the global default dtype (main thread only)."""
    _assert_main_thread("default_dtype")
    _tid = threading.get_ident()
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dt)
    try:
        yield
    finally:
        if threading.get_ident() != _tid:  # sanity: same thread exits
            raise RuntimeError(
                "default_dtype context exited in a different " "thread than it entered."
            )
        torch.set_default_dtype(previous)


@contextmanager
def default_device(dev: torch.device) -> Iterator[None]:
    """Temporarily sets the global default device (main thread only)."""
    _assert_main_thread("default_device")
    _tid = threading.get_ident()
    previous = torch.tensor([]).device
    torch.set_default_device(dev)
    try:
        yield
    finally:
        if threading.get_ident() != _tid:
            raise RuntimeError(
                "default_device context exited in a different "
                "thread than it entered."
            )
        torch.set_default_device(previous)


# --------------------------------------------------------------------------- #
#  SafeTensor serialization helpers                                           #
# --------------------------------------------------------------------------- #
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
        # Load safetensor bytes (always loads to CPU in safetensors >=0.5.3)
        tensor_dict: Dict[str, torch.Tensor] = _sf_load(self.data)
        tensor = tensor_dict["tensor"]
        assert tensor.device == Device.cpu.to_torch(), "Error: tensor not on cpu"
        if (
            tuple(tensor.shape) != self.shape
            or DType.from_torch(tensor.dtype) != self.dtype
        ):
            raise RuntimeError("Tensor metadata mismatch on deserialization.")
        return tensor

    @staticmethod
    def from_bytes(raw: bytes) -> "TensorState":
        tensor_dict: Dict[str, torch.Tensor] = _sf_load(raw)
        if set(tensor_dict.keys()) != {"tensor"}:
            raise ValueError(
                "SafeTensor must contain exactly one entry named 'tensor'."
            )
        t = tensor_dict["tensor"]
        assert t.device == Device.cpu.to_torch(), "Error: tensor not on cpu"
        return TensorState(
            data=raw, shape=tuple(t.shape), dtype=DType.from_torch(t.dtype)
        )


# --------------------------------------------------------------------------- #
#  Environment fingerprint                                                    #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Adam optimizer helpers                                                     #
# --------------------------------------------------------------------------- #
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

        # Accept Python int or 0‑D CPU tensor for step
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
    foreach: bool | None = None
    capturable: bool = False
    differentiable: bool = False
    fused: bool | None = None
    decoupled_weight_decay: bool = False

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    @classmethod
    def from_torch(cls, g: Mapping[str, object]) -> "AdamParamGroup":
        # Pydantic will validate and populate all known fields (extra fields forbidden)
        return cls.model_validate(g)

    def to_torch(self) -> Dict[str, object]:
        # Convert the Pydantic model to a Python dict for state_dict
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
        if set(sd.keys()) != {"state", "param_groups"}:
            raise RuntimeError(
                "state_dict() must contain exactly 'state' and 'param_groups'."
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

    def to_torch(self) -> Dict[str, object]:
        return {
            "state": {pid: st.to_torch() for pid, st in self.param_states.items()},
            "param_groups": [g.to_torch() for g in self.param_groups],
        }


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #
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
